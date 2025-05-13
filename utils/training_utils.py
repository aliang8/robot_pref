import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import optim
from copy import deepcopy

from utils.dataset_utils import bradley_terry_loss, PreferenceDataset


def log_wandb_metrics(train_loss, val_loss, epoch, lr=None, wandb=None):
    """Log training metrics to wandb."""
    if not wandb or not wandb.run:
        return
    
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
    }
    
    # Only add validation loss if it exists
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    
    if lr is not None:
        metrics["learning_rate"] = lr
    
    # Log metrics to wandb
    wandb.log(metrics)


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4, wandb=None, is_ensemble=False):
    """Unified training function for both single reward models and ensembles.
    
    Args:
        model: Model to train (either single model or ensemble)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (can be None if no validation is needed)
        device: Device to run training on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        wandb: Optional wandb instance for logging
        is_ensemble: Whether the model is an ensemble (for logging purposes only)
    
    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    max_grad_norm = 1.0  # For gradient clipping
    
    # Check if we have validation data
    has_validation = val_loader is not None and len(val_loader) > 0
    
    # Initialize weights properly for single models
    if not is_ensemble:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    print(f"Training {'ensemble' if is_ensemble else 'reward'} model with gradient clipping at {max_grad_norm}...")
    print(f"Using learning rate: {lr}")
    if not has_validation:
        print("No validation data provided - will train without validation")
    
    # Clear CUDA cache before training 
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        # Use a progress bar with ETA
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", 
                           leave=False, dynamic_ncols=True)
        
        for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(progress_bar):
            # Move data to device
            obs1 = obs1.to(device, non_blocking=True)
            actions1 = actions1.to(device, non_blocking=True)
            obs2 = obs2.to(device, non_blocking=True)
            actions2 = actions2.to(device, non_blocking=True)
            pref = pref.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Compute rewards for both segments using the unified interface
            reward1, reward2 = model.compute_paired_rewards(obs1, actions1, obs2, actions2)
            
            # Compute losses for all models at once using broadcasting
            # reward1 and reward2 have shape [num_models, batch_size]
            # pref has shape [batch_size]
            losses = bradley_terry_loss(reward1, reward2, pref)
            
            # Average loss across all models
            batch_loss = losses.mean()
            
            batch_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            train_loss += batch_loss.item()
            progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation (if we have validation data)
        avg_val_loss = None
        if has_validation:
            model.eval()
            val_loss = 0
            
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", 
                              leave=False, dynamic_ncols=True)
            
            with torch.no_grad():
                for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(val_progress):
                    # Move data to device
                    obs1 = obs1.to(device, non_blocking=True)
                    actions1 = actions1.to(device, non_blocking=True)
                    obs2 = obs2.to(device, non_blocking=True)
                    actions2 = actions2.to(device, non_blocking=True)
                    pref = pref.to(device, non_blocking=True)
                    
                    # Compute rewards for both segments
                    reward1, reward2 = model.compute_paired_rewards(obs1, actions1, obs2, actions2)
                    
                    # Compute losses for all models at once
                    losses = bradley_terry_loss(reward1, reward2, pref)
                    
                    # Average loss across all models
                    batch_loss = losses.mean()
                    
                    val_loss += batch_loss.item()
                    val_progress.set_postfix({"loss": f"{batch_loss.item():.4f}"})
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())
        else:
            # If no validation, just use the training loss for model selection
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_model_state = deepcopy(model.state_dict())
            
            # Add None to val_losses to maintain epoch alignment
            val_losses.append(None)
        
        # Log to wandb
        if wandb:
            log_wandb_metrics(avg_train_loss, avg_val_loss, epoch, lr, wandb)
        
        # Print progress (for ensemble, only print occasionally)
        if not is_ensemble or (epoch + 1) % 5 == 0 or epoch == 0:
            if has_validation:
                print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={lr:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, lr={lr:.6f}")
    
    # Load best model if we found one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("WARNING: No best model found (all had NaN losses). Using the final model state.")
    
    # Filter out NaN values for plotting
    train_losses_clean = [loss for loss in train_losses if not np.isnan(loss)]
    val_losses_clean = [loss for loss in val_losses if loss is not None and not np.isnan(loss)]
    
    # Plot training curve if we have non-NaN values and not an ensemble
    if not is_ensemble and train_losses_clean:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_clean, label='Train Loss')
        if val_losses_clean:
            plt.plot(val_losses_clean, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reward Model Training')
        plt.legend()
        plt.savefig('reward_model_training.png', dpi=300, bbox_inches='tight')
        
        # Log the plot to wandb
        if wandb and wandb.run:
            wandb.log({"training_curve": wandb.Image('reward_model_training.png')})
            
        plt.close()
    
    # Move model back to CPU if using CUDA to save memory for ensembles
    if is_ensemble and device.type == 'cuda':
        model = model.cpu()
    
    return model, train_losses, val_losses


# For backwards compatibility
def train_reward_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4, wandb=None):
    """Wrapper around train_model for single reward models."""
    return train_model(model, train_loader, val_loader, device, num_epochs, lr, wandb, is_ensemble=False) 