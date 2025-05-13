import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import optim

from utils.dataset_utils import bradley_terry_loss


def log_wandb_metrics(train_loss, val_loss, epoch, lr=None, wandb=None):
    """Log training metrics to wandb."""
    if not wandb or not wandb.run:
        return
    
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    
    if lr is not None:
        metrics["learning_rate"] = lr
    
    # Log metrics to wandb
    wandb.log(metrics)


def train_reward_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4, wandb=None):
    """Train the reward model using Bradley-Terry loss with wandb logging.
    
    Args:
        model: Reward model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        wandb: Optional wandb instance for logging
    
    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Add weight decay for regularization
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    max_grad_norm = 1.0  # For gradient clipping
    
    # Initialize weights properly - this can help prevent NaN issues at the start
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
    
    print(f"Training reward model with gradient clipping at {max_grad_norm}...")
    print(f"Using constant learning rate: {lr}")
    
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
            obs1, actions1, obs2, actions2, pref = (
                obs1.to(device, non_blocking=True),
                actions1.to(device, non_blocking=True),
                obs2.to(device, non_blocking=True),
                actions2.to(device, non_blocking=True),
                pref.to(device, non_blocking=True)
            )
            
            optimizer.zero_grad(set_to_none=True)
            
            # Compute rewards
            reward1 = model(obs1, actions1)
            reward2 = model(obs2, actions2)
            
            loss = bradley_terry_loss(reward1, reward2, pref)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", 
                           leave=False, dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(val_progress):
                # Move data to device
                obs1, actions1, obs2, actions2, pref = (
                    obs1.to(device, non_blocking=True),
                    actions1.to(device, non_blocking=True),
                    obs2.to(device, non_blocking=True),
                    actions2.to(device, non_blocking=True),
                    pref.to(device, non_blocking=True)
                )
                
                reward1 = model(obs1, actions1)
                reward2 = model(obs2, actions2)
                
                loss = bradley_terry_loss(reward1, reward2, pref)
                
                val_loss += loss.item()
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Log to wandb
        if wandb:
            log_wandb_metrics(avg_train_loss, avg_val_loss, epoch, lr, wandb)
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={lr:.6f}")
    
    # Load best model if we found one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("WARNING: No best model found (all had NaN losses). Using the final model state.")
    
    # Filter out NaN values for plotting
    train_losses_clean = [loss for loss in train_losses if not np.isnan(loss)]
    val_losses_clean = [loss for loss in val_losses if not np.isnan(loss)]
    
    # Plot training curve if we have non-NaN values
    if train_losses_clean and val_losses_clean:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_clean, label='Train Loss')
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
    
    return model, train_losses, val_losses


def train_ensemble_model(state_dim, action_dim, labeled_pairs, segment_indices, labeled_preferences,
                         data, device, num_models=5, hidden_dims=[256, 256], num_epochs=20,
                         fine_tune=False, prev_ensemble=None, fine_tune_lr=5e-5):
    """Train an ensemble of reward models on the labeled data.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        labeled_pairs: List of labeled segment pairs
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        labeled_preferences: List of preferences for labeled pairs
        data: Data dictionary containing observations and actions (should be on CPU)
        device: Device to run training on
        num_models: Number of models in the ensemble
        hidden_dims: Hidden dimensions for each model
        num_epochs: Number of epochs to train each model
        fine_tune: Whether to fine-tune from previous model
        prev_ensemble: Previous ensemble model to fine-tune from
        fine_tune_lr: Learning rate for fine-tuning
        
    Returns:
        ensemble: Trained ensemble model
    """
    from models.reward_models import EnsembleRewardModel
    from utils.dataset_utils import PreferenceDataset, bradley_terry_loss
    
    # Ensure data is on CPU for dataset creation
    if not isinstance(data, dict) or any(isinstance(v, torch.Tensor) and v.device.type != 'cpu' 
                                       for v in data.values()):
        print("Warning: Data should be on CPU for indexing in PreferenceDataset. Converting to CPU...")
        data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    # Create dataset from labeled pairs
    dataset = PreferenceDataset(data, labeled_pairs, segment_indices, labeled_preferences)
    
    # Create or reuse ensemble model
    if fine_tune and prev_ensemble is not None:
        print("Fine-tuning from previous ensemble model")
        ensemble = prev_ensemble
        # Ensure the ensemble has the right number of models
        if len(ensemble.models) != num_models:
            print(f"Warning: Previous ensemble has {len(ensemble.models)} models, but {num_models} requested. "
                 f"Creating new ensemble.")
            ensemble = EnsembleRewardModel(state_dim, action_dim, hidden_dims, num_models)
        lr = fine_tune_lr  # Use lower learning rate for fine-tuning
    else:
        print("Training new ensemble model from scratch")
        ensemble = EnsembleRewardModel(state_dim, action_dim, hidden_dims, num_models)
        lr = 1e-4  # Use standard learning rate for training from scratch
    
    # Move entire ensemble to device at once
    ensemble = ensemble.to(device)
    
    # Create a combined optimizer for all models in the ensemble
    combined_params = list(ensemble.parameters())
    optimizer = optim.Adam(combined_params, lr=lr, weight_decay=1e-4)
    
    # Create train/val split for the dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    
    print(f"Training ensemble of {num_models} models for {num_epochs} epochs (lr={lr})")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        ensemble.train()
        train_loss = 0.0
        
        for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(train_loader):
            # Move data to device
            obs1, actions1 = obs1.to(device), actions1.to(device)
            obs2, actions2 = obs2.to(device), actions2.to(device)
            pref = pref.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass through all models
            batch_loss = 0.0
            for i in range(num_models):
                model = ensemble.models[i]
                
                # Compute rewards
                reward1 = model(obs1, actions1)
                reward2 = model(obs2, actions2)
                
                # Compute loss
                loss = bradley_terry_loss(reward1, reward2, pref)
                batch_loss += loss
            
            # Average loss across models
            batch_loss /= num_models
            
            # Backward pass
            batch_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(combined_params, max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            train_loss += batch_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        ensemble.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(val_loader):
                # Move data to device
                obs1, actions1 = obs1.to(device), actions1.to(device)
                obs2, actions2 = obs2.to(device), actions2.to(device)
                pref = pref.to(device)
                
                # Forward pass through all models
                batch_loss = 0.0
                for i in range(num_models):
                    model = ensemble.models[i]
                    
                    # Compute rewards
                    reward1 = model(obs1, actions1)
                    reward2 = model(obs2, actions2)
                    
                    # Compute loss
                    loss = bradley_terry_loss(reward1, reward2, pref)
                    batch_loss += loss
                
                # Average loss across models
                batch_loss /= num_models
                
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
    
    # Move ensemble back to CPU if using CUDA to save memory
    if device.type == 'cuda':
        ensemble = ensemble.cpu()
    
    return ensemble 