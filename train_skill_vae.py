import os
import time
from pathlib import Path
import json

import hydra
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from models.skill_vae import SkillVAE
from utils.skill_dataset import create_skill_data_loaders
from utils.data import load_tensordict
from utils.seed import set_seed

# Set up plotting style
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)


def train_skill_vae(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    lr=1e-4,
    beta=1.0,
    alpha=1.0,
    gradient_clip=1.0,
    eval_every=10,
    save_every=25,
    patience=20,
    min_delta=1e-4,
    output_dir=None,
    wandb_run=None
):
    """
    Train the skill VAE model.
    
    Args:
        model: SkillVAE model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs to train
        lr: Learning rate
        beta: Weight for KL divergence term
        alpha: Weight for temporal predictability loss
        gradient_clip: Gradient clipping value
        eval_every: Evaluate every N epochs
        save_every: Save model every N epochs
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        output_dir: Directory to save outputs
        wandb_run: Wandb run for logging
        
    Returns:
        Trained model and training history
    """
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training history
    history = {
        'train_total_loss': [],
        'train_vae_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_temporal_loss': [],
        'val_total_loss': [],
        'val_vae_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_temporal_loss': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Training SkillVAE for {num_epochs} epochs...")
    print(f"Beta (KL weight): {beta}, Alpha (temporal weight): {alpha}")
    print(f"Learning rate: {lr}, Gradient clip: {gradient_clip}")
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = {
            'total': 0, 'vae': 0, 'recon': 0, 'kl': 0, 'temporal': 0
        }
        
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} (Train)",
            leave=False
        )
        
        for batch_idx, batch in enumerate(train_progress):
            # Move data to device
            obs1 = batch['obs1'].to(device)
            actions1 = batch['actions1'].to(device)
            obs2 = batch['obs2'].to(device)
            actions2 = batch['actions2'].to(device)
            time_diff = batch['time_diff'].to(device)
            
            optimizer.zero_grad()
            
            # Compute total loss
            losses = model.compute_total_loss(
                obs1, actions1, obs2, actions2, time_diff,
                beta=beta, alpha=alpha
            )
            
            total_loss = losses['total_loss']
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Accumulate losses
            train_losses['total'] += total_loss.item()
            train_losses['vae'] += losses['vae_loss'].item()
            train_losses['recon'] += losses['recon_loss'].item()
            train_losses['kl'] += losses['kl_loss'].item()
            train_losses['temporal'] += losses['temporal_loss'].item()
            
            # Update progress bar
            train_progress.set_postfix({
                'Total': f"{total_loss.item():.4f}",
                'VAE': f"{losses['vae_loss'].item():.4f}",
                'Temporal': f"{losses['temporal_loss'].item():.4f}"
            })
            
            # Log to wandb
            if wandb_run and batch_idx % 50 == 0:
                wandb_run.log({
                    'train/batch_total_loss': total_loss.item(),
                    'train/batch_vae_loss': losses['vae_loss'].item(),
                    'train/batch_recon_loss': losses['recon_loss'].item(),
                    'train/batch_kl_loss': losses['kl_loss'].item(),
                    'train/batch_temporal_loss': losses['temporal_loss'].item(),
                    'epoch': epoch
                })
        
        # Average training losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Store training losses
        history['train_total_loss'].append(train_losses['total'])
        history['train_vae_loss'].append(train_losses['vae'])
        history['train_recon_loss'].append(train_losses['recon'])
        history['train_kl_loss'].append(train_losses['kl'])
        history['train_temporal_loss'].append(train_losses['temporal'])
        
        # Validation phase
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            model.eval()
            val_losses = {
                'total': 0, 'vae': 0, 'recon': 0, 'kl': 0, 'temporal': 0
            }
            
            val_progress = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} (Val)",
                leave=False
            )
            
            with torch.no_grad():
                for batch in val_progress:
                    obs1 = batch['obs1'].to(device)
                    actions1 = batch['actions1'].to(device)
                    obs2 = batch['obs2'].to(device)
                    actions2 = batch['actions2'].to(device)
                    time_diff = batch['time_diff'].to(device)
                    
                    # Compute losses
                    losses = model.compute_total_loss(
                        obs1, actions1, obs2, actions2, time_diff,
                        beta=beta, alpha=alpha
                    )
                    
                    # Accumulate losses
                    val_losses['total'] += losses['total_loss'].item()
                    val_losses['vae'] += losses['vae_loss'].item()
                    val_losses['recon'] += losses['recon_loss'].item()
                    val_losses['kl'] += losses['kl_loss'].item()
                    val_losses['temporal'] += losses['temporal_loss'].item()
            
            # Average validation losses
            for key in val_losses:
                val_losses[key] /= len(val_loader)
            
            # Store validation losses
            history['val_total_loss'].append(val_losses['total'])
            history['val_vae_loss'].append(val_losses['vae'])
            history['val_recon_loss'].append(val_losses['recon'])
            history['val_kl_loss'].append(val_losses['kl'])
            history['val_temporal_loss'].append(val_losses['temporal'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}, VAE: {train_losses['vae']:.4f}, "
                  f"Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f}, "
                  f"Temporal: {train_losses['temporal']:.4f}")
            print(f"  Val   - Total: {val_losses['total']:.4f}, VAE: {val_losses['vae']:.4f}, "
                  f"Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f}, "
                  f"Temporal: {val_losses['temporal']:.4f}")
            
            # Log to wandb
            if wandb_run:
                wandb_run.log({
                    'train/epoch_total_loss': train_losses['total'],
                    'train/epoch_vae_loss': train_losses['vae'],
                    'train/epoch_recon_loss': train_losses['recon'],
                    'train/epoch_kl_loss': train_losses['kl'],
                    'train/epoch_temporal_loss': train_losses['temporal'],
                    'val/epoch_total_loss': val_losses['total'],
                    'val/epoch_vae_loss': val_losses['vae'],
                    'val/epoch_recon_loss': val_losses['recon'],
                    'val/epoch_kl_loss': val_losses['kl'],
                    'val/epoch_temporal_loss': val_losses['temporal'],
                    'epoch': epoch
                })
            
            # Early stopping check
            if val_losses['total'] < best_val_loss - min_delta:
                best_val_loss = val_losses['total']
                epochs_without_improvement = 0
                
                # Save best model
                if output_dir:
                    best_model_path = os.path.join(output_dir, 'best_model.pt')
                    torch.save(model.state_dict(), best_model_path)
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save model periodically
        if output_dir and epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final model and history
    if output_dir:
        final_model_path = os.path.join(output_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    return model, history


def plot_training_curves(history, output_path):
    """Plot training curves for skill VAE."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Total loss
    axes[0, 0].plot(history['train_total_loss'], label='Train', linewidth=2)
    if history['val_total_loss']:
        eval_epochs = np.arange(0, len(history['train_total_loss']), 
                               len(history['train_total_loss']) // len(history['val_total_loss']))[:len(history['val_total_loss'])]
        axes[0, 0].plot(eval_epochs, history['val_total_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # VAE loss
    axes[0, 1].plot(history['train_vae_loss'], label='Train', linewidth=2)
    if history['val_vae_loss']:
        axes[0, 1].plot(eval_epochs, history['val_vae_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('VAE Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 2].plot(history['train_recon_loss'], label='Train', linewidth=2)
    if history['val_recon_loss']:
        axes[0, 2].plot(eval_epochs, history['val_recon_loss'], label='Val', linewidth=2)
    axes[0, 2].set_title('Reconstruction Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # KL loss
    axes[1, 0].plot(history['train_kl_loss'], label='Train', linewidth=2)
    if history['val_kl_loss']:
        axes[1, 0].plot(eval_epochs, history['val_kl_loss'], label='Val', linewidth=2)
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Temporal loss
    axes[1, 1].plot(history['train_temporal_loss'], label='Train', linewidth=2)
    if history['val_temporal_loss']:
        axes[1, 1].plot(eval_epochs, history['val_temporal_loss'], label='Val', linewidth=2)
    axes[1, 1].set_title('Temporal Predictability Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    # Remove spines
    for ax in axes.flat:
        if ax.has_data():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_reconstructions(model, test_loader, device, output_path, num_samples=4):
    """Visualize action reconstructions."""
    model.eval()
    
    # Get a batch from test loader
    batch = next(iter(test_loader))
    obs1 = batch['obs1'][:num_samples].to(device)
    actions1 = batch['actions1'][:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(obs1, actions1)
        reconstructed_actions = outputs['reconstructed_actions']
    
    # Convert to numpy
    original_actions = actions1.cpu().numpy()
    recon_actions = reconstructed_actions.cpu().numpy()
    
    # Plot reconstructions
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Plot action dimensions
        seq_len = original_actions.shape[1]
        time_steps = np.arange(seq_len)
        
        for dim in range(min(4, original_actions.shape[2])):  # Plot up to 4 action dimensions
            axes[i].plot(time_steps, original_actions[i, :, dim], 
                        label=f'Original Action Dim {dim}', linestyle='-', alpha=0.7)
            axes[i].plot(time_steps, recon_actions[i, :, dim], 
                        label=f'Reconstructed Action Dim {dim}', linestyle='--', alpha=0.7)
        
        axes[i].set_title(f'Sample {i+1}: Action Reconstruction')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Action Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_latent_space(model, test_loader, device, output_path, num_samples=1000):
    """Analyze the learned latent space."""
    model.eval()
    
    latent_codes = []
    time_diffs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(latent_codes) * batch['obs1'].shape[0] >= num_samples:
                break
                
            obs1 = batch['obs1'].to(device)
            actions1 = batch['actions1'].to(device)
            time_diff = batch['time_diff'].to(device)
            
            # Encode to latent space
            z_mean, z_logvar = model.encode(obs1, actions1)
            latent_codes.append(z_mean.cpu().numpy())
            time_diffs.append(time_diff.cpu().numpy())
    
    # Concatenate all latent codes
    latent_codes = np.concatenate(latent_codes, axis=0)[:num_samples]
    time_diffs = np.concatenate(time_diffs, axis=0)[:num_samples]
    
    # Create subplots for analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Latent space distribution (first 2 dimensions)
    axes[0, 0].scatter(latent_codes[:, 0], latent_codes[:, 1], 
                       c=time_diffs, cmap='viridis', alpha=0.6)
    axes[0, 0].set_title('Latent Space (First 2 Dims) Colored by Time Diff')
    axes[0, 0].set_xlabel('Latent Dim 0')
    axes[0, 0].set_ylabel('Latent Dim 1')
    cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar.set_label('Time Difference')
    
    # 2. Latent dimension variances
    latent_vars = np.var(latent_codes, axis=0)
    axes[0, 1].bar(range(len(latent_vars)), latent_vars)
    axes[0, 1].set_title('Latent Dimension Variances')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Variance')
    
    # 3. Time difference distribution
    axes[1, 0].hist(time_diffs, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Time Difference Distribution')
    axes[1, 0].set_xlabel('Time Difference')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Latent code norms
    latent_norms = np.linalg.norm(latent_codes, axis=1)
    axes[1, 1].hist(latent_norms, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Latent Code Norm Distribution')
    axes[1, 1].set_xlabel('L2 Norm')
    axes[1, 1].set_ylabel('Frequency')
    
    # Remove spines
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


@hydra.main(config_path="config", config_name="skill_vae", version_base=None)
def main(cfg: DictConfig):
    # Set random seed
    set_seed(cfg.random_seed)
    
    # Get dataset name for output directory naming
    dataset_name = Path(cfg.data.data_path).stem
    cfg.output.model_dir_name = cfg.output.model_dir_name.replace("DATASET_NAME", dataset_name)
    
    print("\n" + "=" * 50)
    print("Training Skill VAE with Temporal Predictability")
    print("=" * 50)
    
    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb_run = None
    if cfg.wandb.use_wandb:
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"skill_vae_{dataset_name}_latent{cfg.model.latent_dim}_seg{cfg.data.segment_length}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes
        )
        print(f"Wandb initialized: {wandb_run.name}")
    
    # Print model architecture after wandb initialization
    if wandb_run is not None:
        print("\n" + "=" * 50)
        print("MODEL ARCHITECTURE AFTER WANDB INITIALIZATION:")
        print("=" * 50)
        print("Wandb initialized - model architecture will be printed after model creation")
    
    # Load data
    print(f"\nLoading data from: {cfg.data.data_path}")
    data = load_tensordict(cfg.data.data_path)
    
    # Create data loaders
    print("Creating data loaders...")
    dataloaders = create_skill_data_loaders(
        data=data,
        segment_length=cfg.data.segment_length,
        max_temporal_diff=cfg.data.max_temporal_diff,
        batch_size=cfg.training.batch_size,
        train_ratio=cfg.training.train_ratio,
        val_ratio=cfg.training.val_ratio,
        obs_key=cfg.data.obs_key,
        action_key=cfg.data.action_key,
        normalize_obs=cfg.data.normalize_obs,
        norm_method=cfg.data.norm_method,
        num_workers=cfg.training.num_workers,
        seed=cfg.random_seed
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    dataset = dataloaders['dataset']
    
    # Get dimensions
    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim
    
    print(f"Data dimensions - Obs: {obs_dim}, Action: {action_dim}")
    print(f"Segment length: {cfg.data.segment_length}")
    
    # Initialize model
    print("\nInitializing SkillVAE model...")
    model = SkillVAE(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model architecture after model initialization
    print("\n" + "=" * 50)
    print("MODEL ARCHITECTURE AFTER MODEL INITIALIZATION:")
    print("=" * 50)
    print("SkillVAE Model:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50)
    
    # Create output directory
    output_dir = os.path.join(cfg.output.output_dir, cfg.output.model_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    model, history = train_skill_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.model.lr,
        beta=cfg.model.beta,
        alpha=cfg.model.alpha,
        gradient_clip=cfg.training.gradient_clip,
        eval_every=cfg.training.eval_every,
        save_every=cfg.training.save_every,
        patience=cfg.training.patience,
        min_delta=cfg.training.min_delta,
        output_dir=output_dir,
        wandb_run=wandb_run
    )
    
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Plot training curves
    print("Generating training curves...")
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(history, curves_path)
    
    if wandb_run:
        wandb_run.log({"training_curves": wandb.Image(curves_path)})
    
    # Generate visualizations
    if cfg.output.save_reconstructions:
        print("Generating reconstruction visualizations...")
        recon_path = os.path.join(output_dir, 'reconstructions.png')
        visualize_reconstructions(model, test_loader, device, recon_path)
        
        if wandb_run:
            wandb_run.log({"reconstructions": wandb.Image(recon_path)})
    
    if cfg.output.save_latent_analysis:
        print("Generating latent space analysis...")
        latent_path = os.path.join(output_dir, 'latent_analysis.png')
        analyze_latent_space(model, test_loader, device, latent_path)
        
        if wandb_run:
            wandb_run.log({"latent_analysis": wandb.Image(latent_path)})
    
    print(f"\nSkill VAE training complete!")
    print(f"Results saved to: {output_dir}")
    
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main() 