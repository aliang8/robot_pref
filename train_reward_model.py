import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    create_segments,
    sample_segment_pairs
)

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)  # For multi-GPU
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up CUDA memory management for better stability
if torch.cuda.is_available():
    # Reduce memory fragmentation
    torch.cuda.empty_cache()
    # Use more aggressive memory caching if available (PyTorch 1.11+)
    if hasattr(torch.cuda, 'memory_stats'):
        print("Enabling memory_stats for better CUDA memory management")
        torch.cuda.memory_stats(device=None)
    # Set memory allocation strategy to avoid fragmentation
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        # Use 80% of available memory to leave room for system
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        print("Set CUDA memory fraction to 80%")

class StateActionRewardModel(nn.Module):
    """MLP-based reward model that takes state and action as input."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], output_dim=1):
        super(StateActionRewardModel, self).__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = state_dim + action_dim  # Concatenated state and action
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Add layer normalization for stability
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with a better scheme for stability."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, obs, action):
        """Forward pass using observation and action."""
        # Check for NaN inputs
        if torch.isnan(obs).any() or torch.isnan(action).any():
            print("WARNING: NaN detected in model inputs")
            
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=-1)
        
        # Apply model
        result = self.model(x).squeeze(-1)  # Squeeze to remove last dimension if batch size = 1
        
        # Check for NaN outputs
        if torch.isnan(result).any():
            print("WARNING: NaN detected in model outputs")
            
        return result
    
    def logpdf(self, obs, action, reward):
        """Compute log probability of reward given observation and action."""
        # Assuming Gaussian distribution with unit variance
        pred_reward = self(obs, action)
        log_prob = -0.5 * ((pred_reward - reward) ** 2) - 0.5 * np.log(2 * np.pi)
        return log_prob

class SegmentRewardModel(nn.Module):
    """Model that computes reward for a segment of observation-action pairs."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(SegmentRewardModel, self).__init__()
        self.reward_model = StateActionRewardModel(state_dim, action_dim, hidden_dims)
    
    def forward(self, observations, actions):
        """Compute reward for a sequence of observation-action pairs."""
        # Handle both single segments and batches
        if observations.dim() == 2:  # Single segment (seq_len, obs_dim)
            # Observations and actions should be the same length now
            if observations.size(0) != actions.size(0):
                # If they're not the same, adjust them
                min_len = min(observations.size(0), actions.size(0))
                observations = observations[:min_len]
                actions = actions[:min_len]
                
            # Process all steps in the segment at once (vectorized)
            batch_obs = observations.unsqueeze(0)  # Add batch dim
            batch_actions = actions.unsqueeze(0)
            
            # Concatenate observations and actions along the feature dimension
            combined_inputs = torch.cat([
                batch_obs.reshape(-1, batch_obs.size(-1)), 
                batch_actions.reshape(-1, batch_actions.size(-1))
            ], dim=1)
            
            # Process through the reward model
            rewards = self.reward_model.model(combined_inputs)
            rewards = rewards.reshape(batch_obs.size(0), -1)
            return rewards.sum(1)[0]  # Sum over sequence length and remove batch dim
        
        elif observations.dim() == 3:  # Batch of segments (batch_size, seq_len, obs_dim)
            batch_size = observations.size(0)
            
            # Observations and actions should be the same length now
            if observations.size(1) != actions.size(1):
                # If they're not the same, adjust them
                min_len = min(observations.size(1), actions.size(1))
                observations = observations[:, :min_len]
                actions = actions[:, :min_len]
            
            # Flatten the batch and sequence dimensions
            flat_obs = observations.reshape(-1, observations.size(-1))
            flat_actions = actions.reshape(-1, actions.size(-1))
            
            # Concatenate observations and actions along the feature dimension
            combined_inputs = torch.cat([flat_obs, flat_actions], dim=1)
            
            # Process through the reward model
            flat_rewards = self.reward_model.model(combined_inputs)
            rewards = flat_rewards.reshape(batch_size, -1)
            return rewards.sum(1)  # Sum over sequence length
        
        else:
            raise ValueError(f"Unexpected input shape: observations {observations.shape}, actions {actions.shape}")
    
    def logpdf(self, observations, actions, rewards):
        """Compute log probability of rewards given observations and actions."""
        # Vectorized implementation
        if observations.dim() == 2:  # Single segment
            segment_reward = self(observations, actions)
            return self.reward_model.logpdf(observations.mean(0, keepdim=True), 
                                          actions.mean(0, keepdim=True) if len(actions) > 0 else actions[0:1], 
                                          rewards)
        elif observations.dim() == 3:  # Batch of segments
            # Compute mean observation and action for each segment
            mean_obs = observations.mean(1)  # Average over sequence length
            mean_actions = actions.mean(1) if actions.size(1) > 0 else actions[:, 0]
            return self.reward_model.logpdf(mean_obs, mean_actions, rewards)
        else:
            raise ValueError(f"Unexpected input shape for logpdf")

class PreferenceDataset(Dataset):
    """Dataset for segment preference pairs."""
    def __init__(self, data, segment_pairs, segment_indices, preferences):
        # Ensure all data is on CPU for multi-process loading
        self.data = data.cpu() if isinstance(data, torch.Tensor) else data
        self.segment_pairs = segment_pairs
        self.segment_indices = segment_indices
        self.preferences = preferences
        
        # Determine the data length for bounds checking
        if isinstance(self.data, dict):
            # For dictionary data, use the observation tensor length
            if "obs" in self.data:
                self.data_length = len(self.data["obs"])
            elif "state" in self.data:
                self.data_length = len(self.data["state"])
            else:
                raise ValueError("Data dictionary must contain 'obs' or 'state' key")
        else:
            # For tensor data, use its length
            self.data_length = len(self.data)
        
        print(f"Dataset initialized with data length: {self.data_length}")
    
    def __len__(self):
        return len(self.segment_pairs)
    
    def __getitem__(self, idx):
        seg_idx1, seg_idx2 = self.segment_pairs[idx]
        start1, end1 = self.segment_indices[seg_idx1]
        start2, end2 = self.segment_indices[seg_idx2]
        
        # Check bounds against the correct data length
        if start1 < 0 or end1 >= self.data_length or start1 > end1:
            raise IndexError(f"Invalid segment indices for segment 1: {start1}:{end1}, data length: {self.data_length}")
        
        if start2 < 0 or end2 >= self.data_length or start2 > end2:
            raise IndexError(f"Invalid segment indices for segment 2: {start2}:{end2}, data length: {self.data_length}")
        
        # Get data for segments (handle dictionary data correctly)
        if isinstance(self.data, dict):
            # Extract from dictionary (TensorDict)
            obs_key = "obs" if "obs" in self.data else "state"
            action_key = "action"
            
            # Safely extract data
            obs1 = self.data[obs_key][start1:end1+1].clone().detach()
            actions1 = self.data[action_key][start1:end1].clone().detach()
            obs2 = self.data[obs_key][start2:end2+1].clone().detach()
            actions2 = self.data[action_key][start2:end2].clone().detach()
        else:
            # Extract directly from tensor
            obs1 = self.data[start1:end1+1].clone().detach()
            actions1 = self.data[start1:end1].clone().detach()
            obs2 = self.data[start2:end2+1].clone().detach()
            actions2 = self.data[start2:end2].clone().detach()
        
        # Make observations and actions have the same length (important for concatenation)
        # Method 1: Remove the last observation
        obs1 = obs1[:-1]
        obs2 = obs2[:-1]
        
        # Ensure shapes are compatible
        if obs1.shape[0] != actions1.shape[0]:
            # Adjust shapes if somehow they're still mismatched
            min_len = min(obs1.shape[0], actions1.shape[0])
            obs1 = obs1[:min_len]
            actions1 = actions1[:min_len]
        
        if obs2.shape[0] != actions2.shape[0]:
            min_len = min(obs2.shape[0], actions2.shape[0])
            obs2 = obs2[:min_len]
            actions2 = actions2[:min_len]
        
        # Convert preference to tensor
        pref = torch.tensor(self.preferences[idx], dtype=torch.long)
        
        # Handle NaN values
        if torch.isnan(obs1).any() or torch.isnan(actions1).any() or torch.isnan(obs2).any() or torch.isnan(actions2).any():
            obs1 = torch.nan_to_num(obs1, nan=0.0)
            actions1 = torch.nan_to_num(actions1, nan=0.0)
            obs2 = torch.nan_to_num(obs2, nan=0.0)
            actions2 = torch.nan_to_num(actions2, nan=0.0)
        
        return obs1, actions1, obs2, actions2, pref

def bradley_terry_loss(rewards1, rewards2, preferences):
    """
    Compute the Bradley-Terry preference learning loss.
    
    Args:
        rewards1: Predicted rewards for the first segments in each pair
        rewards2: Predicted rewards for the second segments in each pair
        preferences: Labels indicating which segment is preferred (1 or 2)
    
    Returns:
        Loss value
    """
    # Convert preferences to probabilities (1 = first segment preferred, 2 = second segment preferred)
    prefs = (preferences == 1).float()
    
    # Compute probability that segment1 is preferred over segment2 using the Bradley-Terry model
    # Add a small epsilon for numerical stability
    eps = 1e-6
    logits = torch.clamp(rewards1 - rewards2, min=-50.0, max=50.0)  # Clip logits to prevent overflow
    pred_probs = torch.sigmoid(logits)
    
    # Use a more numerically stable binary cross-entropy loss
    loss = -torch.mean(prefs * torch.log(pred_probs + eps) + (1 - prefs) * torch.log(1 - pred_probs + eps))
    
    # Check for NaN loss and return a fallback value if needed
    if torch.isnan(loss).any():
        print("WARNING: NaN loss detected. Using fallback MSE loss.")
        # Use a fallback MSE loss instead
        mse_loss = torch.mean((rewards1 - rewards2 - (2 * prefs - 1))**2)
        return mse_loss
    
    return loss

def log_wandb_metrics(train_loss, val_loss, epoch, lr=None):
    """Log training metrics to wandb."""
    if not wandb.run:
        return
    
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    
    if lr is not None:
        metrics["learning_rate"] = lr
    
    wandb.log(metrics)

def train_reward_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4):
    """Train the reward model using Bradley-Terry loss with wandb logging."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Add weight decay for regularization
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    max_grad_norm = 1.0  # For gradient clipping
    
    # Initialize weights properly - this can help prevent NaN issues at the start
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
    
    print(f"Training reward model with gradient clipping at {max_grad_norm}...")
    
    # Clear CUDA cache before training 
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        nan_batches = 0
        
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

            # Skip batches with NaN values
            if (torch.isnan(obs1).any() or torch.isnan(actions1).any() or 
                torch.isnan(obs2).any() or torch.isnan(actions2).any()):
                nan_batches += 1
                continue
            
            optimizer.zero_grad(set_to_none=True)
            
            # Compute rewards
            reward1 = model(obs1, actions1)
            reward2 = model(obs2, actions2)
            
            # Skip if NaN rewards
            if torch.isnan(reward1).any() or torch.isnan(reward2).any():
                nan_batches += 1
                continue
            
            loss = bradley_terry_loss(reward1, reward2, pref)
            
            # Skip problematic batches
            if torch.isnan(loss).any():
                nan_batches += 1
                continue
                
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Check for NaN gradients
            nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_grad = True
                    break
            
            if nan_grad:
                nan_batches += 1
                continue
            
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        if nan_batches > 0:
            print(f"WARNING: {nan_batches}/{len(train_loader)} training batches skipped due to NaN issues")
            
        avg_train_loss = train_loss / (len(train_loader) - nan_batches) if len(train_loader) > nan_batches else float('nan')
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_nan_batches = 0
        
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
                
                # Skip batches with NaN values
                if (torch.isnan(obs1).any() or torch.isnan(actions1).any() or 
                    torch.isnan(obs2).any() or torch.isnan(actions2).any()):
                    val_nan_batches += 1
                    continue
                
                reward1 = model(obs1, actions1)
                reward2 = model(obs2, actions2)
                
                # Skip if NaN rewards
                if torch.isnan(reward1).any() or torch.isnan(reward2).any():
                    val_nan_batches += 1
                    continue
                
                loss = bradley_terry_loss(reward1, reward2, pref)
                
                if torch.isnan(loss).any():
                    val_nan_batches += 1
                    continue
                    
                val_loss += loss.item()
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if val_nan_batches > 0:
            print(f"WARNING: {val_nan_batches}/{len(val_loader)} validation batches skipped due to NaN issues")
            
        avg_val_loss = val_loss / (len(val_loader) - val_nan_batches) if len(val_loader) > val_nan_batches else float('nan')
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model (if not NaN)
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Log to wandb
        log_wandb_metrics(avg_train_loss, avg_val_loss, epoch, scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
    
    # If we had NaN issues the whole time, try with a simpler model
    if all(np.isnan(loss) for loss in val_losses):
        print("All validation losses were NaN. Training a simpler model with fewer parameters...")
        # Initialize a simpler model with smaller hidden layers and more regularization
        simpler_model = SegmentRewardModel(model.reward_model.model[0].in_features // 2, 
                                          model.reward_model.model[0].in_features // 2, 
                                          hidden_dims=[64, 32])
        simpler_model = simpler_model.to(device)
        
        # Log to wandb
        if wandb.run:
            wandb.log({"model_simplified": True})
            
        # Train the simpler model with more regularization and lower learning rate
        return train_reward_model(simpler_model, train_loader, val_loader, device, num_epochs, lr=lr/10)
    
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
        if wandb.run:
            wandb.log({"training_curve": wandb.Image('reward_model_training.png')})
            
        plt.close()
    
    return model, train_losses, val_losses

def load_preferences_data(file_path):
    """Load preference data saved from collect_cluster_preferences.py.
    
    Args:
        file_path: Path to saved preference data pickle file
        
    Returns:
        Tuple of (segment_pairs, segment_indices, preferences, segments)
    """
    print(f"Loading preference data from {file_path}")
    try:
        with open(file_path, 'rb') as f:
            pref_data = pickle.load(f)
        
        # Extract the necessary components
        segment_pairs = pref_data['segment_pairs']
        segment_indices = pref_data['segment_indices']
        
        # Get preferences - different files might have different keys
        if 'preference_labels' in pref_data:
            print("Using preference_labels from dataset")
            preferences = pref_data['preference_labels']
        elif 'human_preferences' in pref_data and len(pref_data['human_preferences']) > 0:
            print("Using collected human preferences")
            preferences = pref_data['human_preferences']
        elif 'synthetic_preferences' in pref_data:
            print("Using synthetic preferences (based on rewards)")
            preferences = pref_data['synthetic_preferences']
        else:
            print("Could not find preferences in dataset! Available keys:", list(pref_data.keys()))
            raise KeyError("No preference data found in file")
        
        # Check if segments are included in the preference data
        segments = None
        if 'segments' in pref_data:
            segments = pref_data['segments']
            print(f"Found {len(segments)} segments in preference data")
        elif 'original_segments' in pref_data:
            segments = pref_data['original_segments']
            print(f"Found {len(segments)} original segments in preference data")
        
        print(f"Loaded {len(segment_pairs)} preference pairs")
        return segment_pairs, segment_indices, preferences, segments
    except Exception as e:
        print(f"Error loading preference data: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to load preference data from {file_path}: {e}")

@hydra.main(config_path="config/train_reward_model", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Train a state-action reward model using BT loss with Hydra config."""
    print("\n" + "=" * 50)
    print("Training reward model with Bradley-Terry preference learning")
    print("=" * 50)
    
    # Print config for visibility
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize wandb
    if cfg.wandb.use_wandb:
        # Generate experiment name based on data path
        dataset_name = Path(cfg.data.data_path).stem
        
        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"reward_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes
        )
        print(f"Wandb initialized: {wandb.run.name}")
    
    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    
    # Setup CUDA device
    if cfg.hardware.use_cpu:
        device = torch.device("cpu")
    else:
        cuda_device = f"cuda:{cfg.hardware.gpu}" if torch.cuda.is_available() else "cpu"
        device = torch.device(cuda_device)
    
    print(f"Using device: {device}")
    
    # Determine effective GPU and CPU settings
    effective_num_workers = cfg.training.num_workers
    effective_pin_memory = cfg.training.pin_memory
    
    # Adjust for CPU-only mode
    if device.type == "cpu":
        print("Running in CPU mode")
        effective_pin_memory = False
    
    # Initialize variables
    segments = None
    
    # Check if we have pre-collected preference data
    if hasattr(cfg.data, 'preferences_data_path') and cfg.data.preferences_data_path:
        # Load pre-collected preferences
        segment_pairs, segment_indices, preferences, loaded_segments = load_preferences_data(cfg.data.preferences_data_path)
        segments = loaded_segments  # Use loaded segments if available
    else:
        # Load data
        print(f"Loading data from {cfg.data.data_path}")
        data = load_tensordict(cfg.data.data_path)
        
        # Get observation and action dimensions
        observations = data["obs"] if "obs" in data else data["state"]
        actions = data["action"]
        state_dim = observations.shape[1]
        action_dim = actions.shape[1]
        
        print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")
        
        # Ensure data is on CPU before creating segments to avoid device mismatch issues
        data_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        # Generate segments and preference pairs from scratch
        print(f"Creating segments of length {cfg.data.segment_length}...")
        num_segments = cfg.data.num_segments if cfg.data.num_segments > 0 else None
        segments, segment_indices = create_segments(data_cpu, segment_length=cfg.data.segment_length, max_segments=num_segments)
        
        # Generate preference pairs
        print(f"Generating {cfg.data.num_pairs} preference pairs...")
        segment_pairs, preferences = sample_segment_pairs(
            segments, 
            segment_indices, 
            data_cpu["reward"], 
            n_pairs=cfg.data.num_pairs
        )
    
    # Create dataset
    preference_dataset = PreferenceDataset(
        data_cpu, 
        segment_pairs,
        segment_indices,
        preferences
    )
    
    # Split into train, validation, and test sets
    total_size = len(preference_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Create random splits
    train_dataset, val_dataset, test_dataset = random_split(
        preference_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Log dataset information to wandb
    if cfg.wandb.use_wandb:
        wandb.config.update({
            "dataset": {
                "name": Path(cfg.data.data_path).stem,
                "total_pairs": total_size,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "observation_dim": state_dim,
                "action_dim": action_dim
            }
        })
    
    print(f"Split data: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        # Add prefetch factor for more efficient loading
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.training.batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    print(f"Train pairs: {len(train_dataset)}, Validation pairs: {len(val_dataset)}, Test pairs: {len(test_dataset)}")
    
    # Initialize reward model
    model = SegmentRewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
    
    # Log model info to wandb
    if cfg.wandb.use_wandb:
        # Use a different key name to avoid conflicts with existing 'model' key
        wandb.config.update({
            "model_info": {
                "hidden_dims": cfg.model.hidden_dims,
                "total_parameters": sum(p.numel() for p in model.parameters())
            }
        })
        
        # Log model graph if possible
        if hasattr(wandb, 'watch'):
            wandb.watch(model, log="all")
    
    print(f"Reward model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Start timing the training
    start_time = time.time()
    
    # Train the model
    print("\nTraining reward model...")
    model, train_losses, val_losses = train_reward_model(
        model, train_loader, val_loader, device, 
        num_epochs=cfg.training.num_epochs, lr=cfg.model.lr
    )
    
    # Calculate and print training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_acc = 0
    test_total = 0
    
    with torch.no_grad():
        for obs1, actions1, obs2, actions2, pref in tqdm(test_loader, desc="Testing"):
            # Skip batches with NaN values
            if (torch.isnan(obs1).any() or torch.isnan(actions1).any() or 
                torch.isnan(obs2).any() or torch.isnan(actions2).any()):
                continue
                
            # Move to device
            obs1, actions1 = obs1.to(device), actions1.to(device)
            obs2, actions2 = obs2.to(device), actions2.to(device)
            pref = pref.to(device)
            
            # Get reward predictions
            reward1 = model(obs1, actions1)
            reward2 = model(obs2, actions2)
            
            # Skip if NaN rewards
            if torch.isnan(reward1).any() or torch.isnan(reward2).any():
                continue
                
            # Compute loss
            loss = bradley_terry_loss(reward1, reward2, pref)
            
            # Skip if NaN loss
            if torch.isnan(loss).any():
                continue
                
            test_loss += loss.item()
            
            # Compute accuracy (prediction matches preference)
            pred_pref = (reward1 > reward2).long() + 1  # 1 if reward1 > reward2, 2 otherwise
            test_acc += (pred_pref == pref).sum().item()
            test_total += pref.size(0)
    
    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else float('nan')
    test_accuracy = test_acc / test_total if test_total > 0 else 0
    
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    
    # Log final test results to wandb
    if cfg.wandb.use_wandb:
        wandb.log({
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy
        })
    
    # Save model
    model_path = f"{cfg.output.output_dir}/state_action_reward_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Log as wandb artifact
    if cfg.wandb.use_wandb:
        artifact = wandb.Artifact(f"reward_model_{wandb.run.id}", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    # Save segment info
    segment_info = {
        "segment_length": cfg.data.segment_length,
        "num_segments": len(segments),
        "num_pairs": len(segment_pairs),
        "observation_dim": state_dim,
        "action_dim": action_dim,
        "training_losses": train_losses,
        "validation_losses": val_losses,
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    with open(f"{cfg.output.output_dir}/reward_model_info.pkl", "wb") as f:
        pickle.dump(segment_info, f)
    
    print(f"Model information saved to {cfg.output.output_dir}/reward_model_info.pkl")
    
    # Finish wandb run
    if cfg.wandb.use_wandb and wandb.run:
        wandb.finish()
        
    print("\nReward model training complete!")

if __name__ == "__main__":
    main() 