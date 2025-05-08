import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    sample_segments
)

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
    def __init__(self, observations, actions, segment_pairs, segment_indices, preference_labels):
        # Ensure all data is on CPU for multi-process loading
        self.observations = observations.cpu() if isinstance(observations, torch.Tensor) else observations
        self.actions = actions.cpu() if isinstance(actions, torch.Tensor) else actions
        self.segment_pairs = segment_pairs
        self.segment_indices = segment_indices
        self.preference_labels = preference_labels
    
    def __len__(self):
        return len(self.segment_pairs)
    
    def __getitem__(self, idx):
        seg_idx1, seg_idx2 = self.segment_pairs[idx]
        start1, end1 = self.segment_indices[seg_idx1]
        start2, end2 = self.segment_indices[seg_idx2]
        
        # Get data for first segment - with safety checks
        if start1 < 0 or end1 >= len(self.observations) or start1 > end1:
            raise IndexError(f"Invalid segment indices for segment 1: {start1}:{end1}")
        
        # Get data for second segment - with safety checks
        if start2 < 0 or end2 >= len(self.observations) or start2 > end2:
            raise IndexError(f"Invalid segment indices for segment 2: {start2}:{end2}")
        
        # Get data for segments
        obs1 = self.observations[start1:end1+1].clone().detach()
        actions1 = self.actions[start1:end1].clone().detach()
        obs2 = self.observations[start2:end2+1].clone().detach()
        actions2 = self.actions[start2:end2].clone().detach()
        
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
        pref = torch.tensor(self.preference_labels[idx], dtype=torch.long)
        
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

def create_state_action_segments(data, H, max_segments=None):
    """Create H-step segments of observations and actions from data."""
    # Extract episode IDs
    episode_ids = data["episode"].cpu()
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    rewards = data["reward"]
    
    # Move data to CPU for indexing operations
    observations_cpu = observations.cpu()
    actions_cpu = actions.cpu()
    rewards_cpu = rewards.cpu()
    
    # Create clean masks to handle NaN values at episode beginnings - use vectorized operations
    print("Creating masks for NaN values...")
    obs_mask = ~torch.isnan(observations_cpu).any(dim=1)
    action_mask = ~torch.isnan(actions_cpu).any(dim=1)
    reward_mask = ~torch.isnan(rewards_cpu)
    
    # Combined mask for valid transitions
    valid_mask = obs_mask & action_mask & reward_mask
    
    # Count NaNs to report
    total_transitions = len(observations)
    valid_transitions = valid_mask.sum().item()
    print(f"Found {total_transitions - valid_transitions} NaN transitions out of {total_transitions} total transitions")
    
    segments = []
    segment_indices = []  # Store start and end indices of each segment
    
    # Process all episodes at once for faster operation
    print("Finding valid episodes and segments...")
    unique_episodes = torch.unique(episode_ids).tolist()
    print(f"Found {len(unique_episodes)} unique episodes")
    
    # Pre-allocate a list for all potential segments to avoid repeated allocations
    potential_segments = []
    
    for episode_id in tqdm(unique_episodes, desc="Processing episodes"):
        # Get indices for this episode
        ep_indices = torch.where(episode_ids == episode_id)[0]
        
        # For each episode, get valid indices (no NaNs)
        ep_valid = valid_mask[ep_indices]
        
        # Skip if no valid indices
        if not ep_valid.any():
            continue
        
        # Get consecutive valid indices for this episode
        valid_ep_indices = ep_indices[ep_valid].numpy()  # Convert to numpy for faster operations
        
        # Skip episodes with too few valid transitions
        if len(valid_ep_indices) < H:
            continue
        
        # Find consecutive segments using numpy operations (faster than PyTorch for this task)
        # Calculate differences between consecutive indices
        diffs = np.diff(valid_ep_indices)
        # Find where the difference is not 1 (indicating a break in consecutive indices)
        break_points = np.where(diffs > 1)[0]
        
        # Process each consecutive group
        if len(break_points) == 0:  # Single consecutive group
            # We can directly create segments from this group
            for i in range(len(valid_ep_indices) - H + 1):
                start_idx = valid_ep_indices[i]
                end_idx = valid_ep_indices[i + H - 1]
                potential_segments.append((start_idx, end_idx))
        else:
            # Process multiple groups
            start_idx = 0
            for bp in break_points:
                # Check if this group is long enough
                if bp - start_idx + 1 >= H:
                    # Create segments from this group
                    for i in range(start_idx, bp - H + 2):
                        s_idx = valid_ep_indices[i]
                        e_idx = valid_ep_indices[i + H - 1]
                        potential_segments.append((s_idx, e_idx))
                start_idx = bp + 1
            
            # Process the last group
            if len(valid_ep_indices) - start_idx >= H:
                for i in range(start_idx, len(valid_ep_indices) - H + 1):
                    s_idx = valid_ep_indices[i]
                    e_idx = valid_ep_indices[i + H - 1]
                    potential_segments.append((s_idx, e_idx))
    
    # Randomly sample segments if max_segments is specified
    if max_segments is not None and max_segments < len(potential_segments):
        print(f"Sampling {max_segments} segments from {len(potential_segments)} potential segments...")
        sampled_indices = random.sample(range(len(potential_segments)), max_segments)
        segment_indices = [potential_segments[i] for i in sampled_indices]
    else:
        segment_indices = potential_segments
    
    print(f"Created {len(segment_indices)} segments across {len(unique_episodes)} episodes after filtering out NaN transitions")
    
    return segment_indices

def generate_synthetic_preferences(segment_indices, rewards, n_pairs=10000):
    """Generate synthetic preference pairs based on cumulative rewards."""
    n_segments = len(segment_indices)
    print(f"Generating preferences from {n_segments} segments")
    
    # Sample random pairs of segment indices
    pairs = []
    preference_labels = []
    
    # Keep generating pairs until we have enough or max attempts reached
    max_attempts = n_pairs * 5  # Allow more attempts to handle cases with equal rewards
    num_attempts = 0
    
    with tqdm(total=n_pairs, desc="Generating preference pairs") as pbar:
        while len(pairs) < n_pairs and num_attempts < max_attempts:
            num_attempts += 1
            
            # Sample two different segments
            idx1, idx2 = random.sample(range(n_segments), 2)
            
            # Get segment indices
            start1, end1 = segment_indices[idx1]
            start2, end2 = segment_indices[idx2]
            
            # Skip if rewards contain NaN values
            if torch.isnan(rewards[start1:end1+1]).any() or torch.isnan(rewards[start2:end2+1]).any():
                continue
            
            # Calculate cumulative reward for each segment
            reward1 = rewards[start1:end1+1].sum().item()
            reward2 = rewards[start2:end2+1].sum().item()
            
            # If rewards are equal, skip this pair
            if abs(reward1 - reward2) < 1e-6:
                continue
            
            # Add pair to the list
            pairs.append((idx1, idx2))
            
            # Assign preference label (1 if segment1 is preferred, 2 if segment2 is preferred)
            if reward1 > reward2:
                preference_labels.append(1)
            else:
                preference_labels.append(2)
                
            pbar.update(1)
            
            # If we've reached the target number of pairs, we're done
            if len(pairs) >= n_pairs:
                break
    
    print(f"Generated {len(pairs)} preference pairs after {num_attempts} attempts")
    return pairs, preference_labels

def train_reward_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4):
    """Train the reward model using Bradley-Terry loss."""
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
                    nan_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    nan_grad = True
                    break
            
            if not nan_grad:
                optimizer.step()
                train_loss += loss.item()
                
                # Update progress bar with current loss
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Log if there were any NaN issues
        if nan_batches > 0:
            print(f"WARNING: {nan_batches}/{len(train_loader)} batches skipped due to NaN issues")
        
        avg_train_loss = train_loss / (len(train_loader) - nan_batches) if len(train_loader) > nan_batches else float('nan')
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_nan_batches = 0
        
        # Use a progress bar for validation too
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
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
    
    # If we had NaN issues the whole time, try with a simpler model
    if all(np.isnan(loss) for loss in val_losses):
        print("All validation losses were NaN. Training a simpler model with fewer parameters...")
        # Initialize a simpler model with smaller hidden layers and more regularization
        simpler_model = SegmentRewardModel(model.reward_model.model[0].in_features // 2, 
                                          model.reward_model.model[0].in_features // 2, 
                                          hidden_dims=[64, 32])
        simpler_model = simpler_model.to(device)
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
        plt.close()
    
    return model, train_losses, val_losses

def evaluate_reward_model(model, test_loader, device):
    """Evaluate the reward model on test data."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    avg_logpdf = 0
    
    with torch.no_grad():
        for obs1, actions1, obs2, actions2, pref in tqdm(test_loader, desc="Evaluating reward model"):
            obs1, actions1 = obs1.to(device), actions1.to(device)
            obs2, actions2 = obs2.to(device), actions2.to(device)
            pref = pref.to(device)
            
            reward1 = model(obs1, actions1)
            reward2 = model(obs2, actions2)
            
            loss = bradley_terry_loss(reward1, reward2, pref)
            test_loss += loss.item()
            
            # Count correct predictions (when model assigns higher reward to preferred segment)
            pred = 1 * (reward1 > reward2) + 2 * (reward1 <= reward2)
            correct += (pred == pref).sum().item()
            total += pref.size(0)
            
            # Compute logpdf for preferred segments
            preferred_obs = torch.where((pref == 1).unsqueeze(1).unsqueeze(2).expand_as(obs1), obs1, obs2)
            preferred_actions = torch.where((pref == 1).unsqueeze(1).unsqueeze(2).expand_as(actions1), actions1, actions2)
            preferred_rewards = torch.where((pref == 1), reward1, reward2)
            
            # Calculate logpdf
            logpdfs = model.logpdf(preferred_obs, preferred_actions, preferred_rewards)
            avg_logpdf += logpdfs.mean().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    avg_logpdf /= len(test_loader)
    
    print(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Avg LogPDF: {avg_logpdf:.4f}")
    return test_loss, accuracy, avg_logpdf

def main():
    parser = argparse.ArgumentParser(description="Train a state-action reward model using BT loss")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATHS[0],
                        help="Path to the PT file containing trajectory data")
    parser.add_argument("--segment_length", type=int, default=20,
                        help="Length of segments (H)")
    parser.add_argument("--num_segments", type=int, default=5000,
                        help="Number of segments to sample (0 for all)")
    parser.add_argument("--num_pairs", type=int, default=5000,
                        help="Number of preference pairs to generate")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256],
                        help="Hidden dimensions of the reward model")
    parser.add_argument("--output_dir", type=str, default="reward_model",
                        help="Directory to save model and results")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Use pin_memory for faster data transfer to GPU")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID to use (e.g., 0 for cuda:0)")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force using CPU even if CUDA is available")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device with explicit GPU selection
    if args.use_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as specified")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_tensordict(args.data_path)
    
    # Extract observations and actions
    print("Extracting observations and actions...")
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    rewards = data["reward"]
    
    # Report NaN statistics before filtering
    obs_nans = torch.isnan(observations).any(dim=1).sum().item()
    action_nans = torch.isnan(actions).any(dim=1).sum().item()
    reward_nans = torch.isnan(rewards).sum().item()
    
    print(f"Before filtering:")
    print(f"  - Observations with NaNs: {obs_nans}/{observations.shape[0]} ({obs_nans/observations.shape[0]*100:.2f}%)")
    print(f"  - Actions with NaNs: {action_nans}/{actions.shape[0]} ({action_nans/actions.shape[0]*100:.2f}%)")
    print(f"  - Rewards with NaNs: {reward_nans}/{rewards.shape[0]} ({reward_nans/rewards.shape[0]*100:.2f}%)")
    print(f"Observations shape: {observations.shape}, Actions shape: {actions.shape}")
    
    # Create segments (now with NaN filtering)
    print("\nCreating observation-action segments with NaN filtering...")
    segment_indices = create_state_action_segments(data, args.segment_length, max_segments=args.num_segments)
    
    # Generate preference pairs
    print("\nGenerating preference pairs (skipping segments with NaNs)...")
    segment_pairs, preference_labels = generate_synthetic_preferences(
        segment_indices, rewards, n_pairs=args.num_pairs
    )
    
    # Split into train, validation, and test sets (80%, 10%, 10%)
    n_pairs = len(segment_pairs)
    indices = list(range(n_pairs))
    random.shuffle(indices)
    
    n_train = int(0.8 * n_pairs)
    n_val = int(0.1 * n_pairs)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_pairs = [segment_pairs[i] for i in train_indices]
    train_prefs = [preference_labels[i] for i in train_indices]
    val_pairs = [segment_pairs[i] for i in val_indices]
    val_prefs = [preference_labels[i] for i in val_indices]
    test_pairs = [segment_pairs[i] for i in test_indices]
    test_prefs = [preference_labels[i] for i in test_indices]
    
    train_dataset = PreferenceDataset(observations, actions, train_pairs, segment_indices, train_prefs)
    val_dataset = PreferenceDataset(observations, actions, val_pairs, segment_indices, val_prefs)
    test_dataset = PreferenceDataset(observations, actions, test_pairs, segment_indices, test_prefs)
    
    # Use DataLoader with multiple workers for faster data loading
    # Note: for CUDA tensors, num_workers must be 0 when tensors are on GPU
    # Safety check for GPU tensors
    on_gpu = (observations.device.type == 'cuda' or actions.device.type == 'cuda')
    
    # Choose appropriate num_workers
    effective_num_workers = 0 if on_gpu else args.num_workers
    if on_gpu and args.num_workers > 0:
        print(f"Warning: Data is on GPU but num_workers={args.num_workers}. Setting num_workers=0 for safety.")
    
    # If on GPU without workers, pin_memory doesn't help
    effective_pin_memory = args.pin_memory and not on_gpu
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        # Add prefetch factor for more efficient loading
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    print(f"Train pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")
    
    # Initialize reward model
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")
    
    model = SegmentRewardModel(state_dim, action_dim, hidden_dims=args.hidden_dims)
    
    print(f"Reward model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Start timing the training
    start_time = time.time()
    
    # Train the model
    print("\nTraining reward model...")
    model, train_losses, val_losses = train_reward_model(
        model, train_loader, val_loader, device, 
        num_epochs=args.num_epochs, lr=args.lr
    )
    
    # Calculate and print training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Evaluate the model
    print("\nEvaluating reward model...")
    test_loss, accuracy, avg_logpdf = evaluate_reward_model(model, test_loader, device)
    
    # Save the model and results
    torch.save(model.state_dict(), f"{args.output_dir}/state_action_reward_model.pt")
    
    results = {
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'avg_logpdf': avg_logpdf,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'training_time': training_time
    }
    
    with open(f"{args.output_dir}/sa_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Model saved to {args.output_dir}/state_action_reward_model.pt")
    print(f"Results saved to {args.output_dir}/sa_results.pkl")
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 