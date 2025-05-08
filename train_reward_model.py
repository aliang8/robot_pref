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

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    sample_segments
)

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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
            rewards = torch.zeros(len(observations), device=observations.device)
            for t in range(len(observations)):
                obs = observations[t]
                action = actions[t] if t < len(actions) else torch.zeros_like(actions[0])  # Last observation has no action
                rewards[t] = self.reward_model(obs.unsqueeze(0), action.unsqueeze(0))
            return rewards.sum()  # Return cumulative reward for segment
        
        elif observations.dim() == 3:  # Batch of segments (batch_size, seq_len, obs_dim)
            batch_rewards = []
            for b in range(observations.size(0)):
                segment_reward = 0
                for t in range(observations.size(1)):
                    obs = observations[b, t]
                    action = actions[b, t] if t < actions.size(1) else torch.zeros_like(actions[b, 0])
                    segment_reward += self.reward_model(obs.unsqueeze(0), action.unsqueeze(0))
                batch_rewards.append(segment_reward)
            return torch.stack(batch_rewards)
        
        else:
            raise ValueError(f"Unexpected input shape: observations {observations.shape}, actions {actions.shape}")
    
    def logpdf(self, observations, actions, rewards):
        """Compute log probability of rewards given observations and actions."""
        # Assuming rewards are per-segment, need to distribute across steps
        if observations.dim() == 2:  # Single segment
            segment_reward = self(observations, actions)
            return self.reward_model.logpdf(observations.mean(0, keepdim=True), 
                                           actions.mean(0, keepdim=True) if len(actions) > 0 else actions[0:1], 
                                           rewards)
        elif observations.dim() == 3:  # Batch of segments
            batch_logpdfs = []
            for b in range(observations.size(0)):
                obs = observations[b]
                action = actions[b]
                reward = rewards[b]
                logp = self.reward_model.logpdf(obs.mean(0, keepdim=True), 
                                              action.mean(0, keepdim=True) if action.size(0) > 0 else action[0:1], 
                                              reward)
                batch_logpdfs.append(logp)
            return torch.stack(batch_logpdfs)
        else:
            raise ValueError(f"Unexpected input shape for logpdf")

class PreferenceDataset(Dataset):
    """Dataset for segment preference pairs."""
    def __init__(self, observations, actions, segment_pairs, segment_indices, preference_labels):
        self.observations = observations
        self.actions = actions
        self.segment_pairs = segment_pairs
        self.segment_indices = segment_indices
        self.preference_labels = preference_labels
    
    def __len__(self):
        return len(self.segment_pairs)
    
    def __getitem__(self, idx):
        seg_idx1, seg_idx2 = self.segment_pairs[idx]
        start1, end1 = self.segment_indices[seg_idx1]
        start2, end2 = self.segment_indices[seg_idx2]
        
        obs1 = self.observations[start1:end1+1]
        actions1 = self.actions[start1:end1]
        obs2 = self.observations[start2:end2+1]
        actions2 = self.actions[start2:end2]
        
        pref = self.preference_labels[idx]
        
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

def create_state_action_segments(data, H):
    """Create H-step segments of observations and actions from data."""
    # Extract episode IDs
    episode_ids = data["episode"].cpu()
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    rewards = data["reward"]
    
    segments = []
    segment_indices = []  # Store start and end indices of each segment
    
    unique_episodes = torch.unique(episode_ids).tolist()
    print(f"Found {len(unique_episodes)} unique episodes")
    
    for episode_id in tqdm(unique_episodes, desc="Processing episodes"):
        # Get indices for this episode
        ep_indices = torch.where(episode_ids == episode_id)[0]
        ep_observations = observations[ep_indices]
        ep_actions = actions[ep_indices]
        
        num_segments_in_episode = max(0, len(ep_observations) - H + 1)
        
        # Create segments for this episode
        for i in range(num_segments_in_episode):
            # Store indices for this segment
            segment_indices.append((int(ep_indices[i].item()), int(ep_indices[i+H-1].item())))
            segments.append(i)  # Just store the index, we'll fetch observations/actions later
        
        if num_segments_in_episode == 0:
            print(f"Episode {episode_id} has {len(ep_observations)} frames (less than H={H}), skipping")
    
    print(f"Created {len(segments)} segments across {len(unique_episodes)} episodes")
    
    return segment_indices

def generate_synthetic_preferences(segment_indices, rewards, n_pairs=10000):
    """Generate synthetic preference pairs based on cumulative rewards."""
    n_segments = len(segment_indices)
    print(f"Generating preferences from {n_segments} segments")
    
    # Sample random pairs of segment indices
    pairs = []
    preference_labels = []
    
    # Keep generating pairs until we have enough
    for _ in tqdm(range(n_pairs), desc="Generating preference pairs"):
        # Sample two different segments
        idx1, idx2 = random.sample(range(n_segments), 2)
        
        # Get segment indices
        start1, end1 = segment_indices[idx1]
        start2, end2 = segment_indices[idx2]
        
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
    
    print(f"Generated {len(pairs)} preference pairs")
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
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        nan_batches = 0
        
        for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")):
            obs1, actions1 = obs1.to(device), actions1.to(device)
            obs2, actions2 = obs2.to(device), actions2.to(device)
            pref = pref.to(device)
            
            optimizer.zero_grad()
            
            # Check for NaN in input
            if torch.isnan(obs1).any() or torch.isnan(actions1).any() or torch.isnan(obs2).any() or torch.isnan(actions2).any():
                print(f"WARNING: NaN values detected in inputs at batch {batch_idx}. Skipping batch.")
                nan_batches += 1
                continue
            
            reward1 = model(obs1, actions1)
            reward2 = model(obs2, actions2)
            
            # Check for NaN in rewards
            if torch.isnan(reward1).any() or torch.isnan(reward2).any():
                print(f"WARNING: NaN rewards detected at batch {batch_idx}. Skipping batch.")
                nan_batches += 1
                continue
            
            loss = bradley_terry_loss(reward1, reward2, pref)
            
            # Skip problematic batches
            if torch.isnan(loss).any():
                print(f"WARNING: NaN loss after bradley_terry_loss computation at batch {batch_idx}. Skipping batch.")
                nan_batches += 1
                continue
                
            loss.backward()
            
            # Clip gradients to help with training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"WARNING: NaN gradients detected at batch {batch_idx}. Skipping optimizer step.")
                nan_batches += 1
                continue
                
            optimizer.step()
            
            train_loss += loss.item()
        
        # Log if there were any NaN issues
        if nan_batches > 0:
            print(f"WARNING: {nan_batches}/{len(train_loader)} batches skipped due to NaN issues")
        
        avg_train_loss = train_loss / (len(train_loader) - nan_batches) if len(train_loader) > nan_batches else float('nan')
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_nan_batches = 0
        
        with torch.no_grad():
            for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)")):
                obs1, actions1 = obs1.to(device), actions1.to(device)
                obs2, actions2 = obs2.to(device), actions2.to(device)
                pref = pref.to(device)
                
                # Check for NaN in input
                if torch.isnan(obs1).any() or torch.isnan(actions1).any() or torch.isnan(obs2).any() or torch.isnan(actions2).any():
                    print(f"WARNING: NaN values detected in validation inputs at batch {batch_idx}. Skipping batch.")
                    val_nan_batches += 1
                    continue
                
                reward1 = model(obs1, actions1)
                reward2 = model(obs2, actions2)
                
                # Check for NaN in rewards
                if torch.isnan(reward1).any() or torch.isnan(reward2).any():
                    print(f"WARNING: NaN rewards detected in validation at batch {batch_idx}. Skipping batch.")
                    val_nan_batches += 1
                    continue
                
                loss = bradley_terry_loss(reward1, reward2, pref)
                
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN validation loss at batch {batch_idx}. Skipping batch.")
                    val_nan_batches += 1
                    continue
                    
                val_loss += loss.item()
        
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
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_tensordict(args.data_path)
    
    # Extract observations and actions
    print("Extracting observations and actions...")
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    rewards = data["reward"]
    
    print(f"Observations shape: {observations.shape}, Actions shape: {actions.shape}")
    
    # Create segments
    print("\nCreating observation-action segments...")
    segment_indices = create_state_action_segments(data, args.segment_length)
    
    # Sample segments if needed
    if args.num_segments > 0 and args.num_segments < len(segment_indices):
        print(f"\nSampling {args.num_segments} segments from {len(segment_indices)} total segments...")
        segment_idx_list = list(range(len(segment_indices)))
        sampled_indices = random.sample(segment_idx_list, args.num_segments)
        segment_indices = [segment_indices[i] for i in sampled_indices]
    
    # Generate preference pairs
    print("\nGenerating preference pairs...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Train pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")
    
    # Initialize reward model
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")
    
    model = SegmentRewardModel(state_dim, action_dim, hidden_dims=args.hidden_dims)
    model = model.to(device)
    
    print(f"Reward model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    print("\nTraining reward model...")
    model, train_losses, val_losses = train_reward_model(
        model, train_loader, val_loader, device, 
        num_epochs=args.num_epochs, lr=args.lr
    )
    
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
        'action_dim': action_dim
    }
    
    with open(f"{args.output_dir}/sa_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Model saved to {args.output_dir}/state_action_reward_model.pt")
    print(f"Results saved to {args.output_dir}/sa_results.pkl")
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 