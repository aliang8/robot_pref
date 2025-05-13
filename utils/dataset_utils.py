import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


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
        if isinstance(self.preferences[idx], torch.Tensor):
            # If it's already a tensor, just clone it
            pref = self.preferences[idx].clone().detach().long()
        else:
            # Otherwise create a new tensor
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
                 Shape can be [batch_size] or [num_models, batch_size]
        rewards2: Predicted rewards for the second segments in each pair
                 Shape can be [batch_size] or [num_models, batch_size]
        preferences: Labels indicating which segment is preferred (1 or 2)
                    Shape is [batch_size]
    
    Returns:
        Loss value if rewards are [batch_size]
        Loss tensor of shape [num_models] if rewards are [num_models, batch_size]
    """
    # Convert preferences to probabilities (1 = first segment preferred, 2 = second segment preferred)
    # Use detach and clone for tensor conversion if input is already a tensor
    if isinstance(preferences, torch.Tensor):
        prefs = (preferences == 1).float()
    else:
        prefs = (torch.tensor(preferences) == 1).float()
    
    # Handle both single model and ensemble outputs
    if rewards1.dim() > 1 and rewards1.shape[0] > 1:
        # For ensemble output [num_models, batch_size]
        # We need to compute loss for each model separately
        
        # Add a new axis to prefs to enable broadcasting
        # [batch_size] -> [1, batch_size]
        prefs = prefs.unsqueeze(0)
        
        # Compute probability that segment1 is preferred over segment2 using the Bradley-Terry model
        # Add a small epsilon for numerical stability
        eps = 1e-6
        logits = torch.clamp(rewards1 - rewards2, min=-50.0, max=50.0)  # Clip logits to prevent overflow
        pred_probs = torch.sigmoid(logits)
        
        # Use a more numerically stable binary cross-entropy loss
        # This will compute loss for each model separately using broadcasting
        # Result shape: [num_models]
        loss = -torch.mean(prefs * torch.log(pred_probs + eps) + 
                          (1 - prefs) * torch.log(1 - pred_probs + eps), 
                          dim=1)
    else:
        # For single model output [batch_size]
        # Compute probability that segment1 is preferred over segment2 using the Bradley-Terry model
        # Add a small epsilon for numerical stability
        eps = 1e-6
        logits = torch.clamp(rewards1 - rewards2, min=-50.0, max=50.0)  # Clip logits to prevent overflow
        pred_probs = torch.sigmoid(logits)
        
        # Use a more numerically stable binary cross-entropy loss
        loss = -torch.mean(prefs * torch.log(pred_probs + eps) + (1 - prefs) * torch.log(1 - pred_probs + eps))
    
    return loss


def create_data_loaders(preference_dataset, train_ratio=0.8, val_ratio=0.1, batch_size=32, num_workers=4, pin_memory=True, seed=42):
    """Create data loaders for training, validation, and testing.
    
    Args:
        preference_dataset: Dataset containing segment preference pairs
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing 'train', 'val', and 'test' data loaders, plus dataset sizes
    """
    # Calculate split sizes
    total_size = len(preference_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Create random splits
    train_dataset, val_dataset, test_dataset = random_split(
        preference_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Split data: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Determine effective settings for CPU vs GPU
    effective_num_workers = num_workers
    effective_pin_memory = pin_memory
    
    # Adjust for CPU-only mode
    if not torch.cuda.is_available():
        effective_pin_memory = False
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }


def evaluate_model_on_test_set(model, test_loader, device):
    """Evaluate model performance on the test set.
    
    Args:
        model: Trained reward model (single model or ensemble)
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_acc = 0
    test_total = 0
    logpdf_values = []
    
    # Determine if model is an ensemble
    is_ensemble = hasattr(model, 'models')
    
    with torch.no_grad():
        for obs1, actions1, obs2, actions2, pref in tqdm(test_loader, desc="Testing"):
            # Move to device
            obs1, actions1 = obs1.to(device), actions1.to(device)
            obs2, actions2 = obs2.to(device), actions2.to(device)
            pref = pref.to(device)
            
            # Get reward predictions using the paired rewards interface
            reward1, reward2 = model.compute_paired_rewards(obs1, actions1, obs2, actions2)
            
            # Compute loss
            loss = bradley_terry_loss(reward1, reward2, pref)
            
            # For ensemble, use mean loss across models
            if is_ensemble:
                test_loss += loss.mean().item()
                
                # For accuracy, use mean prediction across models
                reward1_mean = reward1.mean(dim=0)
                reward2_mean = reward2.mean(dim=0)
                
                # Get model predictions based on mean rewards
                pred_pref = torch.where(reward1_mean > reward2_mean, 
                                       torch.ones_like(pref), 
                                       torch.ones_like(pref) * 2)
            else:
                test_loss += loss.item()
                
                # Get model predictions
                pred_pref = torch.where(reward1.squeeze(0) > reward2.squeeze(0), 
                                       torch.ones_like(pref), 
                                       torch.ones_like(pref) * 2)
            
            # Compute accuracy (prediction matches ground truth preference)
            correct = (pred_pref == pref).sum().item()
            test_acc += correct
            test_total += pref.size(0)
            
            # Calculate log probability for each batch item
            if hasattr(model, 'logpdf'):
                batch_size = pref.size(0)
                for i in range(batch_size):
                    # Select the correctly preferred segments for this batch item
                    if pref[i] == 1:
                        # First segment is preferred
                        segment_obs = obs1[i:i+1]  # Keep batch dimension
                        segment_actions = actions1[i:i+1]
                        segment_reward = reward1.squeeze(0)[i:i+1] if not is_ensemble else reward1.mean(dim=0)[i:i+1]
                    else:
                        # Second segment is preferred
                        segment_obs = obs2[i:i+1]  # Keep batch dimension
                        segment_actions = actions2[i:i+1]
                        segment_reward = reward2.squeeze(0)[i:i+1] if not is_ensemble else reward2.mean(dim=0)[i:i+1]
                    
                    # Calculate logpdf
                    logp = model.logpdf(segment_obs, segment_actions, segment_reward)
                    logpdf_values.append(logp.mean().item())
            elif is_ensemble and hasattr(model.models[0], 'logpdf'):
                # For ensembles, compute logpdf using first model if ensemble doesn't have a logpdf method
                batch_size = pref.size(0)
                for i in range(batch_size):
                    # Select the correctly preferred segments for this batch item
                    if pref[i] == 1:
                        # First segment is preferred
                        segment_obs = obs1[i:i+1]  # Keep batch dimension
                        segment_actions = actions1[i:i+1]
                        segment_reward = reward1.mean(dim=0)[i:i+1]  # Use mean reward across ensemble
                    else:
                        # Second segment is preferred
                        segment_obs = obs2[i:i+1]  # Keep batch dimension
                        segment_actions = actions2[i:i+1]
                        segment_reward = reward2.mean(dim=0)[i:i+1]  # Use mean reward across ensemble
                    
                    # Calculate logpdf using first model
                    logp = model.models[0].logpdf(segment_obs, segment_actions, segment_reward)
                    logpdf_values.append(logp.mean().item())
    
    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else float('nan')
    test_accuracy = test_acc / test_total if test_total > 0 else 0
    avg_logpdf = np.mean(logpdf_values) if logpdf_values else float('nan')
    
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Avg LogPDF: {avg_logpdf:.4f}")
    print(f"Correctly predicted {test_acc} out of {test_total} preference pairs ({test_accuracy:.2%})")
    
    return {
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "avg_logpdf": avg_logpdf,
        "num_test_samples": test_total
    }


def load_preferences_data(file_path):
    """Load preference data saved from collect_preferences.
    
    Args:
        file_path: Path to saved preference data pickle file
        
    Returns:
        Tuple of (segment_pairs, segment_indices, preferences, data)
    """
    print(f"Loading preference data from {file_path}")
    
    # Load data with torch.load instead of pickle
    pref_data = torch.load(file_path, weights_only=False)
    
    # Extract the necessary components
    segment_pairs = pref_data['segment_pairs']
    segment_indices = pref_data['segment_indices']
    
    # Get preferences
    if 'preference_labels' in pref_data:
        print("Using preference_labels from dataset")
        preferences = pref_data['preference_labels']
    else:
        print("Could not find preferences in dataset! Available keys:", list(pref_data.keys()))
        raise KeyError("No preference data found in file")
    
    # Check if data is included in the preference data
    data = None
    if 'data' in pref_data:
        data = pref_data['data']
        print(f"Found embedded data with fields: {list(data.keys())}")
    
    print(f"Loaded {len(segment_pairs)} preference pairs")
    return segment_pairs, segment_indices, preferences, data 