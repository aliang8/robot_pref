import random

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class PreferenceDataset(Dataset):
    """Dataset for segment preference pairs."""
    def __init__(self, data, segment_pairs, segment_indices, preferences, normalize_obs=False, norm_method='standard', norm_stats=None):
        """
        Initialize the dataset for preference learning.
        
        Args:
            data: Dictionary containing observations and actions, or tensor
            segment_pairs: List of pairs of segment indices [(i, j), ...]
            segment_indices: List of segment start/end indices [(start, end), ...]
            preferences: List of preferences (1 = first segment preferred, 2 = second segment preferred)
            normalize_obs: Whether to normalize observations (default: False)
            norm_method: Normalization method ('standard' or 'minmax')
            norm_stats: Pre-computed normalization statistics dict (optional)
        """
        # Ensure all data is on CPU for multi-process loading
        self.data = data.cpu() if isinstance(data, torch.Tensor) else data
        self.segment_pairs = segment_pairs
        self.segment_indices = segment_indices
        self.preferences = preferences
        self.normalize_obs = normalize_obs
        self.norm_method = norm_method
        
        # Determine the data length for bounds checking
        if isinstance(self.data, dict):
            # For dictionary data, use the observation tensor length
            if "obs" in self.data:
                self.data_length = len(self.data["obs"])
                self.obs_key = "obs"
            elif "state" in self.data:
                self.data_length = len(self.data["state"])
                self.obs_key = "state"
            else:
                raise ValueError("Data dictionary must contain 'obs' or 'state' key")
        else:
            # For tensor data, use its length
            self.data_length = len(self.data)
            self.obs_key = None
        
        # Compute or use provided normalization statistics
        self.norm_stats = {}
        if self.normalize_obs:
            if norm_stats is not None:
                # Use provided normalization statistics
                self.norm_stats = norm_stats
                print("Using provided normalization statistics")
            else:
                # Compute normalization statistics from the data
                print("Computing observation normalization statistics...")
                self._compute_normalization_statistics()
            
            # Print summary of normalization statistics
            if self.norm_method == 'standard':
                print(f"Observation mean: {self.norm_stats['mean'].mean().item():.4f}, std: {self.norm_stats['std'].mean().item():.4f}")
            elif self.norm_method == 'minmax':
                print(f"Observation min: {self.norm_stats['min'].mean().item():.4f}, max: {self.norm_stats['max'].mean().item():.4f}")
        
        print(f"Dataset initialized with data length: {self.data_length}, normalize_obs={normalize_obs}")
    
    def _compute_normalization_statistics(self):
        """Compute normalization statistics from the dataset."""
        # Get the observations tensor
        if isinstance(self.data, dict):
            obs = self.data[self.obs_key]
        else:
            obs = self.data
        
        # For standard normalization, compute mean and std
        if self.norm_method == 'standard':
            # Compute along first dimension (time/batch)
            mean = obs.mean(dim=0)
            std = obs.std(dim=0)
            # Ensure std is not zero (replace zeros with ones)
            std = torch.where(std > 1e-6, std, torch.ones_like(std))
            
            self.norm_stats['mean'] = mean
            self.norm_stats['std'] = std
            
        # For min-max normalization, compute min and max
        elif self.norm_method == 'minmax':
            # Compute along first dimension (time/batch)
            min_vals = obs.min(dim=0)[0]
            max_vals = obs.max(dim=0)[0]
            # Ensure range is not zero (replace zero-range with unit range)
            range_vals = max_vals - min_vals
            valid_range = torch.where(range_vals > 1e-6, range_vals, torch.ones_like(range_vals))
            
            self.norm_stats['min'] = min_vals
            self.norm_stats['max'] = max_vals
            self.norm_stats['range'] = valid_range
        
        else:
            raise ValueError(f"Unsupported normalization method: {self.norm_method}")
    
    def _normalize_observations(self, obs):
        """Apply normalization to the observations."""
        if not self.normalize_obs or not self.norm_stats:
            return obs
            
        if self.norm_method == 'standard':
            # Apply standard normalization: (x - mean) / std
            return (obs - self.norm_stats['mean']) / self.norm_stats['std']
            
        elif self.norm_method == 'minmax':
            # Apply min-max normalization: (x - min) / (max - min)
            return (obs - self.norm_stats['min']) / self.norm_stats['range']
            
        return obs
    
    def __len__(self):
        return len(self.segment_pairs)

    def __getitem__(self, idx):
        seg_idx1, seg_idx2 = self.segment_pairs[idx]
        start1, end1 = self.segment_indices[seg_idx1]
        start2, end2 = self.segment_indices[seg_idx2]

        # Check bounds against the correct data length
        if start1 < 0 or end1 >= self.data_length or start1 > end1:
            raise IndexError(
                f"Invalid segment indices for segment 1: {start1}:{end1}, data length: {self.data_length}"
            )

        if start2 < 0 or end2 >= self.data_length or start2 > end2:
            raise IndexError(
                f"Invalid segment indices for segment 2: {start2}:{end2}, data length: {self.data_length}"
            )

        # Get data for segments (handle dictionary data correctly)
        if isinstance(self.data, dict):
            # Extract from dictionary (TensorDict)
            obs_key = self.obs_key
            action_key = "action"

            # Safely extract data
            obs1 = self.data[obs_key][start1 : end1 + 1].clone().detach()
            actions1 = self.data[action_key][start1:end1].clone().detach()
            obs2 = self.data[obs_key][start2 : end2 + 1].clone().detach()
            actions2 = self.data[action_key][start2:end2].clone().detach()
        else:
            # Extract directly from tensor
            obs1 = self.data[start1 : end1 + 1].clone().detach()
            actions1 = self.data[start1:end1].clone().detach()
            obs2 = self.data[start2 : end2 + 1].clone().detach()
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
        
        # Apply normalization to observations
        if self.normalize_obs:
            obs1 = self._normalize_observations(obs1)
            obs2 = self._normalize_observations(obs2)
        
        # Convert preference to tensor
        if isinstance(self.preferences[idx], torch.Tensor):
            # If it's already a tensor, just clone it
            pref = self.preferences[idx].clone().detach().long()
        else:
            # Otherwise create a new tensor
            pref = torch.tensor(self.preferences[idx], dtype=torch.long)

        # Handle NaN values
        if (
            torch.isnan(obs1).any()
            or torch.isnan(actions1).any()
            or torch.isnan(obs2).any()
            or torch.isnan(actions2).any()
        ):
            obs1 = torch.nan_to_num(obs1, nan=0.0)
            actions1 = torch.nan_to_num(actions1, nan=0.0)
            obs2 = torch.nan_to_num(obs2, nan=0.0)
            actions2 = torch.nan_to_num(actions2, nan=0.0)

        return obs1, actions1, obs2, actions2, pref


def bradley_terry_loss(rewards1, rewards2, preferences):
    """
    Compute the Bradley-Terry preference learning loss (binary cross-entropy).
    
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

    # Compute probability that segment1 is preferred over segment2 using the Bradley-Terry model
    # Add a small epsilon for numerical stability
    eps = 1e-6
    logits = torch.clamp(
        rewards1 - rewards2, min=-50.0, max=50.0
    )  # Clip logits to prevent overflow
    pred_probs = torch.sigmoid(logits)
    
    # Standard binary cross-entropy loss: -(y*log(p) + (1-y)*log(1-p))
    # This is negative log likelihood (higher means worse fit)
    bce = -(prefs * torch.log(pred_probs + eps) + (1 - prefs) * torch.log(1 - pred_probs + eps))
    
    # Return mean over batch dimension
    return torch.mean(bce, dim=-1)  # Mean over last dimension (batch)


def shuffle_preference_dataset(dataset, seed=42):
    """
    Shuffle a PreferenceDataset to ensure random sampling in train/val/test splits.
    
    Args:
        dataset: PreferenceDataset to shuffle
        seed: Random seed for reproducibility
        
    Returns:
        A new PreferenceDataset with shuffled segment_pairs and preferences
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create a list of indices for the dataset
    indices = list(range(len(dataset.segment_pairs)))
    
    # Shuffle the indices
    random.shuffle(indices)
    
    # Create shuffled segment_pairs and preferences
    shuffled_segment_pairs = [dataset.segment_pairs[i] for i in indices]
    shuffled_preferences = [dataset.preferences[i] for i in indices]
    
    # Create a new dataset with the shuffled pairs
    shuffled_dataset = PreferenceDataset(
        dataset.data,
        shuffled_segment_pairs,
        dataset.segment_indices,
        shuffled_preferences,
        normalize_obs=dataset.normalize_obs,
        norm_method=dataset.norm_method,
        norm_stats=dataset.norm_stats
    )
    
    print(f"Shuffled preference dataset with {len(shuffled_dataset)} pairs")
    return shuffled_dataset


def create_data_loaders(preference_dataset, train_ratio=0.8, val_ratio=0.1, batch_size=32, 
                    num_workers=4, pin_memory=True, seed=42, normalize_obs=False, 
                    norm_method='standard', shuffle_dataset=True):
    """Create data loaders for training, validation, and testing.

    Args:
        preference_dataset: Dataset containing segment preference pairs
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducibility
        normalize_obs: Whether to normalize observations
        norm_method: Normalization method ('standard' or 'minmax')
        shuffle_dataset: Whether to shuffle the dataset before splitting
        
    Returns:
        Dictionary containing 'train', 'val', and 'test' data loaders, plus dataset sizes
    """
    # Apply normalization to the dataset if requested
    if normalize_obs and not preference_dataset.normalize_obs:
        print(f"Applying {norm_method} normalization to dataset")
        # Create a new dataset with normalization enabled
        preference_dataset = PreferenceDataset(
            preference_dataset.data,
            preference_dataset.segment_pairs,
            preference_dataset.segment_indices,
            preference_dataset.preferences,
            normalize_obs=True,
            norm_method=norm_method
        )
    
    # Shuffle the dataset to ensure random sampling if requested
    if shuffle_dataset:
        print("Shuffling dataset before splitting...")
        preference_dataset = shuffle_preference_dataset(preference_dataset, seed=seed)
    
    # Calculate split sizes
    total_size = len(preference_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Create random splits
    train_dataset, val_dataset, test_dataset = random_split(
        preference_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
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
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
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
    segment_pairs = pref_data["segment_pairs"]
    segment_indices = pref_data["segment_indices"]

    # Get preferences
    if "preference_labels" in pref_data:
        print("Using preference_labels from dataset")
        preferences = pref_data["preference_labels"]
    else:
        print(
            "Could not find preferences in dataset! Available keys:",
            list(pref_data.keys()),
        )
        raise KeyError("No preference data found in file")

    # Check if data is included in the preference data
    data = None
    if "data" in pref_data:
        data = pref_data["data"]
        print(f"Found embedded data with fields: {list(data.keys())}")

    print(f"Loaded {len(segment_pairs)} preference pairs")
    return segment_pairs, segment_indices, preferences, data
