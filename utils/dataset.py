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
        self.data = data
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
            raise ValueError("Data must be a dictionary")
        
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

        if isinstance(self.data, dict):
            obs_key = self.obs_key
            action_key = "action"
            obs1 = self.data[obs_key][start1 : end1]
            actions1 = self.data[action_key][start1:end1]
            obs2 = self.data[obs_key][start2 : end2]
            actions2 = self.data[action_key][start2:end2]
   
        # Apply normalization to observations
        if self.normalize_obs:
            obs1 = self._normalize_observations(obs1)
            obs2 = self._normalize_observations(obs2)
        
        # Convert preference to tensor
        if isinstance(self.preferences[idx], torch.Tensor):
            # If it's already a tensor, just clone it
            pref = self.preferences[idx].long()
        else:
            # Otherwise create a new tensor
            pref = torch.tensor(self.preferences[idx], dtype=torch.long)

        return obs1, actions1, obs2, actions2, pref

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
                        num_workers=4,
                        pin_memory=True, seed=42, normalize_obs=False, 
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
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
