import torch
from torch.utils.data import Dataset
import numpy as np
import random
from typing import Dict, List, Tuple


class SkillDataset(Dataset):
    """Dataset for training skill VAE with temporal predictability."""
    
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        segment_length: int = 32,
        max_temporal_diff: int = 100,
        obs_key: str = "obs",
        action_key: str = "action",
        normalize_obs: bool = True,
        norm_method: str = "min_max"
    ):
        """
        Initialize skill dataset.
        
        Args:
            data: Dictionary containing trajectory data
            segment_length: Length of sub-trajectories (H in the paper)
            max_temporal_diff: Maximum temporal difference between sub-trajectory pairs
            obs_key: Key for observations in data
            action_key: Key for actions in data
            normalize_obs: Whether to normalize observations
            norm_method: Normalization method ("min_max" or "z_score")
        """
        self.segment_length = segment_length
        self.max_temporal_diff = max_temporal_diff
        self.obs_key = obs_key
        self.action_key = action_key
        
        # Extract observations and actions
        observations = data[obs_key] if obs_key in data else data["state"]
        actions = data[action_key]
        
        # Ensure data is float32
        self.observations = observations.float()
        self.actions = actions.float()
        
        self.obs_dim = self.observations.shape[1]
        self.action_dim = self.actions.shape[1]
        self.seq_len = self.observations.shape[0]
        
        # Normalize observations if requested
        if normalize_obs:
            self._normalize_observations(norm_method)
        
        # Generate all possible sub-trajectory segments
        self.segments = self._generate_segments()
        
        # Generate sub-trajectory pairs with temporal differences
        self.pairs = self._generate_pairs()
        
        print(f"Created SkillDataset with {len(self.segments)} segments and {len(self.pairs)} pairs")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
    
    def _normalize_observations(self, method: str):
        """Normalize observations using specified method."""
        if method == "min_max":
            self.obs_min = self.observations.min(dim=0, keepdim=True)[0]
            self.obs_max = self.observations.max(dim=0, keepdim=True)[0]
            
            # Avoid division by zero
            obs_range = self.obs_max - self.obs_min
            obs_range[obs_range == 0] = 1.0
            
            self.observations = (self.observations - self.obs_min) / obs_range
            
        elif method == "z_score":
            self.obs_mean = self.observations.mean(dim=0, keepdim=True)
            self.obs_std = self.observations.std(dim=0, keepdim=True)
            
            # Avoid division by zero
            obs_std = self.obs_std.clone()
            obs_std[obs_std == 0] = 1.0
            
            self.observations = (self.observations - self.obs_mean) / obs_std
    
    def _generate_segments(self) -> List[Tuple[int, int]]:
        """Generate all possible sub-trajectory segments of length H."""
        segments = []
        
        # Generate segments with sliding window
        for start_idx in range(0, self.seq_len - self.segment_length + 1, self.segment_length // 2):
            end_idx = start_idx + self.segment_length
            if end_idx <= self.seq_len:
                segments.append((start_idx, end_idx))
        
        return segments
    
    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """Generate pairs of segments with their temporal differences."""
        pairs = []
        
        for i, (start1, end1) in enumerate(self.segments):
            for j, (start2, end2) in enumerate(self.segments):
                if i != j:  # Don't pair segment with itself
                    # Calculate temporal difference (in timesteps)
                    time_diff = abs(start2 - start1)
                    
                    # Only include pairs within max temporal difference
                    if time_diff <= self.max_temporal_diff:
                        pairs.append((i, j, time_diff))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get a pair of sub-trajectories with temporal difference."""
        seg_idx1, seg_idx2, time_diff = self.pairs[idx]
        
        # Get first segment
        start1, end1 = self.segments[seg_idx1]
        obs1 = self.observations[start1:end1]
        actions1 = self.actions[start1:end1]
        
        # Get second segment
        start2, end2 = self.segments[seg_idx2]
        obs2 = self.observations[start2:end2]
        actions2 = self.actions[start2:end2]
        
        return {
            'obs1': obs1,           # [seq_len, obs_dim]
            'actions1': actions1,   # [seq_len, action_dim]
            'obs2': obs2,           # [seq_len, obs_dim]
            'actions2': actions2,   # [seq_len, action_dim]
            'time_diff': torch.tensor(time_diff, dtype=torch.long),  # Scalar
            'seg_idx1': seg_idx1,   # For debugging
            'seg_idx2': seg_idx2,   # For debugging
        }


def create_skill_data_loaders(
    data: Dict[str, torch.Tensor],
    segment_length: int = 32,
    max_temporal_diff: int = 100,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    obs_key: str = "obs",
    action_key: str = "action",
    normalize_obs: bool = True,
    norm_method: str = "min_max",
    num_workers: int = 0,
    seed: int = 42
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders for skill learning.
    
    Args:
        data: Dictionary containing trajectory data
        segment_length: Length of sub-trajectories
        max_temporal_diff: Maximum temporal difference between pairs
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        obs_key: Key for observations in data
        action_key: Key for actions in data
        normalize_obs: Whether to normalize observations
        norm_method: Normalization method
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    

    # Make sure data is on cpu
    data = {k: v.cpu() for k, v in data.items()}
    
    # Create dataset
    dataset = SkillDataset(
        data=data,
        segment_length=segment_length,
        max_temporal_diff=max_temporal_diff,
        obs_key=obs_key,
        action_key=action_key,
        normalize_obs=normalize_obs,
        norm_method=norm_method
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split dataset
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Check if CUDA is available for pin_memory
    use_pin_memory = torch.cuda.is_available()
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'dataset': dataset  # Return original dataset for access to normalization params
    } 