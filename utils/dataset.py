import random

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class PreferenceDataset(Dataset):
    """Dataset for segment preference pairs."""

    def __init__(
        self,
        data,
        segment_pairs,
        segment_indices,
        preferences,
        costs=None,
        normalize_obs=False,
        norm_method="standard",
        norm_stats=None,
        use_images=False,
        image_key="image",
        obs_key="obs",
        action_key="action",
        normalize_images=True,
        use_image_embeddings=False,
        image_embedding_key="image_embedding"
    ):
        """
        Initialize the dataset for preference learning.

        Args:
            data: Dictionary containing observations, actions, and optionally images/embeddings
            segment_pairs: List of pairs of segment indices [(i, j), ...]
            segment_indices: List of segment start/end indices [(start, end), ...]
            preferences: List of preferences (1 = first segment preferred, 2 = second segment preferred)
            costs: DTW costs for each segment pair (optional)
            normalize_obs: Whether to normalize observations (default: False)
            norm_method: Normalization method ('standard' or 'minmax')
            norm_stats: Pre-computed normalization statistics dict (optional)
            use_images: Whether to include images in the data
            image_key: Key for image data in the data dictionary
            obs_key: Key for observation data in the data dictionary
            action_key: Key for action data in the data dictionary
            normalize_images: Whether to normalize images to [0, 1] range
            use_image_embeddings: Whether to use pre-computed image embeddings
            image_embedding_key: Key for image embeddings in the data dictionary
        """
        # Convert all tensors to float32
        self.data = {
            k: v.float() if isinstance(v, torch.Tensor) else v 
            for k, v in data.items()
        }
        self.segment_pairs = segment_pairs
        self.segment_indices = segment_indices
        self.preferences = preferences
        self.costs = costs
        self.normalize_obs = normalize_obs
        self.norm_method = norm_method
        self.use_images = use_images
        self.image_key = image_key
        self.obs_key = obs_key
        self.action_key = action_key
        self.normalize_images = normalize_images
        self.use_image_embeddings = use_image_embeddings
        self.image_embedding_key = image_embedding_key

        # Verify data contains required keys
        if self.use_images:
            if self.use_image_embeddings:
                if self.image_embedding_key not in self.data:
                    raise ValueError(f"Image embedding key '{self.image_embedding_key}' not found in data")
                print(f"Using pre-computed image embeddings with shape: {self.data[self.image_embedding_key].shape}")
            else:
                if self.image_key not in self.data:
                    raise ValueError(f"Image key '{self.image_key}' not found in data")
                print(f"Using raw images with shape: {self.data[self.image_key].shape}")

        if self.obs_key not in self.data:
            raise ValueError(f"Observation key '{self.obs_key}' not found in data")
        if self.action_key not in self.data:
            raise ValueError(f"Action key '{self.action_key}' not found in data")

        # Get data length
        self.data_length = len(self.data[self.obs_key])

        # Process images if using raw images
        if self.use_images and not self.use_image_embeddings:
            images = self.data[self.image_key]
            
            # Check if images need to be reordered from HWC to CHW
            if images.shape[-1] == 3:  # If last dimension is 3, it's in HWC format
                print("Converting images from HWC to CHW format")
                # Permute dimensions to CHW format
                images = images.permute(0, 3, 1, 2)
                self.data[self.image_key] = images
            
            # Normalize images if needed
            if self.normalize_images:
                if images.max() > 1.0:
                    print("Normalizing images from [0, 255] to [0, 1] range")
                    self.data[self.image_key] = images.float() / 255.0
                print(f"Image range: [{self.data[self.image_key].min():.3f}, {self.data[self.image_key].max():.3f}]")
            
            print(f"Final image shape: {self.data[self.image_key].shape} (N, C, H, W)")

        # Compute or use provided normalization statistics
        self.norm_stats = {}
        if self.normalize_obs:
            if norm_stats is not None:
                self.norm_stats = {k: v.float() for k, v in norm_stats.items()}
                print("Using provided normalization statistics")
            else:
                print("Computing observation normalization statistics...")
                self._compute_normalization_statistics()

            # Print summary of normalization statistics
            if self.norm_method == "standard":
                print(
                    f"Observation mean: {self.norm_stats['mean'].mean().item():.4f}, std: {self.norm_stats['std'].mean().item():.4f}"
                )
            elif self.norm_method == "minmax":
                print(
                    f"Observation min: {self.norm_stats['min'].mean().item():.4f}, max: {self.norm_stats['max'].mean().item():.4f}"
                )

    def _compute_normalization_statistics(self):
        """Compute normalization statistics from the dataset."""
        obs = self.data[self.obs_key]

        if self.norm_method == "standard":
            mean = obs.mean(dim=0)
            std = obs.std(dim=0)
            std = torch.where(std > 1e-6, std, torch.ones_like(std))
            self.norm_stats["mean"] = mean
            self.norm_stats["std"] = std

        elif self.norm_method == "minmax":
            min_vals = obs.min(dim=0)[0]
            max_vals = obs.max(dim=0)[0]
            range_vals = max_vals - min_vals
            valid_range = torch.where(range_vals > 1e-6, range_vals, torch.ones_like(range_vals))
            self.norm_stats["min"] = min_vals
            self.norm_stats["max"] = max_vals
            self.norm_stats["range"] = valid_range

    def _normalize_observations(self, obs):
        """Apply normalization to the observations."""
        if not self.normalize_obs or not self.norm_stats:
            return obs

        if self.norm_method == "standard":
            return (obs - self.norm_stats["mean"]) / self.norm_stats["std"]
        elif self.norm_method == "minmax":
            return (obs - self.norm_stats["min"]) / self.norm_stats["range"]

        return obs

    def __len__(self):
        return len(self.segment_pairs)

    def __getitem__(self, idx):
        # Get segment indices
        seg_idx1, seg_idx2 = self.segment_pairs[idx]
        start1, end1 = self.segment_indices[seg_idx1]
        start2, end2 = self.segment_indices[seg_idx2]

        # Get observations and actions
        obs1 = self.data[self.obs_key][start1:end1]
        obs2 = self.data[self.obs_key][start2:end2]
        actions1 = self.data[self.action_key][start1:end1]
        actions2 = self.data[self.action_key][start2:end2]

        # Apply normalization to observations
        if self.normalize_obs:
            obs1 = self._normalize_observations(obs1)
            obs2 = self._normalize_observations(obs2)

        # Get images or image embeddings if using them
        if self.use_images:
            if self.use_image_embeddings:
                images1 = self.data[self.image_embedding_key][start1:end1]
                images2 = self.data[self.image_embedding_key][start2:end2]
            else:
                images1 = self.data[self.image_key][start1:end1]
                images2 = self.data[self.image_key][start2:end2]
        else:
            images1 = images2 = None

        # Convert preference to tensor
        if isinstance(self.preferences[idx], torch.Tensor):
            pref = self.preferences[idx].float()
        else:
            pref = torch.tensor(self.preferences[idx], dtype=torch.float)

        # Costs
        if self.costs is not None:
            cost = self.costs[idx]
            if not isinstance(cost, torch.Tensor):
                cost = torch.tensor(cost, dtype=torch.float)
            else:
                cost = cost.float()
        else:
            cost = None

        # Build return dictionary with required fields
        return_dict = {
            'obs1': obs1.float(),
            'obs2': obs2.float(),
            'actions1': actions1.float(),
            'actions2': actions2.float(),
            'preference': pref.float(),
        }

        # Only add images if they exist
        if images1 is not None and images2 is not None:
            return_dict['images1'] = images1.float()
            return_dict['images2'] = images2.float()

        # Only add cost if it exists
        if cost is not None:
            return_dict['cost'] = cost.float()

        return return_dict


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
    if dataset.costs is not None:
        shuffled_costs = [dataset.costs[i] for i in indices]
    else:
        shuffled_costs = None

    # Create a new dataset with the shuffled pairs
    shuffled_dataset = PreferenceDataset(
        dataset.data,
        shuffled_segment_pairs,
        dataset.segment_indices,
        shuffled_preferences,
        costs=shuffled_costs,
        normalize_obs=dataset.normalize_obs,
        norm_method=dataset.norm_method,
        norm_stats=dataset.norm_stats,
        use_images=dataset.use_images,
        image_key=dataset.image_key,
        obs_key=dataset.obs_key,
        action_key=dataset.action_key,
        normalize_images=dataset.normalize_images,
        use_image_embeddings=dataset.use_image_embeddings,
        image_embedding_key=dataset.image_embedding_key
    )

    print(f"Shuffled preference dataset with {len(shuffled_dataset)} pairs")
    return shuffled_dataset


def create_data_loaders(
    preference_dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    batch_size=32,
    num_workers=4,
    seed=42,
    normalize_obs=False,
    norm_method="standard",
    shuffle_dataset=True,
):
    """Create data loaders for training, validation, and testing.

    Args:
        preference_dataset: Dataset containing segment preference pairs
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
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
            costs=preference_dataset.costs,
            normalize_obs=True,
            norm_method=norm_method,
            use_images=preference_dataset.use_images,
            image_key=preference_dataset.image_key,
            obs_key=preference_dataset.obs_key,
            action_key=preference_dataset.action_key,
            normalize_images=preference_dataset.normalize_images,
            use_image_embeddings=preference_dataset.use_image_embeddings,
            image_embedding_key=preference_dataset.image_embedding_key
        )

    # Shuffle the dataset to ensure random sampling if requested
    if shuffle_dataset:
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
