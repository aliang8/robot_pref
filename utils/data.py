import numpy as np
import torch
from d3rlpy.datasets import MDPDataset
from tqdm import tqdm


# Define a simple AttrDict class that provides dot access to dictionaries
class AttrDict(dict):
    """A dictionary subclass that allows attribute-style access.

    This provides a more convenient way to access dict elements using
    dot notation (dict.key) in addition to subscript notation (dict['key']).

    It also handles nested dictionaries by recursively converting them
    to AttrDict objects.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        """Create nested AttrDict from nested dict.

        Args:
            data: A dictionary, potentially with nested dictionaries

        Returns:
            An AttrDict with all nested dictionaries also converted to AttrDict
        """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dict(data[key]) for key in data})


def load_tensordict(file_path):
    """Load tensordict data from file."""
    data = torch.load(file_path, weights_only=False)
    print(
        f"Loaded TensorDict with shape: {data['image'].shape}, device: {data['image'].device}"
    )
    print(f"Fields: {list(data.keys())}")
    return data


def process_data_trajectories(data, device="cpu"):
    """
    Load and process data into trajectories based on "episode" key from a data file.

    Args:
        data (str): Raw TensorDict data to process.

    Returns:
        trajectories: List of processed trajectories.
    """
    data = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    # Group data by episode
    unique_episodes = data["episode"].unique()

    trajectories = []
    # Process each unique episode
    for episode_num in unique_episodes:
        episode_mask = data["episode"] == episode_num

        trajectory = {}
        for key in data.keys():
            trajectory[key] = data[key][episode_mask]

        trajectories.append(trajectory)

    print(f"Loaded {len(trajectories)} trajectories")

    return trajectories


def segment_trajectory(trajectory, segment_length, segments_per_trajectory=3):
    """
    Segment a trajectory into segments_per_trajectory parts of equal length.

    Args:
        trajectory (Tensor): The trajectory data.
        segment_length (int): The length of each segment.
        segments_per_trajectory (int): Number of segments to extract per trajectory.

    Returns:
        list: List of segments with length segment_length.
    """
    total_length = len(trajectory["obs"])

    segments = []
    for i in range(segments_per_trajectory):
        # Calculate evenly spaced starting points across the trajectory
        start_idx = (
            i * (total_length - segment_length) // max(1, segments_per_trajectory - 1)
        )
        end_idx = start_idx + segment_length

        segment = {}
        for key in trajectory.keys():
            segment[key] = trajectory[key][start_idx:end_idx]
        segments.append(segment)

    return segments


def get_gt_preferences(data, segment_indices, pairs):
    """
    Get ground truth preferences for segments based on cumulative rewards.

    Args:
        data (dict): Raw TensorDict data.
        segment_indices (list): List of tuples containing (start_idx, end_idx) for all segments.
        pairs (list): List of tuples containing segment indices to compute ground truth preferences.

    Returns:
        list: List of preference labels (1 if first segment preferred, 2 if second segment preferred).
    """
    preference_labels = []

    for idx1, idx2 in tqdm(pairs):
        # Get the segment indices
        start_idx1, end_idx1 = segment_indices[idx1]
        start_idx2, end_idx2 = segment_indices[idx2]

        return1 = data["reward"][start_idx1:end_idx1].sum().item()
        return2 = data["reward"][start_idx2:end_idx2].sum().item()

        # Determine preference
        if return1 > return2:
            preference_labels.append(1)
        else:
            preference_labels.append(2)

    return preference_labels


def segment_episodes(data, segment_length):
    """Segment episodes into smaller segments.

    Args:
        data: Raw TensorDict data to segment
        segment_length: Length of each segment

    Returns:
        segments: List of segments
    """
    episode_lens = [
        len(np.where(data["episode"] == i)[0]) for i in np.unique(data["episode"])
    ]
    assert len(set(episode_lens)) == 1, "All episodes should be the same length"
    episode_len = episode_lens[0]
    print(f"Episode length: {episode_len}")

    # Calculate segments_per_trajectory based on the episode length
    segments_per_trajectory = episode_len // segment_length + 1

    # Get segments from each episode
    segments = []
    segment_indices = []
    unique_episodes = np.unique(data["episode"])

    # Compute the starting absolute index for each episode in the full dataset
    episode_start_indices = {}
    abs_idx = 0
    print(f"Segmenting {len(unique_episodes)} episodes")
    for episode_idx in unique_episodes:
        episode_mask = data["episode"] == episode_idx
        episode_len = np.sum(episode_mask.numpy())
        episode_start_indices[episode_idx] = abs_idx
        abs_idx += episode_len

    for episode_idx in tqdm(unique_episodes):
        episode_abs_start = episode_start_indices[episode_idx]

        # Create segments_per_trajectory evenly spaced segments
        for i in range(segments_per_trajectory):
            # Calculate evenly spaced starting points across the trajectory
            start_idx = (
                i
                * (episode_len - segment_length)
                // max(1, segments_per_trajectory - 1)
            )
            end_idx = start_idx + segment_length

            # Ensure end_idx doesn't exceed the episode length
            end_idx = min(end_idx, episode_len)

            # Compute absolute indices in the full dataset
            abs_start_idx = episode_abs_start + start_idx
            abs_end_idx = episode_abs_start + end_idx

            segment_indices.append((abs_start_idx, abs_end_idx))

            # Create segment dictionary
            segment = {}
            for key in data.keys():
                segment[key] = data[key][abs_start_idx:abs_end_idx]

            segments.append(segment)

    print(f"Segmented {len(segments)} segments")
    return segments, segment_indices


def load_dataset(
    data,
    reward_model=None,
    device=None,
    use_ground_truth=False,
    max_segments=None,
    reward_batch_size=32,
    scale_rewards=False,
    reward_min=None,
    reward_max=None,
    use_zero_rewards=False,
):
    """Load and process dataset for either IQL or BC training.

    Args:
        data: TensorDict with observations, actions, rewards, and episode IDs
        reward_model: Trained reward model (required for IQL, None for BC)
        device: Device to run the reward model on (required for IQL)
        use_ground_truth: If True, use ground truth rewards for IQL instead of reward model predictions
        max_segments: Maximum number of segments to process (optional)
        reward_batch_size: Batch size for reward computation (for IQL)
        scale_rewards: If True, scales rewards to specified min/max range
        reward_min: Minimum value for scaled rewards (default: -1)
        reward_max: Maximum value for scaled rewards (default: 1)
        use_zero_rewards: If True, replace all rewards with zeros (sanity check)

    Returns:
        d3rlpy MDPDataset with observations, actions, rewards, and terminals
    """
    # Set default scaling values if not provided
    if scale_rewards:
        reward_min = reward_min if reward_min is not None else -1.0
        reward_max = reward_max if reward_max is not None else 1.0
        print(f"Scaling rewards to range [{reward_min}, {reward_max}]")

    # Extract necessary data
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    episode_ids = data["episode"]

    # For BC or ground truth rewards, extract original rewards
    if reward_model is None or use_ground_truth:
        if "reward" not in data:
            raise ValueError(
                "Ground truth rewards requested but 'reward' not found in data."
            )
        rewards = data["reward"].cpu()

    # Make sure data is on CPU for preprocessing
    observations = observations.cpu()
    actions = actions.cpu()
    episode_ids = episode_ids.cpu()

    # Filter out observations with NaN values
    valid_mask = ~torch.isnan(observations).any(dim=1) & ~torch.isnan(actions).any(
        dim=1
    )
    if reward_model is None or use_ground_truth:
        valid_mask = valid_mask & ~torch.isnan(rewards)

    if not valid_mask.any():
        raise ValueError("No valid observations found in the dataset.")

    # Extract valid data
    valid_obs = observations[valid_mask]
    valid_actions = actions[valid_mask]
    valid_episodes = episode_ids[valid_mask]

    if reward_model is None or use_ground_truth:
        valid_rewards = rewards[valid_mask].numpy()

    print(
        f"Using {valid_obs.shape[0]} valid observations out of {observations.shape[0]} total"
    )

    # Process rewards based on algorithm and options
    if reward_model is not None and not use_ground_truth:
        # IQL with reward model - Process in manageable batches
        process_batch_size = reward_batch_size or 1024
        all_rewards = []

        # Compute rewards using the trained reward model
        reward_model.eval()  # Ensure model is in evaluation mode

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, len(valid_obs), process_batch_size), desc="Computing rewards"
            ):
                end_idx = min(start_idx + process_batch_size, len(valid_obs))

                # Move batch to device
                batch_obs = valid_obs[start_idx:end_idx].to(device)
                batch_actions = valid_actions[start_idx:end_idx].to(device)

                # Compute rewards, need the per step reward not the summed reward
                batch_rewards = reward_model(batch_obs, batch_actions).cpu().numpy()
                all_rewards.append(batch_rewards)
        # Combine all rewards
        if len(all_rewards) == 1:
            rewards_np = all_rewards[0]
        else:
            rewards_np = np.concatenate(all_rewards)
    else:
        # BC or IQL with ground truth - use the extracted rewards
        rewards_np = valid_rewards

    print(
        f"Rewards max: {np.max(rewards_np)}, min: {np.min(rewards_np)}, mean: {np.mean(rewards_np)}"
    )

    # Apply zero rewards if requested (sanity check)
    if use_zero_rewards:
        print("\n" + "=" * 60)
        print("⚠️ SANITY CHECK MODE: USING ZERO REWARDS FOR ALL TRANSITIONS ⚠️")
        print(
            "This mode replaces all rewards with zeros to test if policy learning depends on rewards."
        )
        print("=" * 60 + "\n")
        original_rewards = rewards_np.copy()
        rewards_np = np.zeros_like(rewards_np)
        print(
            f"Reward stats before zeroing - Mean: {np.mean(original_rewards):.4f}, Min: {np.min(original_rewards):.4f}, Max: {np.max(original_rewards):.4f}"
        )
        print(
            f"Reward stats after zeroing - Mean: {np.mean(rewards_np):.4f}, Min: {np.min(rewards_np):.4f}, Max: {np.max(rewards_np):.4f}"
        )

    # Scale rewards if requested
    if scale_rewards:
        original_min = np.min(rewards_np)
        original_max = np.max(rewards_np)

        # Avoid division by zero
        if original_max - original_min > 1e-8:
            # Scale to [0, 1] first, then to target range
            rewards_np = (rewards_np - original_min) / (original_max - original_min)
            rewards_np = rewards_np * (reward_max - reward_min) + reward_min
            print(
                f"Scaled rewards from [{original_min:.4f}, {original_max:.4f}] to [{reward_min:.4f}, {reward_max:.4f}]"
            )
        else:
            # If all rewards are the same, set to the middle of the target range
            middle_value = (reward_max + reward_min) / 2
            rewards_np = np.ones_like(rewards_np) * middle_value
            print(
                f"All rewards have the same value ({original_min:.4f}), setting to {middle_value:.4f}"
            )

    # Create terminals array (True at the end of each episode)
    episode_ends = torch.cat(
        [
            valid_episodes[1:] != valid_episodes[:-1],
            torch.tensor([True]),  # Last observation is always an episode end
        ]
    )
    terminals_np = episode_ends.numpy()

    # Convert to numpy for d3rlpy
    observations_np = valid_obs.numpy()
    actions_np = valid_actions.numpy()

    # Create MDPDataset with the rewards
    dataset = MDPDataset(
        observations=observations_np,
        actions=actions_np,
        rewards=rewards_np,
        terminals=terminals_np,
    )

    # Print final dataset statistics
    print(
        f"Final dataset size: {dataset.size()} transitions with {dataset.size() - np.sum(terminals_np)} non-terminal transitions"
    )
    reward_stats = {
        "mean": np.mean(rewards_np),
        "std": np.std(rewards_np),
        "min": np.min(rewards_np),
        "max": np.max(rewards_np),
    }
    print(
        f"Reward statistics: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}, min={reward_stats['min']:.4f}, max={reward_stats['max']:.4f}"
    )

    return dataset
