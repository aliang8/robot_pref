import numpy as np
import torch
# from d3rlpy.datasets import MDPDataset
from tqdm import tqdm
import random


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
    # print(
    #     f"Loaded TensorDict with shape: {data['image'].shape}, device: {data['image'].device}"
    # )
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
        list: List of preference labels (1 if first segment preferred, 0 if second segment preferred, 0.5 if equal).
    """
    if "reward" not in data:
        raise ValueError("Dataset does not contain 'reward' key. Can't compute ground truth preferences! ")
    
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
        elif return1 < return2:
            preference_labels.append(0)
        else:  # equal
            preference_labels.append(0.5)

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
    # assert len(set(episode_lens)) == 1, "All episodes should be the same length"
    unique_episode_lens = np.unique(episode_lens)
    print(f"Unique episode lengths: {unique_episode_lens}")
    
    # only keep segments that are the same length
    episode_len_gt = episode_lens[10]
    print(f"Episode length use: {episode_len_gt}")

    # Calculate segments_per_trajectory based on the episode length
    segments_per_trajectory = episode_len_gt // segment_length + 1

    # Get segments from each episode
    segments = []
    segment_indices = []
    unique_episodes = np.unique(data["episode"])

    # Compute the starting absolute index for each episode in the full dataset
    episode_start_indices = {}
    abs_idx = 0
    print(f"Segmenting {len(unique_episodes)} episodes")
    for episode_idx in tqdm(unique_episodes):
        episode_mask = data["episode"] == episode_idx
        episode_len = np.sum(episode_mask.numpy())
        if episode_len != episode_len_gt:
            print(f"Episode length {episode_len} != {episode_len_gt}, skipping")
            continue
        episode_start_indices[episode_idx] = abs_idx
        episode_abs_start = abs_idx

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

        abs_idx += episode_len

    print(f"Segmented {len(segments)} segments")
    return segments, segment_indices


def segment_episodes_random(data, segment_length, num_segments=None):
    """Segment episodes into smaller segments, handling variable episode lengths.

    Args:
        data: Raw TensorDict data to segment
        segment_length: Base length of each segment
        num_segments: Number of segments to sample (if None, uses all possible segments)

    Returns:
        segments: List of segments
        segment_indices: List of (start_idx, end_idx) tuples for each segment
    """
    # Get unique episodes and their lengths
    unique_episodes = np.unique(data["episode"])
    episode_lens = {
        int(ep): len(np.where(data["episode"] == ep)[0]) 
        for ep in unique_episodes
    }
    
    print(f"Found {len(unique_episodes)} episodes")
    print(f"Episode lengths range: min={min(episode_lens.values())}, max={max(episode_lens.values())}")
    
    # Create list of valid episodes (those long enough for at least one segment)
    valid_episodes = []
    episode_start_indices = {}  # Track start index of each episode
    current_idx = 0
    
    for episode_idx in unique_episodes:
        episode_len = episode_lens[int(episode_idx)]
        if episode_len >= segment_length:
            valid_episodes.append((episode_idx, episode_len))
            episode_start_indices[int(episode_idx)] = current_idx
        current_idx += episode_len
    
    print(f"Found {len(valid_episodes)} valid episodes (length >= {segment_length})")
    
    if not valid_episodes:
        raise ValueError(f"No episodes found with length >= {segment_length}")
    
    # Sample segments
    segments = []
    segment_indices = []
    
    # Sample episodes and their segments
    remaining_segments = num_segments if num_segments is not None else float('inf')
    attempts = 0
    max_attempts = len(valid_episodes) * 100  # Increased max attempts
    
    while remaining_segments > 0 and valid_episodes and attempts < max_attempts:
        attempts += 1
        
        # Randomly select an episode
        episode_idx, episode_len = random.choice(valid_episodes)
        episode_abs_start = episode_start_indices[int(episode_idx)]
        
        # Calculate number of segments to take from this episode
        max_start = episode_len - segment_length
        if max_start < 0:
            continue
            
        # Randomly sample a starting point
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + segment_length
        
        # Compute absolute indices in the full dataset
        abs_start_idx = episode_abs_start + start_idx
        abs_end_idx = episode_abs_start + end_idx
        
        # Validate segment length
        if abs_end_idx - abs_start_idx != segment_length:
            continue
            
        # Create segment dictionary and validate data
        segment = {}
        valid_segment = True
        for key in data.keys():
            segment_data = data[key][abs_start_idx:abs_end_idx]
            if len(segment_data) != segment_length:
                valid_segment = False
                break
            segment[key] = segment_data
        
        if not valid_segment:
            continue
            
        segment_indices.append((abs_start_idx, abs_end_idx))
        segments.append(segment)
        remaining_segments -= 1
    
    if not segments:
        raise ValueError("Failed to create any valid segments")
        
    print(f"Created {len(segments)} valid segments")
    return segments, segment_indices


def load_dataset(
    data,
    reward_model=None,
    device=None,
    use_ground_truth=False,
    max_segments=None,
    reward_batch_size=32,
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
        use_zero_rewards: If True, replace all rewards with zeros (sanity check)

    Returns:
        d3rlpy MDPDataset with observations, actions, rewards, and terminals
    """

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
