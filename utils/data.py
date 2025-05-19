import torch
import numpy as np
import tqdm

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

def process_data_trajectories(data_path, device="cpu"):
    """
    Load and process data into trajectories based on "episode" key from a data file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        trajectories: List of processed trajectories.
    """
    data = load_tensordict(data_path)
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
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
        data (dict): Raw data containing all trajectories.
        segment_indices (list): List of tuples containing (episode_idx, start_idx, end_idx) for each segment.
        pairs (list): List of tuples containing segment indices to compute ground truth preferences.

    Returns:
        list: List of preference labels (1 if first segment preferred, 2 if second segment preferred).
    """
    preference_labels = []

    for idx1, idx2 in tqdm.tqdm(pairs):
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
    episode_lengths = [len(np.where(data["episode"] == i)[0]) for i in np.unique(data["episode"])]
    assert len(set(episode_lengths)) == 1, "All episodes should be the same length"

    # Calculate segments_per_trajectory based on the episode length
    segments_per_trajectory = episode_lengths[0] // segment_length + 1

    # Get segments from each episode
    segments = []
    segment_indices = []
    unique_episodes = np.unique(data["episode"])

    # Compute the starting absolute index for each episode in the full dataset
    episode_start_indices = {}
    abs_idx = 0
    for episode_idx in unique_episodes:
        episode_mask = data["episode"] == episode_idx
        episode_len = np.sum(episode_mask.numpy())
        episode_start_indices[episode_idx] = abs_idx
        abs_idx += episode_len

    for episode_idx in unique_episodes:
        # Extract the current episode data
        episode_mask = data["episode"] == episode_idx
        episode_data = {k: v[episode_mask] for k, v in data.items()}

        total_length = len(episode_data["obs"])
        episode_abs_start = episode_start_indices[episode_idx]

        # Create segments_per_trajectory evenly spaced segments
        for i in range(segments_per_trajectory):
            # Calculate evenly spaced starting points across the trajectory
            start_idx = (
                i * (total_length - segment_length) // max(1, segments_per_trajectory - 1)
            )
            end_idx = start_idx + segment_length

            # Ensure end_idx doesn't exceed the episode length
            end_idx = min(end_idx, total_length)

            # Compute absolute indices in the full dataset
            abs_start_idx = episode_abs_start + start_idx
            abs_end_idx = episode_abs_start + end_idx

            segment_indices.append((abs_start_idx, abs_end_idx))

            # Create segment dictionary
            segment = {}
            for key in episode_data.keys():
                segment[key] = episode_data[key][start_idx:end_idx]

            segments.append(segment)

    return segments, segment_indices

