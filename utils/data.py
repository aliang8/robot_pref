import numpy as np


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

def process_data_trajectories(data_path):
    """
    Load and process data into trajectories based on "episode" key from a data file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        trajectories: List of processed trajectories.
    """
    data = load_tensordict(data_path)

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
