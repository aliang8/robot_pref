import random

import numpy as np
import torch
import random

import random

import numpy as np
import torch

import utils.env as utils_env

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from tqdm import tqdm
import utils.env as utils_env
import cv2


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
        raise ValueError(
            "Dataset does not contain 'reward' key. Can't compute ground truth preferences! "
        )

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


def segment_episodes_random(data, segment_length, num_segments=None, val_split=0.1):
    """Segment episodes into num_segments segments of segment_length randomly. Split into test and train segments.

    Args:
        data: Raw TensorDict data to segment
        segment_length: Base length of each segment
        num_segments: Number of segments to sample
        val_split: Percentage of data to hold our for validation. Additioanlly, samples val_split * num_segments segments.

    Returns:
        segments: List of segments
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        val_segments: List of validation segments
        val_segment_indices: List of (start_idx, end_idx) tuples for each validation segment
        val_samples: List of unique episode IDs used for validation
    """
    # Get unique episodes and their lengths
    unique_episodes = np.unique(data["episode"])

    # Randomly sample episodes to be held out for validation
    num_val_samples = int(len(unique_episodes) * val_split)
    val_samples = np.random.choice(unique_episodes, size=num_val_samples, replace=False)

    episode_lens = {
        int(ep): len(np.where(data["episode"] == ep)[0]) for ep in unique_episodes
    }

    print(f"Found {len(unique_episodes)} episodes")
    print(
        f"Episode lengths range: min={min(episode_lens.values())}, max={max(episode_lens.values())}"
    )

    # Create list of train episodes and val episodes
    train_episodes = []
    val_episodes = []
    episode_start_indices = {}  # Track start index of each episode
    current_idx = 0

    for episode_idx in unique_episodes:
        episode_len = episode_lens[int(episode_idx)]
        if episode_len <= segment_length:
            raise ValueError(
                f"Episode {episode_idx} is too short, use a shorter segment_length"
            )
        if episode_idx not in val_samples:
            train_episodes.append((episode_idx, episode_len))
        else:
            val_episodes.append((episode_idx, episode_len))
        episode_start_indices[episode_idx] = current_idx
        current_idx += episode_len

    print(
        f"Using {len(train_episodes)} train episodes and {len(val_episodes)} val episodes"
    )

    # Sample segments
    segments = []
    segment_indices = []

    # Sample num_segments segments
    total_segments = 0
    while total_segments < num_segments:
        # Randomly select an episode
        episode_idx, episode_len = random.choice(train_episodes)
        episode_abs_start = episode_start_indices[episode_idx]

        # Sample a random valid segment
        max_start = episode_len - segment_length - 1  # we don't want to sample next obs
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + segment_length

        # Compute absolute indices in the full dataset
        abs_start_idx = episode_abs_start + start_idx
        abs_end_idx = episode_abs_start + end_idx

        # Create segment dictionary
        segment = {}
        for key in data.keys():
            segment_data = data[key][abs_start_idx:abs_end_idx]
            segment[key] = segment_data

        segment_indices.append((abs_start_idx, abs_end_idx))
        segments.append(segment)
        total_segments += 1

    print(f"Created {len(segments)} valid train segments")

    # Sample val segments as well
    num_val_segments = num_segments * val_split
    total_segments = 0

    val_segments = []
    val_segment_indices = []

    while total_segments < num_val_segments:
        # Randomly select an episode
        episode_idx, episode_len = random.choice(val_episodes)
        episode_abs_start = episode_start_indices[episode_idx]

        # Sample a random valid segment
        max_start = episode_len - segment_length
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + segment_length

        # Compute absolute indices in the full dataset
        abs_start_idx = episode_abs_start + start_idx
        abs_end_idx = episode_abs_start + end_idx

        # Create segment dictionary
        segment = {}
        for key in data.keys():
            segment_data = data[key][abs_start_idx:abs_end_idx]
            segment[key] = segment_data

        val_segment_indices.append((abs_start_idx, abs_end_idx))
        val_segments.append(segment)
        total_segments += 1

    print(f"Created {len(val_segments)} valid val segments")

    return segments, segment_indices, val_segments, val_segment_indices, val_samples


def load_datasets(config):
    """Load main dataset and cross dataset if needed."""
    # Load main dataset
    if "metaworld" in config.env:
        dataset = utils_env.MetaWorld_dataset(config)
    elif "dmc" in config.env:
        dataset = utils_env.DMC_dataset(config)
        config.threshold *= 0.1  # Different reward scaling
    elif "robomimic" in config.env:
        dataset = Robomimic_dataset(config.data_path, return_images=True)
    else:
        raise ValueError(f"Unsupported environment type: {config.env}")

    # Load cross dataset if needed
    cross_dataset = None
    if config.get("cross_data_path"):
        cross_dataset = Robomimic_dataset(config.cross_data_path, return_images=True)

    return dataset, cross_dataset


def normalize_datasets(dataset):
    """Normalize observations and goal points for both datasets."""

    def normalize_dataset(data):
        # Normalize observations
        state_mean = data["observations"].mean(axis=0)
        state_std = np.maximum(data["observations"].std(axis=0) + 1e-8, 1e-2)

        data["observations"] = (data["observations"] - state_mean) / state_std
        data["next_observations"] = (data["next_observations"] - state_mean) / state_std

        # Normalize goal points
        if "goal_points" in data:
            goal_points_mean = data["goal_points"].mean(axis=0)
            goal_points_std = data["goal_points"].std(axis=0) + 1e-8
            data["goal_points"] = (
                data["goal_points"] - goal_points_mean
            ) / goal_points_std

        return state_mean, state_std

    return normalize_dataset(dataset)


def convert_labels_to_array(label_list):
    """Convert preference labels to array format."""
    label_map = {1: [1, 0], 0: [0, 1]}
    return np.array([label_map.get(label, [0.5, 0.5]) for label in label_list])


def create_segment_indices(idx_st_list, segment_size):
    """Create segment indices from start indices."""
    return [[j for j in range(i, i + segment_size)] for i in idx_st_list]


def get_obs_act_data(dataset, idx_list):
    """Get concatenated observations and actions for segments."""
    return np.concatenate(
        (dataset["observations"][idx_list], dataset["actions"][idx_list]), axis=-1
    )


def get_images_data(dataset, idx_list):
    """Get images for segments if available."""
    return dataset["images"][idx_list] if "images" in dataset else None


def get_eef_data(dataset, idx_list, with_goal=False):
    """Get end-effector positions for segments."""
    if with_goal:
        return np.concatenate(
            (
                dataset["observations"][:, :3][idx_list],
                dataset["goal_points"][idx_list],
            ),
            axis=-1,
        )
    return dataset["observations"][:, :3][idx_list]


def Robomimic_dataset(
    data_path, return_images=False, clip_last=False, filter_data=False
):
    """
    Load Robomimic dataset and build:
    If clip_last, we don't use the last transition for IQL
    """

    print(f"Loading data from: {data_path}")

    with h5py.File(data_path, "r") as f:
        data = f["data"]

        all_observations = []
        all_next_observations = []
        all_actions = []
        all_rewards = []
        all_images = []
        all_terminals = []

        all_goal_points = []

        print(f"Found {len(data.keys())} trajectories in dataset")

        for demo in tqdm(
            sorted(data.keys(), key=lambda x: int(x.split("_")[1])),
            desc="Processing demos",
        ):
            demo_data = data[demo]
            # Concatenate observation components
            obs = np.concatenate(
                [
                    demo_data["obs"]["robot0_eef_pos"][:],
                    demo_data["obs"]["robot0_eef_quat"][:],
                    demo_data["obs"]["robot0_gripper_qpos"][:],
                    demo_data["obs"]["object"][:],
                ],
                axis=1,
            )

            if clip_last:
                next_obs = obs[1:]
                obs = obs[:-1]

                acts = demo_data["actions"][:-1]
                rewards = demo_data["rewards"][:-1]
                images = demo_data["obs"]["agentview_image"][:-1]

                goal_points = demo_data["goal_points"][:-1]
            else:
                next_obs = obs

                acts = demo_data["actions"]
                rewards = demo_data["rewards"]
                images = demo_data["obs"]["agentview_image"]

                goal_points = demo_data["goal_points"]

            all_observations.append(obs)
            all_next_observations.append(next_obs)
            all_actions.append(acts)
            all_rewards.append(rewards)

            all_goal_points.append(goal_points)

            if images.shape[1] != 84:
                # Assuming images are in format (batch, height, width, channels)
                resized_images = np.array([cv2.resize(img, (84, 84)) for img in images])
                all_images.append(resized_images)
            else:
                all_images.append(images)
            # Create terminals array - True only for the last step of each episode
            episode_length = len(acts)
            terminals = np.zeros(episode_length, dtype=bool)
            terminals[-1] = True  # Mark the last step as terminal
            all_terminals.append(terminals)

        if filter_data:
            # Filter out trajectories that fall one standard deviation below the mean
            rewards_sum = np.array([rewards.mean() for rewards in all_rewards])
            mu, std = rewards_sum.mean(), rewards_sum.std()

            keep_mask = rewards_sum >= (mu - std)

            all_observations = [
                all_observations[i] for i in range(len(keep_mask)) if keep_mask[i]
            ]
            all_next_observations = [
                all_next_observations[i] for i in range(len(keep_mask)) if keep_mask[i]
            ]
            all_actions = [
                all_actions[i] for i in range(len(keep_mask)) if keep_mask[i]
            ]
            all_rewards = [
                all_rewards[i] for i in range(len(keep_mask)) if keep_mask[i]
            ]
            all_images = [all_images[i] for i in range(len(keep_mask)) if keep_mask[i]]
            all_terminals = [
                all_terminals[i] for i in range(len(keep_mask)) if keep_mask[i]
            ]
            all_goal_points = [
                all_goal_points[i] for i in range(len(keep_mask)) if keep_mask[i]
            ]

            print(
                f"Filtered out {len(rewards_sum) - sum(keep_mask)} trajectories based on rewards"
            )

        # Convert to numpy arrays
        observations = np.concatenate(all_observations, axis=0)
        next_observations = np.concatenate(all_next_observations, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        images = np.concatenate(all_images, axis=0)
        terminals = np.concatenate(all_terminals, axis=0)
        goal_points = np.concatenate(all_goal_points, axis=0)

    print(f"Total number of transitions: {len(observations)}")

    dataset = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "goal_points": goal_points,
    }
    if return_images:
        dataset["images"] = images

    return dataset
