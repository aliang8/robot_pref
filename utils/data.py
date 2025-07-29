import os
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import models.reward_model as reward_model
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from utils.env import get_robomimic_env
import cv2
import h5py

TensorBatch = List[torch.Tensor]


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
    Load and process data into trajectories based on "traj" key from a data file.

    Args:
        data (str): Raw TensorDict data to process.

    Returns:
        trajectories: List of processed trajectories.
    """
    data = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    # Group data by traj
    unique_trajs = data["traj"].unique()

    trajectories = []
    # Process each unique traj
    for traj_num in unique_trajs:
        traj_mask = data["traj"] == traj_num

        trajectory = {}
        for key in data.keys():
            trajectory[key] = data[key][traj_mask]

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


def segment_trajs(data, segment_length):
    """Segment trajs into smaller segments.

    Args:
        data: Raw TensorDict data to segment
        segment_length: Length of each segment

    Returns:
        segments: List of segments
    """
    traj_lens = [
        len(np.where(data["traj"] == i)[0]) for i in np.unique(data["traj"])
    ]
    # assert len(set(traj_lens)) == 1, "All trajs should be the same length"
    unique_traj_lens = np.unique(traj_lens)
    print(f"Unique traj lengths: {unique_traj_lens}")

    # only keep segments that are the same length
    traj_len_gt = traj_lens[10]
    print(f"Episode length use: {traj_len_gt}")

    # Calculate segments_per_trajectory based on the traj length
    segments_per_trajectory = traj_len_gt // segment_length + 1

    # Get segments from each traj
    segments = []
    segment_indices = []
    unique_trajs = np.unique(data["traj"])

    # Compute the starting absolute index for each traj in the full dataset
    traj_start_indices = {}
    abs_idx = 0
    print(f"Segmenting {len(unique_trajs)} trajs")
    for traj_idx in tqdm(unique_trajs):
        traj_mask = data["traj"] == traj_idx
        traj_len = np.sum(traj_mask.numpy())
        if traj_len != traj_len_gt:
            print(f"Episode length {traj_len} != {traj_len_gt}, skipping")
            continue
        traj_start_indices[traj_idx] = abs_idx
        traj_abs_start = abs_idx

        # Create segments_per_trajectory evenly spaced segments
        for i in range(segments_per_trajectory):
            # Calculate evenly spaced starting points across the trajectory
            start_idx = (
                i
                * (traj_len - segment_length)
                // max(1, segments_per_trajectory - 1)
            )
            end_idx = start_idx + segment_length

            # Ensure end_idx doesn't exceed the traj length
            end_idx = min(end_idx, traj_len)

            # Compute absolute indices in the full dataset
            abs_start_idx = traj_abs_start + start_idx
            abs_end_idx = traj_abs_start + end_idx

            segment_indices.append((abs_start_idx, abs_end_idx))

            # Create segment dictionary
            segment = {}
            for key in data.keys():
                segment[key] = data[key][abs_start_idx:abs_end_idx]

            segments.append(segment)

        abs_idx += traj_len

    print(f"Segmented {len(segments)} segments")
    return segments, segment_indices


def segment_trajs_random(data, segment_length, num_segments=None, val_split=0.1):
    """Segment trajs into num_segments segments of segment_length randomly. Split into test and train segments.

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
        val_samples: List of unique traj IDs used for validation
    """
    # Get unique trajs and their lengths
    unique_trajs = np.unique(data["traj"])

    # Randomly sample trajs to be held out for validation
    num_val_samples = int(len(unique_trajs) * val_split)
    val_samples = np.random.choice(unique_trajs, size=num_val_samples, replace=False)

    traj_lens = {
        int(ep): len(np.where(data["traj"] == ep)[0]) for ep in unique_trajs
    }

    print(f"Found {len(unique_trajs)} trajs")
    print(
        f"Episode lengths range: min={min(traj_lens.values())}, max={max(traj_lens.values())}"
    )

    # Create list of train trajs and val trajs
    train_trajs = []
    val_trajs = []
    traj_start_indices = {}  # Track start index of each traj
    current_idx = 0

    for traj_idx in unique_trajs:
        traj_len = traj_lens[int(traj_idx)]
        if traj_len <= segment_length:
            raise ValueError(
                f"Episode {traj_idx} is too short, use a shorter segment_length"
            )
        if traj_idx not in val_samples:
            train_trajs.append((traj_idx, traj_len))
        else:
            val_trajs.append((traj_idx, traj_len))
        traj_start_indices[traj_idx] = current_idx
        current_idx += traj_len

    print(
        f"Using {len(train_trajs)} train trajs and {len(val_trajs)} val trajs"
    )

    # Sample segments
    segments = []
    segment_indices = []

    # Sample num_segments segments
    total_segments = 0
    while total_segments < num_segments:
        # Randomly select an traj
        traj_idx, traj_len = random.choice(train_trajs)
        traj_abs_start = traj_start_indices[traj_idx]

        # Sample a random valid segment
        max_start = traj_len - segment_length - 1  # we don't want to sample next obs
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + segment_length

        # Compute absolute indices in the full dataset
        abs_start_idx = traj_abs_start + start_idx
        abs_end_idx = traj_abs_start + end_idx

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
        # Randomly select an traj
        traj_idx, traj_len = random.choice(val_trajs)
        traj_abs_start = traj_start_indices[traj_idx]

        # Sample a random valid segment
        max_start = traj_len - segment_length
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + segment_length

        # Compute absolute indices in the full dataset
        abs_start_idx = traj_abs_start + start_idx
        abs_end_idx = traj_abs_start + end_idx

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


def Robomimic_dataset(data_path, return_images=False):
    """
    Load Robomimic dataset.
    """

    print(f"Loading data from: {data_path}")

    with h5py.File(data_path, "r") as f:
        data = f["data"]

        all_observations = []
        all_actions = []
        all_rewards = []
        all_images = []
        all_terminals = []
        all_timesteps = []

        all_goal_points = []

        print(f"Found {len(data.keys())} trajectories in dataset")

        for demo in tqdm(
            sorted(data.keys(), key=lambda x: int(x.split("_")[1])),
            desc="Processing demos",
        ):
            demo_data = data[demo]
            obs = np.concatenate(
                [
                    demo_data["obs"]["robot0_eef_pos"][:],
                    demo_data["obs"]["robot0_eef_quat"][:],
                    demo_data["obs"]["robot0_gripper_qpos"][:],
                    demo_data["obs"]["object"][:],
                ],
                axis=1,
            )
            acts = demo_data["actions"]
            rewards = demo_data["rewards"]
            images = demo_data["obs"]["agentview_image"]
            # goal_points = demo_data["goal_points"]

            all_observations.append(obs)
            all_actions.append(acts)
            all_rewards.append(rewards)
            # all_goal_points.append(goal_points)

            if images.shape[1] != 84:
                # Assuming images are in format (batch, height, width, channels)
                resized_images = np.array([cv2.resize(img, (84, 84)) for img in images])
                all_images.append(resized_images)
            else:
                all_images.append(images)
            # Create terminals array - True only for the last step of each traj
            traj_length = len(acts)
            terminals = np.zeros(traj_length, dtype=bool)
            terminals[-1] = True  # Mark the last step as terminal
            all_terminals.append(terminals)
            all_timesteps.append(np.arange(len(terminals)))

        # Convert to numpy arrays
        observations = np.concatenate(all_observations, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0).reshape(-1)
        images = np.concatenate(all_images, axis=0)
        terminals = np.concatenate(all_terminals, axis=0).reshape(-1)
        timesteps = np.concatenate(all_timesteps, axis=0).reshape(-1)
        # goal_points = np.concatenate(all_goal_points, axis=0)

    print(f"Total number of transitions: {len(observations)}")

    dataset = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timesteps": timesteps,
        # "goal_points": goal_points,
    }
    if return_images:
        dataset["images"] = images

    return dataset


class ReplayBuffer:
    """Replay buffer for storing and sampling experience transitions."""

    def __init__(
        self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device

        # Initialize buffers
        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self._rtgs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self._timesteps = torch.zeros((buffer_size,), dtype=torch.long, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]) -> None:
        """Load dataset in d4rl format."""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._rtgs[:n_transitions] = self._to_tensor(data["rtgs"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"])
        self._timesteps[:n_transitions] = torch.tensor(
            data["timesteps"], dtype=torch.long, device=self._device
        )

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._rtgs[indices],
            self._next_states[indices],
            self._dones[indices],
            self._timesteps[indices],
        ]

    def add_transition(self):
        """Add new transition (not implemented for offline RL)."""
        raise NotImplementedError("Fine-tuning not implemented")


class SequentialReplayBuffer:
    """Samples s_t, a_t, r_t, a_t+1, r_t+1, ..., a_t+H, r_t+H"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        seq_len: int,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        self._seq_len = seq_len

        # Initialize buffers
        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self._timesteps = torch.zeros((buffer_size,), dtype=torch.long, device=device)

        # Track traj boundaries for sequential sampling
        self._traj_starts = []
        self._traj_ends = []

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]) -> None:
        """Load dataset in d4rl format and track traj boundaries."""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"])
        self._timesteps[:n_transitions] = torch.tensor(
            data["timesteps"], dtype=torch.long, device=self._device
        )

        # Track traj boundaries
        self._track_traj_boundaries(data["terminals"])
        # Find valid start indices
        self._find_valid_starts()

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(
            f"Dataset size: {n_transitions}, Number of valid start indices {len(self.valid_starts)}"
        )
        print(f"Number of trajs: {len(self._traj_starts)}")

    def _track_traj_boundaries(self, terminals: np.ndarray) -> None:
        """Track start and end indices of each traj."""
        traj_start = 0

        for i, terminal in enumerate(terminals):
            if terminal:
                self._traj_starts.append(traj_start)
                self._traj_ends.append(i)
                traj_start = i + 1

    def _find_valid_starts(self):
        self.valid_starts = []
        for ep_start, ep_end in zip(self._traj_starts, self._traj_ends):
            # Only allow starts that have at least seq_len + 1 steps remaining in traj (we need to be able to sample a valid next obs)
            for start_idx in range(ep_start, max(ep_start, ep_end - self._seq_len + 1)):
                self.valid_starts.append(start_idx)

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """
        Sample sequences of length seq_len.

        Args:
            batch_size: Number of sequences to sample

        Returns:
            List of tensors: [states, actions, rewards, next_states, dones]
        """

        # Sample batch_size starting indices
        batch_starts = np.random.choice(
            self.valid_starts, size=batch_size, replace=True
        )

        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        for start_idx in batch_starts:
            # Single initial state
            state = self._states[start_idx]  # (state_dim,)

            # Sequence indices for this sample
            seq_indices = np.arange(start_idx, start_idx + self._seq_len)

            # Action and reward sequences
            action_seq = self._actions[seq_indices]  # (seq_len, action_dim)
            reward_seq = self._rewards[seq_indices]  # (seq_len, 1)
            done_seq = self._dones[seq_indices]  # (seq_len,)

            next_state = self._states[start_idx + self._seq_len]  # (state_dim,)

            assert done_seq[-1] == 1 if done_seq.sum() == 1 else True

            states.append(state)
            actions.append(action_seq)
            rewards.append(reward_seq)
            next_states.append(next_state)
            dones.append(done_seq)

        # Stack into batch tensors
        states = torch.stack(states, dim=0)  # (batch_size, state_dim)
        actions = torch.stack(actions, dim=0)  # (batch_size, seq_len, action_dim)
        rewards = torch.stack(rewards, dim=0)  # (batch_size, seq_len, 1)
        next_states = torch.stack(next_states, dim=0)  # (batch_size, state_dim)
        dones = torch.stack(dones, dim=0)  # (batch_size, seq_len)

        return [states, actions, rewards, next_states, dones]


class DTSequentialReplayBuffer:
    """Optimized version with pre-computed RTGs while maintaining exact functionality"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        K: int,
        device: str = "cpu",
        scale: float = 10.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        self._K = K
        self._scale = scale

        # Initialize buffers
        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size,), dtype=torch.long, device=device)
        self._timesteps = torch.zeros((buffer_size,), dtype=torch.long, device=device)
        self._rtgs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)

        # traj boundaries
        self._traj_starts = []
        self._traj_ends = []
        self._traj_lengths = []

    def load_dataset(self, data: Dict[str, np.ndarray]) -> None:
        """Load dataset and PRE-COMPUTE all RTGs"""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        # Load data
        self._states[:n_transitions] = torch.from_numpy(data["observations"]).to(dtype=torch.float32, device=self._device)
        self._actions[:n_transitions] = torch.from_numpy(data["actions"]).to(dtype=torch.float32, device=self._device)
        self._rewards[:n_transitions] = torch.from_numpy(data["rewards"]).to(dtype=torch.float32, device=self._device)
        self._dones[:n_transitions] = torch.from_numpy(data["terminals"]).to(dtype=torch.long, device=self._device)
        self._timesteps[:n_transitions] = torch.from_numpy(data["timesteps"]).to(dtype=torch.long, device=self._device)

        # Pre-computing data
        self._track_traj_boundaries(data["terminals"])
        self._traj_lengths = [end - start + 1 for start, end in zip(self._traj_starts, self._traj_ends)]
        self._precompute_rtgs()
        
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")
        print(f"Number of trajs: {len(self._traj_starts)}")

    def _track_traj_boundaries(self, terminals: np.ndarray) -> None:
        """Track start and end indices of each traj using NumPy vectorized operations."""
        terminal_indices = np.flatnonzero(terminals)
        self._traj_starts = [0] + (terminal_indices[:-1] + 1).tolist()
        self._traj_ends = terminal_indices.tolist()

    def _precompute_rtgs(self, gamma=1.0):
        """Pre-compute RTGs for all trajs using original logic"""
        print("Pre-computing RTGs...")
        for start, end in zip(self._traj_starts, self._traj_ends):
            traj_rewards = self._rewards[start:end+1]
            traj_rtgs = self._discount_cumsum(traj_rewards, gamma)
            self._rtgs[start:end+1] = traj_rtgs
        print("RTGs pre-computed!")

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        traj_lengths = torch.tensor(self._traj_lengths, dtype=torch.float32, device=self._device)
        p_sample = traj_lengths / traj_lengths.sum()
        batch_inds = torch.multinomial(p_sample, batch_size, replacement=True)

        s = torch.zeros((batch_size, self._K, self.state_dim), device=self._device)
        a = torch.ones((batch_size, self._K, self.action_dim), device=self._device) * -10.
        r = torch.zeros((batch_size, self._K, 1), device=self._device)
        d = torch.ones((batch_size, self._K), device=self._device) * 2
        rtg = torch.zeros((batch_size, self._K, 1), device=self._device)
        timesteps = torch.zeros((batch_size, self._K), dtype=torch.long, device=self._device)
        mask = torch.zeros((batch_size, self._K), device=self._device)

        for i in range(batch_size):
            ep_idx = batch_inds[i].item()
            ep_start = self._traj_starts[ep_idx]
            ep_end = self._traj_ends[ep_idx]
            ep_len = self._traj_lengths[ep_idx]

            si = torch.randint(0, ep_len, (1,)).item()
            idx_start = ep_start + si
            idx_end = min(idx_start + self._K, ep_end + 1)

            tlen = idx_end - idx_start

            s[i, -tlen:] = self._states[idx_start:idx_end]
            a[i, -tlen:] = self._actions[idx_start:idx_end]
            r[i, -tlen:] = self._rewards[idx_start:idx_end].unsqueeze(-1)
            d[i, -tlen:] = self._dones[idx_start:idx_end]
            rtg[i, -tlen:] = self._rtgs[idx_start:idx_end].unsqueeze(-1) / self._scale
            timesteps[i, -tlen:] = torch.arange(si, si + tlen, device=self._device)
            mask[i, -tlen:] = 1

        return [s, a, r, rtg, d.long(), timesteps, mask]


    # def sample(self, batch_size: int) -> List[torch.Tensor]:
    #     import random
    #     # Sample trajectories proportionally to their lengths
    #     traj_lengths = np.array(self._traj_lengths)
    #     p_sample = traj_lengths / traj_lengths.sum()
    #     num_trajs = len(self._traj_starts)
    #     batch_inds = np.random.choice(np.arange(num_trajs), size=batch_size, replace=True, p=p_sample)

    #     s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        
    #     for i in range(batch_size):
    #         ep_idx = batch_inds[i]
    #         ep_start = self._traj_starts[ep_idx]
    #         ep_end = self._traj_ends[ep_idx]
    #         ep_len = self._traj_lengths[ep_idx]

    #         # Random start index [ep_start, ep_end]
    #         si = random.randint(0, ep_len - 1)
            
    #         idx_start = ep_start + si
    #         idx_end = idx_start + self._K

    #         # Compute how many valid steps are left in the traj
    #         tlen = min(self._K, ep_end - idx_start + 1)

    #         s_seq = self._states[idx_start:idx_start + tlen].detach().cpu().numpy().reshape(1, -1, self.state_dim)
    #         a_seq = self._actions[idx_start:idx_start + tlen].detach().cpu().numpy().reshape(1, -1, self.action_dim)
    #         r_seq = self._rewards[idx_start:idx_start + tlen].detach().cpu().numpy().reshape(1, -1, 1)
    #         d_seq = self._dones[idx_start:idx_start + tlen].detach().cpu().numpy().reshape(1, -1)
    #         t_seq = np.arange(si, si + tlen).reshape(1, -1)
    #         rtg_seq = self._rtgs[idx_start:idx_start+ tlen].detach().cpu().numpy().reshape(1, -1, 1)

    #         # Padding
    #         pad_len = self._K - tlen
    #         s_pad = np.zeros((1, pad_len, self.state_dim))
    #         a_pad = np.ones((1, pad_len, self.action_dim)) * -10.
    #         r_pad = np.zeros((1, pad_len, 1))
    #         d_pad = np.ones((1, pad_len)) * 2
    #         rtg_pad = np.zeros((1, pad_len, 1))  # divide by scale
    #         t_pad = np.zeros((1, pad_len))
    #         m_pad = np.zeros((1, pad_len))

    #         s.append(np.concatenate([s_pad, s_seq], axis=1))
    #         a.append(np.concatenate([a_pad, a_seq], axis=1))
    #         r.append(np.concatenate([r_pad, r_seq], axis=1))
    #         d.append(np.concatenate([d_pad, d_seq], axis=1))
    #         rtg.append(np.concatenate([rtg_pad, rtg_seq], axis=1) / self._scale)
    #         timesteps.append(np.concatenate([t_pad, t_seq], axis=1))
    #         mask.append(np.concatenate([m_pad, np.ones((1, tlen))], axis=1))

    #     # Convert to tensors
    #     s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self._device)
    #     a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self._device)
    #     r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self._device)
    #     d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self._device)
    #     rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self._device)
    #     timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self._device)
    #     mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.float32, device=self._device)

    #     return [s, a, r, rtg, d, timesteps, mask]

    def _discount_cumsum(self, rewards, gamma=1.0):
        T = rewards.shape[0]
        rtg = torch.zeros_like(rewards)
        running_sum = 0.0
        for t in reversed(range(T)):
            running_sum = rewards[t] + gamma * running_sum
            rtg[t] = running_sum
        return rtg


def setup_reward_model(config, dataset):
    """Setup reward model if specified in config."""
    if not config.use_reward_model:
        return dataset

    print("Using rewards labeled from trained reward model")

    # Determine reward model dimension
    if config.eef_rm:
        dimension = 5 if config.eef_rm_2d else 3
    else:
        dimension = dataset["observations"].shape[1] + dataset["actions"].shape[1]

    print(f"Using reward model with dimension {dimension}")

    # Load reward model
    model = reward_model.RewardModel(config, None, None, None, None, dimension)
    path = build_rm_checkpoint_path(config)
    path = os.path.join(config.checkpoints_path, path)

    print(f"Loading reward model from {path}")
    model.load_model(path)
    print("Successfully loaded reward model")

    # Apply reward model to dataset
    rewards = model.get_reward(dataset)
    dataset["rewards"] = rewards
    dataset["rtgs"] = np.flip(np.cumsum(np.flip(rewards, axis=0), axis=0))

    return dataset


def print_dataset_statistics(dataset: Dict[str, np.ndarray]) -> None:
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_samples = None
    for key, data in dataset.items():
        if not isinstance(data, np.ndarray):
            continue
        if key != "observations" and key != "actions" and key != "rewards":
            continue

        if total_samples is None:
            total_samples = data.shape[0]

        print(f"\n{key.upper()}:")
        print(f"  Shape: {data.shape}")

        # Basic statistics
        stats = {
            "Mean": np.mean(data),
            "Min": np.min(data),
            "Max": np.max(data),
            "Std": np.std(data),
        }

        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}:  {stat_value:>12.6f}")

        # Per-dimension statistics for reasonable-sized arrays
        if data.ndim == 2 and data.shape[1] <= 10:
            print("  Per-dimension statistics:")
            for i in range(data.shape[1]):
                dim_stats = {
                    "mean": np.mean(data[:, i]),
                    "std": np.std(data[:, i]),
                    "min": np.min(data[:, i]),
                    "max": np.max(data[:, i]),
                }
                print(
                    f"    Dim {i:2d}: "
                    + f"mean={dim_stats['mean']:8.4f}, std={dim_stats['std']:8.4f}, "
                    + f"min={dim_stats['min']:8.4f}, max={dim_stats['max']:8.4f}"
                )
        elif data.ndim == 2 and data.shape[1] > 10:
            print(f"  (Skipping per-dimension stats for {data.shape[1]} dimensions)")

    # Summary information
    if total_samples is not None:
        print("\nSUMMARY:")
        print(f"  Total samples: {total_samples:,}")

        # Episode statistics
        if "terminals" in dataset and isinstance(dataset["terminals"], np.ndarray):
            num_trajs = np.sum(dataset["terminals"])
            avg_traj_length = total_samples / num_trajs
            print(f"  Trajectories: {num_trajs:,}")
            print(f"  Average traj length: {avg_traj_length:.1f} steps")

    print("\n" + "=" * 60)


def setup_checkpoint_paths(config):
    """Setup checkpoint directories and save config."""
    if getattr(config, "checkpoints_path", None) is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")

        checkpoint_name = build_rm_checkpoint_path(config)
        config.checkpoints_path = os.path.join(config.checkpoints_path, checkpoint_name)

        os.makedirs(config.checkpoints_path, exist_ok=True)
        OmegaConf.save(
            config=config, f=os.path.join(config.checkpoints_path, "config.yaml")
        )


def build_rm_checkpoint_path(config: DictConfig) -> str:
    """Build reward learning checkpoint path based on config parameters."""
    components = [
        f"{config.env}",
        f"fn_{config.feedback_num}",
        f"gt_{int(config.single_emb)}",
        f"eef_{int(config.eef_rm)}",
        f"dtw_{int(config.use_cross)}",
        f"s_{config.seed}",
    ]
    return "/".join(components)
