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
    data_path, return_images=False, clip_last=False
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
            # all_rtgs.append(rtgs)

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
            all_timesteps.append(np.arange(len(terminals)))

        # Convert to numpy arrays
        observations = np.concatenate(all_observations, axis=0)
        next_observations = np.concatenate(all_next_observations, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0).reshape(-1, 1)
        images = np.concatenate(all_images, axis=0)
        terminals = np.concatenate(all_terminals, axis=0).reshape(-1)
        timesteps = np.concatenate(all_timesteps, axis=0).reshape(-1)
        goal_points = np.concatenate(all_goal_points, axis=0)

    print(f"Total number of transitions: {len(observations)}")

    dataset = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timesteps": timesteps,
        "goal_points": goal_points,
    }
    if return_images:
        dataset["images"] = images

    return dataset


class ReplayBuffer:
    """Replay buffer for storing and sampling experience transitions."""
    
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device

        # Initialize buffers
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self._rtgs = torch.zeros((buffer_size,1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
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
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._rtgs[:n_transitions] = self._to_tensor(data["rtgs"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"])
        self._timesteps[:n_transitions] = torch.tensor(data["timesteps"], dtype=torch.long, device=self._device)

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
            self._timesteps[indices]
        ]

    def add_transition(self):
        """Add new transition (not implemented for offline RL)."""
        raise NotImplementedError("Fine-tuning not implemented")

class SequentialReplayBuffer:
    """Replay buffer for storing and sampling sequential experience transitions."""

    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, K: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        self._K = K

        # Initialize buffers
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # self._rtgs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self._timesteps = torch.zeros((buffer_size,), dtype=torch.long, device=device)
        
        # Track episode boundaries for sequential sampling
        self._episode_starts = []
        self._episode_ends = []

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]) -> None:
        """Load dataset in d4rl format and track episode boundaries."""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"])
        self._timesteps[:n_transitions] = torch.tensor(data["timesteps"], dtype=torch.long, device=self._device)

        # Track episode boundaries
        self._track_episode_boundaries(data["terminals"])
        
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        
        print(f"Dataset size: {n_transitions}")
        print(f"Number of episodes: {len(self._episode_starts)}")

    def _track_episode_boundaries(self, terminals: np.ndarray) -> None:
        """Track start and end indices of each episode."""
        episode_start = 0
        
        for i, terminal in enumerate(terminals):
            if terminal:
                self._episode_starts.append(episode_start)
                self._episode_ends.append(i)
                episode_start = i + 1

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """
        Sample sequences of length K.
        
        Args:
            batch_size: Number of sequences to sample
            K: Length of each sequence (K)
            
        Returns:
            List of tensors: [states, actions, rewards, rtgs, next_states, dones, timesteps]
            Each tensor has shape (batch_size, K, ...)
        """
        # Sample batch_size indices
        batch_inds = np.random.randint(0, self._size, size=batch_size)
        
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        
        for ep_idx in batch_inds:
            # If episode is too close to the end, we left pad
            ep_end = self._episode_ends[self._find_episode_for_transition(ep_idx)]
            valid_len = ep_end + 1 - ep_idx
            k_start = max(0, self._K - valid_len)

            # Padding
            zero_state = torch.zeros((1, self.state_dim), device=self._device)
            zero_action = torch.zeros((1, self.action_dim), device=self._device)
            state_pad = [zero_state] * k_start
            action_pad = [zero_action] * k_start
            reward_pad = [0.0] * k_start
            rtg_pad = [0.0] * k_start
            done_pad = [0.0] * k_start
            timestep_pad = [0] * k_start
            mask_pad = [0] * k_start

            # Actual sequence
            seq_slice = slice(ep_idx, min(ep_idx + self._K, ep_end + 1))
            s_seq = self._states[seq_slice]
            a_seq = self._actions[seq_slice]
            r_seq = self._rewards[seq_slice]
            rtg_seq = self._discount_cumsum(r_seq, gamma=1.0)
            d_seq = self._dones[seq_slice]
            t_seq = self._timesteps[seq_slice]

            seq_len = s_seq.shape[0]
            m_seq = [1] * seq_len

            # Append padded + actual
            s.append(torch.cat([*state_pad, s_seq]))
            a.append(torch.cat([*action_pad, a_seq]))
            r.append(torch.tensor(reward_pad + r_seq.view(-1).tolist(), device=self._device).view(-1, 1))
            rtg.append(torch.tensor(rtg_pad + rtg_seq.view(-1).tolist(), device=self._device).view(-1, 1))
            d.append(torch.tensor(done_pad + d_seq.view(-1).tolist(), device=self._device).view(-1, 1))
            timesteps.append(torch.tensor(timestep_pad + t_seq.view(-1).tolist(), device=self._device))
            mask.append(torch.tensor(mask_pad + m_seq, dtype=torch.long, device=self._device))

        s = torch.stack(s, dim=0)
        a = torch.stack(a, dim=0)
        r = torch.stack(r, dim=0)
        rtg = torch.stack(rtg, dim=0)
        d = torch.stack(d, dim=0)
        timesteps = torch.stack(timesteps, dim=0)
        mask = torch.stack(mask, dim=0)

        return [s, a, r, rtg, d, timesteps, mask]


    def _find_episode_for_transition(self, transition_idx: int) -> int:
        """Find which episode a given transition belongs to."""
        for i, (start, end) in enumerate(zip(self._episode_starts, self._episode_ends)):
            if start <= transition_idx <= end:
                return i
        raise ValueError(f"Transition index {transition_idx} not found in any episode")

    def _discount_cumsum(self, rewards, gamma=1.0):
        """
        Compute discounted cumulative sums of rewards.

        Args:
            rewards (torch.Tensor): shape (T,) or (T, 1)
            gamma (float): discount factor

        Returns:
            torch.Tensor: discounted cumulative sum of rewards, same shape as input
        """
        T = rewards.shape[0]
        rtg = torch.zeros_like(rewards)
        running_sum = 0.0

        for t in reversed(range(T)):
            running_sum = rewards[t] + gamma * running_sum
            rtg[t] = running_sum

        return rtg

def setup_environment_and_dataset(config):
    """Setup environment and dataset based on configuration."""
    if "metaworld" in config.env:
        env = make_metaworld_env(config.env, config.seed)
        dataset = MetaWorld_dataset(config)
    elif "robomimic" in config.env:
        env = get_robomimic_env(config.data_path, seed=config.seed, render_hw=config.render_hw)
        dataset = Robomimic_dataset(config.data_path, clip_last=True)
    else:
        env = gym.make(config.env)
        # Add dataset loading for standard gym environments if needed
        raise NotImplementedError("Dataset loading for standard gym environments not implemented")
    
    return env, dataset


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
                print(f"    Dim {i:2d}: " + 
                      f"mean={dim_stats['mean']:8.4f}, std={dim_stats['std']:8.4f}, " +
                      f"min={dim_stats['min']:8.4f}, max={dim_stats['max']:8.4f}")
        elif data.ndim == 2 and data.shape[1] > 10:
            print(f"  (Skipping per-dimension stats for {data.shape[1]} dimensions)")
    
    # Summary information
    if total_samples is not None:
        print("\nSUMMARY:")
        print(f"  Total samples: {total_samples:,}")
        
        # Episode statistics
        if "terminals" in dataset and isinstance(dataset["terminals"], np.ndarray):
            num_episodes = np.sum(dataset["terminals"]) + 1
            avg_episode_length = total_samples / num_episodes
            print(f"  Estimated episodes: {num_episodes:,}")
            print(f"  Average episode length: {avg_episode_length:.1f} steps")

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