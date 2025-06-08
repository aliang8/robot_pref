# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import multiprocessing as mp
from functools import partial

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import utils_env

import rich
import sys

# sys.path.append("./Reward_learning")
import models.reward_model as reward_model

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "metaworld_box-close-v2"  # environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    data_quality: float = 5.0  # Replay buffer size (data_quality * 100000)
    trivial_reward: int = (
        0  # 0: GT reward, 1: zero reward, 2: constant reward, 3: negative reward
    )
    # Video recording
    record_video: bool = False  # Whether to record evaluation videos
    video_dir: Optional[str] = None  # Directory to save evaluation videos
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # reward model
    feedback_num: int = 1000
    use_reward_model: bool = False
    epochs: int = 0
    batch_size: int = 256
    activation: str = "tanh"
    lr: float = 1e-3
    threshold: float = 0.0
    segment_size: int = 25
    data_aug: str = "none"
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"
    q_budget: int = 100
    feedback_type: str = "RLT"
    model_type: str = "BT"
    noise: float = 0.0
    human: bool = False
    use_relative_eef: bool = False
    # DTW augmentations
    use_dtw_augmentations: bool = False
    dtw_augment_before_training: bool = False  # If True: augment first then train, If False: train then augment
    dtw_subsample_size: int = 10000  # Number of segments to sample for DTW matrix
    dtw_augmentation_size: int = 2000  # Number of augmentation pairs to create from DTW matrix
    dtw_k_augment: int = 5  # Number of similar segments to find for each original preference pair
    dtw_preference_ratios: List[float] = None  # Ratios for [seg1_better, seg2_better, equal_pref] sampling. None = use all available
    acquisition_threshold_low: float = 0.25  # 25th percentile for acquisition filtering
    acquisition_threshold_high: float = 0.75  # 75th percentile for acquisition filtering
    acquisition_method: str = "entropy"  # "entropy", "disagreement", "combined", "variance"
    # Class weighting for preference balancing
    use_class_weights: bool = False  # Apply inverse frequency weighting to balance preference types
    # Wandb logging
    project: str = "robot_pref"
    entity: str = "clvr"
    group: str = "IQL-MetaWorld"
    name: str = "IQL"

    def __post_init__(self):
        # Set default equal ratios for DTW preference sampling if not specified
        if self.dtw_preference_ratios is None:
            self.dtw_preference_ratios = [0.0, 1.0, 0.0]  # [seg1_better, seg2_better, equal_pref]
        
        # Validate ratios sum to 1.0
        if self.dtw_preference_ratios is not None:
            ratio_sum = sum(self.dtw_preference_ratios)
            if abs(ratio_sum - 1.0) > 1e-6:
                print(f"Warning: DTW preference ratios sum to {ratio_sum:.6f}, normalizing to sum to 1.0")
                self.dtw_preference_ratios = [r / ratio_sum for r in self.dtw_preference_ratios]
        
        if self.use_reward_model:
            # Build comprehensive group and checkpoint names including all new parameters
            # Create shorter group name to stay under 128 char limit
            group_parts = [
                f"env_{self.env.replace('metaworld_', 'mw_')}",  # Shorten metaworld prefix
                f"d{self.data_quality}",  # Shorter data quality
                f"fn{self.feedback_num}",  # Shorter feedback num
                f"qb{self.q_budget}",  # Shorter q budget
                f"{self.feedback_type}",  # Remove ft_ prefix
                f"{self.model_type}",  # Remove m_ prefix
                f"n{self.noise}",  # Shorter noise
                f"e{self.epochs}",  # Shorter epochs
                f"th{self.threshold}",  # Shorter threshold
                f"tr{self.trivial_reward}",  # Shorter trivial reward
                f"cw{int(self.use_class_weights)}"  # Shorter class weights
            ]
            
            # Add DTW components if enabled (much shorter version)
            if self.use_dtw_augmentations:
                dtw_mode = "B" if self.dtw_augment_before_training else "A"  # Before/After
                dtw_component = f"dtw{int(self.use_dtw_augmentations)}{dtw_mode}{self.dtw_subsample_size//1000}k{self.dtw_augmentation_size}"
                group_parts.append(dtw_component)
            
            self.group = "_".join(group_parts)
            
            # Build checkpoint path components to match learn_reward.py exactly
            checkpoint_components = [
                f"{self.name}",
                f"{self.env}",
                f"data_{self.data_quality}",
                f"fn_{self.feedback_num}",
                f"qb_{self.q_budget}",
                f"ft_{self.feedback_type}",
                f"m_{self.model_type}",
                f"n_{self.noise}",
                f"e_{self.epochs}",
                f"th_{self.threshold}"
            ]
            
            # Add DTW components if enabled (match learn_reward.py format)
            if self.use_dtw_augmentations:
                dtw_mode = "before" if self.dtw_augment_before_training else "after"
                checkpoint_components.extend([
                    f"dtw_{dtw_mode}",
                    f"sub_{self.dtw_subsample_size//1000}k",
                    f"aug_{self.dtw_augmentation_size}"
                ])
            
            # Use same seed format as learn_reward.py
            checkpoint_components.append(f"s_{self.seed}")
            checkpoint_name = "/".join(checkpoint_components)
            print(f"Checkpoint name: {checkpoint_name}")
            print(f"Group name length: {len(self.group)} chars: {self.group}")
        else:
            self.group = f"IQL/{self.env}_quality_{self.data_quality}_trivial_{self.trivial_reward}"
            checkpoint_name = f"{self.name}/{self.env}/quality_{self.data_quality}_trivial_{self.trivial_reward}/seed_{self.seed}_{str(uuid.uuid4())[:8]}"
        
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, checkpoint_name)
        self.name = f"seed_{self.seed}_{str(uuid.uuid4())[:8]}"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        # if state 2 dim
        if len(state) == 2:
            state = state[0]
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        entity=config["entity"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def _eval_episode(env, actor, device, max_steps, seed, record_video=False, video_path=None):
    """Helper function to evaluate a single episode.
    
    Args:
        env: The environment to evaluate on
        actor: The actor network to evaluate
        device: Device to run the actor on
        max_steps: Maximum number of steps per episode
        seed: Random seed for evaluation
        record_video: Whether to record video of the episode
        video_path: Path to save the video file
    """
    env.seed(seed)
    
    # Setup video recording if requested
    video_writer = None
    if record_video and video_path:
        try:
            import imageio
            import os
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=30)
        except Exception as e:
            print(f"Error setting up video recording: {e}")
            video_writer = None
    
    state, info = env.reset()
    episode_reward = 0.0
    episode_success = 0
    steps = 0
    
    # Record initial frame if recording
    if video_writer:
        try:
            frame = env.render(mode="rgb_array")
            video_writer.append_data(frame)
        except Exception as e:
            print(f"Warning: Could not capture initial frame: {e}")
    
    while steps < max_steps:
        action = actor.act(state, device)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        episode_reward += reward
        # if "metaworld" in env.env_name:
        #     episode_success = max(episode_success, info["success"])
            
        # Record frame if recording
        if video_writer:
            try:
                frame = env.render(mode="rgb_array")
                video_writer.append_data(frame)
            except Exception as e:
                print(f"Warning: Could not capture frame: {e}")
                
        if done:
            break
    
    # Close video writer if recording
    if video_writer:
        try:
            video_writer.close()
        except Exception as e:
            print(f"Warning: Error closing video writer: {e}")
            
    return episode_reward, episode_success


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    env_name: str,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
    max_steps: int = 1000,
    parallel: bool = False,
    record_video: bool = False,
    video_dir: str = None,
) -> np.ndarray:
    """Evaluate the actor on the environment."""
    actor.eval()
    
    # Create a wrapper class for the actor to make it picklable
    class ActorWrapper:
        def __init__(self, actor, device):
            self.actor = actor
            self.device = device
            
        def predict(self, obs):
            with torch.no_grad():
                action = self.actor.act(obs, self.device)
            return action
    
    # Wrap the actor
    algo = ActorWrapper(actor, device)
    
    # Create video directory if needed
    if record_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)
    
    # Determine if we should use parallel evaluation
    use_parallel = parallel and n_episodes > 1
    
    if use_parallel:
        try:
            from multiprocessing import get_context
            from utils.eval import evaluate_episode_worker, PicklableEnvCreator
            
            # Determine number of workers
            n_workers = min(n_episodes, mp.cpu_count())
            print(f"Running parallel evaluation ({n_workers} workers, {n_episodes} episodes)")
            
            # Prepare arguments for workers
            worker_args = []
            for i in range(n_episodes):
                # Create a distinct environment creator for each episode with appropriate seed
                episode_seed = seed + i
                
                # Try to determine if this is a MetaWorld environment
                is_metaworld = False
                try:
                    import metaworld
                    if isinstance(env, metaworld.envs.base.Env) or "metaworld" in env.__class__.__module__:
                        is_metaworld = True
                except (ImportError, AttributeError):
                    if hasattr(env, "model") and hasattr(env, "data"):
                        is_metaworld = True
                
                # Create environment creator with appropriate type
                env_type = "metaworld" if is_metaworld else None
                env_creator = PicklableEnvCreator(env_id=env_name, seed=episode_seed, env_type=env_type)
                
                # Determine if this episode should be recorded
                should_record = record_video and i < 5  # Record up to 5 episodes
                video_path = None
                if should_record:
                    video_path = os.path.join(video_dir, f"episode_{i}.mp4")
                
                # Add to worker arguments
                worker_args.append((env_creator, algo, i, should_record, video_path, 30))
            
            # Use 'spawn' context for better compatibility across platforms
            ctx = get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                results = list(pool.map(evaluate_episode_worker, worker_args))
            
            # Process results
            episode_rewards = [r["return"] for r in results]
            episode_success_list = [r["success"] for r in results]
            
        except Exception as e:
            print(f"Error in parallel evaluation: {e}. Falling back to sequential evaluation.")
            import traceback
            traceback.print_exc()
            use_parallel = False
    
    # Sequential evaluation (fallback or if parallel not requested)
    if not use_parallel:
        episode_rewards = []
        episode_success_list = []
        
        for i in range(n_episodes):
            # Setup video path for this episode
            video_path = None
            if record_video and video_dir:
                video_path = os.path.join(video_dir, f"episode_{i}.mp4")
            
            reward, success = _eval_episode(
                env, 
                actor, 
                device, 
                max_steps, 
                seed + i,
                record_video=record_video,
                video_path=video_path
            )
            episode_rewards.append(reward)
            episode_success_list.append(success)
    
    actor.train()
    return np.array(episode_rewards), np.array(episode_success_list)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(
    dataset,
    max_episode_steps=1000,
    trivial_reward=0,
):
    # if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
    #     min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
    #     dataset["rewards"] /= max_ret - min_ret
    #     dataset["rewards"] *= max_episode_steps
    # elif "antmaze" in env_name:
    #     dataset["rewards"] -= 1.0
    min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
    # GT reward
    if trivial_reward == 0:
        dataset["rewards"] = (dataset["rewards"] - min(dataset["rewards"])) / (
            max(dataset["rewards"]) - min(dataset["rewards"])
        )
    # zero reward
    elif trivial_reward == 1:
        dataset["rewards"] *= 0.0
    # random reward
    elif trivial_reward == 2:
        dataset["rewards"] = (dataset["rewards"] - min(dataset["rewards"])) / (
            max(dataset["rewards"]) - min(dataset["rewards"])
        )
        min_reward, max_reward = min(dataset["rewards"]), max(dataset["rewards"])
        dataset["rewards"] = np.random.uniform(
            min_reward, max_reward, size=dataset["rewards"].shape
        )
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
    # negative reward
    elif trivial_reward == 3:
        dataset["rewards"] = 1 - (dataset["rewards"] - min(dataset["rewards"])) / (
            max(dataset["rewards"]) - min(dataset["rewards"])
        )


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(
                self(state) * self.max_action, -self.max_action, self.max_action
            )
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        # targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        # MetaWolrd has no terminals (only time limit)
        targets = rewards + self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    rich.print("config ", config)
    if "metaworld" in config.env:
        env = utils_env.make_metaworld_env(config.env, config.seed)
        dataset = utils_env.MetaWorld_dataset(config)
    elif "dmc" in config.env:
        env = utils_env.make_dmc_env(config.env, config.seed)
        dataset = utils_env.DMC_dataset(config)
    elif "robomimic" in config.env:
        env = utils_env.get_robomimic_env(config.env, seed=config.seed)
        dataset = utils_env.Robomimic_dataset(config)
    else:
        env = gym.make(config.env)

    # state_dim = env.observation_space.shape[0]  # 39 for metaworld
    state_dim = env.observation_space["state"].shape[0] # robomimic
    action_dim = env.action_space.shape[0]  # 4 for metaworld


    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    dimension = dataset["observations"].shape[1] + dataset["actions"].shape[1]
    if config.use_reward_model:
        model = reward_model.RewardModel(config, None, None, None, dimension)
        
        # Use the same checkpoint path structure as learn_reward.py
        base_path = os.getcwd() + "/logs/Reward"
        
        # Build path components to match learn_reward.py exactly
        checkpoint_components = [
            "Reward",  # Name from learn_reward.py
            config.env,
            f"data_{config.data_quality}",
            f"fn_{config.feedback_num}",
            f"qb_{config.q_budget}",
            f"ft_{config.feedback_type}",
            f"m_{config.model_type}",
            f"n_{config.noise}",
            f"e_{config.epochs}",
            f"th_{config.threshold}"
        ]
        
        # Add DTW components if enabled (match learn_reward.py format)
        if config.use_dtw_augmentations:
            dtw_mode = "before" if config.dtw_augment_before_training else "after"
            checkpoint_components.extend([
                f"dtw_{dtw_mode}",
                f"sub_{config.dtw_subsample_size//1000}k",
                f"aug_{config.dtw_augmentation_size}"
            ])
        
        # Use same seed format as learn_reward.py
        checkpoint_components.append(f"s_{config.seed}")
        
        path = os.path.join(base_path, *checkpoint_components[1:])  # Skip "Reward" for path construction
        print(f"Loading reward model from: {path}")
        
        model.load_model(path)
        dataset["rewards"] = model.get_reward(dataset)

    if config.normalize_reward:
        modify_reward(
            dataset,
            max_episode_steps=500,
            trivial_reward=config.trivial_reward,
        )

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    config.buffer_size = dataset["observations"].shape[0]

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    # Initialize wandb
    wandb_init(asdict(config))

    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        if (t + 1) % 5000 == 0:
            wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            
            # Create video directory for this evaluation
            video_dir = None
            if config.record_video:
                if config.video_dir:
                    video_dir = os.path.join(config.video_dir, f"eval_{t}")
                else:
                    video_dir = os.path.join("videos", f"eval_{t}")
                os.makedirs(video_dir, exist_ok=True)
            
            eval_scores, eval_success = eval_actor(
                env,
                config.env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                record_video=config.record_video,
                video_dir=video_dir,
            )
            eval_score = eval_scores.mean()  # For DMControl
            eval_success = eval_success.mean() * 100  # For MetaWorld
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , success: {eval_success:.3f}"
            )
            print("---------------------------------------")
            
            # Log metrics to wandb
            wandb.log(
                {
                    "eval/eval_score": eval_score,
                    "eval/eval_success": eval_success,
                },
                step=trainer.total_it,
            )
            
            # Log videos to wandb if recording
            if config.record_video and video_dir:
                video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
                if video_files:
                    # Log each video individually
                    for video_file in video_files:
                        episode_num = int(os.path.basename(video_file).split("_")[1].split(".")[0])
                        wandb.log(
                            {
                                f"eval/video_episode_{episode_num}": wandb.Video(
                                    video_file,
                                    fps=30,
                                    format="mp4",
                                )
                            },
                            step=trainer.total_it,
                        )
            
            if (config.checkpoints_path is not None) and (t + 1) % (
                20 * config.eval_freq
            ) == 0:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )


if __name__ == "__main__":
    train()
