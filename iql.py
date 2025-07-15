# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import hydra
import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

import models.reward_model as reward_model
import utils.env as utils_env
import wandb
from learn_reward import build_rm_checkpoint_path
from models.flow_policy import FlowNoisePredictionNet, FlowPolicy
from utils.eval import eval_actor
from utils.wandb import wandb_init
from utils.seed import set_seed

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0



def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


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

        action_shape = (buffer_size, action_dim)
        self._actions = torch.zeros(action_shape, dtype=torch.float32, device=device)

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


class TwinQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
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
        log_dict["target_q_mean"] = target_q.mean().item()
        log_dict["v_mean"] = v.mean().item()
        log_dict["adv_mean"] = adv.mean().item()
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
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        # MetaWolrd has no terminals (only time limit)
        # targets = rewards + self.discount * next_v.detach()
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
        # policy_out = self.actor(observations)
        # if isinstance(policy_out, torch.distributions.Distribution):
        #     bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        # elif torch.is_tensor(policy_out):
        #     if policy_out.shape != actions.shape:
        #         raise RuntimeError("Actions shape missmatch")
        #     bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        # else:
        #     raise NotImplementedError

        # bc_losses, loss_dict = self.actor.compute_loss(observations, actions)
        loss = self.actor(observations, actions)
        log_dict.update({"bc_loss": loss.mean().item()})

        # AWR
        policy_loss = torch.mean(exp_adv * loss)
        log_dict["policy_loss"] = policy_loss.item()
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


def print_dataset_statistics(dataset: Dict[str, np.ndarray]) -> None:
    """Print statistics for each key in the dataset in a legible format."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_samples = None
    for key, data in dataset.items():
        if isinstance(data, np.ndarray):
            if total_samples is None:
                total_samples = data.shape[0]

            print(f"\n{key.upper()}:")
            print(f"  Shape: {data.shape}")

            # Calculate statistics
            mean_val = np.mean(data)
            min_val = np.min(data)
            max_val = np.max(data)
            std_val = np.std(data)

            # Print statistics in a formatted way
            print(f"  Mean:  {mean_val:>12.6f}")
            print(f"  Min:   {min_val:>12.6f}")
            print(f"  Max:   {max_val:>12.6f}")
            print(f"  Std:   {std_val:>12.6f}")

            # If it's a 2D array, also show per-dimension statistics
            if (
                data.ndim == 2 and data.shape[1] <= 10
            ):  # Only for reasonable number of dimensions
                print("  Per-dimension statistics:")
                for i in range(data.shape[1]):
                    dim_mean = np.mean(data[:, i])
                    dim_std = np.std(data[:, i])
                    dim_min = np.min(data[:, i])
                    dim_max = np.max(data[:, i])
                    print(
                        f"    Dim {i:2d}: mean={dim_mean:8.4f}, std={dim_std:8.4f}, min={dim_min:8.4f}, max={dim_max:8.4f}"
                    )
            elif data.ndim == 2 and data.shape[1] > 10:
                print(
                    f"  (Skipping per-dimension stats for {data.shape[1]} dimensions)"
                )
        else:
            print(f"\n{key.upper()}:")
            print(f"  Type: {type(data)}")
            if hasattr(data, "__len__"):
                print(f"  Length: {len(data)}")

    # Print summary information
    if total_samples is not None:
        print("\nSUMMARY:")
        print(f"  Total samples: {total_samples:,}")

        # Try to estimate number of episodes if terminals are available
        if "terminals" in dataset:
            terminals = dataset["terminals"]
            if isinstance(terminals, np.ndarray):
                num_episodes = np.sum(terminals) + 1  # +1 for the last episode
                avg_episode_length = total_samples / num_episodes
                print(f"  Estimated episodes: {num_episodes:,}")
                print(f"  Average episode length: {avg_episode_length:.1f} steps")

        # Calculate total memory usage
        total_memory = sum(
            data.nbytes for data in dataset.values() if isinstance(data, np.ndarray)
        )
        total_memory_mb = total_memory / (1024 * 1024)
        print(f"  Total memory usage: {total_memory_mb:.2f} MB")

    print("\n" + "=" * 60)


@hydra.main(config_path="configs", config_name="iql", version_base=None)
def train(config):
    # Initialize wandb
    wandb_init(config) if config.use_wandb else None

    rich.print("config ", config)
    if "metaworld" in config.env:
        env = utils_env.make_metaworld_env(config.env, config.seed)
        dataset = utils_env.MetaWorld_dataset(config)
    elif "robomimic" in config.env:
        env = utils_env.get_robomimic_env(config.data_path, seed=config.seed)
        dataset = utils_env.Robomimic_dataset(config.data_path, clip_last=True)
    else:
        env = gym.make(config.env)

    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0]

    if config.normalize:
        state_mean = dataset["observations"].mean(axis=0)
        state_std = dataset["observations"].std(axis=0) + 1e-8

        # Bound the std to prevent large values
        min_std = 1e-2
        state_std = np.maximum(state_std, min_std)

        dataset["observations"] = (dataset["observations"] - state_mean) / state_std
        dataset["next_observations"] = (
            dataset["next_observations"] - state_mean
        ) / state_std

        # max_action = np.abs(dataset["actions"]).max()
        # dataset["actions"] = dataset["actions"] / max_action
    else:
        state_mean = 0
        state_std = 1
        # max_action = 1.0

    max_action = float(env.action_space.high[0])

    # label with trained rm rewards
    if config.use_reward_model:
        print("Using relabeled rewards from trained reward model")
        # end effector reward model
        if config.eef_rm:
            dimension = 3 + dataset["actions"].shape[1]
        else:
            dimension = (
                dataset["observations"].shape[1] + dataset["actions"].shape[1]
            )  # TODO
        print(f"Using reward model with dimension {dimension}")

        model = reward_model.RewardModel(config, None, None, None, None, dimension)

        # Use the helper function to build checkpoint path
        path = build_rm_checkpoint_path(config)
        path = os.path.join(config.checkpoints_path, path)
        print(f"Loading reward model from {path}")
        model.load_model(path)
        print("Successfully loaded reward model")
        dataset["rewards"] = model.get_reward(dataset)
    # iql zero
    elif config.trivial_reward == 1:
        print("Using zero rewards (trivial reward)")
        dataset["rewards"] *= 0.0
    # iql gt rewards
    else:
        print("Using ground truth rewards (no reward model)")

    if config.normalize_reward:
        print("Normalizing rewards")
        reward_mean = dataset["rewards"].mean(axis=0)
        reward_std = dataset["rewards"].std(axis=0) + 1e-8
        dataset["rewards"] = (dataset["rewards"] - reward_mean) / reward_std

    print_dataset_statistics(dataset)

    env = utils_env.wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        device=config.device,
    )
    replay_buffer.load_dataset(dataset)

    # if config.checkpoints_path is not None:
    #     print(f"Checkpoints path: {config.checkpoints_path}")
    #     checkpoint_name = build_iql_checkpoint_path(config)
    #     config.checkpoints_path = os.path.join(config.checkpoints_path, checkpoint_name)

    #     os.makedirs(config.checkpoints_path, exist_ok=True)
    #     OmegaConf.save(
    #         config=config, f=os.path.join(config.checkpoints_path, "config.yaml")
    #     )

    # Set seed
    set_seed(config.seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)

    noise_pred_net = FlowNoisePredictionNet(
        action_dim=action_dim, global_cond_dim=state_dim
    ).to(config.device)

    actor = FlowPolicy(
        action_dim=action_dim, noise_pred_net=noise_pred_net, max_action=max_action
    ).to(config.device)

    # Print model architecture after model initialization
    print("\n" + "=" * 50)
    print("Model Network:")
    print("=" * 50)
    print("Q-Network (TwinQ):")
    print(q_network)
    print(f"Q-Network parameters: {sum(p.numel() for p in q_network.parameters()):,}")
    print("\nValue Network:")
    print(v_network)
    print(
        f"Value Network parameters: {sum(p.numel() for p in v_network.parameters()):,}"
    )
    print("\nActor Network:")
    print(actor)
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print("=" * 50)

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

    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)
        # Log training/validation metrics
        wandb.log(log_dict, step=trainer.total_it) if wandb.run is not None else None
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Eval at step: {t + 1}")

            eval_mean_rewards, eval_success, eval_frames = eval_actor(
                env,
                actor,
                config.n_episodes,
                config.seed,
                record_video=config.record_video,
            )
            eval_mean_reward = eval_mean_rewards.mean()
            eval_mean_success = eval_success.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_mean_reward:.3f} , success: {eval_mean_success * 100:.3f}"
            )
            print("---------------------------------------")

            # Log to wandb
            if config.use_wandb:
                # Metrics
                wandb.log(
                    {
                        "eval/mean_rewards": eval_mean_reward,
                        "eval/success": eval_mean_success,
                    },
                    step=trainer.total_it,
                )

                # Rollout vids
                if config.record_video:
                    for i, frames in enumerate(eval_frames):
                        frames_array = np.stack(frames)  # (T, H, W, C)
                        frames_array = np.transpose(
                            frames_array, (0, 3, 1, 2)
                        )  # (T, C, H, W)
                        mean_reward = eval_mean_rewards[i]
                        success = eval_success[i]
                        wandb.log(
                            {
                                f"eval_vids/ep_{i + 1}": wandb.Video(
                                    frames_array,
                                    fps=30,
                                    format="mp4",
                                    caption=f"Mean Reward: {mean_reward:.2f}, Success: {success:.2f}",
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
