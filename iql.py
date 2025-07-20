# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf

import copy
import os
import time
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

import wandb
from models.action_chunking_transformer import (
    ActionChunkingTransformer,
    TwinTransformerQ,
)
from utils.common import MLP
from utils.data import (
    Robomimic_dataset,
    SequentialReplayBuffer,
    print_dataset_statistics,
    setup_reward_model,
)
from utils.env import get_robomimic_env
from utils.eval import eval_actor
from utils.log import print_model_info
from utils.seed import set_seed
from utils.wandb import wandb_init

# Type aliases
TensorBatch = List[torch.Tensor]

# Constants
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Soft update of target network parameters."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2 loss for IQL value function."""
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class TwinQ(nn.Module):
    """Twin Q-network for double Q-learning."""

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
        """Return both Q-values."""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return minimum of both Q-values."""
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    """Value function network."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    """Implicit Q-Learning implementation."""

    def __init__(
        self,
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
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor

        # Optimizers
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)

        # Hyperparameters
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        # Training state
        self.total_it = 0
        self.device = device

    def _update_v(
        self, observations: torch.Tensor, actions: torch.Tensor, log_dict: Dict
    ) -> torch.Tensor:
        """Update value function."""
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v

        v_loss = asymmetric_l2_loss(adv, self.iql_tau)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Log metrics
        log_dict.update(
            {
                "target_q_mean": target_q.mean().item(),
                "v_mean": v.mean().item(),
                "adv_mean": adv.mean().item(),
                "v_loss": v_loss.item(),
            }
        )

        # Log adv histogram
        if wandb.run is not None:
            wandb.log({"adv_hist": wandb.Histogram(adv.detach().cpu().numpy())})

        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ) -> None:
        """Update Q-function."""

        if rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if terminals.shape[-1] == 1:
            terminals = terminals.squeeze(-1)

        # Sum over action chunk dim
        if len(rewards.shape) > 1:
            S = rewards.shape[-1]
            rewards = rewards * self.discount ** torch.arange(S, device=rewards.device)
            rewards = rewards.sum(1)

            terminals = terminals[:, -1]
            discount = self.discount**S
        else:
            discount = self.discount

        targets = rewards + (1.0 - terminals.float()) * discount * next_v.detach()
        qs = self.qf.both(observations, actions)

        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

        log_dict["q_loss"] = q_loss.item()

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ) -> None:
        """Update policy using advantage-weighted regression."""
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        pred_actions = self.actor(observations)

        loss = F.mse_loss(pred_actions, actions, reduction="none")
        loss = loss.mean(dim=(1, 2))

        # Advantage-weighted regression
        policy_loss = torch.mean(exp_adv * loss)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        log_dict.update(
            {
                "bc_loss": loss.mean().item(),
                "policy_loss": policy_loss.item(),
                "lr": self.actor_optimizer.param_groups[0]["lr"],
            }
        )

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """Train the model on a batch of data."""
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        log_dict = {}

        # Compute next value
        with torch.no_grad():
            next_v = self.vf(next_observations)

        # Update networks
        adv = self._update_v(observations, actions, log_dict)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def setup_networks(config, state_dim, action_dim, seq_len):
    """Setup networks for IQL."""

    q_network = (
        TwinTransformerQ(state_dim, action_dim, seq_len)
        if seq_len > 1
        else TwinQ(state_dim, action_dim)
    )
    q_network = q_network.to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)

    # noise_pred_net = FlowNoisePredictionNet(
    #     action_dim=action_dim,
    #     global_cond_dim=state_dim
    # ).to(config.device)

    # actor = FlowPolicy(
    #     action_dim=action_dim,
    #     noise_pred_net=noise_pred_net,
    #     max_action=max_action
    # ).to(config.device)

    actor = ActionChunkingTransformer(state_dim, action_dim, seq_len).to(config.device)

    return q_network, v_network, actor


@hydra.main(config_path="configs", config_name="iql", version_base=None)
def train(config):
    wandb_init(config) if config.use_wandb else None

    rich.print("config ", config)

    # Load datasets
    dataset = Robomimic_dataset(config.data_path)
    print_dataset_statistics(dataset)

    # Setup env
    def make_env_fn(seed):
        def _init():
            env = get_robomimic_env(config.data_path, seed=seed)
            return env

        return _init

    env_fns = [make_env_fn(seed) for seed in range(config.n_envs)]
    env = DummyVecEnv(env_fns)

    
    if config.use_reward_model:
        dataset = setup_reward_model(config, dataset)
    elif config.trivial_reward == 1:
        print("Using zero rewards (trivial reward)")
        dataset["rewards"] *= 0.0
    else:
        print("Using ground truth rewards")

    print_dataset_statistics(dataset)

    # Replay Buffer
    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = SequentialReplayBuffer(
        state_dim, action_dim, config.buffer_size, config.seq_len, device=config.device
    )
    replay_buffer.load_dataset(dataset)

    # Set seed
    set_seed(config.seed, env)

    # Setup networks
    q_network, v_network, actor = setup_networks(
        config, state_dim, action_dim, config.seq_len
    )
    print_model_info(
        {"Q-Network": q_network, "Value Network": v_network, "Actor Network": actor}
    )

    # Setup optimizers
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.actor_lr, weight_decay=1e-4
    )

    # Initialize trainer
    trainer = ImplicitQLearning(
        actor=actor,
        actor_optimizer=actor_optimizer,
        q_network=q_network,
        q_optimizer=q_optimizer,
        v_network=v_network,
        v_optimizer=v_optimizer,
        discount=config.discount,
        tau=config.tau,
        device=config.device,
        beta=config.beta,
        iql_tau=config.iql_tau,
        max_steps=config.max_timesteps,
    )

    start_time = time.time()

    # Training loop
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)

        batch_time = time.time() - start_time
        log_dict["batch_time"] = batch_time

        wandb.log(log_dict, step=trainer.total_it) if wandb.run is not None else None
        if (t + 1) % config.eval_freq == 0:
            print(f"Eval at step: {t + 1}")

            eval_reward, eval_sr, eval_frames = eval_actor(
                env, actor, record_video=config.record_video, seq_len=config.seq_len
            )

            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_envs} episodes: "
                f"{eval_reward:.3f} , success: {eval_sr * 100:.3f}"
            )
            print("---------------------------------------")

            # Log to wandb
            if config.use_wandb:
                wandb.log(
                    {
                        "eval/mean_rewards": eval_reward,
                        "eval/success": eval_sr,
                    },
                    step=trainer.total_it,
                )

                if config.record_video:
                    for i, frames in enumerate(eval_frames):
                        frames_array = np.stack(frames)  # (T, H, W, C)
                        frames_array = np.transpose(frames_array, (0, 3, 1, 2))
                        wandb.log(
                            {
                                f"eval_vids/ep_{i + 1}": wandb.Video(
                                    frames_array,
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
