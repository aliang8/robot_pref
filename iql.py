# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf

import copy
import os
from typing import Any, Dict, List, Tuple

import hydra
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from utils.env import wrap_env
import wandb
from models.flow_policy import FlowNoisePredictionNet, FlowPolicy
from utils.eval import eval_actor, log_evaluation_results
from utils.common import MLP
from utils.wandb import wandb_init
from utils.seed import set_seed
from utils.data import normalize_datasets, ReplayBuffer, setup_environment_and_dataset, setup_reward_model, print_dataset_statistics


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
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _update_v(self, observations: torch.Tensor, actions: torch.Tensor, log_dict: Dict) -> torch.Tensor:
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
        log_dict.update({
            "target_q_mean": target_q.mean().item(),
            "v_mean": v.mean().item(),
            "adv_mean": adv.mean().item(),
            "v_loss": v_loss.item(),})

        # Log adv histogram
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
        import ipdb; ipdb.set_trace()  # CHECK SHAPES HERE
        if rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if terminals.shape[-1] == 1:
            terminals = terminals.squeeze(-1)

        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
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
        loss = self.actor(observations, actions)
        
        # Advantage-weighted regression
        policy_loss = torch.mean(exp_adv * loss)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        log_dict.update({
            "bc_loss": loss.mean().item(),
            "policy_loss": policy_loss.item(),
            "lr": self.actor_optimizer.param_groups[0]["lr"],
        })

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """Train the model on a batch of data."""
        self.total_it += 1
        observations, actions, rewards, _, next_observations, dones, _ = batch
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


def setup_networks(config, state_dim, action_dim, max_action):
    """Setup neural networks for IQL."""
    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    
    noise_pred_net = FlowNoisePredictionNet(
        action_dim=action_dim, 
        global_cond_dim=state_dim
    ).to(config.device)
    
    actor = FlowPolicy(
        action_dim=action_dim,
        noise_pred_net=noise_pred_net,
        max_action=max_action
    ).to(config.device)
    
    return q_network, v_network, actor


def print_model_info(q_network, v_network, actor):
    """Print model architecture information."""
    print("\n" + "=" * 50)
    print("MODEL ARCHITECTURES")
    print("=" * 50)
    
    models = [
        ("Q-Network (TwinQ)", q_network),
        ("Value Network", v_network),
        ("Actor Network", actor)
    ]
    
    for name, model in models:
        print(f"\n{name}:")
        print(model)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name} parameters: {param_count:,}")
    
    print("=" * 50)


@hydra.main(config_path="configs", config_name="iql", version_base=None)
def train(config):
    """Main training function."""
    # Initialize wandb
    if config.use_wandb:
        wandb_init(config)
    
    rich.print("Config:", config)
    
    # Setup environment and dataset
    env, dataset = setup_environment_and_dataset(config)
    
    # Get dimensions
    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Normalize dataset
    state_mean, state_std = normalize_datasets(dataset)
    
    # Setup reward model and label rewards if needed
    if config.use_reward_model:
        dataset = setup_reward_model(config, dataset)
    elif config.trivial_reward == 1:
        print("Using zero rewards (trivial reward)")
        dataset["rewards"] *= 0.0
    else:
        print("Using ground truth rewards")
    
    # Print dataset statistics
    print_dataset_statistics(dataset)
    
    # Wrap environment
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    
    # Setup replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, device=config.device)
    replay_buffer.load_dataset(dataset)
    
    # Set seed
    set_seed(config.seed, env)
    
    # Setup networks
    q_network, v_network, actor = setup_networks(config, state_dim, action_dim, max_action)
    print_model_info(q_network, v_network, actor)
    
    # Setup optimizers
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr, weight_decay=1e-4)

    # Initialize trainer
    trainer = ImplicitQLearning(
        max_action=max_action,
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
    
    # Training loop
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)
        
        # Log training metrics
        if wandb.run is not None:
            wandb.log(log_dict, step=trainer.total_it)
        
        # Evaluate periodically
        if (t + 1) % config.eval_freq == 0:
            print(f"Evaluation at step: {t + 1}")
            
            eval_results = eval_actor(
                env, actor, config.n_episodes, config.seed, 
                record_video=config.record_video
            )
            
            eval_mean_rewards, eval_success, _ = eval_results
            eval_mean_reward = eval_mean_rewards.mean()
            eval_mean_success = eval_success.mean()
            
            print("---------------------------------------")
            print(f"Evaluation over {config.n_episodes} episodes: "
                  f"Reward: {eval_mean_reward:.3f}, Success: {eval_mean_success * 100:.1f}%")
            print("---------------------------------------")
            
            # Log evaluation results
            log_evaluation_results(config, trainer.total_it, eval_results)

            # Save checkpoint
            if (config.checkpoints_path is not None and 
                (t + 1) % (20 * config.eval_freq) == 0):
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt")
                )


if __name__ == "__main__":
    train()