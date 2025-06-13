import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_env
import wandb
from iql import ReplayBuffer, eval_actor

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    # Video recording
    record_video: bool = False  # Whether to record evaluation videos
    video_dir: Optional[str] = None  # Directory to save videos, if None, uses "bc_videos"

    # BC
    buffer_size: int = 1_000_000  # Replay buffer size
    frac: float = 0.1  # Best data fraction to use
    max_traj_len: int = 1000  # Max trajectory length
    normalize: bool = False  # Normalize states
    # Wandb logging
    project: str = "robot_pref"
    entity: str = "clvr"
    group: str = "BC-Robomimic"
    name: str = "BC"

    data_path: str = ""

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


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
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


# @torch.no_grad()
# def eval_actor(
#     env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
# ) -> np.ndarray:
#     env.seed(seed)
#     actor.eval()
#     episode_rewards = []
#     for _ in range(n_episodes):
#         state, done = env.reset(), False
#         episode_reward = 0.0
#         while not done:
#             action = actor.act(state, device)
#             state, reward, done, _ = env.step(action)
#             episode_reward += reward
#         episode_rewards.append(episode_reward)

#     actor.train()
#     return np.asarray(episode_rewards)


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminals"] = dataset["terminals"][order]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, pos_weight: float = 1.0):
        super(Actor, self).__init__()

        # Main action network (excluding gripper)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim - 1),  # Exclude gripper dimension
            nn.Tanh(),
        )

        # Separate gripper network - outputs a single value for binary classification
        self.gripper_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Use sigmoid for binary classification
        )

        self.max_action = max_action
        self.pos_weight = pos_weight  # Weight for gripper loss

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        main_actions = self.net(state)
        gripper_logits = self.gripper_net(state)
        # Convert sigmoid output (0-1) to -1 or 1
        gripper_action = 2 * gripper_logits - 1
        return torch.cat([main_actions, gripper_action], dim=-1)

    def compute_loss(self, state: torch.Tensor, target_action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pi = self(state)
        
        # Split actions into main actions and gripper
        arm_actions = target_action[:, :-1]
        gripper_action = target_action[:, -1:]
        arm_pi = pi[:, :-1]
        gripper_pi = pi[:, -1:]
        
        # Compute separate losses
        arm_loss = F.mse_loss(arm_pi, arm_actions)
        
        # Convert target gripper action from [-1,1] to [0,1] for BCE
        gripper_target = (gripper_action + 1) / 2
        # Convert predicted gripper action from [-1,1] to [0,1] for BCE
        gripper_pred = (gripper_pi + 1) / 2
        gripper_loss = F.binary_cross_entropy(gripper_pred, gripper_target)
        
        # Combine losses with gripper weight
        total_loss = arm_loss + self.pos_weight * gripper_loss
        
        log_dict = {
            "arm_loss": arm_loss.item(),
            "gripper_loss": gripper_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, log_dict

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions = self(state)
        
        # Round gripper action to -1 or 1
        main_actions = actions[:, :-1]
        gripper_action = actions[:, -1:]
        gripper_action = torch.sign(gripper_action)  # This will give -1 or 1
        
        actions = torch.cat([main_actions, gripper_action], dim=-1)
        return actions.cpu().data.numpy().flatten()


class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss using the actor's compute_loss method
        actor_loss, loss_dict = self.actor.compute_loss(state, action)
        log_dict.update(loss_dict)
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
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
        env = utils_env.get_robomimic_env(config.data_path, seed=config.seed)
        dataset = utils_env.Robomimic_dataset(config.data_path)
    else:
        env = gym.make(config.env)

    # state_dim = env.observation_space.shape[0]  # 39 for metaworld
    state_dim = env.observation_space["state"].shape[0] # robomimic
    action_dim = env.action_space.shape[0]  # 4 for metaworld


    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-4)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Eval at step: {t + 1}")
            
            eval_scores, eval_success, eval_frames = eval_actor(
                env,
                config.env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                record_video=config.record_video,
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
            if config.record_video:
                for i, frames in enumerate(eval_frames):
                    if frames is not None:
                        # Convert frames list to numpy array and transpose to (T, C, H, W)
                        frames_array = np.stack(frames)  # (T, H, W, C)
                        frames_array = np.transpose(frames_array, (0, 3, 1, 2))  # (T, C, H, W)
                        wandb.log(
                            {
                                f"eval_vids/video_episode_{i+1}": wandb.Video(
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