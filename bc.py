import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import hydra
import numpy as np
import rich
import torch
import torch.nn as nn
from tqdm import trange

import utils_env
import wandb
from iql import ReplayBuffer, eval_actor
from models.act import ActionChunkingTransformerPolicy, ACTTemporalEnsembler
from utils.loss import arm_gripper_loss

TensorBatch = List[torch.Tensor]

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
        project=config.wandb.project,
        name=config.wandb.name,
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

class Actor(nn.Module):
    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        max_action:  float,
        model_config: Dict[str, Any]
    ):
        super().__init__()

        self.state_embedder = nn.Linear(state_dim, model_config.d_model)
        self.model = ActionChunkingTransformerPolicy(cfg=model_config, output_dim=action_dim)
        self.loss_fn = nn.L1Loss(reduction="none")
        if model_config.use_separate_gripper:
            self.gripper_pos_weight = getattr(model_config, "gripper_pos_weight", 1.0)
            self.gripper_pos_weight = torch.tensor(self.gripper_pos_weight)
            self.gripper_loss_fn = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=self.gripper_pos_weight
            )

        self.max_action = float(max_action)

        self.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff=model_config.temporal_ensemble_coeff, chunk_size=model_config.seq_len)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_embs = self.state_embedder(state)
        action_preds = self.model(state_embs)
        
        return action_preds
    
    def compute_loss(
        self, state: torch.Tensor, target_action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        state_embs = self.state_embedder(state)
        action_preds = self.model(state_embs)

        arm_loss, gripper_loss, gripper_acc = arm_gripper_loss(action_preds=action_preds, actions=target_action, arm_loss_fn=self.loss_fn, gripper_loss_fn=self.gripper_loss_fn, gaussian_output=False)

        total_loss = arm_loss + gripper_loss

        return total_loss, {
            "arm_loss": arm_loss.item(),
            "gripper_loss": gripper_loss.item(),
            "gripper_acc": gripper_acc.item(),
            "loss": total_loss.item(),
        }

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> torch.Tensor:
        state_embs = self.state_embedder(state)
        action_preds = self.model(state_embs).actions
        action = self.temporal_ensembler.update(action_preds)

        return action.squeeze()

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

        if state.dim() == 3:
            state = state[:, 0:1] # take only the first frame for now

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


@hydra.main(config_path="configs", config_name="bc", version_base=None)
def train(config):
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

    # Handle both standard and dict observation spaces
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space["state"].shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


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
        seq_len=config.model.seq_len,
    )
    replay_buffer.load_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        # Save config using Hydra's utilities
        from omegaconf import OmegaConf
        OmegaConf.save(config=config, f=os.path.join(config.checkpoints_path, "config.yaml"))

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action,model_config=config.model).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

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

    wandb_init(config) if config.use_wandb else None

    evaluations = []
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it) if wandb.run is not None else None
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
            ) if wandb.run is not None else None
            
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
                        ) if wandb.run is not None else None
            
            if (config.checkpoints_path is not None) and (t + 1) % (
                20 * config.eval_freq
            ) == 0:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

if __name__ == "__main__":
    train()