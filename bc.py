import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import rich
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import trange

import utils_env
import wandb
from iql import ReplayBuffer, eval_actor, print_dataset_statistics, wrap_env
from models.flow_policy import FlowNoisePredictionNet, FlowPolicy
from reward_utils import normalize_states
from utils.wandb import wandb_init

TensorBatch = List[torch.Tensor]

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std





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
        loss = self.actor(state, action)
        log_dict.update({"loss": loss.item()})

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        loss.backward()
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

class EnvFactory:
    def __init__(self, data_path):
        self.data_path = data_path

    def __call__(self, seed):
        return utils_env.get_robomimic_env(self.data_path, seed=seed)

@hydra.main(config_path="configs", config_name="bc", version_base=None)
def train(config):
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    wandb_init(config) if config.use_wandb else None

    rich.print("config ", config)
    if "metaworld" in config.env:
        env = utils_env.make_metaworld_env(config.env, config.seed)
        dataset = utils_env.MetaWorld_dataset(config)
    elif "dmc" in config.env:
        env = utils_env.make_dmc_env(config.env, config.seed)
        dataset = utils_env.DMC_dataset(config)
    elif "robomimic" in config.env:
        env = utils_env.get_robomimic_env(config.data_path, seed=config.seed)
        # env_fn = EnvFactory(config.data_path)
        dataset = utils_env.Robomimic_dataset(config.data_path, seq_len=config.seq_len)
    else:
        env = gym.make(config.env)

    # Handle both standard and dict observation spaces
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space["state"].shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-5)
        dataset["observations"] = normalize_states(
            dataset["observations"], state_mean, state_std
        )
        dataset["next_observations"] = normalize_states(
            dataset["next_observations"], state_mean, state_std
        )

    print_dataset_statistics(dataset)
    
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        seq_len=config.seq_len if hasattr(config, 'seq_len') else None,
        device=config.device,
    )
    replay_buffer.load_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        # Save config using Hydra's utilities
        OmegaConf.save(config=config, f=os.path.join(config.checkpoints_path, "config.yaml"))

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    
    noise_pred_net = FlowNoisePredictionNet(
        action_dim=action_dim,
        global_cond_dim=state_dim
    ).to(config.device)

    actor = FlowPolicy(action_len=config.seq_len, action_dim=action_dim, noise_pred_net=noise_pred_net)

    # Print model architecture after model initialization
    print("\n" + "=" * 50)
    print("MODEL ARCHITECTURE:")
    print("=" * 50)
    print("Actor Network:")
    print(actor)
    print(f"Total parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print("=" * 50)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

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

    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it) if wandb.run is not None else None
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Eval at step: {t + 1}")

            
            eval_scores, eval_success, eval_frames = eval_actor(
                env,
                # env_fn,
                actor,
                config.n_episodes,
                config.seed,
                seq_len=config.seq_len,
                # record_video=config.record_video,
                record_video=False,  # Disable video recording for now, TODO: run with rendering in series
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