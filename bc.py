import os
import time
from pathlib import Path
from typing import Any, Dict, List

import gym
import hydra
import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm import trange

import wandb
from models.action_chunking_transformer import ActionChunkingTransformer
from models.flow_policy import FlowNoisePredictionNet, FlowPolicy
from utils.data import Robomimic_dataset, SequentialReplayBuffer, normalize_datasets
from utils.env import get_robomimic_env, wrap_env
from utils.eval import eval_actor, log_evaluation_results
from utils.log import print_model_info
from utils.seed import set_seed
from utils.wandb import wandb_init

TensorBatch = List[torch.Tensor]


class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        observations, actions, _, _, _ = batch

        pred_actions = self.actor(observations)
        loss = F.mse_loss(pred_actions, actions)
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


@hydra.main(config_path="configs", config_name="bc", version_base=None)
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

    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = SequentialReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        device=config.device,
        seq_len=config.seq_len,
    )
    replay_buffer.load_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        # Save config using Hydra's utilities
        OmegaConf.save(
            config=config, f=os.path.join(config.checkpoints_path, "config.yaml")
        )

    max_action = float(env.action_space.high[0])

    # Set seeds
    set_seed(config.seed)

    # noise_pred_net = FlowNoisePredictionNet(
    #     action_dim=action_dim, global_cond_dim=state_dim
    # ).to(config.device)
    # actor = FlowPolicy(
    #     action_dim=action_dim, noise_pred_net=noise_pred_net, max_action=max_action
    # ).to(config.device)
    actor = ActionChunkingTransformer(state_dim, action_dim, config.seq_len).to(
        config.device
    )

    print_model_info({"Actor": actor})

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    start_time = time.time()

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
