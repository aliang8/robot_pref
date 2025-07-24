import copy
import os
import time
from pathlib import Path
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
    ValueFunction,
)
from models.decision_transformer import DecisionTransformer
from utils.common import MLP
from utils.data import (
    DTSequentialReplayBuffer,
    Robomimic_dataset,
    print_dataset_statistics,
    setup_reward_model,
)
from utils.env import get_robomimic_env, wrap_env
from utils.eval import eval_actor_dt, log_evaluation_results
from utils.log import print_model_info
from utils.seed import set_seed
from utils.wandb import wandb_init


@hydra.main(config_path="configs", config_name="dt", version_base=None)
def train(config):
    wandb_init(config) if config.use_wandb else None
    rich.print("config", config)

    set_seed(config.seed)

    # Load datasets
    dataset = Robomimic_dataset(config.data_path)
    print_dataset_statistics(dataset)

    normalization_path = Path(config.data_path).parent / "normalization.npz"

    # Setup envs
    def create_env(seed):
        env = get_robomimic_env(config.data_path, seed=seed, normalization_path=normalization_path)
        return env
    envs = [create_env(seed) for seed in range(config.n_envs)]

    dataset["observations"] = envs[0].normalize_obs(dataset["observations"])
    # dataset["actions"] = envs[0].normalize_action(dataset["actions"])

    if config.use_reward_model:
        dataset = setup_reward_model(config, dataset)
    elif config.trivial_reward == 1:
        print("Using zero rewards (trivial reward)")
        dataset["rewards"] *= 0.0
    else:
        print("Using ground truth rewards")

    # Replay Buffer
    state_dim = envs[0].observation_space["state"].shape[0]
    action_dim = envs[0].action_space.shape[0]
    replay_buffer = DTSequentialReplayBuffer(
        state_dim, action_dim, config.buffer_size, K=config.K, device=config.device, scale=config.scale
    )
    replay_buffer.load_dataset(dataset)

    print_dataset_statistics(dataset)

    # Networks
    dt = DecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=config.hidden_size,
        max_length=config.K,
        nhead=config.nhead,
        nlayer=config.nlayer,
    ).to(config.device)
    print_model_info({"Decision Transformer": dt})
    optimizer = torch.optim.Adam(dt.parameters(), lr=config.lr, weight_decay=1e-4)

    # Training loop
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        action_preds = dt(batch)

        loss = F.mse_loss(action_preds, batch[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if wandb.run is not None:
            wandb.log({"loss": loss.item()}, step=t)

        if (t + 1) % config.eval_freq == 0:
            print(f"Evaluation at step: {t + 1}")

            eval_results = eval_actor_dt(
                envs,
                dt,
                record_video=config.record_video,
                target_return=config.target_return,
                scale=config.scale,
            )

            eval_mean_rewards, eval_success, eval_videos = eval_results
            eval_mean_reward = eval_mean_rewards.mean()
            eval_mean_success = eval_success.mean()

            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_envs} envs: "
                f"Reward: {eval_mean_reward:.3f}, Success: {eval_mean_success * 100:.1f}%"
            )
            print("---------------------------------------")
            log_evaluation_results(config, t, eval_results)


if __name__ == "__main__":
    train()
