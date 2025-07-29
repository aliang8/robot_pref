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
from torch.cuda.amp import GradScaler, autocast
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

    if not normalization_path.exists():
        # create normalization file
        norm_dict = {
            "obs_mean": dataset["observations"].mean(axis=0),
            "obs_std": dataset["observations"].std(axis=0) + 1e-6,
        }
        np.savez(normalization_path,
            obs_mean=norm_dict["obs_mean"],
            obs_std=norm_dict["obs_std"])

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
        state_dim, action_dim, config.buffer_size, K=config.K, scale=config.scale
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
    torch.backends.cudnn.benchmark = True
    # scaler = GradScaler()
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        states, actions, rewards, rtg, dones, timesteps, mask = [x.to(config.device) for x in batch]

        # with autocast():
        action_preds = dt(states, actions, rtg, timesteps, mask)

        action_preds = action_preds.reshape(-1, action_dim)[mask.reshape(-1) > 0]
        action_target = actions.reshape(-1, action_dim)[mask.reshape(-1) > 0]

        loss = torch.mean(
            (action_preds - action_target) ** 2
        )

        # loss = F.mse_loss(action_preds, actions, reduction="none")
        # loss = loss.mean(dim=-1)
        # loss = (loss * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        if wandb.run is not None:
            wandb.log({"loss": loss.cpu().detach().item()}, step=t)

        if (t + 1) % config.eval_freq == 0:
            dt.eval()
            print(f"Evaluation at step: {t + 1}")
            for target_return in config.target_returns:
                
                with torch.no_grad():
                    eval_results = eval_actor_dt(
                        envs,
                        dt,
                        record_video=config.record_video,
                        target_return=target_return / config.scale,
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
                log_evaluation_results(config, t, eval_results, tag=str(target_return))

            dt.train()

if __name__ == "__main__":
    train()
