import os
from typing import List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from utils.viz import create_video_grid
from utils.wandb import log_to_wandb


os.environ["MUJOCO_GL"] = "egl"


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: nn.Module,
    n_episodes: int,
    seed: int,
    max_steps: int = 500,
    record_video: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Evaluate the actor on the environment."""
    # TODO: implement parallel
    actor.eval()

    episode_mean_rewards = []
    episode_success_list = []
    episode_frames = []

    for i in range(n_episodes):
        mean_reward, success, frames = _eval_episode(
            env,
            actor,
            max_steps,
            seed + (i * 5),  # so sequential seeds don't overlap
            record_video=record_video,  # and i < 5,  # Record up to 5 episodes
        )
        episode_mean_rewards.append(mean_reward)
        episode_success_list.append(success)
        episode_frames.append(frames)

    actor.train()

    return (
        np.array(episode_mean_rewards),
        np.array(episode_success_list),
        episode_frames,
    )


def _eval_episode(env, actor, max_steps, seed, record_video=False, device="cuda"):
    env.seed(seed)
    state, _ = env.reset()

    frames = [env.render(mode="rgb_array")] if record_video else []
    total_reward, success = 0.0, False

    for _ in range(max_steps):
        action = (
            actor.sample(torch.from_numpy(state).unsqueeze(0).to(device))
            .cpu()
            .numpy()[0]
        )
        state, reward, _, _, info = env.step(action)
        if record_video:
            frame = env.render(mode="rgb_array")
            frames.append(frame)
        total_reward += reward
        if info.get("success", False):
            success = True
            break

    mean_reward = total_reward / (len(frames) or max_steps)
    return mean_reward, int(success), frames if record_video else None
