import os
from typing import List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import wandb
import cv2

os.environ["MUJOCO_GL"] = "egl"


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: nn.Module,
    max_steps: int = 500,
    record_video: bool = True,
    seq_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Evaluate the actor on the environment."""
    actor.eval()
    if record_video:
        [
            setattr(env, "render_mode", "rgb_array") for env in env.envs
        ]  # needed for rendering

    if seq_len > 1:
        mean_reward, success, frames = _eval_episode_seq(
            env, actor, max_steps, record_video=record_video
        )
    # else:
    #     # Standard evaluation
    #     mean_reward, success, frames = _eval_episode(
    #         env,
    #         actor,
    #         max_steps,
    #         seed + (i * 5),  # so sequential seeds don't overlap
    #         record_video=record_video,  # and i < 5,  # Record up to 5 episodes
    # )
    # episode_mean_rewards.append(mean_reward)
    # episode_success_list.append(success)
    # episode_frames.append(frames)

    actor.train()

    return (
        mean_reward,
        success,
        frames,
    )


@torch.no_grad()
def eval_actor_dt(
    envs: List[gym.Env],
    actor: nn.Module,
    target_return: float,
    max_steps: int = 500,
    record_video: bool = True,
    scale: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Evaluate the actor on the environment."""
    actor.eval()
    if record_video:
        [
            setattr(env, "render_mode", "rgb_array") for env in envs
        ]  # needed for rendering
    
    mean_rewards = []
    successes = []
    videos = []

    for i, env in enumerate(envs):
        mean_reward, success, frames = _eval_episode_dt_seq(
            env, actor, max_steps, target_return/scale, record_video=record_video, scale=scale
        )
        mean_rewards.append(mean_reward)
        successes.append(success)

        if i < 5: 
            videos.append(frames)
        
    actor.train()

    return (
        np.array(mean_rewards),
        np.array(successes),
        videos,
    )


# def _eval_episode(env, actor, max_steps, seed, record_video=False, device="cuda"):
#     env.seed(seed)
#     state, _ = env.reset()

#     frames = [env.render(mode="rgb_array")] if record_video else []
#     total_reward, success = 0.0, False

#     for _ in range(max_steps):
#         with torch.no_grad():
#             action = (
#                 actor.sample(torch.from_numpy(state).unsqueeze(0).to(device))
#                 .cpu()
#                 .numpy()[0]
#             )
#         state, reward, _, _, info = env.step(action)
#         if record_video:
#             frame = env.render(mode="rgb_array")
#             frames.append(frame)
#         total_reward += reward
#         if info.get("success", False):
#             success = True
#             break

#     mean_reward = total_reward / (len(frames) or max_steps)
#     return mean_reward, int(success), frames if record_video else None


def _eval_episode_seq(env, actor, max_steps, record_video=False, device="cuda"):
    states = env.reset()
    num_envs = len(states["state"])

    frames = (
        [[env.envs[i].render()] for i in range(min(5, num_envs))]
        if record_video
        else None
    )

    # total_reward, success = 0.0, False
    total_steps = 0
    total_rewards = np.zeros(num_envs)
    success_flags = np.zeros(num_envs, dtype=bool)
    done_flags = np.zeros(num_envs, dtype=bool)

    for _ in range(max_steps):
        if isinstance(states, dict):
            states = states["state"]
        else:
            states = states

        with torch.no_grad():
            actions = actor(torch.from_numpy(states).to(device)).cpu().numpy()

        for i in range(actions.shape[1]):
            action = actions[:, i]  # (B, D)
            if total_steps >= max_steps:
                break

            states, rewards, dones, infos = env.step(action)
            total_rewards += rewards
            done_flags = np.logical_or(done_flags, dones)

            if record_video:
                for env_i in range(min(5, num_envs)):
                    frame = env.envs[env_i].render()
                    frames[env_i].append(frame)

            for i, info in enumerate(infos):
                if info.get("success", False):
                    success_flags[i] = True
            if all(done_flags):
                break

            total_steps += 1

    avg_reward = total_rewards.mean()
    success_rate = success_flags.mean()

    return avg_reward, success_rate, frames if record_video else None


# Decision Transformer eval
def _eval_episode_dt_seq(env, actor, max_steps, target_return, record_video=False, device="cuda", mode='normal', scale=10.0):
    actor = actor.to(device)
    state = env.reset()

    state = state["state"]

    state_dim = env.observation_space["state"].shape[0]
    act_dim = env.action_space.shape[0]

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).repeat(1, 1)
    timesteps = torch.zeros((1, 1), device=device, dtype=torch.long)

    # eval metrics
    total_reward, success = 0.0, False
    frames = [env.render(mode="rgb_array")] if record_video else []

    for t in range(max_steps):
        # latest action and reward is "padding"
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = actor.get_action(
            states,
            actions,
            rewards,
            target_return,
            timesteps,
        )
        # update
        actions[-1] = action

        # step
        action = action.detach().cpu().numpy()
        state, reward, _, info = env.step(action)
        state = state["state"]
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0, -1] - (reward/scale)
            # pred_return = target_return[0, -1] - reward
        else:
            pred_return = target_return[0, -1]

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        total_reward += reward

        if record_video:
            frame = env.render(mode="rgb_array")
            frames.append(frame)

        if info.get("success", False):
            success = True
            break

    mean_reward = total_reward / (len(frames) or max_steps)

    return mean_reward, int(success), frames if record_video else None


def log_evaluation_results(config, total_it, eval_results):
    """Log evaluation results to wandb."""
    if not config.use_wandb:
        return

    eval_mean_rewards, eval_success, eval_frames = eval_results
    eval_mean_reward = eval_mean_rewards.mean()
    eval_mean_success = eval_success.mean()

    # Log metrics
    wandb.log(
        {
            "eval/mean_rewards": eval_mean_reward,
            "eval/success": eval_mean_success,
        },
        step=total_it,
    )

    # Log videos if requested
    if config.record_video and eval_frames:
        for i, frames in enumerate(eval_frames):
            frames_array = np.stack(frames)  # (T, H, W, C)
            frames_array = np.transpose(frames_array, (0, 3, 1, 2))  # (T, C, H, W)

            wandb.log(
                {
                    f"eval_vids/ep_{i + 1}": wandb.Video(
                        frames_array,
                        fps=30,
                        format="mp4",
                        caption=f"Reward: {eval_mean_rewards[i]:.2f}, Success: {eval_success[i]:.2f}",
                    )
                },
                step=total_it,
            )
