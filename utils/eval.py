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
    n_episodes: int,
    seed: int,
    max_steps: int = 500,
    record_video: bool = True,
    seq_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Evaluate the actor on the environment."""
    # TODO: implement parallel
    actor.eval()

    episode_mean_rewards = []
    episode_success_list = []
    episode_frames = []

    for i in range(n_episodes):
        if seq_len > 1:
            mean_reward, success, frames = _eval_episode_seq(
                env, actor, max_steps, seed + (i * 5), record_video=record_video
            )
        else:
            # Standard evaluation
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
        with torch.no_grad():
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

def _eval_episode_seq(env, actor, max_steps, seed, record_video=False, device="cuda"):
    env.seed(seed)
    state, _ = env.reset()

    frames = [env.render(mode="rgb_array")] if record_video else []
    total_reward, success = 0.0, False
    total_steps = 0

    for _ in range(max_steps):
        with torch.no_grad():
            actions = (
                actor(torch.from_numpy(state).unsqueeze(0).to(device))
                .cpu()
                .numpy()
            ).squeeze(0)
        
        for action in actions:
            if total_steps >= max_steps:
                break
                
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1

            if record_video:
                frame = env.render(mode="rgb_array")
                frames.append(frame)

            if info.get("success", False):
                success = True
            if terminated or truncated:
                break
        
        if terminated or truncated or success:
            break
    
    return total_reward / total_steps, int(success), frames if record_video else None

# Decision Transformer eval
# def _eval_episode_dt_seq(env, actor, max_steps, seed, record_video=False, device="cuda", target_return=30.0, mode='normal', scale=1000.):
#     actor = actor.to(device)

#     env.seed(seed)
#     state, _ = env.reset()

#     state_dim = env.observation_space["state"].shape[0]
#     act_dim = env.action_space.shape[0]

#     states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
#     actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
#     rewards = torch.zeros(0, device=device, dtype=torch.float32)
#     target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
#     timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

#     # eval metrics
#     total_reward, success = 0.0, False
#     frames = [env.render(mode="rgb_array")] if record_video else []

#     for t in range(max_steps):
#         # latest action and reward is "padding"
#         actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
#         rewards = torch.cat([rewards, torch.zeros(1, device=device)])

#         action = actor.sample(
#             states,
#             actions,
#             rewards,
#             target_return,
#             timesteps,
#         )
#         # update
#         actions[-1] = action

#         # step
#         action = action.detach().cpu().numpy()
#         state, reward, _, _, info = env.step(action)
#         cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
#         states = torch.cat([states, cur_state], dim=0)
#         rewards[-1] = reward

#         if mode != 'delayed':
#             pred_return = target_return[0, -1] - (reward/scale)
#         else:
#             pred_return = target_return[0, -1]
#         target_return = torch.cat(
#             [target_return, pred_return.reshape(1, 1)], dim=1)
#         timesteps = torch.cat(
#             [timesteps,
#              torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

#         total_reward += reward

#         if record_video:
#             frame = env.render(mode="rgb_array")
#             frames.append(frame)
        
#         if info.get("success", False):
#             success = True
#             break

#     mean_reward = total_reward / (len(frames) or max_steps)

#     return mean_reward, int(success), frames if record_video else None

def log_evaluation_results(config, total_it, eval_results):
    """Log evaluation results to wandb."""
    if not config.use_wandb:
        return
    
    eval_mean_rewards, eval_success, eval_frames = eval_results
    eval_mean_reward = eval_mean_rewards.mean()
    eval_mean_success = eval_success.mean()
    
    # Log metrics
    wandb.log({
        "eval/mean_rewards": eval_mean_reward,
        "eval/success": eval_mean_success,
    }, step=total_it)
    
    # Log videos if requested
    if config.record_video and eval_frames:
        for i, frames in enumerate(eval_frames):
            frames_array = np.stack(frames)  # (T, H, W, C)
            frames_array = np.transpose(frames_array, (0, 3, 1, 2))  # (T, C, H, W)
            
            wandb.log({
                f"eval_vids/ep_{i + 1}": wandb.Video(
                    frames_array,
                    fps=30,
                    format="mp4",
                    caption=f"Reward: {eval_mean_rewards[i]:.2f}, Success: {eval_success[i]:.2f}",
                )
            }, step=total_it)