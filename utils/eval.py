import gc
import os
from typing import List, Tuple

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn

import wandb

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
    record_video: bool,
    scale: float,
    max_steps: int = 250,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Evaluate the actor on the environment."""
    if record_video:
        [
            setattr(env, "render_mode", "rgb_array") for env in envs
        ]  # needed for rendering
    
    mean_rewards = []
    successes = []
    videos = []

    for i, env in enumerate(envs):
        mean_reward, success, frames = _eval_episode_dt_seq(
            env, actor, max_steps, target_return, scale, record_video=record_video,
        )
        mean_rewards.append(mean_reward)
        successes.append(success)

        if i < 5: 
            videos.append(frames)
            del frames
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   
        

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
def _eval_episode_dt_seq(env, actor, max_steps, target_return, scale, record_video=False, device="cuda"):
    state = env.reset()
    state = state["state"]

    state_dim = env.observation_space["state"].shape[0]
    act_dim = env.action_space.shape[0]

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    # eval metrics
    episode_return, episode_length, success = 0, 0, False
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
        action = action.detach().cpu().numpy()

        state, reward, _, info = env.step(action)
        state = state["state"]

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # if mode != 'delayed':
        pred_return = target_return[0, -1] - (reward/scale)
        pred_return = target_return[0, -1]
        # pred_return = target_return[0, -1] - reward
        # else:
        #     pred_return = target_return[0, -1]

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if record_video:
            frame = env.render(mode="rgb_array")
            # Convert to BGR if needed (OpenCV expects BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Overlay return as text
            cv2.putText(
                frame_bgr,
                f"Return: {pred_return:.2f}",
                org=(10, 30),  # (x, y)
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Convert back to RGB for logging/saving
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            frames.append(frame_rgb)
            # frames.append(frame)

        if info.get("success", False):
            success = True
            break

    mean_reward = episode_return / (episode_length or max_steps)

    return mean_reward, int(success), frames if record_video else None


def log_evaluation_results(config, total_it, eval_results, tag=None):
    """Log evaluation results to wandb."""
    if not config.use_wandb:
        return

    eval_mean_rewards, eval_success, eval_frames = eval_results
    eval_mean_reward = eval_mean_rewards.mean()
    eval_mean_success = eval_success.mean()

    # Log metrics
    if tag:
        wandb.log(
            {
                f"eval/{tag}_mean_rewards": eval_mean_reward,
                f"eval/{tag}_success": eval_mean_success,
            },
            step=total_it,
        )
    else:
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
                    f"eval_vids/{tag}_ep_{i + 1}": wandb.Video(
                        frames_array,
                        fps=30,
                        format="mp4",
                        caption=f"Reward: {eval_mean_rewards[i]:.2f}, Success: {eval_success[i]:.2f}",
                    )
                },
                step=total_it,
            )
