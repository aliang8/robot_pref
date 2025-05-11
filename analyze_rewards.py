#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict
)

from train_reward_model import SegmentRewardModel

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def split_into_episodes(data):
    """Split data into episodes based on episode IDs.
    
    Args:
        data: TensorDict with observations, actions, and episode IDs
        
    Returns:
        List of dictionaries, each containing data for one episode
    """
    # Extract the observations and episode IDs
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    episode_ids = data["episode"]
    
    # Get unique episodes
    unique_episodes = torch.unique(episode_ids).tolist()
    print(f"Found {len(unique_episodes)} unique episodes")
    
    # Split data by episodes
    episodes = []
    for episode_id in unique_episodes:
        # Get indices for this episode
        ep_indices = torch.where(episode_ids == episode_id)[0]
        
        # Skip empty episodes
        if len(ep_indices) == 0:
            continue
        
        # Get data for this episode
        ep_obs = observations[ep_indices]
        ep_actions = actions[ep_indices[:-1]] if len(ep_indices) > 1 else actions[ep_indices]
        
        # Get rewards if available
        ep_rewards = None
        if "reward" in data:
            ep_rewards = data["reward"][ep_indices]
        
        # Create episode dictionary
        episode = {
            "id": episode_id,
            "obs": ep_obs,
            "action": ep_actions,
            "reward": ep_rewards,
            "length": len(ep_indices)
        }
        episodes.append(episode)
    
    # Sort episodes by length (descending)
    episodes.sort(key=lambda x: x["length"], reverse=True)
    
    return episodes

def predict_rewards(model, episodes, device, batch_size=32):
    """Predict rewards for each step in the episodes using the reward model.
    
    Args:
        model: Trained reward model
        episodes: List of episode dictionaries
        device: Device to run the model on
        batch_size: Batch size for prediction
        
    Returns:
        List of episode dictionaries with predicted rewards added
    """
    model.eval()
    
    # Process each episode
    for i, episode in enumerate(tqdm(episodes, desc="Predicting rewards")):
        obs = episode["obs"]
        actions = episode["action"]
        
        # Handle case where obs and actions have different lengths
        if len(obs) != len(actions) + 1:
            # If they're not off by 1, adjust to match
            min_len = min(len(obs), len(actions))
            obs = obs[:min_len]
            actions = actions[:min_len]
        
        # Ensure we have valid data
        if len(obs) == 0 or len(actions) == 0:
            episode["predicted_rewards"] = torch.tensor([])
            continue
        
        # Process in batches
        all_rewards = []
        
        with torch.no_grad():
            for start_idx in range(0, len(actions), batch_size):
                end_idx = min(start_idx + batch_size, len(actions))
                
                # Get batch data
                batch_obs = obs[start_idx:end_idx].to(device)
                batch_actions = actions[start_idx:end_idx].to(device)
                
                # Predict rewards
                batch_rewards = model.reward_model(batch_obs, batch_actions).cpu()
                all_rewards.append(batch_rewards)
        
        # Combine all rewards
        if all_rewards:
            episode["predicted_rewards"] = torch.cat(all_rewards) if len(all_rewards) > 1 else all_rewards[0]
        else:
            episode["predicted_rewards"] = torch.tensor([])
    
    return episodes

def plot_episode_rewards(episode, episode_idx, output_dir, ground_truth=True, smooth_window=5):
    """Plot the rewards for a single episode.
    
    Args:
        episode: Episode dictionary with rewards
        episode_idx: Index of the episode (for filename)
        output_dir: Directory to save the plot
        ground_truth: Whether to plot ground truth rewards if available
        smooth_window: Window size for smoothing the rewards
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Get predicted rewards
    steps = np.arange(len(episode["predicted_rewards"]))
    rewards = episode["predicted_rewards"].cpu().numpy()
    
    # Smooth rewards
    if smooth_window > 1 and len(rewards) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        rewards_smooth = np.convolve(rewards, kernel, mode='valid')
        steps_smooth = steps[smooth_window-1:]
    else:
        rewards_smooth = rewards
        steps_smooth = steps
    
    # Plot predicted rewards (not normalized)
    plt.plot(steps, rewards, 'b-', alpha=0.3, label='Predicted Reward')
    plt.plot(steps_smooth, rewards_smooth, 'b-', linewidth=2, label='Smoothed Predicted Reward')
    
    # Plot ground truth rewards if available
    if ground_truth and episode["reward"] is not None:
        gt_rewards = episode["reward"].cpu().numpy()
        
        # Replace NaN values with zeros
        gt_rewards = np.nan_to_num(gt_rewards, nan=0.0)
        
        # Make sure steps and gt_rewards have the same length
        gt_steps = np.arange(len(gt_rewards))
        
        # Normalize ground truth rewards to [0, 1]
        gt_min, gt_max = gt_rewards.min(), gt_rewards.max()
        if gt_max > gt_min:  # Avoid division by zero
            normalized_gt = (gt_rewards - gt_min) / (gt_max - gt_min)
        else:
            normalized_gt = np.zeros_like(gt_rewards)
        
        # Smooth ground truth rewards
        if smooth_window > 1 and len(normalized_gt) > smooth_window:
            gt_rewards_smooth = np.convolve(normalized_gt, kernel, mode='valid')
            gt_steps_smooth = gt_steps[smooth_window-1:]
            plt.plot(gt_steps, normalized_gt, 'g-', alpha=0.3, label='Ground Truth Reward (Normalized)')
            plt.plot(gt_steps_smooth, gt_rewards_smooth, 'g-', linewidth=2, label='Smoothed Ground Truth Reward')
        else:
            plt.plot(gt_steps, normalized_gt, 'g-', linewidth=2, label='Ground Truth Reward (Normalized)')
        
        # Add original range information to the title
        plt.title(f'Episode {episode["id"]} Rewards (Length: {episode["length"]})\n'
                 f'GT Range: [{gt_min:.3f}, {gt_max:.3f}]')
    else:
        plt.title(f'Episode {episode["id"]} Rewards (Length: {episode["length"]})')
    
    # Create twin axis for predicted rewards
    ax1 = plt.gca()
    ax1.set_ylabel('Predicted Reward')
    
    if ground_truth and episode["reward"] is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Ground Truth Reward [0, 1]', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
    # Add labels
    plt.xlabel('Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/episode_{episode_idx}_rewards.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_rewards(data_path, model_path, output_dir, num_episodes=5, device=None):
    """Analyze rewards for episodes in the dataset.
    
    Args:
        data_path: Path to the dataset
        model_path: Path to the trained reward model
        output_dir: Directory to save the plots
        num_episodes: Number of episodes to analyze
        device: Device to run the model on
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {data_path}")
    data = load_tensordict(data_path)
    
    # Get observation and action dimensions
    observations = data["obs"] if "obs" in data else data["state"]
    state_dim = observations.shape[1]
    action_dim = data["action"].shape[1]
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Split data into episodes
    print("Splitting data into episodes")
    episodes = split_into_episodes(data)
    print(f"Found {len(episodes)} episodes")
    
    # Load reward model
    print(f"Loading reward model from {model_path}")
    model = SegmentRewardModel(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Sample random episodes if we have more than requested
    if len(episodes) > num_episodes:
        print(f"Sampling {num_episodes} random episodes from {len(episodes)} total")
        sampled_episodes = random.sample(episodes, num_episodes)
    else:
        print(f"Using all {len(episodes)} episodes (less than requested {num_episodes})")
        sampled_episodes = episodes
    
    # Predict rewards for each episode
    print("Predicting rewards for episodes")
    sampled_episodes = predict_rewards(model, sampled_episodes, device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot rewards for each episode
    print("Plotting reward curves")
    for i, episode in enumerate(sampled_episodes):
        plot_episode_rewards(episode, i, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze rewards predicted by a reward model")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATHS[0], help="Path to the dataset")
    parser.add_argument("--model_path", type=str, default="/scr/aliang80/robot_pref/reward_model/state_action_reward_model.pt", help="Path to the trained reward model")
    parser.add_argument("--output_dir", type=str, default="reward_analysis", help="Directory to save the plots")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to analyze")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of CUDA")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cpu" if args.use_cpu else "cuda")
    
    # Run analysis
    analyze_rewards(args.data_path, args.model_path, args.output_dir, args.num_episodes, device)

if __name__ == "__main__":
    main() 