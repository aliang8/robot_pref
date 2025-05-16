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
from utils.trajectory import DEFAULT_DATA_PATHS, load_tensordict
from utils.seed import set_seed
from models.reward_models import RewardModel
from utils.data import process_data_trajectories


def predict_rewards(model, episodes):
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

        # Predict rewards
        predicted_rewards = model(obs, actions)
        episode["predicted_rewards"] = predicted_rewards

    return episodes


def plot_reward_grid(
    episodes,
    output_dir,
    grid_size=(3, 3),
    smooth_window=5,
    reward_min=None,
    reward_max=None,
):
    """Plot a grid of reward curves for multiple episodes.

    Args:
        episodes: List of episode dictionaries with rewards
        output_dir: Directory to save the plot
        grid_size: Tuple of (rows, cols) for the grid layout
        smooth_window: Window size for smoothing the rewards
        reward_min: Global minimum reward for normalization
        reward_max: Global maximum reward for normalization
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    # Define smoothing kernel once
    kernel = np.ones(smooth_window) / smooth_window

    # Process each episode in the grid
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j

            # Skip if we've run out of episodes
            if idx >= len(episodes):
                axes[i, j].set_visible(False)
                continue

            episode = episodes[idx]
            ax = axes[i, j]

            # Skip if no predicted rewards
            if len(episode["predicted_rewards"]) == 0:
                ax.set_visible(False)
                continue

            pred_rewards = episode["predicted_rewards"].numpy()
            steps = np.arange(len(pred_rewards))

            # Smooth predicted rewards
            if smooth_window > 1 and len(pred_rewards) > smooth_window:
                pred_rewards_smooth = np.convolve(pred_rewards, kernel, mode="valid")
                steps_smooth = steps[smooth_window - 1 :]
            else:
                pred_rewards_smooth = pred_rewards
                steps_smooth = steps

            # Plot predicted rewards
            ax.plot(steps, pred_rewards, "b-", alpha=0.3, label="Predicted")
            ax.plot(
                steps_smooth,
                pred_rewards_smooth,
                "b-",
                linewidth=2,
                label="Pred (Smoothed)",
            )

            # Plot ground truth rewards if available
            if episode["reward"] is not None:
                gt_rewards = episode["reward"].numpy()

                # Replace NaN values with zeros
                gt_rewards = np.nan_to_num(gt_rewards, nan=0.0)

                # Bound GT rewards to [-1, 1] with min-max scaling
                if (
                    reward_min is not None
                    and reward_max is not None
                    and reward_min != reward_max
                ):
                    # Scale to [-1, 1] range
                    normalized_gt = (
                        2 * (gt_rewards - reward_min) / (reward_max - reward_min) - 1
                    )

                    # Add range information to the title
                    ax.set_title(
                        f"Episode {episode['id']} (L: {episode['length']})\nGT Range: [{gt_rewards.min():.2f}, {gt_rewards.max():.2f}]",
                        fontsize=10,
                    )
                else:
                    # If min and max are the same or not provided, use tanh as fallback
                    normalized_gt = np.tanh(gt_rewards)

                    # Add range information to the title
                    raw_min, raw_max = gt_rewards.min(), gt_rewards.max()
                    ax.set_title(
                        f"Episode {episode['id']} (L: {episode['length']})\nGT Range: [{raw_min:.2f}, {raw_max:.2f}]",
                        fontsize=10,
                    )

                # Smooth ground truth rewards
                if smooth_window > 1 and len(normalized_gt) > smooth_window:
                    gt_rewards_smooth = np.convolve(normalized_gt, kernel, mode="valid")
                    gt_steps_smooth = steps[smooth_window - 1 :]

                    # Plot on same axis with different color
                    ax.plot(
                        steps, normalized_gt, "g--", alpha=0.3, label="Ground Truth"
                    )
                    ax.plot(
                        gt_steps_smooth,
                        gt_rewards_smooth,
                        "g--",
                        linewidth=2,
                        label="GT (Smoothed)",
                    )
                else:
                    # Plot on same axis with different color
                    ax.plot(
                        steps, normalized_gt, "g--", linewidth=2, label="Ground Truth"
                    )
            else:
                ax.set_title(
                    f"Episode {episode['id']} (L: {episode['length']})", fontsize=10
                )

            # Set labels and grid
            ax.set_xlabel("Step", fontsize=9)
            ax.set_ylabel("Reward [-1, 1]", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.05, 1.05)  # Consistent y-range for all plots

            # Add legend with smaller font
            if idx == 0:  # Only add legend to the first plot
                ax.legend(loc="best", fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/reward_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


def analyze_rewards(
    data_path, model_path, output_dir=None, num_episodes=9, device=None, random_seed=42
):
    """Analyze rewards for episodes in the dataset.

    Args:
        data_path: Path to the dataset
        model_path: Path to the trained reward model
        output_dir: Directory to save the plots. If None, uses the model directory.
        num_episodes: Number of episodes to analyze
        device: Device to run the model on
        random_seed: Random seed for reproducibility
    """
    set_seed(random_seed)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Using random seed: {random_seed}")

    # If output_dir is not provided, use the directory of the model path
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
        print(f"Using model directory for output: {output_dir}")

    # Load data
    print(f"Loading data from {data_path}")
    data = load_tensordict(data_path)

    # Get observation and action dimensions
    observations = data["obs"] if "obs" in data else data["state"]
    state_dim = observations.shape[1]
    action_dim = data["action"].shape[1]
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")

    # Compute global reward min/max for normalization
    reward_min = None
    reward_max = None
    if "reward" in data:
        rewards = data["reward"]
        # Filter out NaN values
        valid_rewards = rewards[~torch.isnan(rewards)]
        if len(valid_rewards) > 0:
            reward_min = valid_rewards.min().item()
            reward_max = valid_rewards.max().item()
            print(f"Global reward range: [{reward_min:.4f}, {reward_max:.4f}]")
        else:
            print("Warning: No valid rewards found in dataset")

    # Split data into episodes
    print("Splitting data into episodes")
    episodes = process_data_trajectories(data_path)
    print(f"Found {len(episodes)} episodes")

    # Load reward model
    print(f"Loading reward model from {model_path}")
    model = RewardModel(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    print(f"Sampling {num_episodes} random episodes from {len(episodes)} total")
    sampled_episodes = random.sample(episodes, num_episodes)

    # Predict rewards for each episode
    print("Predicting rewards for episodes")
    sampled_episodes = predict_rewards(model, sampled_episodes, device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot rewards in a grid
    print("Creating reward grid plot")
    plot_reward_grid(
        sampled_episodes,
        output_dir,
        grid_size=(3, 3),
        reward_min=reward_min,
        reward_max=reward_max,
    )

    print(f"Analysis complete. Results saved to {output_dir}")