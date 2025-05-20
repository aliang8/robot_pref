#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random
import wandb
import seaborn as sns
from utils.seed import set_seed

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)  # camera-ready formatting + latex in plots


def predict_rewards(model, episodes):
    """Predict rewards for each step in the episodes using the reward model.

    Args:
        model: Trained reward model
        episodes: List of episode dictionaries

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
    rand_indices,
    output_file,
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
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),
        squeeze=False,
        sharex="col",
        sharey=True,
    )

    # Define smoothing kernel once
    kernel = np.ones(smooth_window) / smooth_window

    # Keep track of the line objects for the legend
    pred_line = None
    gt_line = None

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

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            pred_rewards = episode["predicted_rewards"].detach().cpu().numpy()
            steps = np.arange(len(pred_rewards))

            # Smooth predicted rewards
            if smooth_window > 1 and len(pred_rewards) > smooth_window:
                pred_rewards_smooth = np.convolve(pred_rewards, kernel, mode="valid")
                steps_smooth = steps[smooth_window - 1 :]
            else:
                pred_rewards_smooth = pred_rewards
                steps_smooth = steps

            # Plot predicted rewards (save the line object for legend)
            pred_line = ax.plot(
                steps_smooth,
                pred_rewards_smooth,
                "b-",
                linewidth=3,
                label="Predicted Rewards",
            )[0]  # Get the line object

            # Plot ground truth rewards if available
            gt_rewards = episode["reward"].detach().cpu().numpy()
            ep_len = len(gt_rewards)
            ep_id = rand_indices[idx]

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
                    f"Episode {ep_id}, GT Range: [{gt_rewards.min():.2f}, {gt_rewards.max():.2f}]",
                    fontsize=14,
                )
            else:
                # If min and max are the same or not provided, use tanh as fallback
                normalized_gt = np.tanh(gt_rewards)

                # Add range information to the title
                raw_min, raw_max = gt_rewards.mibn(), gt_rewards.max()
                ax.set_title(
                    f"Episode {ep_id}, GT Range: [{raw_min:.2f}, {raw_max:.2f}]",
                    fontsize=14,
                )

            # Plot on same axis with different color
            gt_line = ax.plot(
                steps, normalized_gt, "g--", linewidth=3, label="Ground Truth"
            )[0]  # Get the line object

            # Set grid
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.05, 1.05)  # Consistent y-range for all plots
            ax.tick_params(axis="both", labelsize=14)

    # First do a tight layout to get proper spacing
    plt.tight_layout()

    # Adjust layout with more space at bottom for labels and legend
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)

    fig.supxlabel("Environment Steps", fontsize=20, y=0.1)
    fig.supylabel("Per-step reward", fontsize=20, x=0.05)

    # Add a common legend at the bottom of the figure with two columns
    if pred_line and gt_line:
        fig.legend(
            [pred_line, gt_line],
            ["Predicted Rewards", "Ground Truth"],
            loc="lower center",
            ncol=2,
            fontsize=14,
            frameon=False,
            bbox_to_anchor=(0.5, 0.05),
        )

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_rewards(
    model,
    episodes,
    output_file=None,
    num_episodes=9,
    reward_min=None,
    reward_max=None,
    wandb_run=None,
    random_seed=None,
):
    """Analyze rewards for episodes in the dataset.

    Args:
        model: Trained reward model
        episodes: List of episode dictionaries
        output_file: Directory to save the plots. If None, uses the model directory.
        num_episodes: Number of episodes to analyze
        wandb_run: Wandb run to log
    """
    if random_seed is not None:
        set_seed(random_seed)

    print(f"Sampling {num_episodes} random episodes from {len(episodes)} total")
    rand_indices = random.sample(range(len(episodes)), num_episodes)
    sampled_episodes = [episodes[i] for i in rand_indices]

    # Predict rewards for each episode
    print("Predicting rewards for episodes")
    sampled_episodes = predict_rewards(model, sampled_episodes)

    # Plot rewards in a grid
    print("Creating reward grid plot")
    plot_reward_grid(
        sampled_episodes,
        rand_indices,
        output_file,
        grid_size=(3, 3),
        reward_min=reward_min,
        reward_max=reward_max,
    )

    print(f"Analysis complete. Results saved to {output_file}")

    if wandb_run:
        wandb_run.log({"reward_grid": wandb.Image(output_file)})
