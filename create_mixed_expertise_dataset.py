#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
from trajectory_utils import DEFAULT_DATA_PATHS, load_tensordict


def list_available_datasets():
    """List all available datasets from DEFAULT_DATA_PATHS."""
    print("\nAvailable datasets:")
    for i, path in enumerate(DEFAULT_DATA_PATHS):
        dataset_name = Path(path).stem
        print(f"  [{i}] {dataset_name}: {path}")
    print()


def extract_episode_returns(data):
    """Extract return (sum of rewards) for each episode.

    Args:
        data: TensorDict containing "episode" and "reward" keys

    Returns:
        dict: Dictionary with episode returns information
    """
    # Get episode IDs and rewards
    episode_ids = data["episode"].cpu().numpy()
    rewards = data["reward"].cpu().numpy()

    # Find unique episodes
    unique_episodes = np.unique(episode_ids)
    num_episodes = len(unique_episodes)

    print(f"Extracting returns for {num_episodes} episodes...")

    # Calculate return for each episode
    episode_returns = []
    episode_lengths = []
    episode_indices = []
    valid_episode_ids = []

    for ep_id in tqdm(unique_episodes):
        # Get indices for this episode
        indices = np.where(episode_ids == ep_id)[0]
        ep_rewards = rewards[indices]

        # Skip empty episodes
        if len(indices) == 0:
            continue

        # Handle NaN values in rewards
        if np.isnan(ep_rewards).all():
            # Skip episodes with all NaN rewards
            continue
        elif np.isnan(ep_rewards).any():
            # Replace NaN rewards with 0
            ep_rewards = np.nan_to_num(ep_rewards, nan=0.0)

        # Calculate episode return
        episode_return = np.sum(ep_rewards)

        episode_returns.append(episode_return)
        episode_lengths.append(len(indices))
        episode_indices.append(indices)
        valid_episode_ids.append(ep_id)

    # Create a dictionary with episode information
    episode_data = {
        "episode_ids": np.array(valid_episode_ids),
        "returns": np.array(episode_returns),
        "lengths": np.array(episode_lengths),
        "indices": episode_indices,
    }

    print(f"Extracted returns for {len(episode_returns)} episodes")
    print(f"Mean return: {np.mean(episode_returns):.2f}")
    print(f"Min return: {np.min(episode_returns):.2f}")
    print(f"Max return: {np.max(episode_returns):.2f}")

    return episode_data


def check_data_for_nans(data):
    """Check a tensordict for NaN values.

    Args:
        data: TensorDict containing trajectory data

    Returns:
        dict: Dictionary with NaN statistics for each key
    """
    print("\nChecking for NaN values in data...")
    nan_stats = {}

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            nan_count = torch.isnan(value).sum().item()
            total_elements = value.numel()
            nan_percentage = (
                100 * nan_count / total_elements if total_elements > 0 else 0
            )

            nan_stats[key] = {
                "nan_count": nan_count,
                "total_elements": total_elements,
                "nan_percentage": nan_percentage,
            }

            if nan_count > 0:
                print(
                    f"  Field '{key}' contains {nan_count}/{total_elements} ({nan_percentage:.2f}%) NaN values"
                )

    # Return NaN statistics
    return nan_stats


def plot_returns_histogram(returns, output_path, bins=50, title=None):
    """Plot a histogram of episode returns."""
    plt.figure(figsize=(10, 6))

    # Create histogram
    n, bins, patches = plt.hist(returns, bins=bins, alpha=0.7, color="blue")

    # Add mean line
    mean_return = np.mean(returns)
    plt.axvline(
        mean_return,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {mean_return:.2f}",
    )

    # Add median line
    median_return = np.median(returns)
    plt.axvline(
        median_return,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Median: {median_return:.2f}",
    )

    # Add percentile lines (25% and 75%)
    pct_25 = np.percentile(returns, 25)
    pct_75 = np.percentile(returns, 75)
    plt.axvline(
        pct_25,
        color="orange",
        linestyle="dotted",
        linewidth=1,
        label=f"25th percentile: {pct_25:.2f}",
    )
    plt.axvline(
        pct_75,
        color="purple",
        linestyle="dotted",
        linewidth=1,
        label=f"75th percentile: {pct_75:.2f}",
    )

    # Set labels and title
    plt.xlabel("Episode Return")
    plt.ylabel("Frequency")
    if title:
        plt.title(title)
    else:
        plt.title("Distribution of Episode Returns")

    plt.legend()
    plt.grid(alpha=0.3)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved histogram to {output_path}")

    plt.close()


def classify_episodes_by_return(returns, num_bins=3):
    """Classify episodes into skill levels based on their returns.

    Args:
        returns: Array of episode returns
        num_bins: Number of bins for classification (default: 3 for random/medium/expert)

    Returns:
        ndarray: Array of skill level labels (0=random, 1=medium, 2=expert)
    """
    min_return = np.min(returns)
    max_return = np.max(returns)

    # Create equally spaced bin edges
    bin_edges = np.linspace(min_return, max_return, num_bins + 1)

    # Classify episodes into bins
    skill_levels = np.zeros(len(returns), dtype=int)

    for i in range(len(returns)):
        # Find the bin this return falls into
        bin_idx = np.digitize(returns[i], bin_edges) - 1

        # Ensure bin_idx is within valid range
        bin_idx = min(bin_idx, num_bins - 1)
        bin_idx = max(bin_idx, 0)

        skill_levels[i] = bin_idx

    # Count episodes per skill level
    for level in range(num_bins):
        count = np.sum(skill_levels == level)
        percentage = 100 * count / len(returns)

        level_name = (
            ["random", "medium", "expert"][level] if num_bins == 3 else f"level_{level}"
        )
        print(f"  {level_name}: {count} episodes ({percentage:.1f}%)")

    return skill_levels


def balance_episodes(episode_data, seed=42):
    """Balance episodes by taking equal numbers from each skill level.

    Args:
        episode_data: Dictionary with episode information including skill_levels
        seed: Random seed for reproducibility

    Returns:
        list: Indices of selected episodes for a balanced dataset
    """
    # Get skill levels
    skill_levels = episode_data["skill_levels"]
    unique_levels = np.unique(skill_levels)

    # Count episodes per skill level
    counts = []
    for level in unique_levels:
        count = np.sum(skill_levels == level)
        counts.append(count)

    # Determine how many episodes to sample from each skill level
    min_count = min(counts)
    print(f"\nBalancing dataset: taking {min_count} episodes from each skill level")

    # Set random seed for reproducibility
    random.seed(seed)

    # Sample episodes from each skill level
    selected_indices = []

    for level in unique_levels:
        # Get indices of episodes with this skill level
        level_indices = np.where(skill_levels == level)[0]

        # Randomly sample from this skill level
        sampled_indices = random.sample(list(level_indices), min_count)
        selected_indices.extend(sampled_indices)

    print(f"Selected {len(selected_indices)} episodes for balanced dataset")
    return selected_indices


def create_balanced_dataset(data, episode_data, selected_indices, output_path):
    """Create a new balanced dataset with the selected episodes.

    Args:
        data: Original TensorDict with trajectory data
        episode_data: Dictionary with episode information
        selected_indices: Indices of selected episodes (into episode_data)
        output_path: Path to save the new dataset
    """
    print("\nCreating balanced dataset...")

    # Get original episode IDs for selected episodes
    selected_episode_ids = episode_data["episode_ids"][selected_indices]

    # Create a mask for the original data
    episode_ids = data["episode"].cpu().numpy()
    keep_mask = np.zeros_like(episode_ids, dtype=bool)

    # Mark the selected episodes in the mask
    for ep_id in tqdm(selected_episode_ids, desc="Creating selection mask"):
        keep_mask |= episode_ids == ep_id

    # Create a new TensorDict with only the selected episodes
    new_data = {}
    total_kept = np.sum(keep_mask)
    print(f"Keeping {total_kept} transitions out of {len(episode_ids)}")

    # Check for NaN values in the selected data
    has_nans = False

    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            # Keep only the selected indices
            selected_tensor = data[key][keep_mask]

            # Check for NaN values
            nan_count = torch.isnan(selected_tensor).sum().item()
            if nan_count > 0:
                has_nans = True
                print(
                    f"  Warning: {key} has {nan_count}/{selected_tensor.numel()} NaN values. Replacing with zeros."
                )
                selected_tensor = torch.nan_to_num(selected_tensor, nan=0.0)

            new_data[key] = selected_tensor
            print(f"  {key}: {new_data[key].shape}")

    if has_nans:
        print("Warning: NaN values were found and replaced with zeros.")

    # Save the new dataset
    print(f"Saving balanced dataset to {output_path}_balanced")
    torch.save(new_data, output_path)

    # Plot the distribution of returns in the balanced dataset
    balanced_returns = []
    balanced_episode_ids = new_data["episode"].cpu().numpy()
    balanced_rewards = new_data["reward"].cpu().numpy()

    unique_balanced_episodes = np.unique(balanced_episode_ids)
    for ep_id in unique_balanced_episodes:
        indices = np.where(balanced_episode_ids == ep_id)[0]
        ep_rewards = balanced_rewards[indices]
        ep_rewards = np.nan_to_num(ep_rewards, nan=0.0)
        balanced_returns.append(np.sum(ep_rewards))

    # Plot histogram of balanced returns
    balanced_histogram_path = os.path.join(
        os.path.dirname(output_path), f"{Path(output_path).stem}_returns_histogram.png"
    )
    plot_returns_histogram(
        np.array(balanced_returns),
        balanced_histogram_path,
        title=f"Return Distribution: {Path(output_path).stem}",
    )

    return {
        "num_episodes": len(selected_episode_ids),
        "num_transitions": total_kept,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced dataset with equal amounts of random/medium/expert episodes."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to trajectory data file (if not specified, will list and use the first available dataset)",
    )
    parser.add_argument(
        "--dataset_idx",
        type=int,
        default=None,
        help="Index of dataset from DEFAULT_DATA_PATHS to use (alternative to --data_path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="balanced_datasets",
        help="Directory to save balanced dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for selecting episodes"
    )

    args = parser.parse_args()

    # List available datasets
    list_available_datasets()

    # If no data path or index is specified, use the first dataset
    if args.data_path is None and args.dataset_idx is None:
        args.dataset_idx = 0
        print(
            f"No dataset specified. Using first dataset (index 0): {DEFAULT_DATA_PATHS[0]}"
        )

    # Get data path from index if specified
    if args.dataset_idx is not None:
        if args.dataset_idx < 0 or args.dataset_idx >= len(DEFAULT_DATA_PATHS):
            print(f"Error: Dataset index {args.dataset_idx} is out of range")
            return
        args.data_path = DEFAULT_DATA_PATHS[args.dataset_idx]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_tensordict(args.data_path)

    # Check for NaN values
    check_data_for_nans(data)

    # Extract episode returns
    episode_data = extract_episode_returns(data)

    # Create dataset name for output files
    dataset_name = Path(args.data_path).stem

    # Plot histogram of returns
    histogram_path = os.path.join(
        args.output_dir, f"{dataset_name}_returns_histogram.png"
    )
    plot_returns_histogram(
        episode_data["returns"],
        histogram_path,
        title=f"Return Distribution: {dataset_name}",
    )

    # Classify episodes by return
    print("\nClassifying episodes by return...")
    skill_levels = classify_episodes_by_return(episode_data["returns"], num_bins=3)
    episode_data["skill_levels"] = skill_levels

    # Balance episodes across skill levels
    selected_indices = balance_episodes(episode_data, seed=args.seed)

    # Create and save the balanced dataset
    output_path = os.path.join(args.output_dir, f"{dataset_name}_balanced.pt")
    create_balanced_dataset(data, episode_data, selected_indices, output_path)

    print("\nBalanced dataset creation complete!")


if __name__ == "__main__":
    main()
