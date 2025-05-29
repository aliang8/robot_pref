#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
from pathlib import Path
import random
from tqdm import tqdm
from utils.data import load_tensordict


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


def combine_datasets(data1, data2, num_episodes1=None, num_episodes2=None, seed=42):
    """Combine two datasets by randomly sampling episodes from each.

    Args:
        data1: First dataset (TensorDict)
        data2: Second dataset (TensorDict)
        num_episodes1: Number of episodes to sample from first dataset (default: all)
        num_episodes2: Number of episodes to sample from second dataset (default: all)
        seed: Random seed for reproducibility

    Returns:
        TensorDict: Combined dataset
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Extract episode information for both datasets
    print("\nProcessing first dataset...")
    episode_data1 = extract_episode_returns(data1)
    print("\nProcessing second dataset...")
    episode_data2 = extract_episode_returns(data2)

    # Determine number of episodes to sample
    if num_episodes1 is None:
        num_episodes1 = len(episode_data1["episode_ids"])
    if num_episodes2 is None:
        num_episodes2 = len(episode_data2["episode_ids"])

    # Randomly sample episodes
    indices1 = random.sample(range(len(episode_data1["episode_ids"])), num_episodes1)
    indices2 = random.sample(range(len(episode_data2["episode_ids"])), num_episodes2)

    # Get selected episode IDs
    selected_episodes1 = episode_data1["episode_ids"][indices1]
    selected_episodes2 = episode_data2["episode_ids"][indices2]

    # Create masks for selected episodes
    mask1 = np.zeros(len(data1["episode"]), dtype=bool)
    mask2 = np.zeros(len(data2["episode"]), dtype=bool)

    for ep_id in tqdm(selected_episodes1, desc="Creating mask for dataset 1"):
        mask1 |= data1["episode"].cpu().numpy() == ep_id

    for ep_id in tqdm(selected_episodes2, desc="Creating mask for dataset 2"):
        mask2 |= data2["episode"].cpu().numpy() == ep_id

    # Create combined dataset
    combined_data = {}
    for key in data1.keys():
        if key not in data2.keys():
            print(f"Warning: Key {key} not found in second dataset, skipping")
            continue

        # Combine data from both datasets
        data1_selected = data1[key][mask1]
        data2_selected = data2[key][mask2]

        # For episode IDs, we need to offset the second dataset's IDs
        if key == "episode":
            max_ep_id = data1_selected.max().item()
            data2_selected = data2_selected + max_ep_id + 1

        # Concatenate the data
        if isinstance(data1_selected, torch.Tensor):
            combined_data[key] = torch.cat([data1_selected, data2_selected])
        else:
            combined_data[key] = np.concatenate([data1_selected, data2_selected])

    print("\nCombined dataset statistics:")
    for key, value in combined_data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: shape {value.shape}")

    return combined_data


def main():
    parser = argparse.ArgumentParser(
        description="Combine two datasets by randomly sampling episodes from each."
    )
    parser.add_argument(
        "--data_path1",
        type=str,
        required=True,
        help="Path to first dataset file",
    )
    parser.add_argument(
        "--data_path2",
        type=str,
        required=True,
        help="Path to second dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="combined_datasets",
        help="Directory to save combined dataset",
    )
    parser.add_argument(
        "--num_episodes1",
        type=int,
        default=None,
        help="Number of episodes to sample from first dataset (default: all)",
    )
    parser.add_argument(
        "--num_episodes2",
        type=int,
        default=None,
        help="Number of episodes to sample from second dataset (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting episodes",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="combined",
        help="Suffix to add to output filename",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print(f"\nLoading first dataset from {args.data_path1}")
    data1 = load_tensordict(args.data_path1)
    print(f"\nLoading second dataset from {args.data_path2}")
    data2 = load_tensordict(args.data_path2)

    # Combine datasets
    combined_data = combine_datasets(
        data1,
        data2,
        num_episodes1=args.num_episodes1,
        num_episodes2=args.num_episodes2,
        seed=args.seed,
    )

    # Create output filename
    dataset1_name = Path(args.data_path1).stem
    dataset2_name = Path(args.data_path2).stem
    output_path = os.path.join(
        args.output_dir,
        f"{dataset1_name}_{dataset2_name}_{args.output_suffix}.pt",
    )

    # Save combined dataset
    print(f"\nSaving combined dataset to {output_path}")
    torch.save(combined_data, output_path)

    print("\nDataset combination complete!")


if __name__ == "__main__":
    main() 