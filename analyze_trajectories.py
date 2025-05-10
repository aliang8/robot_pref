#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
from collections import defaultdict

# Import DEFAULT_DATA_PATHS from trajectory_utils
try:
    from trajectory_utils import DEFAULT_DATA_PATHS, load_tensordict as load_from_utils
    USE_IMPORTED_LOAD = True
except ImportError:
    # Default paths in case trajectory_utils is not available
    DEFAULT_DATA_PATHS = [
        "/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt",
        "/scr/shared/clam/datasets/metaworld/bin-picking-v2/buffer_bin-picking-v2.pt",
        "/scr/shared/clam/datasets/metaworld/peg-insert-side-v2/buffer_peg-insert-side-v2.pt",
    ]
    USE_IMPORTED_LOAD = False

def list_available_datasets():
    """List all available datasets from DEFAULT_DATA_PATHS."""
    print("\nAvailable datasets:")
    for i, path in enumerate(DEFAULT_DATA_PATHS):
        dataset_name = Path(path).stem
        print(f"  [{i}] {dataset_name} - {path}")
    print()
    return DEFAULT_DATA_PATHS

def load_tensordict(file_path):
    """Load data from a .pt file."""
    print(f"Loading data from {file_path}")
    if USE_IMPORTED_LOAD:
        try:
            return load_from_utils(file_path)
        except Exception as e:
            print(f"Error using imported load_tensordict: {e}")
            # Fall back to standard loading
    
    try:
        data = torch.load(file_path)
        print(f"Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_episode_returns(data):
    """Extract returns for each episode from the TensorDict."""
    print("Extracting episode returns...")
    
    # Get episode IDs and rewards
    episode_ids = data["episode"].cpu().numpy()
    rewards = data["reward"].cpu().numpy()
    
    # Find episode boundaries
    unique_episodes = np.unique(episode_ids)
    n_episodes = len(unique_episodes)
    print(f"Found {n_episodes} unique episodes")
    
    # Calculate cumulative rewards for each episode
    episode_returns = []
    episode_lengths = []
    episode_start_indices = []
    episode_end_indices = []
    valid_episodes = []
    
    for episode_id in tqdm(unique_episodes, desc="Calculating episode returns"):
        # Get indices for this episode
        indices = np.where(episode_ids == episode_id)[0]
        episode_rewards = rewards[indices]
        
        # Skip episodes with NaN rewards
        if np.isnan(episode_rewards).any():
            nan_count = np.sum(np.isnan(episode_rewards))
            if nan_count == len(episode_rewards):
                # All rewards are NaN, skip this episode
                continue
            else:
                # Some rewards are NaN, fill with zeros for now
                episode_rewards = np.nan_to_num(episode_rewards, nan=0.0)
                print(f"Warning: Episode {episode_id} has {nan_count}/{len(episode_rewards)} NaN rewards. Replaced with zeros.")
        
        # Calculate return (sum of rewards)
        episode_return = np.sum(episode_rewards)
        
        # Store results
        episode_returns.append(episode_return)
        episode_lengths.append(len(indices))
        episode_start_indices.append(indices[0])
        episode_end_indices.append(indices[-1])
        valid_episodes.append(episode_id)
    
    # Check if we have any valid episodes
    if len(valid_episodes) == 0:
        raise ValueError("No valid episodes found after NaN filtering. Check your data.")
    
    print(f"Kept {len(valid_episodes)}/{n_episodes} episodes after NaN filtering")
    
    return {
        'episode_ids': np.array(valid_episodes),
        'returns': np.array(episode_returns),
        'lengths': np.array(episode_lengths),
        'start_indices': np.array(episode_start_indices),
        'end_indices': np.array(episode_end_indices)
    }

def check_data_for_nans(data):
    """Check data for NaN values and print summary statistics."""
    print("Checking data for NaN values...")
    
    nan_counts = {}
    total_elements = {}
    
    # Check each tensor in the data
    for key, tensor in data.items():
        if isinstance(tensor, torch.Tensor):
            # Move to CPU for processing
            cpu_tensor = tensor.cpu()
            
            # Count NaN values
            nan_count = torch.isnan(cpu_tensor).sum().item()
            total = cpu_tensor.numel()
            
            # Store counts
            nan_counts[key] = nan_count
            total_elements[key] = total
            
            # Print summary
            if nan_count > 0:
                print(f"  {key}: {nan_count}/{total} NaN values ({nan_count/total*100:.2f}%)")
    
    # Print overall summary
    total_nans = sum(nan_counts.values())
    total_elements_all = sum(total_elements.values())
    
    if total_nans > 0:
        print(f"Total NaN values: {total_nans}/{total_elements_all} ({total_nans/total_elements_all*100:.2f}%)")
    else:
        print("No NaN values found in the data.")
    
    return nan_counts, total_elements

def plot_returns_histogram(returns, output_path, bins=50, title=None):
    """Plot a histogram of episode returns."""
    print(f"Creating returns histogram with {bins} bins...")
    
    plt.figure(figsize=(12, 6))
    counts, edges, bars = plt.hist(returns, bins=bins, alpha=0.7)
    
    # Calculate statistics
    mean_return = np.mean(returns)
    median_return = np.median(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    
    # Add lines for mean and median
    plt.axvline(mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.2f}')
    plt.axvline(median_return, color='g', linestyle='-.', label=f'Median: {median_return:.2f}')
    
    # Add statistics text box
    stats_text = f"""
    Mean: {mean_return:.2f}
    Median: {median_return:.2f}
    Std Dev: {std_return:.2f}
    Min: {min_return:.2f}
    Max: {max_return:.2f}
    Total Episodes: {len(returns)}
    """
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    plt.xlabel('Episode Return')
    plt.ylabel('Count')
    if title is None:
        plt.title('Distribution of Episode Returns')
    else:
        plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Histogram saved to {output_path}")
    
    return counts, edges

def classify_episodes_by_return(returns, num_bins=3):
    """Classify episodes into skill levels based on returns."""
    print(f"Classifying episodes into {num_bins} skill levels...")
    
    # Create bins based on return values
    min_return = np.min(returns)
    max_return = np.max(returns)
    bin_edges = np.linspace(min_return, max_return, num_bins + 1)
    
    # Classify episodes
    episode_classes = np.digitize(returns, bin_edges[1:])  # 0 to num_bins-1
    
    # Count episodes in each class
    class_counts = np.bincount(episode_classes, minlength=num_bins)
    
    # Print classification results
    print("Episode classification:")
    for i in range(num_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1] if i < num_bins-1 else float('inf')
        class_name = ['random', 'medium', 'expert'][i] if num_bins == 3 else f"class_{i}"
        print(f"  {class_name}: {class_counts[i]} episodes (return range: {bin_start:.2f} - {bin_end:.2f})")
    
    return episode_classes, bin_edges

def balance_episodes(episode_data, max_per_class=None, seed=42):
    """Balance episodes by taking equal numbers from each class."""
    print("Balancing episodes across skill classes...")
    
    random.seed(seed)
    episode_classes = episode_data['classes']
    unique_classes = np.unique(episode_classes)
    
    # Count episodes in each class
    class_counts = {}
    for cls in unique_classes:
        class_counts[cls] = np.sum(episode_classes == cls)
    
    # Determine how many episodes to take from each class
    if max_per_class is None:
        # Use the minimum count across classes
        max_per_class = min(class_counts.values())
    
    # Select episodes from each class
    selected_indices = []
    
    for cls in unique_classes:
        # Get indices of episodes in this class
        class_indices = np.where(episode_classes == cls)[0]
        
        # Randomly select episodes if we have more than needed
        if len(class_indices) > max_per_class:
            selected = random.sample(list(class_indices), max_per_class)
        else:
            selected = class_indices
        
        selected_indices.extend(selected)
    
    # Return selected episode indices
    print(f"Selected {len(selected_indices)} episodes after balancing")
    return selected_indices

def filter_episodes_by_return(episode_data, min_return=None, max_return=None, percentile_range=None):
    """Filter episodes based on return value criteria."""
    returns = episode_data['returns']
    
    # Initialize masks
    keep_mask = np.ones_like(returns, dtype=bool)
    
    # Apply minimum return filter
    if min_return is not None:
        keep_mask &= (returns >= min_return)
    
    # Apply maximum return filter
    if max_return is not None:
        keep_mask &= (returns <= max_return)
    
    # Apply percentile range filter
    if percentile_range is not None:
        min_percentile, max_percentile = percentile_range
        min_val = np.percentile(returns, min_percentile)
        max_val = np.percentile(returns, max_percentile)
        keep_mask &= (returns >= min_val) & (returns <= max_val)
        print(f"Percentile range {min_percentile}% - {max_percentile}% corresponds to returns: {min_val:.2f} - {max_val:.2f}")
    
    # Get indices of episodes to keep
    kept_indices = np.where(keep_mask)[0]
    print(f"Kept {len(kept_indices)} episodes after filtering")
    
    return kept_indices

def create_balanced_dataset(data, episode_data, selected_indices, output_path):
    """Create a new dataset with only the selected episodes."""
    print(f"Creating balanced dataset with {len(selected_indices)} episodes...")
    
    # Get original episode IDs for selected episodes
    selected_episode_ids = episode_data['episode_ids'][selected_indices]
    
    # Create a mask for the original data
    episode_ids = data["episode"].cpu().numpy()
    keep_mask = np.zeros_like(episode_ids, dtype=bool)
    
    # Mark the selected episodes in the mask
    for ep_id in tqdm(selected_episode_ids, desc="Creating selection mask"):
        keep_mask |= (episode_ids == ep_id)
    
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
                print(f"  Warning: {key} has {nan_count}/{selected_tensor.numel()} NaN values. Replacing with zeros.")
                selected_tensor = torch.nan_to_num(selected_tensor, nan=0.0)
            
            new_data[key] = selected_tensor
            print(f"  {key}: {new_data[key].shape}")
    
    if has_nans:
        print("Warning: The dataset contains NaN values that have been replaced with zeros.")
    
    # Save the new dataset
    print(f"Saving balanced dataset to {output_path}")
    torch.save(new_data, output_path)
    
    # Also save the episode statistics
    stats_file = output_path.replace('.pt', '_stats.npz')
    selected_returns = episode_data['returns'][selected_indices]
    selected_lengths = episode_data['lengths'][selected_indices]
    
    np.savez(
        stats_file,
        episode_ids=selected_episode_ids,
        returns=selected_returns,
        lengths=selected_lengths,
        indices=selected_indices
    )
    print(f"Saved episode statistics to {stats_file}")
    
    return new_data

def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory returns and create balanced datasets")
    parser.add_argument("--data_path", type=str, help="Path to the PT file containing trajectory data")
    parser.add_argument("--dataset_index", type=int, help="Index of the dataset in DEFAULT_DATA_PATHS (alternative to --data_path)")
    parser.add_argument("--list_datasets", action="store_true", help="List available datasets and exit")
    parser.add_argument("--output_dir", type=str, default="./trajectory_analysis", help="Directory to save output files")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for the returns histogram")
    parser.add_argument("--skill_levels", type=int, default=3, help="Number of skill levels to classify episodes (default: 3 for random/medium/expert)")
    parser.add_argument("--balance", action="store_true", help="Balance episodes across skill levels")
    parser.add_argument("--max_per_class", type=int, default=None, help="Maximum number of episodes to keep per skill class")
    parser.add_argument("--filter_min_return", type=float, default=None, help="Minimum return to keep an episode")
    parser.add_argument("--filter_max_return", type=float, default=None, help="Maximum return to keep an episode")
    parser.add_argument("--filter_percentile", type=str, default=None, help="Percentile range to keep (e.g., '10-90' for 10th to 90th percentile)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # List datasets if requested
    if args.list_datasets:
        list_available_datasets()
        return
    
    # Determine which dataset to use
    data_path = None
    if args.data_path:
        data_path = args.data_path
    elif args.dataset_index is not None:
        if 0 <= args.dataset_index < len(DEFAULT_DATA_PATHS):
            data_path = DEFAULT_DATA_PATHS[args.dataset_index]
        else:
            print(f"Error: Dataset index {args.dataset_index} is out of range.")
            list_available_datasets()
            return
    else:
        # If no data path or index provided, list available datasets and use the first one
        print("No dataset specified. Available datasets:")
        list_available_datasets()
        data_path = DEFAULT_DATA_PATHS[0]
        print(f"Using default dataset: {Path(data_path).stem}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_tensordict(data_path)
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Get dataset name from file path
    dataset_name = Path(data_path).stem
    
    # Extract episode returns
    episode_data = extract_episode_returns(data)
    
    # Plot returns histogram
    hist_path = os.path.join(args.output_dir, f"{dataset_name}_returns_histogram.png")
    counts, edges = plot_returns_histogram(
        episode_data['returns'], 
        hist_path, 
        bins=args.bins,
        title=f"Distribution of Episode Returns for {dataset_name}"
    )
    
    # Classify episodes by return
    classes, bin_edges = classify_episodes_by_return(
        episode_data['returns'], 
        num_bins=args.skill_levels
    )
    episode_data['classes'] = classes
    episode_data['bin_edges'] = bin_edges
    
    # Initialize selected indices
    selected_indices = None
    
    # Apply filtering if requested
    if args.filter_min_return is not None or args.filter_max_return is not None or args.filter_percentile is not None:
        # Parse percentile range if provided
        percentile_range = None
        if args.filter_percentile is not None:
            try:
                min_pct, max_pct = map(float, args.filter_percentile.split('-'))
                percentile_range = (min_pct, max_pct)
            except:
                print(f"Invalid percentile range format: {args.filter_percentile}. Expected format: 'min-max'")
        
        # Filter episodes
        selected_indices = filter_episodes_by_return(
            episode_data,
            min_return=args.filter_min_return,
            max_return=args.filter_max_return,
            percentile_range=percentile_range
        )
    
    # Balance episodes if requested
    if args.balance:
        if selected_indices is not None:
            # Apply balancing only to the already filtered episodes
            filtered_data = {k: v[selected_indices] if isinstance(v, np.ndarray) else v
                           for k, v in episode_data.items()}
            balanced_indices = balance_episodes(filtered_data, max_per_class=args.max_per_class, seed=args.seed)
            selected_indices = selected_indices[balanced_indices]
        else:
            # Balance all episodes
            selected_indices = balance_episodes(episode_data, max_per_class=args.max_per_class, seed=args.seed)
    
    # If no filtering or balancing was done, use all episodes
    if selected_indices is None:
        selected_indices = np.arange(len(episode_data['returns']))
        print(f"Using all {len(selected_indices)} episodes")
    
    # Plot histogram of selected episodes
    if len(selected_indices) < len(episode_data['returns']):
        selected_hist_path = os.path.join(args.output_dir, f"{dataset_name}_selected_returns_histogram.png")
        plot_returns_histogram(
            episode_data['returns'][selected_indices], 
            selected_hist_path, 
            bins=args.bins,
            title=f"Distribution of Selected Episode Returns for {dataset_name}"
        )
    
    # Create balanced dataset if filtering or balancing was applied
    if len(selected_indices) < len(episode_data['returns']):
        output_path = os.path.join(args.output_dir, f"{dataset_name}_balanced.pt")
        create_balanced_dataset(data, episode_data, selected_indices, output_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 