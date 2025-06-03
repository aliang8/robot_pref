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


def predict_rewards(model, episodes, is_distributional=False):
    """Predict rewards for each step in the episodes using the reward model.

    Args:
        model: Trained reward model
        episodes: List of episode dictionaries
        is_distributional: Whether the model is distributional

    Returns:
        List of episode dictionaries with predicted rewards added
    """
    model.eval()

    # Process each episode
    for i, episode in enumerate(tqdm(episodes, desc="Predicting rewards")):
        obs = episode["obs"]
        actions = episode["action"]
        images = None
        if "image" in episode:
            images = episode["image"]
        elif "image_embedding" in episode:
            images = episode["image_embedding"]

        # Predict rewards based on model type
        if is_distributional:
            # For distributional models, get both mean and variance
            predicted_mean, predicted_variance = model(obs, actions, images, return_distribution=True)
            episode["predicted_rewards"] = predicted_mean
            episode["predicted_variance"] = predicted_variance
            
            # Also compute confidence intervals (mean ± 2*std)
            predicted_std = torch.sqrt(predicted_variance)
            episode["predicted_std"] = predicted_std
            episode["predicted_upper"] = predicted_mean + 2 * predicted_std
            episode["predicted_lower"] = predicted_mean - 2 * predicted_std
        else:
            # For regular models, just get the reward prediction
            predicted_rewards = model(obs, actions, images)
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
    is_distributional=False,
):
    """Plot a grid of reward curves for multiple episodes.

    Args:
        episodes: List of episode dictionaries with rewards
        rand_indices: Random indices for episode selection
        output_file: File path to save the plot
        grid_size: Tuple of (rows, cols) for the grid layout
        smooth_window: Window size for smoothing the rewards
        reward_min: Global minimum reward for normalization
        reward_max: Global maximum reward for normalization
        is_distributional: Whether the model is distributional
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
    uncertainty_patch = None

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

            # Add uncertainty bands for distributional models (plot first so it's behind the line)
            if is_distributional and "predicted_upper" in episode and "predicted_lower" in episode:
                pred_upper = episode["predicted_upper"].detach().cpu().numpy()
                pred_lower = episode["predicted_lower"].detach().cpu().numpy()
                
                # Smooth uncertainty bounds if needed
                if smooth_window > 1 and len(pred_upper) > smooth_window:
                    pred_upper_smooth = np.convolve(pred_upper, kernel, mode="valid")
                    pred_lower_smooth = np.convolve(pred_lower, kernel, mode="valid")
                else:
                    pred_upper_smooth = pred_upper
                    pred_lower_smooth = pred_lower
                
                # Plot uncertainty band
                uncertainty_patch = ax.fill_between(
                    steps_smooth,
                    pred_lower_smooth,
                    pred_upper_smooth,
                    alpha=0.2,
                    color="blue",
                    label="Uncertainty (2std)"
                )

            # Plot predicted rewards (save the line object for legend)
            pred_line = ax.plot(
                steps_smooth,
                pred_rewards_smooth,
                "b-",
                linewidth=3,
                label="Predicted Rewards",
            )[0]  # Get the line object

            if "reward" in episode:
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
                    raw_min, raw_max = gt_rewards.min(), gt_rewards.max()
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

    # Add a common legend at the bottom of the figure
    legend_elements = []
    legend_labels = []
    
    if pred_line:
        legend_elements.append(pred_line)
        legend_labels.append("Predicted Rewards")
    
    if gt_line:
        legend_elements.append(gt_line)
        legend_labels.append("Ground Truth")
    
    if uncertainty_patch and is_distributional:
        legend_elements.append(uncertainty_patch)
        legend_labels.append("Uncertainty (2std)")

    if legend_elements:
        fig.legend(
            legend_elements,
            legend_labels,
            loc="lower center",
            ncol=len(legend_elements),
            fontsize=14,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_preference_return_analysis(model, test_loader, data, segment_indices, output_file, wandb_run=None, is_distributional=False):
    """
    Create scatter plots showing ground truth preference labels vs return deltas.
    Creates two plots: one with ground truth returns, one with model predicted returns.
    
    Args:
        model: Trained reward model
        test_loader: DataLoader for test data
        data: Raw data dictionary containing rewards
        segment_indices: List of (start, end) indices for segments
        output_file: Path to save the plot
        wandb_run: Optional wandb run for logging
        is_distributional: Whether the model is distributional
    """
    print("\nCreating preference return analysis plots (ground truth vs predicted)...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Check if we have ground truth rewards
    if "reward" not in data:
        print("Warning: No ground truth rewards available, skipping preference return analysis")
        return
    
    # Get the test dataset - handle both Subset and regular dataset
    test_dataset = test_loader.dataset
    if hasattr(test_dataset, 'dataset'):
        # This is a Subset object, get the original dataset and indices
        original_dataset = test_dataset.dataset
        subset_indices = test_dataset.indices
        print(f"Processing {len(subset_indices)} test samples from subset...")
    else:
        # This is the original dataset
        original_dataset = test_dataset
        subset_indices = list(range(len(test_dataset)))
        print(f"Processing {len(subset_indices)} test samples...")
    
    gt_preferences = []
    gt_return_deltas = []
    pred_return_deltas = []
    
    # Iterate through the test subset indices
    with torch.no_grad():
        for subset_idx in tqdm(subset_indices, desc="Computing ground truth and predicted returns"):
            # Get segment pair indices for this test sample
            seg1_idx, seg2_idx = original_dataset.segment_pairs[subset_idx]
            
            # Get segment boundaries
            start1, end1 = segment_indices[seg1_idx]
            start2, end2 = segment_indices[seg2_idx]
            
            # Calculate ground truth returns
            gt_return1 = data["reward"][start1:end1].sum().item()
            gt_return2 = data["reward"][start2:end2].sum().item()
            gt_return_delta = gt_return1 - gt_return2
            
            # Get observations and actions for model prediction
            obs1 = data[original_dataset.obs_key][start1:end1].unsqueeze(0).float().to(device)  # Add batch dimension
            obs2 = data[original_dataset.obs_key][start2:end2].unsqueeze(0).float().to(device)
            actions1 = data[original_dataset.action_key][start1:end1].unsqueeze(0).float().to(device)
            actions2 = data[original_dataset.action_key][start2:end2].unsqueeze(0).float().to(device)
            
            # Get images if using them
            images1, images2 = None, None
            if original_dataset.use_images:
                if original_dataset.use_image_embeddings:
                    images1 = data[original_dataset.image_embedding_key][start1:end1].unsqueeze(0).float().to(device)
                    images2 = data[original_dataset.image_embedding_key][start2:end2].unsqueeze(0).float().to(device)
                else:
                    images1 = data[original_dataset.image_key][start1:end1].unsqueeze(0).float().to(device)
                    images2 = data[original_dataset.image_key][start2:end2].unsqueeze(0).float().to(device)
            
            # Get model predictions
            if is_distributional:
                pred_rewards1 = model(obs1, actions1, images1, return_distribution=False)  # Get means only
                pred_rewards2 = model(obs2, actions2, images2, return_distribution=False)
            else:
                pred_rewards1 = model(obs1, actions1, images1)
                pred_rewards2 = model(obs2, actions2, images2)
            
            # Calculate predicted returns (sum over time dimension)
            pred_return1 = pred_rewards1.sum(dim=1).item()  # Sum over time, get scalar
            pred_return2 = pred_rewards2.sum(dim=1).item()
            pred_return_delta = pred_return1 - pred_return2
            
            # Get the ground truth preference for this sample
            preference = original_dataset.preferences[subset_idx]
            
            gt_preferences.append(preference)
            gt_return_deltas.append(gt_return_delta)
            pred_return_deltas.append(pred_return_delta)
    
    # Convert to numpy arrays
    gt_preferences = np.array(gt_preferences)
    gt_return_deltas = np.array(gt_return_deltas)
    pred_return_deltas = np.array(pred_return_deltas)
    
    # Sort by preferences for better visualization
    sort_indices = np.argsort(gt_preferences)
    gt_preferences_sorted = gt_preferences[sort_indices]
    gt_return_deltas_sorted = gt_return_deltas[sort_indices]
    pred_return_deltas_sorted = pred_return_deltas[sort_indices]
    
    # Create the plots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define colors and labels based on preference encoding
    unique_prefs = np.unique(gt_preferences_sorted)
    print(f"Unique preference values found: {unique_prefs}")
    
    color_map = {}
    label_map = {}
    
    for pref in unique_prefs:
        if pref == 0:
            color_map[pref] = 'red'
            label_map[pref] = 'Segment 2 Preferred'
        elif pref == 0.5:
            color_map[pref] = 'orange'  
            label_map[pref] = 'Equal'
        elif pref == 1:
            color_map[pref] = 'green'
            label_map[pref] = 'Segment 1 Preferred'
        elif pref == 2:
            color_map[pref] = 'red'
            label_map[pref] = 'Segment 2 Preferred'
        else:
            color_map[pref] = 'blue'
            label_map[pref] = f'Preference {pref}'
    
    # Prepare data for box plots by preference
    gt_data_by_pref = {}
    pred_data_by_pref = {}
    
    for pref in unique_prefs:
        mask = gt_preferences_sorted == pref
        gt_data_by_pref[pref] = gt_return_deltas_sorted[mask]
        pred_data_by_pref[pref] = pred_return_deltas_sorted[mask]
    
    # Plot 1: Ground Truth Returns (Hybrid scatter + box plot)
    
    # Create box plots for each preference category
    box_positions = []
    box_data_gt = []
    box_colors_gt = []
    
    for i, pref in enumerate(sorted(unique_prefs)):
        box_positions.append(pref)
        box_data_gt.append(gt_data_by_pref[pref])
        box_colors_gt.append(color_map[pref])
    
    # Create box plots
    bp1 = ax1.boxplot(box_data_gt, positions=box_positions, widths=0.15, 
                      patch_artist=True,
                      medianprops=dict(color='black', linewidth=2))
    
    # Color the box plots
    for patch, color in zip(bp1['boxes'], box_colors_gt):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    
    # Overlay scatter points with jitter for better visibility
    np.random.seed(42)  # For reproducible jitter
    for pref in unique_prefs:
        mask = gt_preferences_sorted == pref
        n_points = np.sum(mask)
        jitter = np.random.normal(0, 0.03, n_points)  # Small jitter around x position
        x_pos = np.full(n_points, pref) + jitter
        
        ax1.scatter(x_pos, gt_return_deltas_sorted[mask], 
                   alpha=0.6, s=30, c=color_map[pref], 
                   label=label_map[pref], edgecolors='black', linewidth=0.5)
    
    # Add reference line
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Return Delta = 0')
    
    # Customize ground truth plot
    ax1.set_xlabel('Ground Truth Preference Label', fontsize=14)
    ax1.set_ylabel('Ground Truth Return Delta (Seg1 - Seg2)', fontsize=14)
    ax1.set_title('Ground Truth: Preference Categories vs Return Deltas', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis ticks for ground truth plot
    tick_positions = []
    tick_labels = []
    for pref in sorted(unique_prefs):
        tick_positions.append(pref)
        if pref == 0:
            tick_labels.append(f'{pref}\n(Segment 2)')
        elif pref == 0.5:
            tick_labels.append(f'{pref}\n(Equal)')
        elif pref == 1:
            tick_labels.append(f'{pref}\n(Segment 1)')
        elif pref == 2:
            tick_labels.append(f'{pref}\n(Segment 2)')
        else:
            tick_labels.append(f'{pref}')
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    
    # Add statistics for ground truth plot
    gt_corr_coef = np.corrcoef(gt_preferences, gt_return_deltas)[0, 1]
    gt_quartiles = np.percentile(gt_return_deltas, [0, 25, 50, 75, 100])
    
    # Simplified correlation text
    ax1.text(0.02, 0.98, f'GT Correlation: {gt_corr_coef:.3f}\nN = {len(gt_preferences)}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Predicted Returns (Hybrid scatter + box plot)
    
    # Create box plots for predicted data
    box_data_pred = []
    box_colors_pred = []
    
    for pref in sorted(unique_prefs):
        box_data_pred.append(pred_data_by_pref[pref])
        box_colors_pred.append(color_map[pref])
    
    # Create box plots
    bp2 = ax2.boxplot(box_data_pred, positions=box_positions, widths=0.15,
                      patch_artist=True,
                      medianprops=dict(color='black', linewidth=2))
    
    # Color the box plots
    for patch, color in zip(bp2['boxes'], box_colors_pred):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    
    # Overlay scatter points with jitter
    for pref in unique_prefs:
        mask = gt_preferences_sorted == pref
        n_points = np.sum(mask)
        jitter = np.random.normal(0, 0.03, n_points)  # Small jitter around x position
        x_pos = np.full(n_points, pref) + jitter
        
        ax2.scatter(x_pos, pred_return_deltas_sorted[mask], 
                   alpha=0.6, s=30, c=color_map[pref], 
                   label=label_map[pref], edgecolors='black', linewidth=0.5)
    
    # Add reference line
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Return Delta = 0')
    
    # Customize predicted plot
    ax2.set_xlabel('Ground Truth Preference Label', fontsize=14)
    ax2.set_ylabel('Predicted Return Delta (Seg1 - Seg2)', fontsize=14)
    ax2.set_title('Model Predicted: Preference Categories vs Return Deltas', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis ticks for predicted plot
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    
    # Add statistics for predicted plot
    pred_corr_coef = np.corrcoef(gt_preferences, pred_return_deltas)[0, 1]
    pred_quartiles = np.percentile(pred_return_deltas, [0, 25, 50, 75, 100])
    
    # Simplified correlation text
    ax2.text(0.02, 0.98, f'Pred Correlation: {pred_corr_coef:.3f}\nN = {len(gt_preferences)}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legends (avoiding duplicate labels)
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Preference return analysis plots saved to: {output_file}")
    print(f"Ground truth correlation: {gt_corr_coef:.3f}")
    print(f"Predicted correlation: {pred_corr_coef:.3f}")
    print(f"\nGround Truth Return Delta Statistics:")
    print(f"  Min: {gt_quartiles[0]:.2f}, Q1: {gt_quartiles[1]:.2f}, Median: {gt_quartiles[2]:.2f}, Q3: {gt_quartiles[3]:.2f}, Max: {gt_quartiles[4]:.2f}")
    print(f"Predicted Return Delta Statistics:")
    print(f"  Min: {pred_quartiles[0]:.2f}, Q1: {pred_quartiles[1]:.2f}, Median: {pred_quartiles[2]:.2f}, Q3: {pred_quartiles[3]:.2f}, Max: {pred_quartiles[4]:.2f}")
    
    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log({
            "preference_return_analysis": wandb.Image(output_file),
            "gt_return_correlation": gt_corr_coef,
            "pred_return_correlation": pred_corr_coef,
            "gt_return_delta_min": gt_quartiles[0],
            "gt_return_delta_q1": gt_quartiles[1], 
            "gt_return_delta_median": gt_quartiles[2],
            "gt_return_delta_q3": gt_quartiles[3],
            "gt_return_delta_max": gt_quartiles[4],
            "pred_return_delta_min": pred_quartiles[0],
            "pred_return_delta_q1": pred_quartiles[1],
            "pred_return_delta_median": pred_quartiles[2], 
            "pred_return_delta_q3": pred_quartiles[3],
            "pred_return_delta_max": pred_quartiles[4]
        })
    
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
    is_distributional=False,
):
    """Analyze rewards for episodes in the dataset.

    Args:
        model: Trained reward model
        episodes: List of episode dictionaries
        output_file: Directory to save the plots. If None, uses the model directory.
        num_episodes: Number of episodes to analyze
        reward_min: Minimum reward for normalization
        reward_max: Maximum reward for normalization
        wandb_run: Wandb run to log
        random_seed: Random seed for reproducibility
        is_distributional: Whether the model is distributional
    """
    if random_seed is not None:
        set_seed(random_seed)

    print(f"Sampling {num_episodes} random episodes from {len(episodes)} total")
    rand_indices = random.sample(range(len(episodes)), num_episodes)
    sampled_episodes = [episodes[i] for i in rand_indices]

    # Predict rewards for each episode
    print("Predicting rewards for episodes")
    sampled_episodes = predict_rewards(model, sampled_episodes, is_distributional=is_distributional)

    # Plot rewards in a grid
    print("Creating reward grid plot")
    plot_reward_grid(
        sampled_episodes,
        rand_indices,
        output_file,
        grid_size=(3, 3),
        reward_min=reward_min,
        reward_max=reward_max,
        is_distributional=is_distributional,
    )

    print(f"Analysis complete. Results saved to {output_file}")

    if wandb_run:
        wandb_run.log({"reward_grid": wandb.Image(output_file)})
