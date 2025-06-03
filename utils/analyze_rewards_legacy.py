#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import wandb
import seaborn as sns
from utils.seed import set_seed

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
try:
    plt.rc("text", usetex=False)  # camera-ready formatting + latex in plots
except:
    pass  # Skip if latex not available


def predict_rewards_legacy(reward_model, episodes):
    """Predict rewards for each step in the episodes using the legacy RewardModel.

    Args:
        reward_model: Legacy RewardModel instance with ensemble_model_forward method
        episodes: List of episode dictionaries

    Returns:
        List of episode dictionaries with predicted rewards added
    """
    # Set ensemble models to eval mode
    for member in range(reward_model.ensemble_num):
        reward_model.ensemble_model[member].eval()

    # Process each episode
    for i, episode in enumerate(tqdm(episodes, desc="Predicting rewards")):
        obs = episode["obs"]
        actions = episode["action"]
        
        # Concatenate obs and actions as expected by legacy model
        obs_act = torch.cat([obs, actions], dim=-1).to(reward_model.device)
        
        with torch.no_grad():
            # Use ensemble forward method
            predicted_rewards = reward_model.ensemble_model_forward(obs_act)
            episode["predicted_rewards"] = predicted_rewards.squeeze(-1)  # Remove last dimension if needed

    return episodes


def plot_reward_grid_legacy(
    episodes,
    rand_indices,
    output_file,
    grid_size=(3, 3),
    smooth_window=5,
    reward_min=None,
    reward_max=None,
):
    """Plot a grid of reward curves for multiple episodes (legacy version).

    Args:
        episodes: List of episode dictionaries with rewards
        rand_indices: Random indices for episode selection
        output_file: File path to save the plot
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
            if "predicted_rewards" not in episode or len(episode["predicted_rewards"]) == 0:
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

            if "reward" in episode:
                # Plot ground truth rewards if available
                gt_rewards = episode["reward"].detach().cpu().numpy()
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
            else:
                # No ground truth available
                ep_id = rand_indices[idx] if idx < len(rand_indices) else idx
                ax.set_title(f"Episode {ep_id} (No GT)", fontsize=14)

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


def analyze_rewards_legacy(
    reward_model,
    episodes,
    output_file=None,
    num_episodes=9,
    reward_min=None,
    reward_max=None,
    wandb_run=None,
    random_seed=None,
):
    """Analyze rewards for episodes in the dataset using legacy RewardModel.

    Args:
        reward_model: Legacy RewardModel instance
        episodes: List of episode dictionaries
        output_file: File path to save the plots
        num_episodes: Number of episodes to analyze
        reward_min: Minimum reward for normalization
        reward_max: Maximum reward for normalization
        wandb_run: Wandb run to log
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        set_seed(random_seed)

    print(f"Sampling {num_episodes} random episodes from {len(episodes)} total")
    rand_indices = random.sample(range(len(episodes)), min(num_episodes, len(episodes)))
    sampled_episodes = [episodes[i] for i in rand_indices]

    # Predict rewards for each episode
    print("Predicting rewards for episodes")
    sampled_episodes = predict_rewards_legacy(reward_model, sampled_episodes)

    # Plot rewards in a grid
    print("Creating reward grid plot")
    plot_reward_grid_legacy(
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


def create_episodes_from_dataset(dataset, device="cpu", episode_length=500):
    """Create episodes from the dataset format used by legacy code.
    
    Args:
        dataset: Dictionary with observations, actions, rewards
        device: Device to place tensors on
        episode_length: Length of each episode
        
    Returns:
        List of episode dictionaries
    """
    episodes = []
    
    # Convert to tensors if needed
    observations = torch.tensor(dataset["observations"], dtype=torch.float32, device=device)
    actions = torch.tensor(dataset["actions"], dtype=torch.float32, device=device)
    
    # Check if rewards are available
    rewards = None
    if "rewards" in dataset:
        rewards = torch.tensor(dataset["rewards"], dtype=torch.float32, device=device)
    
    # Create episodes
    num_episodes = len(observations) // episode_length
    print(f"Creating {num_episodes} episodes of length {episode_length}")
    
    for i in range(num_episodes):
        start_idx = i * episode_length
        end_idx = start_idx + episode_length
        
        episode = {
            "obs": observations[start_idx:end_idx],
            "action": actions[start_idx:end_idx],
            "episode": torch.full((episode_length,), i, dtype=torch.long, device=device)
        }
        
        if rewards is not None:
            episode["reward"] = rewards[start_idx:end_idx]
            
        episodes.append(episode)
    
    return episodes


def plot_preference_return_analysis_legacy(
    reward_model, 
    obs_act_1, 
    obs_act_2, 
    labels, 
    gt_return_1,
    gt_return_2,
    segment_size, 
    output_file, 
    wandb_run=None
):
    """
    Create scatter plots showing ground truth preference labels vs return deltas for legacy RewardModel.
    Creates two plots: one with ground truth returns, one with model predicted returns.
    
    Args:
        reward_model: Legacy RewardModel instance
        obs_act_1: First segments (observations + actions concatenated) [N, segment_size, obs+act_dim]
        obs_act_2: Second segments (observations + actions concatenated) [N, segment_size, obs+act_dim]
        labels: Preference labels [N, 2] where each row is [preference_for_seg1, preference_for_seg2]
        gt_return_1: Ground truth returns for first segments [N]
        gt_return_2: Ground truth returns for second segments [N]
        segment_size: Length of each segment
        output_file: Path to save the plot
        wandb_run: Optional wandb run for logging
    """
    print("\nCreating preference return analysis plots (ground truth vs predicted)...")
    
    # Set ensemble models to eval mode
    for member in range(reward_model.ensemble_num):
        reward_model.ensemble_model[member].eval()
    
    # Convert data to tensors if needed
    if not isinstance(obs_act_1, torch.Tensor):
        obs_act_1 = torch.tensor(obs_act_1, dtype=torch.float32)
    if not isinstance(obs_act_2, torch.Tensor):
        obs_act_2 = torch.tensor(obs_act_2, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    obs_act_1 = obs_act_1.to(reward_model.device)
    obs_act_2 = obs_act_2.to(reward_model.device)
    
    # Convert GT returns to numpy if needed
    if isinstance(gt_return_1, torch.Tensor):
        gt_return_1 = gt_return_1.cpu().numpy()
    if isinstance(gt_return_2, torch.Tensor):
        gt_return_2 = gt_return_2.cpu().numpy()
    
    gt_preferences = []
    gt_return_deltas = []
    pred_return_deltas = []
    
    # Convert preference labels to single values
    # labels is [N, 2] where [1, 0] means segment 1 preferred, [0, 1] means segment 2 preferred, [0.5, 0.5] means equal
    preference_values = []
    for label in labels:
        if label[0] > label[1]:
            preference_values.append(1)  # Segment 1 preferred
        elif label[1] > label[0]:
            preference_values.append(0)  # Segment 2 preferred
        else:
            preference_values.append(0.5)  # Equal
    preference_values = np.array(preference_values)
    
    print(f"Processing {len(obs_act_1)} preference pairs...")
    
    with torch.no_grad():
        for i in tqdm(range(len(obs_act_1)), desc="Computing returns"):
            # Get segments for this pair
            seg1 = obs_act_1[i]  # [segment_size, obs+act_dim]
            seg2 = obs_act_2[i]  # [segment_size, obs+act_dim]
            
            # Predict rewards for both segments
            pred_rewards1 = reward_model.ensemble_model_forward(seg1)  # [segment_size, 1]
            pred_rewards2 = reward_model.ensemble_model_forward(seg2)  # [segment_size, 1]
            
            # Calculate predicted returns (sum over time dimension)
            pred_return1 = pred_rewards1.sum().item()
            pred_return2 = pred_rewards2.sum().item()
            pred_return_delta = pred_return1 - pred_return2
            
            # Calculate ground truth return delta using the actual GT returns
            gt_return_delta = gt_return_1[i] - gt_return_2[i]
            
            gt_preferences.append(preference_values[i])
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
    
    # Plot 1: Ground Truth Returns
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
        else:
            tick_labels.append(f'{pref}')
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    
    # Add statistics for ground truth plot
    if len(gt_return_deltas) > 1 and np.std(gt_return_deltas) > 0:
        gt_corr_coef = np.corrcoef(gt_preferences, gt_return_deltas)[0, 1]
        gt_quartiles = np.percentile(gt_return_deltas, [0, 25, 50, 75, 100])
        
        # Simplified correlation text
        ax1.text(0.02, 0.98, f'GT Correlation: {gt_corr_coef:.3f}\nN = {len(gt_preferences)}', 
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax1.text(0.02, 0.98, f'N = {len(gt_preferences)}\n(No GT variance)', 
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Predicted Returns
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
    print(f"Predicted correlation: {pred_corr_coef:.3f}")
    if len(gt_return_deltas) > 1 and np.std(gt_return_deltas) > 0:
        print(f"Ground truth correlation: {gt_corr_coef:.3f}")
        print(f"Ground Truth Return Delta Statistics:")
        print(f"  Min: {gt_quartiles[0]:.2f}, Q1: {gt_quartiles[1]:.2f}, Median: {gt_quartiles[2]:.2f}, Q3: {gt_quartiles[3]:.2f}, Max: {gt_quartiles[4]:.2f}")
    print(f"Predicted Return Delta Statistics:")
    print(f"  Min: {pred_quartiles[0]:.2f}, Q1: {pred_quartiles[1]:.2f}, Median: {pred_quartiles[2]:.2f}, Q3: {pred_quartiles[3]:.2f}, Max: {pred_quartiles[4]:.2f}")
    
    # Log to wandb if available
    if wandb_run is not None:
        log_data = {
            "preference_return_analysis": wandb.Image(output_file),
            "pred_return_correlation": pred_corr_coef,
            "pred_return_delta_min": pred_quartiles[0],
            "pred_return_delta_q1": pred_quartiles[1],
            "pred_return_delta_median": pred_quartiles[2], 
            "pred_return_delta_q3": pred_quartiles[3],
            "pred_return_delta_max": pred_quartiles[4]
        }
        
        # Add GT statistics if available
        if len(gt_return_deltas) > 1 and np.std(gt_return_deltas) > 0:
            log_data.update({
                "gt_return_correlation": gt_corr_coef,
                "gt_return_delta_min": gt_quartiles[0],
                "gt_return_delta_q1": gt_quartiles[1], 
                "gt_return_delta_median": gt_quartiles[2],
                "gt_return_delta_q3": gt_quartiles[3],
                "gt_return_delta_max": gt_quartiles[4]
            })
        
        wandb_run.log(log_data)
    
    plt.close()


def plot_segment_return_scatter_analysis_legacy(
    reward_model,
    obs_act_1,
    obs_act_2,
    gt_return_1,
    gt_return_2,
    segment_size,
    output_file,
    max_samples=5000,
    wandb_run=None,
    random_seed=None
):
    """
    Create a scatter plot comparing predicted segment returns vs ground truth segment returns.
    Both are normalized to [0, 25] range.
    
    Args:
        reward_model: Legacy RewardModel instance
        obs_act_1: First segments (observations + actions concatenated) [N, segment_size, obs+act_dim]
        obs_act_2: Second segments (observations + actions concatenated) [N, segment_size, obs+act_dim]
        gt_return_1: Ground truth returns for first segments [N]
        gt_return_2: Ground truth returns for second segments [N]
        segment_size: Length of each segment
        output_file: Path to save the plot
        max_samples: Maximum number of segments to plot (for performance)
        wandb_run: Optional wandb run for logging
        random_seed: Random seed for reproducibility
    """
    print("\nCreating segment return scatter analysis plot (predicted vs ground truth)...")
    
    if random_seed is not None:
        set_seed(random_seed)
    
    # Set ensemble models to eval mode
    for member in range(reward_model.ensemble_num):
        reward_model.ensemble_model[member].eval()
    
    # Convert data to tensors if needed
    if not isinstance(obs_act_1, torch.Tensor):
        obs_act_1 = torch.tensor(obs_act_1, dtype=torch.float32)
    if not isinstance(obs_act_2, torch.Tensor):
        obs_act_2 = torch.tensor(obs_act_2, dtype=torch.float32)
    
    obs_act_1 = obs_act_1.to(reward_model.device)
    obs_act_2 = obs_act_2.to(reward_model.device)
    
    # Convert GT returns to numpy if needed
    if isinstance(gt_return_1, torch.Tensor):
        gt_return_1 = gt_return_1.cpu().numpy()
    if isinstance(gt_return_2, torch.Tensor):
        gt_return_2 = gt_return_2.cpu().numpy()
    
    # Combine both segments and their GT returns for analysis
    all_segments = torch.cat([obs_act_1, obs_act_2], dim=0)  # [2*N, segment_size, obs+act_dim]
    all_gt_returns = np.concatenate([gt_return_1, gt_return_2], axis=0)  # [2*N]
    
    # Sample segments if too many
    n_total_segments = len(all_segments)
    if n_total_segments > max_samples:
        print(f"Sampling {max_samples} segments from {n_total_segments} total segments")
        indices = np.random.choice(n_total_segments, size=max_samples, replace=False)
        sampled_segments = all_segments[indices]
        sampled_gt_returns = all_gt_returns[indices]
    else:
        sampled_segments = all_segments
        sampled_gt_returns = all_gt_returns
    
    print(f"Analyzing {len(sampled_segments)} segments...")
    
    # Predict returns for segments
    predicted_returns = []
    
    with torch.no_grad():
        for i in tqdm(range(len(sampled_segments)), desc="Computing predicted segment returns"):
            segment = sampled_segments[i]  # [segment_size, obs+act_dim]
            
            # Predict rewards for this segment
            pred_rewards = reward_model.ensemble_model_forward(segment)  # [segment_size, 1]
            pred_return = pred_rewards.sum().item()  # Sum to get return
            predicted_returns.append(pred_return)
    
    predicted_returns = np.array(predicted_returns)
    gt_returns = sampled_gt_returns
    
    # Normalize both to [0, 25] range
    def normalize_to_range(values, target_min=0, target_max=25):
        """Normalize values to target range [target_min, target_max]"""
        if len(values) == 0:
            return values
        val_min, val_max = values.min(), values.max()
        if val_max - val_min == 0:
            return np.full_like(values, (target_max + target_min) / 2)
        normalized = (values - val_min) / (val_max - val_min) * (target_max - target_min) + target_min
        return normalized
    
    pred_returns_norm = normalize_to_range(predicted_returns)
    gt_returns_norm = normalize_to_range(gt_returns)
    
    # Calculate Pearson correlation
    if len(pred_returns_norm) > 1 and np.std(pred_returns_norm) > 0 and np.std(gt_returns_norm) > 0:
        correlation = np.corrcoef(pred_returns_norm, gt_returns_norm)[0, 1]
    else:
        correlation = 0.0
    
    # Calculate additional statistics
    mse = np.mean((pred_returns_norm - gt_returns_norm) ** 2)
    mae = np.mean(np.abs(pred_returns_norm - gt_returns_norm))
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with some transparency
    plt.scatter(gt_returns_norm, pred_returns_norm, alpha=0.6, s=20, edgecolors='none')
    
    # Add perfect correlation line (y=x)
    plt.plot([0, 25], [0, 25], 'r--', linewidth=2, label='Perfect Correlation (y=x)')
    
    # Add trend line (linear regression) if we have variation
    if len(pred_returns_norm) > 1 and np.std(gt_returns_norm) > 0:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(gt_returns_norm, pred_returns_norm)
        line_x = np.array([0, 25])
        line_y = slope * line_x + intercept
        # Clip to plot bounds
        line_y = np.clip(line_y, 0, 25)
        plt.plot(line_x, line_y, 'g-', linewidth=2, label=f'Trend Line (slope={slope:.3f})')
    
    # Customize plot
    plt.xlabel('Ground Truth Segment Returns (Normalized [0,25])', fontsize=14)
    plt.ylabel('Predicted Segment Returns (Normalized [0,25])', fontsize=14)
    plt.title('Predicted vs Ground Truth Segment Returns', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    
    # Add statistics text box
    stats_text = f'Pearson r = {correlation:.4f}\nMSE = {mse:.4f}\nMAE = {mae:.4f}\nn = {len(predicted_returns):,} segments'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12)
    
    # Make plot square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Segment return scatter analysis saved to: {output_file}")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"Predicted returns range: [{predicted_returns.min():.3f}, {predicted_returns.max():.3f}] -> [0, 25]")
    print(f"GT returns range: [{gt_returns.min():.3f}, {gt_returns.max():.3f}] -> [0, 25]")
    
    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log({
            "segment_return_scatter_analysis": wandb.Image(output_file),
            "segment_return_correlation": correlation,
            "segment_return_mse": mse,
            "segment_return_mae": mae,
            "n_segments_analyzed": len(predicted_returns)
        })
    
    plt.close() 