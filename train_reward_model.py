import glob
import itertools
import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np

import wandb
from models.reward_models import RewardModel, DistributionalRewardModel
from utils.analyze_rewards import analyze_rewards
from utils.data import (
    get_gt_preferences,
    load_tensordict,
    process_data_trajectories,
    segment_episodes,
)
from utils.dataset import (
    PreferenceDataset,
    create_data_loaders,
    custom_collate,
)
from utils.seed import set_seed
from utils.training import evaluate_model_on_test_set, train_model, train_distributional_model
from utils.wandb import log_to_wandb


def load_preferences_from_directory(pref_dir):
    """
    Load all preference files from a directory and return unique preferences.
    
    Args:
        pref_dir (str): Path to directory containing preference JSON files
        
    Returns:
        list: List of unique preferences with pair indices and choices
        dict: Statistics about preferences
    """

    print("="*50)
    print(f"Loading preferences from {pref_dir}")
    print("="*50)

    all_preferences = []
    stats = {
        'total_files': 0,
        'total_preferences': 0,
        'unique_pairs': set(),
        'preference_counts': {'A': 0, 'B': 0, 'equal': 0}
    }
    
    # Load all JSON files in the directory
    for pref_file in glob.glob(os.path.join(pref_dir, '*.json')):
        try:
            with open(pref_file, 'r') as f:
                data = json.load(f)
                stats['total_files'] += 1
                
                # Extract preferences from each file
                for pref in data.get('preferences', []):
                    pair_idx = pref.get('pair_index')
                    preference = pref.get('preference')
                    timestamp = pref.get('timestamp')
                    
                    if pair_idx is not None and preference:
                        all_preferences.append({
                            'pair_index': pair_idx,
                            'preference': preference,
                            'timestamp': timestamp
                        })
                        stats['unique_pairs'].add(pair_idx)
                        stats['preference_counts'][preference] += 1
                        stats['total_preferences'] += 1
                        
        except Exception as e:
            print(f"Error loading preferences from {pref_file}: {e}")
            continue
    
    # Sort preferences by timestamp and get unique latest preference for each pair
    all_preferences.sort(key=lambda x: x['timestamp'])
    unique_preferences = {}
    for pref in all_preferences:
        unique_preferences[pref['pair_index']] = pref
    
    # Convert to list of unique preferences
    unique_pref_list = list(unique_preferences.values())
    
    print(f"Loaded {len(unique_pref_list)} unique preferences from {stats['total_files']} files")
    print(f"Preference distribution: A: {stats['preference_counts']['A']}, "
          f"B: {stats['preference_counts']['B']}, "
          f"Equal: {stats['preference_counts']['equal']}")
    
    return unique_pref_list, stats

def load_segments_from_files(segment_pairs_path, segment_indices_path):
    """
    Load segment pairs and indices from numpy files.
    
    Args:
        segment_pairs_path (str): Path to segment pairs .npy file
        segment_indices_path (str): Path to segment indices .npy file
        
    Returns:
        tuple: (segment_pairs, segment_indices)
    """
    segment_pairs = np.load(segment_pairs_path)
    segment_indices = np.load(segment_indices_path)
    segment_indices = segment_indices.reshape(-1, 2)  # Reshape to (N, 2) for start/end indices
    
    print(f"Loaded {len(segment_pairs)} segment pairs")
    print(f"Loaded {len(segment_indices)} segment indices")
    
    return segment_pairs, segment_indices

def filter_pairs_with_preferences(segment_pairs, preferences):
    """
    Filter segment pairs to only include those with preferences.
    
    Args:
        segment_pairs (np.ndarray): Array of segment pairs
        preferences (list): List of preference dictionaries
        
    Returns:
        tuple: (filtered_pairs, filtered_preferences)
    """
    preference_dict = {p['pair_index']: p['preference'] for p in preferences}
    filtered_pairs = []
    filtered_preferences = []
    
    for idx, pair in enumerate(segment_pairs):
        if idx in preference_dict:
            filtered_pairs.append(pair)
            pref = preference_dict[idx]

            # Convert preference to numerical value
            if pref == 'A':
                filtered_preferences.append(1)
            elif pref == 'B':
                filtered_preferences.append(0)
            else:  # equal
                filtered_preferences.append(0.5)
    
    return np.array(filtered_pairs), np.array(filtered_preferences)

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

@hydra.main(config_path="config", config_name="reward_model", version_base=None)
def main(cfg: DictConfig):
    # Get the dataset name
    dataset_name = Path(cfg.data.data_path).stem

    # Replace only the dataset name placeholder in the template strings
    if hasattr(cfg.output, "model_dir_name"):
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace(
            "DATASET_NAME", dataset_name
        )

    print("\n" + "=" * 50)
    print("Training reward model with Bradley-Terry preference learning")
    print("=" * 50)

    # Print config for visibility
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Get number of seeds to run (default to 1 if not specified)
    num_seeds = cfg.get("num_seeds", 1)
    print(f"Training {num_seeds} models with different seeds")

    # Base random seed
    base_seed = cfg.get("random_seed", 42)
    
    all_test_metrics = []
    seed_metrics = []
    
    for seed_idx in tqdm(range(num_seeds)):
        # Calculate seed for this run
        current_seed = base_seed + seed_idx
        print("\n" + "=" * 50)
        print(f"Training model {seed_idx+1}/{num_seeds} with seed {current_seed}")
        print("=" * 50)
        
        # Set random seed for reproducibility
        set_seed(current_seed)

        # Initialize wandb for this seed
        if cfg.wandb.use_wandb:
            # Generate experiment name based on data path
            dataset_name = Path(cfg.data.data_path).stem

            # Set up a run name if not specified
            run_name = cfg.wandb.name
            if run_name is None:
                run_name = f"reward_{dataset_name}_{cfg.data.num_pairs}_seed{current_seed}_{time.strftime('%Y%m%d_%H%M%S')}"

            # Initialize wandb
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.wandb.tags,
                notes=cfg.wandb.notes,
                reinit=True,  # Allow reinit for multiple runs
                group=f"reward_{dataset_name}_{cfg.data.num_pairs}_multiseed",  # Group runs together
            )

            print(f"Wandb initialized: {wandb_run.name}")
        else:
            wandb_run = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading data from file: {cfg.data.data_path}")
        data = load_tensordict(cfg.data.data_path)

        # Get observation and action dimensions
        observations = data["obs"] if "obs" in data else data["state"]
        actions = data["action"]
        state_dim = observations.shape[1]
        action_dim = actions.shape[1]

        # Ensure data is on CPU for processing
        data_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        print(f"Loaded data with {len(observations)} observations")

        # Load segments and preferences if paths are provided
        if hasattr(cfg.data, 'segment_pairs_path') and hasattr(cfg.data, 'segment_indices_path') and cfg.data.segment_pairs_path is not None and cfg.data.segment_indices_path is not None:
            print("Loading segments from files...")
            all_segment_pairs, segment_indices = load_segments_from_files(
                cfg.data.segment_pairs_path,
                cfg.data.segment_indices_path
            )
            all_segment_pairs = all_segment_pairs.tolist()
        else:
            print("Generating segments from data...")
            segments, segment_indices = segment_episodes(data_cpu, cfg.data.segment_length)
            all_segment_pairs = list(itertools.combinations(range(len(segment_indices)), 2))

        # all_segment_pairs = random.sample(all_segment_pairs, cfg.data.num_pairs)

        # Load preferences if path is provided
        if hasattr(cfg.data, 'preferences_dir') and cfg.data.preferences_dir is not None:
            print("Loading preferences from files...")
            preferences, pref_stats = load_preferences_from_directory(cfg.data.preferences_dir)
            all_segment_pairs, preferences = filter_pairs_with_preferences(all_segment_pairs, preferences)
        else:
            print("Computing ground truth preferences...")
            preferences = get_gt_preferences(data_cpu, segment_indices, all_segment_pairs)

        print(
            f"Final data stats - Observation dimension: {state_dim}, Action dimension: {action_dim}"
        )
        print(
            f"Working with {len(all_segment_pairs)} preference pairs across {len(segment_indices)} segments"
        )

        print("="*50)
        print("Data stats:")
        print("="*50)
        for k, v in data_cpu.items():
            print(f"{k}: {v.shape}")
        
        # Hold out test pairs first (similar to active learning script)
        num_test_pairs = cfg.data.get('num_test_pairs', 200)
        print(f"Holding out {num_test_pairs} pairs for testing...")
        
        all_segment_pair_indices = list(range(len(all_segment_pairs)))
        test_pair_indices = random.sample(all_segment_pair_indices, num_test_pairs)

        # Get test pairs and preferences
        test_pairs = [all_segment_pairs[i] for i in test_pair_indices]
        test_preferences = [preferences[i] for i in test_pair_indices]

        # Use remaining pairs for training/validation
        train_val_pair_indices = [i for i in all_segment_pair_indices if i not in test_pair_indices]
        train_val_pair_indices = random.sample(train_val_pair_indices, cfg.data.num_pairs)
        train_val_pairs = [all_segment_pairs[i] for i in train_val_pair_indices]
        train_val_preferences = [preferences[i] for i in train_val_pair_indices]
        
        print(f"Train/Val pairs: {len(train_val_pairs)}, Test pairs: {len(test_pairs)}")
        
        # Create test dataset manually
        test_dataset = PreferenceDataset(
            data_cpu, 
            test_pairs, 
            segment_indices, 
            test_preferences,
            normalize_obs=cfg.data.normalize_obs,
            norm_method=cfg.data.norm_method,
            use_images=cfg.data.use_images,
            image_key=cfg.data.image_key,
            obs_key=cfg.data.obs_key,
            action_key=cfg.data.action_key,
            use_image_embeddings=cfg.data.use_image_embeddings
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=cfg.training.batch_size, 
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=custom_collate
        )
        
        # Create train/val dataset with remaining pairs
        train_val_dataset = PreferenceDataset(
            data_cpu, 
            train_val_pairs, 
            segment_indices, 
            train_val_preferences,
            normalize_obs=cfg.data.normalize_obs,
            norm_method=cfg.data.norm_method,
            use_images=cfg.data.use_images,
            image_key=cfg.data.image_key,
            obs_key=cfg.data.obs_key,
            action_key=cfg.data.action_key,
            use_image_embeddings=cfg.data.use_image_embeddings
        )

        # Create data loaders for train/val split only (no test split)
        dataloaders = create_data_loaders(
            train_val_dataset,
            train_ratio=0.9,  # 90% of remaining data for training
            val_ratio=0.1,    # 10% of remaining data for validation
            batch_size=cfg.training.batch_size,
            seed=current_seed,
            normalize_obs=cfg.data.normalize_obs,
            norm_method=cfg.data.norm_method,
            num_workers=cfg.training.num_workers,
            # pin_memory=cfg.training.pin_memory
        )

        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        # test_loader is already created above

        # Initialize reward model (regular or distributional)
        if cfg.model.is_distributional:
            print("Initializing distributional reward model...")
            model = DistributionalRewardModel(
                state_dim, 
                action_dim, 
                hidden_dims=cfg.model.hidden_dims,
                use_images=cfg.data.use_images,
                image_model=cfg.model.image_model,
                embedding_dim=cfg.model.embedding_dim,
                use_image_embeddings=cfg.data.use_image_embeddings,
                device=device
            )
        else:
            print("Initializing regular reward model...")
            model = RewardModel(
                state_dim, 
                action_dim, 
                hidden_dims=cfg.model.hidden_dims,
                use_images=cfg.data.use_images,
                image_model=cfg.model.image_model,
                embedding_dim=cfg.model.embedding_dim,
                use_image_embeddings=cfg.data.use_image_embeddings,
                device=device
            )
        
        model = model.to(device)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        start_time = time.time()
        dataset_name = Path(cfg.data.data_path).stem
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace("DATASET_NAME", dataset_name)

        model_dir = os.path.join(cfg.output.output_dir, cfg.output.model_dir_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{current_seed}.pt")

        training_curve_path = os.path.join(model_dir, f"training_curve_{current_seed}.png")

        print(f"\nTraining {'distributional' if cfg.model.is_distributional else 'regular'} reward model...")
        
        if cfg.model.is_distributional:
            # Use distributional training function with additional parameters
            lambda_weight = cfg.model.get("lambda_weight", 1.0)
            alpha_reg = cfg.model.get("alpha_reg", 0.1)
            eta = cfg.model.get("eta", 1.0)
            num_samples = cfg.model.get("num_samples", 5)
            
            model, *_ = train_distributional_model(
                model,
                train_loader,
                val_loader,
                device,
                num_epochs=cfg.training.num_epochs,
                lr=cfg.model.lr,
                wandb_run=wandb_run,
                output_path=training_curve_path,
                lambda_weight=lambda_weight,
                alpha_reg=alpha_reg,
                eta=eta,
                num_samples=num_samples,
            )
        else:
            # Use regular training function
            model, *_ = train_model(
                model,
                train_loader,
                val_loader,
                device,
                num_epochs=cfg.training.num_epochs,
                lr=cfg.model.lr,
                wandb_run=wandb_run,
                output_path=training_curve_path,
            )

        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # Evaluate on test set
        test_metrics = evaluate_model_on_test_set(model, test_loader, device, is_distributional=cfg.model.is_distributional, num_vis=0)
        test_metrics["seed"] = current_seed
        all_test_metrics.append(test_metrics)
        
        if wandb_run is not None:
            log_to_wandb(test_metrics, prefix="test")

        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

        seed_metrics.append({
            "seed": current_seed,
            "test_accuracy": test_metrics["test_accuracy"],
            "model_path": model_path,
        })

        # Run reward analysis
        episodes = process_data_trajectories(data, device)    

        if "reward" in data_cpu:
            reward_max = data_cpu["reward"].max().item()
            reward_min = data_cpu["reward"].min().item()
        else:
            reward_max = None
            reward_min = None
        
        analyze_rewards(
            model=model,
            episodes=episodes,
            output_file=os.path.join(model_dir, f"reward_grid_{current_seed}.png"),
            wandb_run=wandb_run,
            reward_max=reward_max,
            reward_min=reward_min,
            is_distributional=cfg.model.is_distributional
        )
        
        # Add preference return analysis plot
        plot_preference_return_analysis(
            model=model,
            test_loader=test_loader,
            data=data_cpu,
            segment_indices=segment_indices,
            output_file=os.path.join(model_dir, f"preference_return_analysis_{current_seed}.png"),
            wandb_run=wandb_run,
            is_distributional=cfg.model.is_distributional
        )

    if len(seed_metrics) > 0:
        metrics_df = pd.DataFrame(seed_metrics)
        
        accuracy_mean = metrics_df["test_accuracy"].mean()
        accuracy_std = metrics_df["test_accuracy"].std()
        log_prob_mean = metrics_df["test_log_prob"].mean()
        log_prob_std = metrics_df["test_log_prob"].std()
        
        print("\n" + "=" * 50)
        print(f"Summary Statistics Across {num_seeds} Seeds:")
        print(f"Test Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
        print(f"Test LogPDF: {log_prob_mean:.4f} ± {log_prob_std:.4f}")
        print("=" * 50)
        
        with open(os.path.join(model_dir, "summary.txt"), "w") as f:
            f.write(f"Summary Statistics Across {num_seeds} Seeds:\n")
            f.write(f"Test Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}\n")
            f.write(f"Test LogPDF: {log_prob_mean:.4f} ± {log_prob_std:.4f}\n")
            f.write("=" * 50 + "\n")
        
    print(f"\n{'distributional' if cfg.model.is_distributional else 'regular'} reward model training complete!")
    print("Model saved to: ", model_path)

if __name__ == "__main__":
    main()
