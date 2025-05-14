import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from copy import deepcopy

# Import utility functions
from trajectory_utils import (
    load_tensordict,
    create_segments,
    sample_segment_pairs,
    compute_dtw_distance_matrix
)
from utils.wandb_utils import log_to_wandb, log_artifact

# Import shared models and utilities
from models import EnsembleRewardModel, RewardModel
from utils.dataset_utils import PreferenceDataset, bradley_terry_loss
from utils.active_learning_utils import (
    select_uncertain_pairs_comprehensive,
    create_initial_dataset
)
from utils.training_utils import train_model
from utils import evaluate_model_on_test_set
from utils.dataset_utils import create_data_loaders
from utils.seed_utils import set_seed


def find_similar_segments_dtw(query_idx, k, distance_matrix):
    """Find the k most similar segments to the query segment using a pre-computed distance matrix.
    
    Args:
        query_idx: Index of the query segment in the distance_matrix.
        k: Number of similar segments to find.
        distance_matrix: Pre-computed DTW distance matrix.
        
    Returns:
        list: Indices of the k most similar segments (relative to the distance_matrix).
    """
    if query_idx < 0 or query_idx >= distance_matrix.shape[0]:
        # Query index is out of bounds for the distance matrix (e.g., segment not in subsample)
        return []
        
    distances = distance_matrix[query_idx].copy()
    distances[query_idx] = float('inf')  # Exclude self
    
    # Ensure k is not larger than the number of available other segments
    k = min(k, len(distances) -1) 
    if k < 0: # handles case where distances has only 1 element after excluding self
        return []

    similar_indices = np.argsort(distances)[:k]
    return similar_indices.tolist()


def run_reward_analysis(model_path, data_path, output_dir, num_episodes=9, device=None, random_seed=42, wandb_run=None):
    """Run reward analysis on the trained model and log results to wandb.
    
    Args:
        model_path: Path to the trained reward model
        data_path: Path to the dataset
        output_dir: Directory to save the analysis plots
        num_episodes: Number of episodes to analyze
        device: Device to run the analysis on
        random_seed: Random seed for reproducibility
        wandb_run: Wandb run object for logging
    """
    # Import analyze_rewards here to avoid circular imports
    from analyze_rewards import analyze_rewards
    
    print("\n--- Running Reward Analysis ---")
    
    # Run the analysis
    analyze_rewards(
        data_path=data_path,
        model_path=model_path,
        output_dir=output_dir,
        num_episodes=num_episodes,
        device=device,
        random_seed=random_seed
    )
    
    # Log the analysis results to wandb
    if wandb_run is not None and wandb_run.run:
        reward_grid_path = os.path.join(output_dir, "reward_grid.png")
        if os.path.exists(reward_grid_path):
            print(f"Logging reward analysis grid to wandb")
            wandb_run.log({"reward_analysis/grid": wandb.Image(reward_grid_path)})
        else:
            print(f"Warning: Could not find reward grid image at {reward_grid_path}")
    
    print("Reward analysis completed successfully")


def train_final_reward_model(labeled_pairs, segment_indices, labeled_preferences, data_cpu, 
                           state_dim, action_dim, device, cfg, random_seed):
    """Train the final reward model on all labeled data.
    
    Args:
        labeled_pairs: List of labeled segment pairs
        segment_indices: List of segment indices
        labeled_preferences: List of preferences for labeled pairs
        data_cpu: Data dictionary containing observations and actions
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        device: Device to run training on
        cfg: Configuration object
        random_seed: Random seed for reproducibility
        
    Returns:
        final_model: Trained final model
        final_metrics: Evaluation metrics on test set
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    print("\n--- Training Final Model ---")
    print(f"Training on all {len(labeled_pairs)} labeled pairs")
    
    # Create dataset from all labeled pairs
    final_dataset = PreferenceDataset(data_cpu, labeled_pairs, segment_indices, labeled_preferences)    
    # Use the utility function to create data loaders with appropriate splits
    data_loaders = create_data_loaders(
        final_dataset,
        train_ratio=1.0,
        val_ratio=0,
        batch_size=min(cfg.training.batch_size, len(final_dataset)),
        num_workers=cfg.training.get('num_workers', 4),
        pin_memory=cfg.training.get('pin_memory', True),
        seed=random_seed
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    print(f"Using all {data_loaders['train_size']} labeled samples for training the final model (no validation split)")
    # Train final model
    final_model = RewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
    
    # Create descriptive output directory structure
    dataset_name = Path(cfg.data.data_path).stem
    uncertainty_method = cfg.active_learning.uncertainty_method
    fine_tune_str = "finetune" if cfg.active_learning.fine_tune else "scratch"
    
    # Add augmentation tag if enabled
    aug_str = "_aug" if cfg.dtw_augmentation.enabled else ""
    
    # Create descriptive subdirectory path
    output_subdir = f"{dataset_name}_active_{uncertainty_method}_init{cfg.active_learning.initial_size}_max{cfg.active_learning.max_queries}_batch{cfg.active_learning.batch_size}_{fine_tune_str}{aug_str}"
    model_dir = os.path.join(cfg.output.output_dir, output_subdir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Path for training curve
    training_curve_path = os.path.join(model_dir, "training_curve.png")
    
    # No ensemble training for final model
    final_model, train_losses, val_losses = train_model(
        final_model,
        train_loader,
        val_loader,
        device,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.model.lr,
        wandb=wandb if cfg.wandb.use_wandb else None,
        is_ensemble=False,
        output_path=training_curve_path
    )
    return final_model, None, train_losses, val_losses

@hydra.main(config_path="config", config_name="reward_model_sampling", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for active preference learning."""
    # Set random seed for reproducibility at the beginning
    random_seed = cfg.get('random_seed', 42)
    set_seed(random_seed)
    print(f"Global random seed set to {random_seed}")
    
    active_preference_learning(cfg)

def active_preference_learning(cfg):
    """Main function for active preference learning.
    
    Args:
        cfg: Configuration object from Hydra
    """
    print("\n" + "=" * 50)
    print("Active Preference Learning with Uncertainty Sampling")
    print("=" * 50)
    
    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Print a summary of active learning parameters
    print("\nActive Learning Parameters:")
    print(f"  Uncertainty method: {cfg.active_learning.uncertainty_method}")
    print(f"  Initial labeled pairs: {cfg.active_learning.initial_size}")
    print(f"  Maximum queries: {cfg.active_learning.max_queries}")
    print(f"  Batch size: {cfg.active_learning.batch_size}")
    print(f"  Fine-tuning: {'Enabled' if cfg.active_learning.fine_tune else 'Disabled'}")
    if cfg.active_learning.fine_tune:
        print(f"  Fine-tuning learning rate: {cfg.active_learning.fine_tune_lr}")
    print(f"  Ensemble models: {cfg.active_learning.num_models}")
    print(f"  Training epochs per iteration: {cfg.training.num_epochs}")
    
    # DTW Augmentation Parameters
    print("\nDTW Augmentation Parameters:")
    dtw_enabled = cfg.dtw_augmentation.enabled
    dtw_k_augment = cfg.dtw_augmentation.k_augment
    dtw_max_segments_for_matrix = cfg.dtw_augmentation.max_dtw_segments
    print(f"  Enabled: {dtw_enabled}")
    if dtw_enabled:
        print(f"  K augment (similar segments): {dtw_k_augment}")
        print(f"  Max segments for DTW matrix computation: {dtw_max_segments_for_matrix if dtw_max_segments_for_matrix is not None else 'All'}")

    # Get random seed from config (already set in main)
    random_seed = cfg.get('random_seed', 42)
    
    # Setup device
    if cfg.hardware.use_cpu:
        device = torch.device("cpu")
    else:
        cuda_device = f"cuda:{cfg.hardware.gpu}" if torch.cuda.is_available() else "cpu"
        device = torch.device(cuda_device)
    
    print(f"Using device: {device}")
    
    # Initialize wandb
    if cfg.wandb.use_wandb:
        # Generate experiment name based on data path
        dataset_name = Path(cfg.data.data_path).stem
        
        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"active_reward_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags + ["active_learning"],
            notes=cfg.wandb.notes
        )
        
        print(f"Wandb initialized: {wandb.run.name}")
    
    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {cfg.data.data_path}")
    data = load_tensordict(cfg.data.data_path)
    
    # Keep a CPU version of the data for indexing operations
    data_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    # Get observation and action dimensions
    observations = data_cpu["obs"] if "obs" in data_cpu else data_cpu["state"]
    actions = data_cpu["action"]
    rewards = data_cpu["reward"]
    
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create segments
    print(f"Creating segments of length {cfg.data.segment_length}...")
    # Pass CPU data to create_segments to ensure proper indexing
    segments, segment_indices = create_segments(
        data_cpu, 
        segment_length=cfg.data.segment_length,
        max_segments=cfg.data.num_segments
    )
    
    # Generate all possible segment pairs
    print(f"Generating {cfg.data.num_pairs} segment pairs...")
    # Pass CPU rewards to sample_segment_pairs
    all_segment_pairs, gt_preferences = sample_segment_pairs(
        segments, 
        segment_indices, 
        data_cpu["reward"], 
        n_pairs=cfg.data.num_pairs
    )
    
    # Compute DTW distance matrix if augmentation is enabled
    distance_matrix = None
    idx_mapping_dtw = None # Maps new_idx in subsampled_segments (used for DTW matrix) to old_idx in original segments
    original_to_dtw_idx = None # Maps old_idx to new_idx in subsampled_segments

    if dtw_enabled:
        print("\nComputing DTW distance matrix for augmentation...")
        # 'segments' is a list of dicts {'obs': tensor, 'action': tensor}
        # compute_dtw_distance_matrix from trajectory_utils handles this format.
        # It uses segment['obs'] for DTW.
        distance_matrix, idx_mapping_dtw = compute_dtw_distance_matrix(
            segments, # Original full list of segments
            max_segments=dtw_max_segments_for_matrix,
            random_seed=random_seed
        )
        if distance_matrix is not None:
            print(f"DTW distance matrix computed with shape: {distance_matrix.shape}")
            if idx_mapping_dtw:
                print(f"DTW matrix computed on a subset of {len(idx_mapping_dtw)} segments.")
                # Create reverse mapping: original segment index to DTW matrix index
                original_to_dtw_idx = {old_idx: new_idx for new_idx, old_idx in idx_mapping_dtw.items()}
        else:
            print("Warning: DTW distance matrix computation failed or yielded no result. Augmentation will be disabled.")
            dtw_enabled = False # Disable if matrix computation failed

    # Create test dataset from a separate set of pairs for consistent evaluation
    # We use 20% of all pairs for testing
    test_size = min(int(0.2 * len(all_segment_pairs)), len(all_segment_pairs))
    
    if test_size == 0:
        print("Warning: Not enough pairs for testing. Using all pairs for training.")
        test_indices = []
        test_pairs = []
        test_preferences = []
    else:
        # Use the same random seed for reproducibility
        test_indices = random.sample(range(len(all_segment_pairs)), test_size)
        test_pairs = [all_segment_pairs[i] for i in test_indices]
        test_preferences = [gt_preferences[i] for i in test_indices]
    
    # Create a set of test indices for faster lookup
    test_indices_set = set(test_indices)
    
    # Make sure test pairs are not in labeled or unlabeled sets
    # First create initial dataset excluding test pairs
    remaining_indices = [i for i in range(len(all_segment_pairs)) if i not in test_indices_set]
    remaining_pairs = [all_segment_pairs[i] for i in remaining_indices]
    remaining_preferences = [gt_preferences[i] for i in remaining_indices]
    
    # Now create the labeled/unlabeled split from the remaining pairs
    print(f"Creating initial dataset with {cfg.active_learning.initial_size} labeled pairs...")
    print(f"Using {len(remaining_pairs)} pairs after excluding test set")
    
    labeled_pairs, labeled_preferences, unlabeled_pairs, unlabeled_indices = create_initial_dataset(
        remaining_pairs,
        segment_indices,
        remaining_preferences,
        data_cpu,  # Use CPU data for indexing
        cfg.active_learning.initial_size
    )
    
    # Map unlabeled_indices back to original indices
    unlabeled_indices = [remaining_indices[i] for i in unlabeled_indices]
    
    # Create test dataset if we have test pairs
    test_dataset = PreferenceDataset(data_cpu, test_pairs, segment_indices, test_preferences)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    print(f"Created test dataset with {len(test_dataset)} pairs")

    print(f"Starting with {len(labeled_pairs)} labeled and {len(unlabeled_pairs)} unlabeled pairs")
    
    # Initialize metrics tracking
    metrics = {
        "num_labeled": [],
        "test_accuracy": [],
        "test_loss": [],
        "avg_logpdf": [],
        "iterations": []
    }
    
    # Main active learning loop
    iteration = 0
    total_labeled = len(labeled_pairs)
    max_queries = cfg.active_learning.max_queries
    
    # Keep track of the previous ensemble for fine-tuning
    prev_ensemble = None

    # Create a mapping from pair to original index for preference lookup
    pair_to_original_idx = {}
    for i, pair_val in enumerate(all_segment_pairs): # Renamed pair to pair_val to avoid conflict
        pair_to_original_idx[tuple(pair_val)] = i
    
    while total_labeled < max_queries and len(unlabeled_pairs) > 0:
        
        iteration += 1
        print(f"\n--- Active Learning Iteration {iteration} ---")
        print(f"Currently have {total_labeled} labeled pairs")
        
        # Train ensemble model on current labeled dataset
        print("Training ensemble model...")
        
        # Create dataset for training the ensemble
        ensemble_dataset = PreferenceDataset(data_cpu, labeled_pairs, segment_indices, labeled_preferences)
        
        # Use utility function to create data loaders with all data for training (no validation split)
        data_loaders = create_data_loaders(
            ensemble_dataset,
            train_ratio=1.0,  # Use all data for training
            val_ratio=0.0,    # No validation set
            batch_size=min(cfg.training.batch_size, len(ensemble_dataset)),
            num_workers=cfg.training.get('num_workers', 4),
            pin_memory=cfg.training.get('pin_memory', True),
            seed=random_seed
        )
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        print(f"Using all {data_loaders['train_size']} labeled samples for training (no validation split)")
        
        if cfg.active_learning.fine_tune and prev_ensemble is not None:
            print("Fine-tuning from previous ensemble model")
            ensemble = prev_ensemble
            # Ensure the ensemble has the right number of models
            if len(ensemble.models) != cfg.active_learning.num_models:
                print(f"Warning: Previous ensemble has {len(ensemble.models)} models, but {cfg.active_learning.num_models} requested. "
                     f"Creating new ensemble.")
                ensemble = EnsembleRewardModel(state_dim, action_dim, cfg.model.hidden_dims, cfg.active_learning.num_models)
            lr = cfg.active_learning.fine_tune_lr  # Use lower learning rate for fine-tuning
        else:
            print("Training new ensemble model from scratch")
            ensemble = EnsembleRewardModel(state_dim, action_dim, cfg.model.hidden_dims, cfg.active_learning.num_models)
            lr = cfg.model.lr  # Use standard learning rate for training from scratch
        
        # Train the ensemble using the unified training function
        ensemble, _, _ = train_model(
            ensemble,
            train_loader,
            val_loader,
            device,
            num_epochs=cfg.training.num_epochs,
            lr=lr,
            wandb=None,     # Don't log ensemble training to wandb
            is_ensemble=True
        )
        
        # Store the current ensemble for potential fine-tuning in the next iteration
        if cfg.active_learning.fine_tune:
            prev_ensemble = deepcopy(ensemble)
        
        # Move ensemble to device
        ensemble = ensemble.to(device)
        
        # Evaluate ensemble on test set
        print("Evaluating ensemble on test set...")
        
        # Use the first model in the ensemble for evaluation
        test_metrics = evaluate_model_on_test_set(ensemble.models[0], test_loader, device)
        
        # Log metrics
        metrics["num_labeled"].append(total_labeled)
        metrics["test_accuracy"].append(test_metrics["test_accuracy"])
        metrics["test_loss"].append(test_metrics["test_loss"])
        metrics["avg_logpdf"].append(test_metrics["avg_logpdf"])
        metrics["iterations"].append(iteration)
        
        # Log to wandb
        if cfg.wandb.use_wandb:
            log_to_wandb({
                "num_labeled": total_labeled,
                "test_accuracy": test_metrics["test_accuracy"],
                "test_loss": test_metrics["test_loss"],
                "avg_logpdf": test_metrics["avg_logpdf"],
                "active_iteration": iteration
            })
        
        # Select next batch of uncertain pairs
        batch_size = min(cfg.active_learning.batch_size, max_queries - total_labeled)
        if batch_size <= 0 or len(unlabeled_pairs) == 0:
            print("Reached maximum number of queries or no more unlabeled data")
            break
        
        print(f"Computing uncertainty scores using {cfg.active_learning.uncertainty_method} method...")

        # Use the unified function for uncertainty-based selection
        # Instead of passing all unlabeled pairs, we'll directly use them
        ranked_pairs = select_uncertain_pairs_comprehensive(
            ensemble,
            segments,
            segment_indices,
            data_cpu,
            device,
            uncertainty_method=cfg.active_learning.uncertainty_method,
            max_pairs=batch_size,
            use_random_candidate_sampling=cfg.active_learning.use_random_sampling, 
            n_candidates=cfg.active_learning.n_candidates,
            candidate_pairs=unlabeled_pairs  # Pass unlabeled_pairs directly
        )

        # Extract the selected pairs
        selected_pairs = ranked_pairs[:batch_size]
        
        # Check if we were able to select any pairs
        if len(selected_pairs) == 0:
            print("No pairs were selected. Ending active learning loop.")
            break
        
        print(f"Selected {len(selected_pairs)} pairs based on uncertainty")
        print(f"Selected pairs: {selected_pairs}")
        
        # Get ground truth preferences for selected pairs
        # In a real system, this would be where we query the human
        selected_preferences = []
        valid_selected_pairs = []
        for pair_val_loop in selected_pairs: # Renamed pair to pair_val_loop
            pair_tuple = tuple(pair_val_loop)
            if pair_tuple in pair_to_original_idx:
                selected_preferences.append(gt_preferences[pair_to_original_idx[pair_tuple]])
                valid_selected_pairs.append(list(pair_val_loop)) # Ensure it's a list of lists
            else:
                print(f"Warning: Pair {pair_val_loop} not found in original pairs mapping")
        
        # Update selected_pairs to only include valid ones
        selected_pairs = valid_selected_pairs
        
        if len(selected_pairs) == 0:
            print("No valid pairs selected. Ending active learning loop.")
            break
            
        # --- DTW Augmentation Start ---
        current_iter_augmented_pairs = []
        current_iter_augmented_preferences = []

        if dtw_enabled and distance_matrix is not None and len(selected_pairs) > 0:
            print(f"Augmenting {len(selected_pairs)} selected pairs using DTW (k={dtw_k_augment})...")
            
            for pair_original_indices, preference_val in zip(selected_pairs, selected_preferences):
                original_i, original_j = pair_original_indices # These are original segment indices
                
                dtw_i, dtw_j = -1, -1
                can_augment_this_pair = True

                if idx_mapping_dtw: # Using a subsampled matrix
                    if original_to_dtw_idx is None: # Should not happen if idx_mapping_dtw exists
                         print("Error: original_to_dtw_idx not created despite idx_mapping_dtw. Skipping augmentation.")
                         can_augment_this_pair = False
                    else:
                        dtw_i = original_to_dtw_idx.get(original_i, -1)
                        dtw_j = original_to_dtw_idx.get(original_j, -1)
                        if dtw_i == -1 or dtw_j == -1:
                            # print(f"Warning: Pair ({original_i}, {original_j}) involves segments not in DTW subset. Skipping augmentation for this pair.")
                            can_augment_this_pair = False
                else: # No subsampling, direct mapping for distance_matrix indices
                    dtw_i, dtw_j = original_i, original_j

                if not can_augment_this_pair:
                    continue
                
                # Ensure dtw_i and dtw_j are valid indices for distance_matrix
                if not (0 <= dtw_i < distance_matrix.shape[0] and 0 <= dtw_j < distance_matrix.shape[0]):
                    # print(f"Warning: Invalid DTW indices {dtw_i}, {dtw_j} for pair ({original_i}, {original_j}) derived. Skipping augmentation.")
                    continue
                
                # preference_val == 1 means original_i is preferred over original_j
                if preference_val == 1:
                    similar_to_dtw_i_indices = find_similar_segments_dtw(dtw_i, dtw_k_augment, distance_matrix)
                    for sim_dtw_idx in similar_to_dtw_i_indices:
                        sim_original_idx = idx_mapping_dtw[sim_dtw_idx] if idx_mapping_dtw else sim_dtw_idx
                        if sim_original_idx != original_j and sim_original_idx != original_i : # Avoid (x,x) and already queried original_j
                            current_iter_augmented_pairs.append([sim_original_idx, original_j])
                            current_iter_augmented_preferences.append(1)

                    similar_to_dtw_j_indices = find_similar_segments_dtw(dtw_j, dtw_k_augment, distance_matrix)
                    for sim_dtw_idx in similar_to_dtw_j_indices:
                        sim_original_idx = idx_mapping_dtw[sim_dtw_idx] if idx_mapping_dtw else sim_dtw_idx
                        if sim_original_idx != original_i and sim_original_idx != original_j: # Avoid (x,x) and already queried original_i
                            current_iter_augmented_pairs.append([original_i, sim_original_idx])
                            current_iter_augmented_preferences.append(1)
                
                elif preference_val == 2: # original_j is preferred over original_i
                    similar_to_dtw_j_indices = find_similar_segments_dtw(dtw_j, dtw_k_augment, distance_matrix)
                    for sim_dtw_idx in similar_to_dtw_j_indices:
                        sim_original_idx = idx_mapping_dtw[sim_dtw_idx] if idx_mapping_dtw else sim_dtw_idx
                        if sim_original_idx != original_i and sim_original_idx != original_j:
                            current_iter_augmented_pairs.append([original_i, sim_original_idx])
                            current_iter_augmented_preferences.append(2)

                    similar_to_dtw_i_indices = find_similar_segments_dtw(dtw_i, dtw_k_augment, distance_matrix)
                    for sim_dtw_idx in similar_to_dtw_i_indices:
                        sim_original_idx = idx_mapping_dtw[sim_dtw_idx] if idx_mapping_dtw else sim_dtw_idx
                        if sim_original_idx != original_j and sim_original_idx != original_i:
                            current_iter_augmented_pairs.append([sim_original_idx, original_j])
                            current_iter_augmented_preferences.append(2)
        # --- DTW Augmentation End ---

        # Combine human-queried and augmented pairs, then add uniquely to labeled_pairs
        all_new_pairs_this_iteration = []
        all_new_preferences_this_iteration = []

        all_new_pairs_this_iteration.extend(selected_pairs) # Human-queried
        all_new_preferences_this_iteration.extend(selected_preferences)

        if current_iter_augmented_pairs:
            all_new_pairs_this_iteration.extend(current_iter_augmented_pairs) # Augmented
            all_new_preferences_this_iteration.extend(current_iter_augmented_preferences)
            print(f"Generated {len(current_iter_augmented_pairs)} raw augmented pairs this iteration.")

        
        # Filter all new pairs to ensure they are unique and not already labeled
        uniquely_added_to_labeled_pairs = []
        uniquely_added_to_labeled_preferences = []
        
        # Set of (sorted_tuple_of_pair_indices) already in the main labeled_pairs list
        current_global_labeled_pairs_tuples = set(tuple(sorted(p)) for p in labeled_pairs)

        for new_pair, new_pref in zip(all_new_pairs_this_iteration, all_new_preferences_this_iteration):
            if new_pair[0] == new_pair[1]: # Skip self-loops
                continue
            
            pair_tuple_sorted = tuple(sorted(new_pair))
            if pair_tuple_sorted not in current_global_labeled_pairs_tuples:
                uniquely_added_to_labeled_pairs.append(new_pair)
                uniquely_added_to_labeled_preferences.append(new_pref)
                current_global_labeled_pairs_tuples.add(pair_tuple_sorted) # Add to set to ensure uniqueness within this batch

        if uniquely_added_to_labeled_pairs:
            labeled_pairs.extend(uniquely_added_to_labeled_pairs)
            labeled_preferences.extend(uniquely_added_to_labeled_preferences)
            
            # Count how many were original human queries vs augmented, among those *actually added*
            num_human_added_this_iter = 0
            selected_pairs_tuples = set(tuple(sorted(p)) for p in selected_pairs)
            for added_p in uniquely_added_to_labeled_pairs:
                if tuple(sorted(added_p)) in selected_pairs_tuples:
                     num_human_added_this_iter +=1
            num_aug_added_this_iter = len(uniquely_added_to_labeled_pairs) - num_human_added_this_iter
            print(f"Added {num_human_added_this_iter} unique human-queried and {num_aug_added_this_iter} unique augmented pairs to labeled set.")

        # total_labeled tracks number of *human queries attempted* in this iteration
        total_labeled += len(selected_pairs) 
        
        # Remove selected pairs (human-queried ones) and any newly labeled (augmented) pairs from the unlabeled set
        # Rebuild the set of all labeled pairs for efficient filtering of unlabeled_pairs
        final_labeled_tuples_for_unlabeled_filtering = set(tuple(sorted(p)) for p in labeled_pairs)
        
        new_unlabeled_pairs_list = []
        for p_unlabeled in unlabeled_pairs:
            if tuple(sorted(p_unlabeled)) not in final_labeled_tuples_for_unlabeled_filtering:
                new_unlabeled_pairs_list.append(p_unlabeled)
        unlabeled_pairs = new_unlabeled_pairs_list
        
        print(f"Now have {len(labeled_pairs)} labeled (human + augmented) and {len(unlabeled_pairs)} unlabeled pairs")
    
    # Train final model on all labeled data
    final_model, final_metrics, train_losses, val_losses = train_final_reward_model(
        labeled_pairs, 
        segment_indices, 
        labeled_preferences, 
        data_cpu, 
        state_dim, 
        action_dim, 
        device, 
        cfg, 
        random_seed
    )
    
    # Also evaluate on the consistent test set if available
    if test_loader is not None:
        consistent_metrics = evaluate_model_on_test_set(final_model, test_loader, device)
        print(f"Final model accuracy on consistent test set: {consistent_metrics['test_accuracy']:.4f}")
    else:
        consistent_metrics = {"test_accuracy": None, "test_loss": None}
        print("No consistent test set available for evaluation")
    
    # Log final metrics
    if cfg.wandb.use_wandb:
        log_to_wandb({
            "final_test_accuracy": final_metrics["test_accuracy"],
            "final_test_loss": final_metrics["test_loss"],
            "consistent_test_accuracy": consistent_metrics["test_accuracy"],
            "consistent_test_loss": consistent_metrics["test_loss"],
            "total_queries": total_labeled,
            "total_iterations": iteration
        })
    
    # Create descriptive output directory structure
    dataset_name = Path(cfg.data.data_path).stem
    uncertainty_method = cfg.active_learning.uncertainty_method
    fine_tune_str = "finetune" if cfg.active_learning.fine_tune else "scratch"
    
    # Create descriptive subdirectory
    aug_str = "_aug" if cfg.dtw_augmentation.enabled else ""
    output_subdir = f"{dataset_name}_active_{uncertainty_method}_init{cfg.active_learning.initial_size}_max{cfg.active_learning.max_queries}_batch{cfg.active_learning.batch_size}_{fine_tune_str}{aug_str}"
    
    # Create model directory
    model_dir = os.path.join(cfg.output.output_dir, output_subdir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Plot learning curve with three subplots: accuracy, loss, and logpdf
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Plot test accuracy
    ax1.plot(metrics["num_labeled"], metrics["test_accuracy"], marker='o', color='blue')
    ax1.set_xlabel("Number of Labeled Pairs")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Test Accuracy vs Labeled Pairs ({cfg.active_learning.uncertainty_method})")
    ax1.grid(True)
    
    # Plot test loss (Bradley-Terry)
    ax2.plot(metrics["num_labeled"], metrics["test_loss"], marker='o', color='red')
    ax2.set_xlabel("Number of Labeled Pairs")
    ax2.set_ylabel("Bradley-Terry Loss (BCE)")
    ax2.set_title(f"Preference Learning Loss vs Labeled Pairs ({cfg.active_learning.uncertainty_method})")
    ax2.grid(True)
    
    # Plot test logpdf (log probability density)
    ax3.plot(metrics["num_labeled"], metrics["avg_logpdf"], marker='o', color='green')
    ax3.set_xlabel("Number of Labeled Pairs") 
    ax3.set_ylabel("Average Log Probability")
    ax3.set_title(f"Avg Log Probability vs Labeled Pairs ({cfg.active_learning.uncertainty_method})")
    ax3.grid(True)
   
    # Add a global title
    fig.suptitle(f"Active Learning Performance Metrics", fontsize=16)
    
    # Save plot in the model directory
    learning_curve_path = os.path.join(model_dir, "learning_curve.png")
    plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
    
    if cfg.wandb.use_wandb:
        wandb.log({"learning_curve": wandb.Image(learning_curve_path)})
    
    # Save final model in model directory
    model_path = os.path.join(model_dir, "final_model.pt")
    torch.save(final_model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    
    # Save configuration
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    print(f"\nActive learning completed. Final model saved to {model_path}")
    
    # Run reward analysis on the final model
    analysis_dir = os.path.join(model_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Check if reward analysis is enabled (default to True)
    run_analysis = cfg.get('run_reward_analysis', True)
    
    if run_analysis:
        try:
            run_reward_analysis(
                model_path=model_path,
                data_path=cfg.data.data_path,
                output_dir=analysis_dir,
                num_episodes=cfg.get('analysis_episodes', 9),
                device=device,
                random_seed=random_seed,
                wandb_run=wandb if cfg.wandb.use_wandb else None
            )
        except Exception as e:
            print(f"Warning: Error during reward analysis: {e}")
    
    # Log artifacts to wandb
    if cfg.wandb.use_wandb:
        log_artifact(
            model_path,
            artifact_type="model",
            metadata={
                "method": cfg.active_learning.uncertainty_method,
                "fine_tune": cfg.active_learning.fine_tune,
                "initial_size": cfg.active_learning.initial_size,
                "max_queries": cfg.active_learning.max_queries,
                "batch_size": cfg.active_learning.batch_size,
                "num_queries": total_labeled,
                "final_accuracy": final_metrics["test_accuracy"],
                "consistent_accuracy": consistent_metrics["test_accuracy"]
            }
        )
        
        # Finish wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 