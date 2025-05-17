import gc
import itertools
import os
import pickle
import random
import time
from copy import deepcopy
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb

from models import EnsembleRewardModel, RewardModel
from utils.active_query_selection import (
    select_active_pref_query,
)
from utils.data import (
    get_gt_preferences,
    load_tensordict,
    segment_episodes,
    process_data_trajectories,
)
from utils.dataset import PreferenceDataset, create_data_loaders
from utils.seed import set_seed
from utils.training import evaluate_model_on_test_set, train_model
from utils.wandb import log_to_wandb
from utils.analyze_rewards import analyze_rewards 


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


def train_final_reward_model(labeled_pairs, segment_indices, labeled_preferences, data, 
                           state_dim, action_dim, device, cfg, random_seed, wandb_run):
    """Train the final reward model on all labeled data.
    
    Args:
        labeled_pairs: List of labeled segment pairs
        segment_indices: List of segment indices
        labeled_preferences: List of preferences for labeled pairs
        data: Data dictionary containing observations and actions
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
    final_dataset = PreferenceDataset(data, labeled_pairs, segment_indices, labeled_preferences)
    
    # Adjust batch size and workers for small datasets
    effective_batch_size = min(cfg.training.batch_size, len(final_dataset))
    if effective_batch_size <= 1:
        effective_batch_size = 1
        
    effective_workers = cfg.training.get('num_workers', 4)
    if effective_batch_size < 4 and effective_workers > 0:
        effective_workers = 0
        print("Reducing worker count to 0 for small batch size")
    
    # Use the utility function to create data loaders
    data_loaders = create_data_loaders(
        final_dataset,
        train_ratio=0.8,
        val_ratio=0.2,
        batch_size=effective_batch_size,
        num_workers=effective_workers,
        pin_memory=True,
        seed=random_seed
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    print(f"Using all {data_loaders['train_size']} labeled samples for training the final model (no validation split)")
    # Train final model
    final_model = RewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
    
    # Get model directory name from config
    model_dir_name = cfg.output.model_dir_name
    
    # Create the model directory
    model_dir = os.path.join(cfg.output.output_dir, model_dir_name)
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
        wandb_run=wandb_run,
        is_ensemble=False,
        output_path=training_curve_path
    )
    return final_model, None, train_losses, val_losses


def active_preference_learning(cfg, dataset_name=None):
    """Main function for active preference learning.
    
    Args:
        cfg: Configuration object from Hydra
        dataset_name: Name of the dataset (extracted from path)
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
    print(f"  Maximum queries: {cfg.active_learning.max_queries}")
    print(f"  Number of queries per iteration: {cfg.active_learning.num_queries_per_iteration}")
    print(f"  Fine-tuning: {'Enabled' if cfg.active_learning.fine_tune else 'Disabled'}")
    if cfg.active_learning.fine_tune:
        print(f"  Fine-tuning learning rate: {cfg.active_learning.fine_tune_lr}")
    print(f"  Ensemble models: {cfg.active_learning.num_models}")
    print(f"  Training epochs per iteration: {cfg.training.num_epochs}")
    
    # DTW Augmentation Parameters
    print("\nDTW Augmentation Parameters:")
    dtw_enabled = cfg.dtw_augmentation.enabled
    dtw_k_augment = cfg.dtw_augmentation.k_augment
    print(f"  DTW Enabled: {dtw_enabled}")
    if dtw_enabled:
        print(f"  K augment (similar segments): {dtw_k_augment}")

    # Get random seed from config (already set in main)
    random_seed = cfg.get('random_seed', 42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if cfg.wandb.use_wandb:
        # Use dataset_name passed from main, or extract if not provided
        if dataset_name is None:
            dataset_name = Path(cfg.data.data_path).stem
            
        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"active_reward_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags + ["active_learning"],
            notes=cfg.wandb.notes
        )
        
        print(f"Wandb initialized: {wandb_run.name}")
    else:
        wandb_run = None
    
    
    # Load data
    print(f"Loading data from {cfg.data.data_path}")
    data = load_tensordict(cfg.data.data_path)
    data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    # Get observation and action dimensions
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    rewards = data["reward"]
    
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")

    # Load pre-computed DTW distance matrix and segment indices if augmentation is enabled
    distance_matrix = None
    segment_indices = None

    if dtw_enabled:
        print("\nLoading pre-computed DTW distance matrix and segment indices...")
        dtw_matrix_file = Path(cfg.data.data_path).parent / f"dtw_matrix_{cfg.data.segment_length}.pkl"

        if not os.path.exists(dtw_matrix_file):
            print(f"Warning: DTW matrix file not found at {dtw_matrix_file}")
            assert False, "DTW matrix file not found. Please run preprocess_dtw_matrix.py"
        else:
            distance_matrix, segment_indices = pickle.load(open(dtw_matrix_file, "rb"))
            print(f"Successfully loaded DTW distance matrix with shape: {distance_matrix.shape}")
    else:
        segments, segment_indices = segment_episodes(data, cfg.data.segment_length)

    num_segments = len(segment_indices)

    # Find all possible segment pairs (num_segments choose 2) and sample data.subsamples from them
    all_segment_pairs = list(itertools.combinations(range(num_segments), 2))
    all_segment_pairs = random.sample(all_segment_pairs, cfg.data.subsamples)
    print(f"Sampled {len(all_segment_pairs)} pairs from {len(all_segment_pairs)} total pairs")

    # Test set
    test_indices = random.sample(range(len(all_segment_pairs)), cfg.data.num_test_pairs)
    test_pairs = [all_segment_pairs[i] for i in test_indices]

    # Compute test preferences using the ground truth function
    test_preferences = get_gt_preferences(data, segment_indices, test_pairs)
    test_dataset = PreferenceDataset(data, test_pairs, segment_indices, test_preferences)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    print(f"Created test dataset with {len(test_dataset)} pairs")

    # Make sure test pairs are not in labeled or unlabeled sets
    all_indices = list(range(len(all_segment_pairs)))
    remaining_indices = list(set(all_indices) - set(test_indices))
    unlabeled_pairs = [all_segment_pairs[i] for i in remaining_indices]
    print(f"Using {len(unlabeled_pairs)} unlabeled pairs after excluding test set")

    # Initialize metrics tracking
    metrics = {
        "num_labeled": [],
        "test_accuracy": [],
        "test_loss": [],
        "avg_logpdf": [],
        "iterations": [],
        "val_losses": []  # Track validation losses across iterations
    }
    
    # Create a mapping from pair to original index for preference lookup
    pair_to_original_idx = {}
    for i, pair_val in enumerate(all_segment_pairs): 
        pair_to_original_idx[tuple(pair_val)] = i

    # Get model directory name from config for checkpoints
    model_dir_name = cfg.output.model_dir_name
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    model_dir = os.path.join(cfg.output.output_dir, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    rm_dir = os.path.join(model_dir, "rm_training")
    os.makedirs(rm_dir, exist_ok=True)

    # Main active learning loop
    iteration = 0
    total_labeled = 0
    max_queries = cfg.active_learning.max_queries
    
    # Keep track of the previous ensemble for fine-tuning
    prev_ensemble = None
    labeled_pairs = []
    labeled_preferences = []

    # If DTW augmentation is enabled, only sample from pairs that are in the DTW matrix
    candidate_pairs = unlabeled_pairs
    if dtw_enabled and distance_matrix is not None:
        # Filter unlabeled pairs to only include those where both segments are in the DTW matrix
        dtw_candidate_pairs = []

        for pair in unlabeled_pairs:
            dtw_candidate_pairs.append(pair)
        
        if len(dtw_candidate_pairs) > 0:
            print(f"Using {len(dtw_candidate_pairs)} pairs from DTW matrix for uncertainty sampling")
            candidate_pairs = dtw_candidate_pairs
        else:
            print("Warning: No pairs found in DTW matrix. Using all unlabeled pairs.")

    while total_labeled < max_queries and len(unlabeled_pairs) > 0:
        
        iteration += 1
        print(f"\n--- Active Learning Iteration {iteration} ---")
        print(f"Currently have {total_labeled} labeled pairs")
        
        print(f"Computing uncertainty scores using {cfg.active_learning.uncertainty_method} method...")
        if iteration == 1:
            rand_idx = random.randint(0, len(candidate_pairs) - 1)
            selected_query_pair = [candidate_pairs[rand_idx]]
        else:
            # Use the unified function for uncertainty-based selection
            selected_query_pair = select_active_pref_query(
                ensemble,
                segment_indices,
                data,
                uncertainty_method=cfg.active_learning.uncertainty_method,
                max_pairs=1,
                candidate_pairs=candidate_pairs
            )
        
        # Check if we were able to select any pairs
        if len(selected_query_pair) == 0:
            print("No query pair was selected. Ending active learning loop.")
            break
        
        selected_query_pref = get_gt_preferences(data, segment_indices, selected_query_pair)

        # --- DTW Augmentation Start ---
        current_iter_augmented_pairs = []
        current_iter_augmented_preferences = []

        if dtw_enabled and distance_matrix is not None and len(selected_query_pair) > 0:
            print(f"Augmenting {len(selected_query_pair)} selected pairs using DTW (k={dtw_k_augment})...")
            
            for qp, preference_val in zip(selected_query_pair, selected_query_pref):
                i, j = qp # These are original segment indices
                
                dtw_i, dtw_j = i, j
                
                # preference_val == 1 means segment i is preferred over segment j
                if preference_val == 1:
                    similar_to_dtw_i_indices = find_similar_segments_dtw(dtw_i, dtw_k_augment, distance_matrix)
                    for sim_idx in similar_to_dtw_i_indices:
                        if sim_idx != j and sim_idx != i:  # Avoid (x,x) and already queried j
                            current_iter_augmented_pairs.append([sim_idx, j])
                            current_iter_augmented_preferences.append(1)

                    similar_to_dtw_j_indices = find_similar_segments_dtw(dtw_j, dtw_k_augment, distance_matrix)
                    for sim_idx in similar_to_dtw_j_indices:
                        if sim_idx != i and sim_idx != j:  # Avoid (x,x) and already queried i
                            current_iter_augmented_pairs.append([i, sim_idx])
                            current_iter_augmented_preferences.append(1)

                elif preference_val == 2:  # original_j is preferred over original_i
                    similar_to_dtw_j_indices = find_similar_segments_dtw(dtw_j, dtw_k_augment, distance_matrix)
                    for sim_idx in similar_to_dtw_j_indices:
                        if sim_idx != i and sim_idx != j:
                            current_iter_augmented_pairs.append([i, sim_idx])
                            current_iter_augmented_preferences.append(2)

                    similar_to_dtw_i_indices = find_similar_segments_dtw(dtw_i, dtw_k_augment, distance_matrix)
                    for sim_idx in similar_to_dtw_i_indices:
                        if sim_idx != j and sim_idx != i:
                            current_iter_augmented_pairs.append([sim_idx, j])
                            current_iter_augmented_preferences.append(2)

            labeled_pairs.extend(current_iter_augmented_pairs)
            labeled_preferences.extend(current_iter_augmented_preferences)
        # --- DTW Augmentation End ---

        # Combine human-queried and augmented pairs, then add uniquely to labeled_pairs
        all_new_pairs_this_iteration = []
        all_new_preferences_this_iteration = []

        all_new_pairs_this_iteration.extend(selected_query_pair) # Human-queried
        all_new_preferences_this_iteration.extend(selected_query_pref)

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
            selected_pairs_tuples = set(tuple(sorted(p)) for p in selected_query_pair)
            for added_p in uniquely_added_to_labeled_pairs:
                if tuple(sorted(added_p)) in selected_pairs_tuples:
                     num_human_added_this_iter +=1
            num_aug_added_this_iter = len(uniquely_added_to_labeled_pairs) - num_human_added_this_iter
            print(f"Added {num_human_added_this_iter} unique human-queried and {num_aug_added_this_iter} unique augmented pairs to labeled set.")

        # total_labeled tracks number of *human queries attempted* in this iteration
        total_labeled += len(selected_query_pair) 
        
        # Remove selected pairs (human-queried ones) and any newly labeled (augmented) pairs from the unlabeled set
        # Rebuild the set of all labeled pairs for efficient filtering of unlabeled_pairs
        final_labeled_tuples_for_unlabeled_filtering = set(tuple(sorted(p)) for p in labeled_pairs)
        
        new_unlabeled_pairs_list = []
        for p_unlabeled in unlabeled_pairs:
            if tuple(sorted(p_unlabeled)) not in final_labeled_tuples_for_unlabeled_filtering:
                new_unlabeled_pairs_list.append(p_unlabeled)
        unlabeled_pairs = new_unlabeled_pairs_list
        
        print(f"Now have {len(labeled_pairs)} labeled (human + augmented) and {len(unlabeled_pairs)} unlabeled pairs")
        
        # Create dataset for training the ensemble
        ensemble_dataset = PreferenceDataset(data, labeled_pairs, segment_indices, labeled_preferences)

        # Use utility function to create data loaders with all data for training
        data_loaders = create_data_loaders(
            ensemble_dataset,
            train_ratio=1.0,  # Use all data for training
            val_ratio=0.0,    # No validation set
            batch_size=cfg.training.batch_size,
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
        ensemble, train_loss, val_loss = train_model(
            ensemble,
            train_loader,
            val_loader,
            device,
            num_epochs=cfg.training.num_epochs,
            lr=lr,
            is_ensemble=True,
            output_path=os.path.join(model_dir, "rm_training", f"reward_model_training_{iteration}.png")
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
                "active_iteration": iteration,
                "val_loss": metrics["val_losses"][-1] if metrics["val_losses"][-1] is not None else 0
            }, wandb_run=wandb_run)

        # Track test loss over number of labeled queries for plotting
        if "test_loss_curve" not in metrics:
            metrics["test_loss_curve"] = []
            metrics["num_labeled_curve"] = []
        metrics["test_loss_curve"].append(test_metrics["test_loss"])
        metrics["num_labeled_curve"].append(total_labeled)

        if iteration % cfg.training.save_model_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
            torch.save(ensemble.models[0].state_dict(), checkpoint_path)
            print(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")
        
        # Select next batch of uncertain pairs
        batch_size = min(cfg.active_learning.num_queries_per_iteration, max_queries - total_labeled)
        if batch_size <= 0 or len(unlabeled_pairs) == 0:
            print("Reached maximum number of queries or no more unlabeled data")
            break
            
    # Train final model on all labeled data
    final_model, final_metrics, train_losses, val_losses = train_final_reward_model(
        labeled_pairs, 
        segment_indices, 
        labeled_preferences, 
        data, 
        state_dim, 
        action_dim, 
        device, 
        cfg, 
        random_seed,
        wandb_run
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
        }, wandb_run=wandb_run)
    
    # Get model directory name from config
    model_dir_name = cfg.output.model_dir_name
    
    # Create the model directory
    model_dir = os.path.join(cfg.output.output_dir, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a 2x2 grid of plots instead of 1x4
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    
    # Common styling for all plots
    dot_size = 8  # Larger dots
    line_color = 'blue'  # Consistent color for all plots
    
    # Plot test accuracy (top-left)
    ax1 = axs[0, 0]
    ax1.plot(metrics["num_labeled"], metrics["test_accuracy"], marker='o', markersize=dot_size, color=line_color)
    ax1.set_xlabel("Number of Labeled Pairs")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Test Accuracy vs Labeled Pairs ({cfg.active_learning.uncertainty_method})")
    ax1.grid(True, alpha=0.3)
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot test loss (Bradley-Terry) (top-right)
    ax2 = axs[0, 1]
    ax2.plot(metrics["num_labeled"], metrics["test_loss"], marker='o', markersize=dot_size, color=line_color)
    ax2.set_xlabel("Number of Labeled Pairs")
    ax2.set_ylabel("Bradley-Terry Loss (BCE)")
    ax2.set_title(f"Preference Learning Loss vs Labeled Pairs ({cfg.active_learning.uncertainty_method})")
    ax2.grid(True, alpha=0.3)
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Save plot in the model directory
    learning_curve_path = os.path.join(model_dir, "active_learning_metrics.png")
    plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
    
    if wandb_run:
        wandb_run.log({"results": wandb.Image(learning_curve_path)})
    
    # Save final model in model directory
    model_path = os.path.join(model_dir, "final_model.pt")
    torch.save(final_model.state_dict(), model_path)
    
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    print(f"\nActive learning completed. Final model saved to {model_path}")
    

    print("\n--- Running Reward Analysis ---")
    episodes = process_data_trajectories(cfg.data.data_path)
    analyze_rewards(
        model=final_model,
        episodes=episodes,
        output_dir=model_dir,
        num_episodes=9,
        wandb_run=wandb_run
    )

@hydra.main(config_path="config", config_name="reward_model_active", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for active preference learning."""
    # Set random seed for reproducibility at the beginning
    set_seed(cfg.random_seed)
    print(f"Global random seed set to {cfg.random_seed}")
    
    # Get the dataset name
    dataset_path = Path(cfg.data.data_path)
    dataset_name = dataset_path.stem
    
    if 'robomimic' in str(dataset_path):
        # Get the task name (like 'can') from the path
        task_name = dataset_path.parent.name
        dataset_name = f"{task_name}_{dataset_name}"
    else:
        # Get parent directory name if needed for other datasets
        parent_dir = dataset_path.parent.name
        if parent_dir and parent_dir not in ['datasets', 'data']:
            dataset_name = f"{parent_dir}_{dataset_name}"

    print(f"Dataset name: {dataset_name}")
    
    # Replace only the dataset name placeholder in the template strings
    if hasattr(cfg.output, "model_dir_name"):
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace("DATASET_NAME", dataset_name)
    
    if hasattr(cfg.output, "artifact_name"):
        cfg.output.artifact_name = cfg.output.artifact_name.replace("DATASET_NAME", dataset_name)
    
    active_preference_learning(cfg)
    
if __name__ == "__main__":
    main() 