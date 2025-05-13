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
    sample_segment_pairs
)
from utils.wandb_utils import log_to_wandb, log_artifact

# Import shared models and utilities
from models import SegmentRewardModel, EnsembleRewardModel
from utils import (
    PreferenceDataset, 
    bradley_terry_loss,
    evaluate_model_on_test_set,
    compute_uncertainty_scores,
    select_uncertain_pairs, 
    select_uncertain_pairs_comprehensive,
    get_ground_truth_preferences,
    create_initial_dataset,
    train_ensemble_model
)

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
    print(f"  Training epochs per iteration: {cfg.active_learning.train_epochs}")
    
    # Set random seed for reproducibility
    random_seed = cfg.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
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
    
    # Create test dataset from a separate set of pairs for consistent evaluation
    # We use 20% of all pairs for testing
    test_size = min(int(0.2 * len(all_segment_pairs)), len(all_segment_pairs))
    
    if test_size == 0:
        print("Warning: Not enough pairs for testing. Using all pairs for training.")
        test_indices = []
        test_pairs = []
        test_preferences = []
    else:
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
    if test_size > 0:
        test_dataset = PreferenceDataset(data_cpu, test_pairs, segment_indices, test_preferences)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
        print(f"Created test dataset with {len(test_dataset)} pairs")
    else:
        # Create a small dummy test dataset from the labeled data if no separate test set
        test_dataset = PreferenceDataset(data_cpu, labeled_pairs[:min(10, len(labeled_pairs))], 
                                        segment_indices, labeled_preferences[:min(10, len(labeled_preferences))])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
        print(f"Created dummy test dataset with {len(test_dataset)} pairs from labeled data")
    
    print(f"Starting with {len(labeled_pairs)} labeled and {len(unlabeled_pairs)} unlabeled pairs")
    
    # Initialize metrics tracking
    metrics = {
        "num_labeled": [],
        "test_accuracy": [],
        "test_loss": [],
        "iterations": []
    }
    
    # Main active learning loop
    iteration = 0
    total_labeled = len(labeled_pairs)
    max_queries = cfg.active_learning.max_queries
    
    # Keep track of the previous ensemble for fine-tuning
    prev_ensemble = None
    
    while total_labeled < max_queries and len(unlabeled_pairs) > 0:
        iteration += 1
        print(f"\n--- Active Learning Iteration {iteration} ---")
        print(f"Currently have {total_labeled} labeled pairs")
        
        # Train ensemble model on current labeled dataset
        print("Training ensemble model...")
        ensemble = train_ensemble_model(
            state_dim,
            action_dim,
            labeled_pairs,
            segment_indices,
            labeled_preferences,
            data_cpu,  # Use CPU data for indexing
            device,
            num_models=cfg.active_learning.num_models,
            hidden_dims=cfg.model.hidden_dims,
            num_epochs=cfg.active_learning.train_epochs,
            fine_tune=cfg.active_learning.fine_tune,
            prev_ensemble=prev_ensemble,
            fine_tune_lr=cfg.active_learning.fine_tune_lr
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
        metrics["iterations"].append(iteration)
        
        # Log to wandb
        if cfg.wandb.use_wandb:
            log_to_wandb({
                "num_labeled": total_labeled,
                "test_accuracy": test_metrics["test_accuracy"],
                "test_loss": test_metrics["test_loss"],
                "active_iteration": iteration
            })
        
        # Select next batch of uncertain pairs
        batch_size = min(cfg.active_learning.batch_size, max_queries - total_labeled)
        if batch_size <= 0 or len(unlabeled_pairs) == 0:
            print("Reached maximum number of queries or no more unlabeled data")
            break
        
        print(f"Computing uncertainty scores using {cfg.active_learning.uncertainty_method} method...")
        
        # Extract configuration parameter for candidate sampling approach
        use_random_sampling = True
        if hasattr(cfg.active_learning, 'use_random_sampling'):
            use_random_sampling = cfg.active_learning.use_random_sampling
        
        # Get number of candidates to consider
        n_candidates = cfg.active_learning.n_candidates if hasattr(cfg.active_learning, 'n_candidates') else 100
        
        # Use the unified function for uncertainty-based selection
        candidate_pairs = []
        candidate_indices = []
        
        # Map unlabeled pairs to their segment indices for the unified function
        for idx, pair_idx in enumerate(unlabeled_indices):
            i, j = unlabeled_pairs[idx]
            candidate_pairs.append((i, j))
            candidate_indices.append(idx)
        
        # Use the unified approach with comprehensive scoring
        ranked_pairs = select_uncertain_pairs_comprehensive(
            ensemble,
            segments,
            segment_indices,
            data_cpu,
            device,
            uncertainty_method=cfg.active_learning.uncertainty_method,
            max_pairs=batch_size,
            use_random_candidate_sampling=False,  # Always use all unlabeled pairs here
            n_candidates=None
        )
        
        # Extract the selected pairs and their indices
        selected_pairs = ranked_pairs[:batch_size]
        selected_indices = []
        
        # Find the indices of selected pairs in the unlabeled set
        for selected_pair in selected_pairs:
            for idx, pair in enumerate(unlabeled_pairs):
                if pair == selected_pair:
                    selected_indices.append(idx)
                    break
        
        # Check if we were able to select any pairs
        if len(selected_pairs) == 0:
            print("No pairs were selected. Ending active learning loop.")
            break
        
        # Get ground truth preferences for selected pairs
        # In a real system, this would be where we query the human
        selected_unlabeled_indices = [unlabeled_indices[i] for i in selected_indices]
        selected_preferences = [gt_preferences[i] for i in selected_unlabeled_indices]
        
        print(f"Selected {len(selected_pairs)} new pairs to label")
        
        # Add newly labeled pairs to labeled set
        labeled_pairs.extend(selected_pairs)
        labeled_preferences.extend(selected_preferences)
        total_labeled += len(selected_pairs)
        
        # Remove selected pairs from unlabeled set
        unlabeled_pairs = [p for i, p in enumerate(unlabeled_pairs) if i not in selected_indices]
        unlabeled_indices = [idx for i, idx in enumerate(unlabeled_indices) if i not in selected_indices]
        
        print(f"Now have {len(labeled_pairs)} labeled and {len(unlabeled_pairs)} unlabeled pairs")
    
    # Train final model on all labeled data
    print("\n--- Training Final Model ---")
    print(f"Training on all {len(labeled_pairs)} labeled pairs")
    
    # Create dataset from all labeled pairs
    final_dataset = PreferenceDataset(data_cpu, labeled_pairs, segment_indices, labeled_preferences)
    
    # Create train/val/test split
    train_size = int(0.8 * len(final_dataset))
    val_size = int(0.1 * len(final_dataset))
    test_size = len(final_dataset) - train_size - val_size
    
    train_dataset, val_dataset, final_test_dataset = torch.utils.data.random_split(
        final_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    final_test_loader = torch.utils.data.DataLoader(final_test_dataset, batch_size=cfg.training.batch_size)
    
    # Train final model
    final_model = SegmentRewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
    
    from utils.training_utils import train_reward_model
    final_model, train_losses, val_losses = train_reward_model(
        final_model,
        train_loader,
        val_loader,
        device,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.model.lr,
        wandb=wandb if cfg.wandb.use_wandb else None
    )
    
    # Evaluate final model
    print("\nEvaluating final model...")
    final_metrics = evaluate_model_on_test_set(final_model, final_test_loader, device)
    
    # Also evaluate on the consistent test set
    consistent_metrics = evaluate_model_on_test_set(final_model, test_loader, device)
    
    print(f"Final model test accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"Final model accuracy on consistent test set: {consistent_metrics['test_accuracy']:.4f}")
    
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
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["num_labeled"], metrics["test_accuracy"], marker='o')
    plt.xlabel("Number of Labeled Pairs")
    plt.ylabel("Test Accuracy")
    plt.title(f"Active Learning Curve ({cfg.active_learning.uncertainty_method})")
    plt.grid(True)
    
    # Create descriptive output directory structure
    dataset_name = Path(cfg.data.data_path).stem
    uncertainty_method = cfg.active_learning.uncertainty_method
    fine_tune_str = "finetune" if cfg.active_learning.fine_tune else "scratch"
    
    # Create descriptive subdirectory
    output_subdir = f"{dataset_name}_active_{uncertainty_method}_init{cfg.active_learning.initial_size}_max{cfg.active_learning.max_queries}_batch{cfg.active_learning.batch_size}_{fine_tune_str}"
    
    # Save plot with descriptive name
    plot_path = os.path.join(cfg.output.output_dir, f"{output_subdir}_learning_curve.png")
    plt.savefig(plot_path)
    
    if cfg.wandb.use_wandb:
        wandb.log({"learning_curve": wandb.Image(plot_path)})
    
    # Save final model in descriptive directory
    model_dir = os.path.join(cfg.output.output_dir, output_subdir)
    os.makedirs(model_dir, exist_ok=True)
    
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

@hydra.main(config_path="config", config_name="reward_model_sampling", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for active preference learning."""
    active_preference_learning(cfg)

if __name__ == "__main__":
    main() 