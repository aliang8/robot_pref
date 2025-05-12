import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn as nn
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

# Import reward model components from train_reward_model.py
from train_reward_model import (
    StateActionRewardModel,
    SegmentRewardModel,
    PreferenceDataset,
    bradley_terry_loss,
    train_reward_model,
    evaluate_model_on_test_set
)

class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], num_models=5):
        super(EnsembleRewardModel, self).__init__()
        self.models = nn.ModuleList([
            SegmentRewardModel(state_dim, action_dim, hidden_dims)
            for _ in range(num_models)
        ])
        self.num_models = num_models
    
    def forward(self, observations, actions):
        """Return rewards from all models in the ensemble."""
        rewards = []
        for model in self.models:
            rewards.append(model(observations, actions))
        
        # Stack rewards from all models
        if isinstance(rewards[0], torch.Tensor) and rewards[0].dim() == 0:
            # Handle single segment case
            return torch.stack(rewards)
        else:
            # Handle batch case
            return torch.stack(rewards, dim=0)
    
    def mean_reward(self, observations, actions):
        """Return mean reward across all models."""
        rewards = self(observations, actions)
        return rewards.mean(dim=0)
    
    def std_reward(self, observations, actions):
        """Return standard deviation of rewards across all models."""
        rewards = self(observations, actions)
        return rewards.std(dim=0)
    
    def disagreement(self, observations, actions):
        """Return disagreement (variance) across models."""
        rewards = self(observations, actions)
        return rewards.var(dim=0)

def compute_entropy(probs):
    """Compute entropy of preference probabilities."""
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs = torch.clamp(probs, min=eps, max=1-eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)

def compute_uncertainty_scores(model, segment_pairs, segment_indices, data, device, method="entropy"):
    """Compute uncertainty scores for segment pairs using specified method.
    
    Args:
        model: Reward model (either single model or ensemble)
        segment_pairs: List of segment pair indices
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        data: Data dictionary containing observations and actions
        device: Device to run computation on
        method: Uncertainty estimation method ("entropy", "disagreement", or "random")
        
    Returns:
        uncertainty_scores: List of uncertainty scores for each segment pair
    """
    # Extract observation and action fields
    obs_key = "obs" if "obs" in data else "state"
    action_key = "action"
    
    uncertainty_scores = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    num_batches = (len(segment_pairs) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc=f"Computing {method} scores"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(segment_pairs))
            batch_pairs = segment_pairs[batch_start:batch_end]
            
            batch_scores = []
            
            for pair_idx, (seg_idx1, seg_idx2) in enumerate(batch_pairs):
                # Get segment indices
                start1, end1 = segment_indices[seg_idx1]
                start2, end2 = segment_indices[seg_idx2]
                
                # Extract observations and actions
                obs1 = data[obs_key][start1:end1].to(device)
                actions1 = data[action_key][start1:end1-1].to(device)
                obs2 = data[obs_key][start2:end2].to(device)
                actions2 = data[action_key][start2:end2-1].to(device)
                
                # Ensure observations and actions have same length
                min_len1 = min(obs1.shape[0]-1, actions1.shape[0])
                min_len2 = min(obs2.shape[0]-1, actions2.shape[0])
                
                obs1 = obs1[:min_len1]
                actions1 = actions1[:min_len1]
                obs2 = obs2[:min_len2]
                actions2 = actions2[:min_len2]
                
                if method == "random":
                    # Random sampling - just assign random score
                    score = random.random()
                
                elif method == "entropy":
                    # Entropy-based uncertainty
                    if isinstance(model, EnsembleRewardModel):
                        # Use mean prediction from ensemble
                        reward1 = model.mean_reward(obs1, actions1)
                        reward2 = model.mean_reward(obs2, actions2)
                    else:
                        # Single model
                        reward1 = model(obs1, actions1)
                        reward2 = model(obs2, actions2)
                    
                    # Compute preference probability
                    logits = reward1 - reward2
                    probs = torch.sigmoid(logits)
                    
                    # Compute entropy of binary prediction
                    probs_2d = torch.stack([probs, 1 - probs], dim=0)
                    score = compute_entropy(probs_2d).item()
                
                elif method == "disagreement":
                    # Disagreement-based uncertainty (requires ensemble)
                    if not isinstance(model, EnsembleRewardModel):
                        raise ValueError("Disagreement method requires an ensemble model")
                    
                    # Get reward predictions from all models
                    rewards1 = model(obs1, actions1)  # Shape: [num_models]
                    rewards2 = model(obs2, actions2)  # Shape: [num_models]
                    
                    # Compute preference probability for each model
                    logits = rewards1 - rewards2  # Shape: [num_models]
                    probs = torch.sigmoid(logits)  # Shape: [num_models]
                    
                    # Disagreement is variance in preference probabilities
                    score = probs.var().item()
                
                batch_scores.append(score)
            
            uncertainty_scores.extend(batch_scores)
    
    return uncertainty_scores

def select_uncertain_pairs(uncertainty_scores, segment_pairs, k):
    """Select top-k most uncertain segment pairs.
    
    Args:
        uncertainty_scores: List of uncertainty scores for each segment pair
        segment_pairs: List of segment pair indices
        k: Number of pairs to select
        
    Returns:
        selected_pairs: List of selected segment pairs
        selected_indices: Indices of selected pairs in the original list
    """
    # Convert to numpy for easier manipulation
    scores = np.array(uncertainty_scores)
    
    # Get indices of top-k highest uncertainty scores
    selected_indices = np.argsort(scores)[-k:]
    
    # Get corresponding segment pairs
    selected_pairs = [segment_pairs[i] for i in selected_indices]
    
    return selected_pairs, selected_indices.tolist()

def get_ground_truth_preferences(segment_pairs, segment_indices, rewards):
    """Generate ground truth preferences based on cumulative rewards.
    
    Args:
        segment_pairs: List of segment pair indices
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        rewards: Tensor of reward values for all transitions
        
    Returns:
        preferences: List of preferences (1 = first segment preferred, 2 = second segment preferred)
    """
    preferences = []
    
    for seg_idx1, seg_idx2 in segment_pairs:
        # Get segment indices
        start1, end1 = segment_indices[seg_idx1]
        start2, end2 = segment_indices[seg_idx2]
        
        # Calculate cumulative rewards for each segment
        reward1 = rewards[start1:end1].sum().item()
        reward2 = rewards[start2:end2].sum().item()
        
        # Determine preference (1 = first segment preferred, 2 = second segment preferred)
        if reward1 > reward2:
            preferences.append(1)
        else:
            preferences.append(2)
    
    return preferences

def create_initial_dataset(segment_pairs, segment_indices, preferences, data, initial_size):
    """Create initial dataset with a small number of labeled pairs.
    
    Args:
        segment_pairs: List of segment pair indices
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        preferences: List of preferences (1 = first segment preferred, 2 = second segment preferred)
        data: Data dictionary containing observations and actions
        initial_size: Initial number of labeled pairs
        
    Returns:
        labeled_pairs: List of initially labeled segment pairs
        labeled_preferences: List of preferences for initially labeled pairs
        unlabeled_pairs: List of remaining unlabeled segment pairs
        unlabeled_indices: Indices of unlabeled pairs in the original list
    """
    # Randomly select initial pairs
    all_indices = list(range(len(segment_pairs)))
    labeled_indices = random.sample(all_indices, initial_size)
    unlabeled_indices = [i for i in all_indices if i not in labeled_indices]
    
    # Get labeled pairs and preferences
    labeled_pairs = [segment_pairs[i] for i in labeled_indices]
    labeled_preferences = [preferences[i] for i in labeled_indices]
    
    # Get unlabeled pairs
    unlabeled_pairs = [segment_pairs[i] for i in unlabeled_indices]
    
    return labeled_pairs, labeled_preferences, unlabeled_pairs, unlabeled_indices

def train_ensemble_model(state_dim, action_dim, labeled_pairs, segment_indices, labeled_preferences, 
                        data, device, num_models=5, hidden_dims=[256, 256], num_epochs=20):
    """Train an ensemble of reward models on the labeled data.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        labeled_pairs: List of labeled segment pairs
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        labeled_preferences: List of preferences for labeled pairs
        data: Data dictionary containing observations and actions
        device: Device to run training on
        num_models: Number of models in the ensemble
        hidden_dims: Hidden dimensions for each model
        num_epochs: Number of epochs to train each model
        
    Returns:
        ensemble: Trained ensemble model
    """
    # Create dataset from labeled pairs
    dataset = PreferenceDataset(data, labeled_pairs, segment_indices, labeled_preferences)
    
    # Create ensemble model
    ensemble = EnsembleRewardModel(state_dim, action_dim, hidden_dims, num_models)
    
    # Train each model in the ensemble
    for i in range(num_models):
        print(f"\nTraining model {i+1}/{num_models} in ensemble")
        
        # Create data loaders for this model
        # Use different random splits for each model to increase diversity
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42 + i)  # Different seed for each model
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Train this model
        train_reward_model(
            ensemble.models[i],
            train_loader,
            val_loader,
            device,
            num_epochs=num_epochs,
            lr=1e-4
        )
    
    return ensemble

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
    
    # Get observation and action dimensions
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    rewards = data["reward"]
    
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create segments
    print(f"Creating segments of length {cfg.data.segment_length}...")
    segments, segment_indices = create_segments(
        data, 
        segment_length=cfg.data.segment_length,
        max_segments=cfg.data.num_segments
    )
    
    # Generate all possible segment pairs
    print(f"Generating {cfg.data.num_pairs} segment pairs...")
    all_segment_pairs, gt_preferences = sample_segment_pairs(
        segments, 
        segment_indices, 
        data["reward"], 
        n_pairs=cfg.data.num_pairs
    )
    
    # Create initial labeled dataset
    print(f"Creating initial dataset with {cfg.active_learning.initial_size} labeled pairs...")
    labeled_pairs, labeled_preferences, unlabeled_pairs, unlabeled_indices = create_initial_dataset(
        all_segment_pairs,
        segment_indices,
        gt_preferences,
        data,
        cfg.active_learning.initial_size
    )
    
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
    
    # Create test dataset from a separate set of pairs for consistent evaluation
    # We use 20% of all pairs for testing
    test_size = int(0.2 * len(all_segment_pairs))
    test_indices = random.sample(range(len(all_segment_pairs)), test_size)
    test_pairs = [all_segment_pairs[i] for i in test_indices]
    test_preferences = [gt_preferences[i] for i in test_indices]
    
    # Make sure test pairs are not in labeled or unlabeled sets
    labeled_pairs = [p for i, p in enumerate(labeled_pairs) if i not in test_indices]
    labeled_preferences = [p for i, p in enumerate(labeled_preferences) if i not in test_indices]
    unlabeled_pairs = [p for i, p in enumerate(unlabeled_pairs) if i not in test_indices]
    unlabeled_indices = [i for i, idx in enumerate(unlabeled_indices) if idx not in test_indices]
    
    # Create test dataset
    test_dataset = PreferenceDataset(data, test_pairs, segment_indices, test_preferences)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    print(f"Created test dataset with {len(test_dataset)} pairs")
    print(f"Starting with {len(labeled_pairs)} labeled and {len(unlabeled_pairs)} unlabeled pairs")
    
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
            data,
            device,
            num_models=cfg.active_learning.num_models,
            hidden_dims=cfg.model.hidden_dims,
            num_epochs=cfg.active_learning.train_epochs
        )
        
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
        uncertainty_scores = compute_uncertainty_scores(
            ensemble,
            unlabeled_pairs,
            segment_indices,
            data,
            device,
            method=cfg.active_learning.uncertainty_method
        )
        
        # Select most uncertain pairs
        print(f"Selecting {batch_size} most uncertain pairs...")
        selected_pairs, selected_indices = select_uncertain_pairs(
            uncertainty_scores,
            unlabeled_pairs,
            batch_size
        )
        
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
    final_dataset = PreferenceDataset(data, labeled_pairs, segment_indices, labeled_preferences)
    
    # Create train/val/test split
    train_size = int(0.8 * len(final_dataset))
    val_size = int(0.1 * len(final_dataset))
    test_size = len(final_dataset) - train_size - val_size
    
    train_dataset, val_dataset, final_test_dataset = random_split(
        final_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    final_test_loader = DataLoader(final_test_dataset, batch_size=cfg.training.batch_size)
    
    # Train final model
    final_model = SegmentRewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
    final_model, train_losses, val_losses = train_reward_model(
        final_model,
        train_loader,
        val_loader,
        device,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.model.lr
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
    
    # Save plot
    plot_path = os.path.join(cfg.output.output_dir, f"active_learning_curve_{cfg.active_learning.uncertainty_method}.png")
    plt.savefig(plot_path)
    
    if cfg.wandb.use_wandb:
        wandb.log({"learning_curve": wandb.Image(plot_path)})
    
    # Save final model
    model_dir = os.path.join(
        cfg.output.output_dir, 
        f"active_{Path(cfg.data.data_path).stem}_{cfg.active_learning.uncertainty_method}"
    )
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