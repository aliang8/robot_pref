import itertools
import pickle
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from models import EnsembleRewardModel
from utils.active_query_selection import (
    select_active_pref_query,
)
from utils.analyze_rewards import analyze_rewards
from utils.data import (
    get_gt_preferences,
    load_tensordict,
    process_data_trajectories,
    segment_episodes,
)
from utils.dataset import PreferenceDataset, create_data_loaders
from utils.seed import set_seed
from utils.training import evaluate_model_on_test_set, train_model
from utils.viz import plot_active_learning_metrics


def find_similar_segments_dtw(query_idx, k, distance_matrix):
    """Find the k most similar segments to the query segment using a pre-computed distance matrix."""
    if query_idx < 0 or query_idx >= distance_matrix.shape[0]:
        assert False, "Query index is out of bounds for the distance matrix"
    distances = distance_matrix[query_idx].copy()
    distances[query_idx] = float('inf')  # Exclude self
    similar_indices = np.argsort(distances)[:k]
    return similar_indices

def active_preference_learning(cfg, dataset_name=None):
    """Main function for active preference learning."""
    # Print some configs
    dtw_enabled = cfg.dtw_augmentation.enabled
    dtw_k_augment = cfg.dtw_augmentation.k_augment
    print(f"DTW enabled: {dtw_enabled}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {cfg.data.data_path}")
    data = load_tensordict(cfg.data.data_path)
    data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    episodes = process_data_trajectories(data, device=device)

    # Get specific keys
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]
    print(f"Obs dim: {state_dim}, Action dim: {action_dim}")

    rewards = data["reward"]
    reward_max = rewards.max().item()
    reward_min = rewards.min().item()

    # Load the DTW Matrix
    distance_matrix = None
    segment_start_end = None

    if dtw_enabled:
        dtw_matrix_file = Path(cfg.data.data_path).parent / f"dtw_matrix_{cfg.data.segment_length}.pkl"
        if not dtw_matrix_file.exists():
            print(f"DTW matrix file not found at {dtw_matrix_file}")
            assert False, "DTW matrix file not found. Please run preprocess_dtw_matrix.py"
        else:
            distance_matrix, segment_start_end = pickle.load(open(dtw_matrix_file, "rb"))
            print(f"Loaded DTW matrix shape: {distance_matrix.shape}")
    else:
        _, segment_start_end = segment_episodes(data, cfg.data.segment_length)

    # Get all possible segment pairs, sample, and create dataset
    all_segment_pairs = list(itertools.combinations(range(len(segment_start_end)), 2))
    sampled_segment_pairs = random.sample(all_segment_pairs, cfg.data.subsamples)
    print(f"Sampled {len(sampled_segment_pairs)} pairs")
    test_indices = random.sample(range(len(sampled_segment_pairs)), cfg.data.num_test_pairs)
    test_pairs = [sampled_segment_pairs[i] for i in test_indices]
    test_preferences = get_gt_preferences(data, segment_start_end, test_pairs)
    test_dataset = PreferenceDataset(data, test_pairs, segment_start_end, test_preferences)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    print(f"Test dataset: {len(test_dataset)} pairs")

    sampled_indices = list(range(len(sampled_segment_pairs)))
    remaining_indices = list(set(sampled_indices) - set(test_indices))
    unlabeled_pairs = [sampled_segment_pairs[i] for i in remaining_indices]
    print(f"Unlabeled pairs after test set: {len(unlabeled_pairs)}")

    # Create directory for saving
    model_dir = Path(cfg.output.output_dir) / cfg.output.model_dir_name
    print(f"Model directory: {model_dir}")
    for subdir in ["checkpoints", "rm_training", "reward_analysis"]:
        (model_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Active learning training loop
    num_queries = 0
    total_queries = cfg.active_learning.total_queries
    labeled_pairs = []
    labeled_preferences = []
    augmented_accuracy = []
    candidate_pairs = unlabeled_pairs
    iteration = 0

    # Initialize metrics dict
    metrics = {
        "num_labeled": [],
        "test_accuracy": [],
        "test_loss": [],
        "avg_logpdf": [],
        "iterations": [],
        "val_losses": [],
        "test_loss_curve": [],
        "num_labeled_curve": [],
        # Add any additional new keys here as needed
    }

    while num_queries < total_queries and len(unlabeled_pairs) > 0:
        iteration += 1
        print(f"\n=== Active Learning Iteration {iteration} ===")
        print(f"Progress: {num_queries}/{total_queries} queries")
        print(f"Candidate pairs: {len(candidate_pairs)}")

        if num_queries == 0:
            selected_query_pair = random.choice(candidate_pairs)
        else:
            selected_query_pair = select_active_pref_query(
                ensemble,
                segment_start_end,
                data,
                uncertainty_method=cfg.active_learning.uncertainty_method,
                max_pairs=1,
                candidate_pairs=candidate_pairs
            )[0]      

        selected_query_pref = get_gt_preferences(data, segment_start_end, [selected_query_pair])[0]
        print(f"Selected pair: {selected_query_pair}, pref: {selected_query_pref}")
        labeled_pairs.append(selected_query_pair)
        labeled_preferences.append(selected_query_pref)
        num_queries += 1
        candidate_pairs.remove(selected_query_pair)

        # DTW preference augmentation
        if dtw_enabled and distance_matrix is not None:
            dtw_augmented_pairs = []
            dtw_augmented_preferences = []
            print(f"DTW augmenting (k={dtw_k_augment})")
            i, j = selected_query_pair
            if selected_query_pref == 1:
                similar_to_i_indices = find_similar_segments_dtw(i, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_i_indices:
                    if sim_idx != j:
                        dtw_augmented_pairs.append((sim_idx, j))
                        dtw_augmented_preferences.append(1)
                similar_to_j_indices = find_similar_segments_dtw(j, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_j_indices:
                    if sim_idx != j:
                        dtw_augmented_pairs.append((i, sim_idx))
                        dtw_augmented_preferences.append(1)
            elif selected_query_pref == 2:
                similar_to_j_indices = find_similar_segments_dtw(j, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_j_indices:
                    if sim_idx != i:
                        dtw_augmented_pairs.append((i, sim_idx))
                        dtw_augmented_preferences.append(2)
                similar_to_i_indices = find_similar_segments_dtw(i, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_i_indices:
                    if sim_idx != j:
                        dtw_augmented_pairs.append((sim_idx, j))
                        dtw_augmented_preferences.append(2)
            labeled_pairs.extend(dtw_augmented_pairs)
            labeled_preferences.extend(dtw_augmented_preferences)
            print(f"DTW augmented pairs: {len(dtw_augmented_pairs)}")
            augmented_pref_with_rewards = get_gt_preferences(data, segment_start_end, dtw_augmented_pairs)
            augmented_acc = (np.array(augmented_pref_with_rewards) == np.array(dtw_augmented_preferences)).mean()
            augmented_accuracy.append(augmented_acc)
            print(f"Augmented accuracy: {augmented_acc:.4f}")
            for dtw_augmented_pair in dtw_augmented_pairs:
                if dtw_augmented_pair in candidate_pairs:
                    candidate_pairs.remove(dtw_augmented_pair)

        ensemble_dataset = PreferenceDataset(data, labeled_pairs, segment_start_end, labeled_preferences)
        data_loaders = create_data_loaders(
            ensemble_dataset,
            train_ratio=1.0,
            val_ratio=0.0,
            batch_size=cfg.training.batch_size,
            seed=cfg.random_seed
        )
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']

        print("Training new ensemble")
        ensemble = EnsembleRewardModel(state_dim, action_dim, cfg.model.hidden_dims, cfg.active_learning.num_models)

        output_path = model_dir / "rm_training" / f"reward_model_training_{iteration}.png"
        ensemble, _, _ = train_model(
            ensemble,
            train_loader,
            val_loader,
            device,
            num_epochs=cfg.training.num_epochs,
            lr=cfg.model.lr,
            is_ensemble=True,
            output_path=str(output_path)
        )

        ensemble = ensemble.to(device)
        print("Evaluating on test set...")
        test_metrics = evaluate_model_on_test_set(ensemble.models[0], test_loader, device)
        metrics["num_labeled"].append(num_queries)
        metrics["test_accuracy"].append(test_metrics["test_accuracy"])
        metrics["test_loss"].append(test_metrics["test_loss"])
        metrics["avg_logpdf"].append(test_metrics["avg_logpdf"])
        metrics["iterations"].append(iteration)

        # Update metrics with new keys if needed
        if "val_losses" in metrics:
            metrics["val_losses"].append(None)  # or actual val_loss if available

        if "test_loss_curve" not in metrics:
            metrics["test_loss_curve"] = []
            metrics["num_labeled_curve"] = []
        metrics["test_loss_curve"].append(test_metrics["test_loss"])
        metrics["num_labeled_curve"].append(num_queries)

        if iteration % cfg.training.save_model_every == 0:
            checkpoint_path = model_dir / "checkpoints" / f"checkpoint_iter_{iteration}.pt"
            torch.save(ensemble.models[0].state_dict(), checkpoint_path)
            print(f"Saved checkpoint at iteration {iteration}")
            print(f"Model saved to: {checkpoint_path}")

        batch_size = min(cfg.active_learning.total_queries_per_iteration, total_queries - num_queries)
        if batch_size <= 0 or len(unlabeled_pairs) == 0:
            print("No more queries or unlabeled data")
            break
        if iteration % cfg.training.reward_analysis_every == 0:
            output_file = model_dir / "reward_analysis" / f"reward_grid_iter_{iteration}.png"
            analyze_rewards(
                model=ensemble.models[0],
                episodes=episodes,
                output_file=str(output_file),
                num_episodes=9,
                reward_max=reward_max,
                reward_min=reward_min,
                random_seed=cfg.random_seed
            )

    # Plot active learning metrics
    plot_active_learning_metrics(model_dir, metrics, augmented_accuracy)

    # Save config
    config_path = model_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print("Active learning completed.")

@hydra.main(config_path="config", config_name="reward_model_active", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.random_seed)
    print(f"Random seed set to {cfg.random_seed}")

    dataset_path = Path(cfg.data.data_path)
    dataset_name = dataset_path.stem
    print(f"Dataset: {dataset_name}")

    if hasattr(cfg.output, "model_dir_name"):
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace("DATASET_NAME", dataset_name)

    active_preference_learning(cfg)

if __name__ == "__main__":
    main() 