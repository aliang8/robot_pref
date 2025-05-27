import itertools
import pickle
import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from models import EnsembleRewardModel
from train_reward_model import (
    filter_pairs_with_preferences,
    load_preferences_from_directory,
)
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
from utils.dataset import PreferenceDataset, create_data_loaders, custom_collate
from utils.seed import set_seed
from utils.training import evaluate_model_on_test_set, train_model
from utils.viz import plot_active_learning_metrics

plt.rcParams['text.usetex'] = False

def find_similar_segments_dtw(query_idx, k, distance_matrix):
    """Find the k most similar segments to the query segment using a pre-computed distance matrix."""
    if query_idx < 0 or query_idx >= distance_matrix.shape[0]:
        assert False, "Query index is out of bounds for the distance matrix"
    distances = distance_matrix[query_idx].copy()
    distances[query_idx] = float('inf')  # Exclude self
    similar_indices = np.argsort(distances)[:k]

    return similar_indices

def compute_segment_return(data, segment_start_end):
    """Compute the return (sum of rewards) for each segment."""
    rewards = data["reward"]
    segment_returns = []
    for (start, end) in segment_start_end:
        # rewards[start:end] is a 1D tensor or numpy array
        segment_returns.append(float(rewards[start:end].sum().item() if hasattr(rewards[start:end], "sum") else np.sum(rewards[start:end])))
    return segment_returns

def save_labeled_pairs_info(labeled_pairs, labeled_preferences, segment_returns, output_path, is_augmented_list=None):
    """
    Save labeled pair info to a text file.
    Each line: pair_id,segment_1_idx,segment_2_idx,ret1,ret2,preference,is_augmented
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("pair_id,segment_1_idx,segment_2_idx,ret1,ret2,preference,is_augmented\n")
        for idx, ((seg1, seg2), pref) in enumerate(zip(labeled_pairs, labeled_preferences)):
            ret1 = segment_returns[seg1]
            ret2 = segment_returns[seg2]
            is_aug = is_augmented_list[idx] if is_augmented_list is not None else "NA"
            f.write(f"{idx},{seg1},{seg2},{ret1},{ret2},{pref},{is_aug}\n")

def log_queries_to_wandb(
    selected_query,
    selected_pref,
    segment_returns,
    iteration,
    dtw_augmented_pairs,
    dtw_augmented_prefs,
    video_dict=None,
    data=None,
    segment_start_end=None,
):
    """
    Log videos of the selected query segments and DTW augmentations.
    """
    # Log selected query
    seg1, seg2 = selected_query
    ret1 = segment_returns[seg1]
    ret2 = segment_returns[seg2]

    # Indicate preference by adding asterisk to the preferred return
    if selected_pref == 1:
        ret1_disp = f"{ret1}*"
        ret2_disp = f"{ret2}"
    elif selected_pref == 2:
        ret1_disp = f"{ret1}"
        ret2_disp = f"{ret2}*"
    else:
        ret1_disp = f"{ret1}"
        ret2_disp = f"{ret2}"

    # Visualize the selected query as videos on wandb if video_dict is provided
    if video_dict is not None:
        seg1_video = video_dict.get("seg1", None)
        seg2_video = video_dict.get("seg2", None)

        # Ensure videos are numpy arrays with shape (T, H, W, C)
        if seg1_video is not None and seg1_video.shape[-1] == 3:
            seg1_video = seg1_video.transpose(0, 3, 1, 2)
        if seg2_video is not None and seg2_video.shape[-1] == 3:
            seg2_video = seg2_video.transpose(0, 3, 1, 2)

        # Log both videos side by side
        if seg1_video is not None and seg2_video is not None:
            wandb.log({
                f"Selected Query Videos/iteration_{iteration}": [
                    wandb.Video(seg1_video, caption=f"Segment {seg1} (Return: {ret1_disp})", fps=8, format="mp4"),
                    wandb.Video(seg2_video, caption=f"Segment {seg2} (Return: {ret2_disp})", fps=8, format="mp4"),
                ],
            }, step=iteration)

    # --- Log DTW augmented pairs as videos ---
    # Only if data and segment_start_end are provided
    if data is not None and segment_start_end is not None and len(dtw_augmented_pairs) > 0 and "image" in data:
        videos_to_log = []
        for idx, ((seg1_aug, seg2_aug), pref) in enumerate(zip(dtw_augmented_pairs, dtw_augmented_prefs)):
            seg1_video = get_segment_video(data, segment_start_end, seg1_aug, video_key="image")
            seg2_video = get_segment_video(data, segment_start_end, seg2_aug, video_key="image")
            
            # Ensure videos are numpy arrays with shape (T, H, W, C)
            if seg1_video is not None and seg1_video.shape[-1] == 3:
                seg1_video = seg1_video.transpose(0, 3, 1, 2)
            if seg2_video is not None and seg2_video.shape[-1] == 3:
                seg2_video = seg2_video.transpose(0, 3, 1, 2)

            # Indicate preference by asterisk
            r1 = segment_returns[seg1_aug]
            r2 = segment_returns[seg2_aug]
            if pref == 1:
                r1_disp = f"{r1}*"
                r2_disp = f"{r2}"
            elif pref == 2:
                r1_disp = f"{r1}"
                r2_disp = f"{r2}*"
            else:
                r1_disp = f"{r1}"
                r2_disp = f"{r2}"

            if seg1_video is not None and seg2_video is not None:
                caption = f"Pair {idx}:\nSegment {seg1_aug} (Return: {r1_disp}) vs Segment {seg2_aug} (Return: {r2_disp})"
                # Combine the two videos side-by-side into one video if desired
                # Otherwise just log both as separate videos in the list
                videos_to_log.append(wandb.Video(seg1_video, caption=f"{caption} - First", fps=8, format="mp4"))
                videos_to_log.append(wandb.Video(seg2_video, caption=f"{caption} - Second", fps=8, format="mp4"))

        if videos_to_log:
            wandb.log({
                f"DTW_Augmented_Pairs/iteration_{iteration}": videos_to_log
            }, step=iteration)

def get_segment_video(data, segment_start_end, seg_idx, video_key="image"):
    """
    Utility to extract a video (np.ndarray) for a segment.
    Assumes data[video_key] is (T, H, W, C)
    Returns a numpy array (segment_T, H, W, C), dtype uint8.
    """
    if video_key not in data:
        return None
    start, end = segment_start_end[seg_idx]
    video = data[video_key][start:end]

    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()

    return video

def active_preference_learning(cfg):
    """Main function for active preference learning."""

    # Set random seed for reproducibility
    set_seed(cfg.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtw_enabled = cfg.dtw_augmentation.enabled
    dtw_k_augment = cfg.dtw_augmentation.k_augment
    
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
    print(f"Loaded data with {len(observations)} observations")

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
            print(f"Pre-norm DTW matrix mean: {np.nanmean(distance_matrix):.4f}, std: {np.nanstd(distance_matrix):.4f}")
            # Normalize the distance matrix to [0, 1], taking into account NaNs
            min_val = np.nanmin(distance_matrix)
            max_val = np.nanmax(distance_matrix)
            print(f"DTW matrix min: {min_val:.4f}, max: {max_val:.4f}")
            if max_val > min_val:
                distance_matrix = (distance_matrix - min_val) / (max_val - min_val)
            else:
                distance_matrix = np.zeros_like(distance_matrix)  # All values are the same, set to 0

            # print mean, ignoring NaNs
            print(f"Post-norm DTW matrix mean: {np.nanmean(distance_matrix):.4f}, std: {np.nanstd(distance_matrix):.4f}")
    else:
        _, segment_start_end = segment_episodes(data, cfg.data.segment_length)

    # Compute returns for each segment for logging
    segment_returns = compute_segment_return(data, segment_start_end)

    # Get all possible segment pairs, sample, and create dataset
    all_segment_pairs = list(itertools.combinations(range(len(segment_start_end)), 2))

    # Prepare segment pairs and preferences for test set
    if hasattr(cfg.data, 'preferences_dir'):
        print("Loading preferences from files...")
        preferences, pref_stats = load_preferences_from_directory(cfg.data.preferences_dir)
        all_segment_pairs, preferences = filter_pairs_with_preferences(all_segment_pairs, preferences)

        # np array to list of tuples
        all_segment_pairs = [tuple(pair) for pair in all_segment_pairs]

        test_pairs = all_segment_pairs[-cfg.data.num_test_pairs:]
        test_preferences = preferences[-cfg.data.num_test_pairs:]

        # Updating the preferences that are used for training
        preferences = preferences[:-cfg.data.num_test_pairs]
    else:
        total_samples = cfg.data.subsamples + cfg.data.num_test_pairs
        print(f"Sampling {cfg.data.subsamples} pairs for training and {cfg.data.num_test_pairs} pairs for testing from {len(all_segment_pairs)} total pairs")
        all_segment_pairs = random.sample(all_segment_pairs, total_samples)
        test_pairs = all_segment_pairs[-cfg.data.num_test_pairs:]
        test_preferences = get_gt_preferences(data, segment_start_end, test_pairs)

    test_dataset = PreferenceDataset(data, test_pairs, segment_start_end, test_preferences)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=custom_collate)
    print(f"Test dataset: {len(test_dataset)} pairs")

    unlabeled_pairs = all_segment_pairs[:-cfg.data.num_test_pairs]
    print(f"Unlabeled pairs after test set: {len(unlabeled_pairs)}")

    # Create directory for saving
    model_dir = Path(cfg.output.output_dir) / cfg.output.model_dir_name
    print(f"Model directory: {model_dir}")
    for subdir in ["checkpoints", "rm_training", "reward_analysis"]:
        (model_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Prepare path for saving labeled pairs info
    labeled_pairs_info_path = model_dir / "labeled_pairs_info.txt"

    # Active learning training loop
    num_queries = 0
    total_queries = cfg.active_learning.total_queries
    labeled_pairs = []
    labeled_costs = []
    labeled_preferences = []
    is_augmented_list = []
    candidate_pairs = unlabeled_pairs
    iteration = 0

    # Initialize metrics dict
    metrics = {
        "num_labeled": [],
        "test_accuracy": [],
        "test_loss": [],
        "iterations": [],
        "val_losses": [],
        "test_loss_curve": [],
        "num_labeled_curve": [],
    }
    # Only add DTW cost tracking if DTW is enabled
    if dtw_enabled:
        metrics["selected_pair_dtw_costs"] = []

        # TODO: once we have more human annotated prefs, we can compute the augmented accuracy, not enough right now
        if not hasattr(cfg.data, 'preferences_dir'):
            metrics["augmented_accuracy"] = []

    while num_queries < total_queries:
        iteration += 1
        print(f"\n=== Active Learning Iteration {iteration} ===")
        print(f"Progress: {num_queries}/{total_queries} queries")
        print(f"Candidate pairs: {len(candidate_pairs)}")

        if num_queries == 0: # First iteration, randomly select a pair
            selected_query_pair = random.choice(candidate_pairs)
        else: # Subsequent iterations, use active sampling
            selected_query_pair = select_active_pref_query(
                ensemble,
                segment_start_end,
                data,
                uncertainty_method=cfg.active_learning.uncertainty_method,
                max_pairs=1,
                candidate_pairs=candidate_pairs
            )[0]

        # Remove the selected query pair from unlabeled pairs
        num_queries += 1
        selected_query_pair_index = candidate_pairs.index(selected_query_pair)
        candidate_pairs.remove(selected_query_pair)
        labeled_pairs.append(selected_query_pair)
        is_augmented_list.append(False)  # Not augmented, original query

        # Get preference for selected query pair from human or use ground truth
        if hasattr(cfg.data, 'preferences_dir'):
            print("Using human preferences for selected query pair...")
            selected_query_pref = preferences[selected_query_pair_index]
            preferences = np.delete(preferences, selected_query_pair_index) # Remove the selected query pair preference from the list
        else:
            print("Computing ground truth preference for selected query pair...")
            selected_query_pref = get_gt_preferences(data, segment_start_end, [selected_query_pair])[0]
        labeled_preferences.append(selected_query_pref)

        # Add cost for selected query pair if DTW is enabled
        selected_query_pair_cost = None
        if dtw_enabled and distance_matrix is not None:
            selected_query_pair_cost = distance_matrix[selected_query_pair[0], selected_query_pair[1]]
            labeled_costs.append(selected_query_pair_cost)
        else:
            labeled_costs.append(None)

        # --- Visualize the selected query as videos on wandb ---
        video_dict = None

        if "image" in data:
            seg1 = selected_query_pair[0]
            seg2 = selected_query_pair[1]

            seg1_video = get_segment_video(data, segment_start_end, seg1, video_key="image")
            seg2_video = get_segment_video(data, segment_start_end, seg2, video_key="image")
            if seg1_video is not None and seg2_video is not None:
                video_dict = {"seg1": seg1_video, "seg2": seg2_video}

        # DTW preference augmentation
        dtw_augmented_pairs = []
        dtw_augmented_preferences = []
        dtw_augmented_preferences_costs = []
        dtw_augmented_is_aug = []
        dtw_augmented_returns = []
        if dtw_enabled and distance_matrix is not None:
            i, j = selected_query_pair
            if selected_query_pref == 1:
                similar_to_i_indices = find_similar_segments_dtw(i, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_i_indices:
                    if sim_idx != j:
                        dtw_augmented_pairs.append((sim_idx, j))
                        dtw_augmented_preferences.append(1)
                        dtw_augmented_is_aug.append(True)
                        cost = distance_matrix[sim_idx, j]
                        dtw_augmented_preferences_costs.append(cost)
                        dtw_augmented_returns.append((segment_returns[sim_idx], segment_returns[j]))
                similar_to_j_indices = find_similar_segments_dtw(j, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_j_indices:
                    if sim_idx != i:
                        dtw_augmented_pairs.append((i, sim_idx))
                        dtw_augmented_preferences.append(1)
                        dtw_augmented_is_aug.append(True)
                        cost = distance_matrix[i, sim_idx]
                        dtw_augmented_preferences_costs.append(cost)
                        dtw_augmented_returns.append((segment_returns[i], segment_returns[sim_idx]))
            elif selected_query_pref == 2:
                similar_to_j_indices = find_similar_segments_dtw(j, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_j_indices:
                    if sim_idx != i:
                        dtw_augmented_pairs.append((i, sim_idx))
                        dtw_augmented_preferences.append(2)
                        dtw_augmented_is_aug.append(True)
                        cost = distance_matrix[i, sim_idx]
                        dtw_augmented_preferences_costs.append(cost)
                        dtw_augmented_returns.append((segment_returns[i], segment_returns[sim_idx]))
                similar_to_i_indices = find_similar_segments_dtw(i, dtw_k_augment, distance_matrix)
                for sim_idx in similar_to_i_indices:
                    if sim_idx != j:
                        dtw_augmented_pairs.append((sim_idx, j))
                        dtw_augmented_preferences.append(2)
                        dtw_augmented_is_aug.append(True)
                        cost = distance_matrix[sim_idx, j]
                        dtw_augmented_preferences_costs.append(cost)
                        dtw_augmented_returns.append((segment_returns[sim_idx], segment_returns[j]))

            labeled_pairs.extend(dtw_augmented_pairs)
            labeled_preferences.extend(dtw_augmented_preferences)
            labeled_costs.extend(dtw_augmented_preferences_costs)
            is_augmented_list.extend(dtw_augmented_is_aug)
            print(f"DTW augmented pairs: {len(dtw_augmented_pairs)}")

            # TODO: once we have more human annotated prefs, we can compute the augmented accuracy, not enough right now
            if hasattr(cfg.data, 'preferences_dir'):
                print("Using human preferences, skipping augmented accuracy calculation.")
                augmented_acc = None
            else:
                augmented_pref_with_rewards = get_gt_preferences(data, segment_start_end, dtw_augmented_pairs)
                augmented_acc = (np.array(augmented_pref_with_rewards) == np.array(dtw_augmented_preferences)).mean()
                print(f"Augmented accuracy: {augmented_acc:.4f}")
                for dtw_augmented_pair in dtw_augmented_pairs:
                    if dtw_augmented_pair in candidate_pairs:
                        candidate_pairs.remove(dtw_augmented_pair)

        # Save labeled pairs info to file after each iteration
        save_labeled_pairs_info(labeled_pairs, labeled_preferences, segment_returns, labeled_pairs_info_path, is_augmented_list=is_augmented_list)

        if wandb.run is not None:
            # Log queries and augmentations to wandb for this iteration
            log_queries_to_wandb(
                selected_query=selected_query_pair,
                selected_pref=selected_query_pref,
                segment_returns=segment_returns,
                iteration=iteration,
                dtw_augmented_pairs=dtw_augmented_pairs,
                dtw_augmented_prefs=dtw_augmented_preferences,
                video_dict=video_dict,  # Pass video_dict for visualization
                data=data,
                segment_start_end=segment_start_end,
            )

        # Create dataset and data loaders
        ensemble_dataset = PreferenceDataset(
            data, 
            labeled_pairs, 
            segment_start_end, 
            labeled_preferences, 
            costs=labeled_costs if cfg.dtw_augmentation.enabled and cfg.dtw_augmentation.use_heuristic_beta else None,
        )
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
        metrics["num_labeled"].append(len(labeled_pairs))
        metrics["test_accuracy"].append(test_metrics["test_accuracy"])
        metrics["test_loss"].append(test_metrics["test_loss"])
        if dtw_enabled:
            # Only log DTW costs if enabled
            metrics["selected_pair_dtw_costs"].append(selected_query_pair_cost)
            if "augmented_accuracy" in metrics:
                metrics["augmented_accuracy"].append(augmented_acc)
        metrics["iterations"].append(iteration)

        if wandb.run is not None:
            # Log test metrics to wandb
            wandb_log_dict = {
                "test_accuracy": test_metrics["test_accuracy"],
                "test_loss": test_metrics["test_loss"],
                "num_labeled": len(labeled_pairs),
                "iteration": iteration,
            }
            if dtw_enabled:
                wandb_log_dict["selected_pair_dtw_cost"] = selected_query_pair_cost
            wandb.log(wandb_log_dict, step=iteration)

        # Update metrics with new keys if needed
        if "val_losses" in metrics:
            metrics["val_losses"].append(None)  # or actual val_loss if available

        if "test_loss_curve" not in metrics:
            metrics["test_loss_curve"] = []
            metrics["num_labeled_curve"] = []
        metrics["test_loss_curve"].append(test_metrics["test_loss"])
        metrics["num_labeled_curve"].append(len(labeled_pairs))

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
    plot_active_learning_metrics(model_dir, metrics)

    # Save config
    config_path = model_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print("Active learning completed.")

    if wandb.run is not None:
        # Finish the wandb run
        wandb.finish()

    # Return final metrics for seed
    return {
        "seed": cfg.random_seed,
        "test_accuracy": test_metrics["test_accuracy"],
        "model_path": checkpoint_path,
    }

@hydra.main(config_path="config", config_name="reward_model_active", version_base=None)
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

    # Initialize wandb for this seed
    if cfg.wandb.use_wandb:
        # Generate experiment name based on data path
        dataset_name = Path(cfg.data.data_path).stem

        # Initialize wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.output.model_dir_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            reinit=True,  # Allow reinit for multiple runs
            group=f"reward_{dataset_name}_{cfg.data.num_pairs}_multiseed",  # Group runs together
        )

        print(f"Wandb initialized: {wandb_run.name}")

    active_preference_learning(cfg)

if __name__ == "__main__":
    main()