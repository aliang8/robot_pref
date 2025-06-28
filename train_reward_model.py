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

import wandb
from models.reward_models import DistributionalRewardModel, RewardModel
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
)
from utils.seed import set_seed
from utils.training import (
    evaluate_model_on_test_set,
    train_distributional_model,
    train_model,
)
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
        if hasattr(cfg.data, 'segment_pairs_path') and hasattr(cfg.data, 'segment_indices_path'):
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

        all_segment_pairs = random.sample(all_segment_pairs, cfg.data.num_pairs)

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
        
        # Create dataset with loaded/filtered data
        preference_dataset = PreferenceDataset(
            data_cpu, 
            all_segment_pairs, 
            segment_indices, 
            preferences,
            normalize_obs=cfg.data.normalize_obs,
            norm_method=cfg.data.norm_method,
            use_images=cfg.data.use_images,
            image_key=cfg.data.image_key,
            obs_key=cfg.data.obs_key,
            action_key=cfg.data.action_key,
            use_image_embeddings=cfg.data.use_image_embeddings
        )

        # Create data loaders
        dataloaders = create_data_loaders(
            preference_dataset,
            train_ratio=0.8,  
            val_ratio=0.1,  
            batch_size=cfg.training.batch_size,
            seed=current_seed,
            normalize_obs=cfg.data.normalize_obs,
            norm_method=cfg.data.norm_method,
            num_workers=cfg.training.num_workers,
            # pin_memory=cfg.training.pin_memory
        )

        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders["test"]

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
        test_metrics = evaluate_model_on_test_set(model, test_loader, device, is_distributional=cfg.model.is_distributional)
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
            "model_path": model_path
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