import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
import random
import itertools
from tqdm import tqdm

from models.reward_models import RewardModel

from utils.data import (
    load_tensordict,
    segment_episodes,
    get_gt_preferences,
    process_data_trajectories,
)
from utils.dataset import (
    PreferenceDataset,
    create_data_loaders,
)
from utils.training import train_model, evaluate_model_on_test_set
from utils.seed import set_seed
from utils.wandb import log_to_wandb
from utils.analyze_rewards import analyze_rewards


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
        print(f"Random seed set to {current_seed}")

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

        segments, segment_indices = segment_episodes(data_cpu, cfg.data.segment_length)

        # Find all possible segment pairs (num_segments choose 2) and sample data.num_pairs from them
        all_segment_pairs = list(itertools.combinations(range(len(segment_indices)), 2))
        # Use current seed to ensure different sampling for each run
        random.seed(current_seed)
        all_segment_pairs = random.sample(all_segment_pairs, cfg.data.num_pairs)
        print(f"Sampled {len(all_segment_pairs)} pairs from {len(segment_indices) * (len(segment_indices) - 1) // 2} total pairs")

        print("Computing preference labels")
        preferences = get_gt_preferences(data_cpu, segment_indices, all_segment_pairs)

        print(
            f"Final data stats - Observation dimension: {state_dim}, Action dimension: {action_dim}"
        )
        print(
            f"Working with {len(all_segment_pairs) if all_segment_pairs is not None else 0} preference pairs across {len(segment_indices) if segment_indices is not None else 0} segments"
        )

        # Create dataset
        preference_dataset = PreferenceDataset(
            data_cpu, all_segment_pairs, segment_indices, preferences
        )

        # Create data loaders
        dataloaders = create_data_loaders(
            preference_dataset,
            train_ratio=0.8,  
            val_ratio=0.1,  
            batch_size=cfg.training.batch_size,
            seed=current_seed,
            normalize_obs=cfg.data.normalize_obs,
            norm_method=cfg.data.norm_method
        )

        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders["test"]

        # Initialize reward model
        model = RewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
        print(f"Reward model: {model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        start_time = time.time()
        dataset_name = Path(cfg.data.data_path).stem
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace("DATASET_NAME", dataset_name)

        model_dir = os.path.join(cfg.output.output_dir, cfg.output.model_dir_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{current_seed}.pt")

        training_curve_path = os.path.join(model_dir, f"training_curve_{current_seed}.png")

        print("\nTraining reward model...")
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
        test_metrics = evaluate_model_on_test_set(model, test_loader, device)
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
            "test_log_prob": test_metrics["avg_logpdf"],
            "model_path": model_path
        })

        # Run reward analysis
        episodes = process_data_trajectories(data, device)    
        reward_max = data_cpu["reward"].max().item()
        reward_min = data_cpu["reward"].min().item()
        analyze_rewards(
            model=model,
            episodes=episodes,
            output_file=os.path.join(model_dir, f"reward_grid_{current_seed}.png"),
            wandb_run=wandb_run,
            reward_max=reward_max,
            reward_min=reward_min
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
        
        with open(os.path.join(model_dir, f"summary.txt"), "w") as f:
            f.write(f"Summary Statistics Across {num_seeds} Seeds:\n")
            f.write(f"Test Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}\n")
            f.write(f"Test LogPDF: {log_prob_mean:.4f} ± {log_prob_std:.4f}\n")
            f.write("=" * 50 + "\n")
        
    print("\nReward model training complete!")


if __name__ == "__main__":
    main()
