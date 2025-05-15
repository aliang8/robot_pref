import os
import pickle
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb

# Import shared models and utilities
from models.reward_models import RewardModel

from utils.trajectory import (
    load_tensordict,
    sample_segment_pairs,
)
from utils.dataset import (
    PreferenceDataset,
    create_data_loaders,
    load_preferences_data,
)
from utils.training import train_model, evaluate_model_on_test_set
from utils.seed import set_seed
from utils.wandb import log_to_wandb


def run_reward_analysis(
    model_path,
    data_path,
    output_dir,
    num_episodes=9,
    device=None,
    random_seed=42,
    wandb_run=None,
):
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
        random_seed=random_seed,
    )

    # Log the analysis results to wandb
    if wandb_run is not None and wandb_run.run:
        reward_grid_path = os.path.join(output_dir, "reward_grid.png")
        if os.path.exists(reward_grid_path):
            print("Logging reward analysis grid to wandb")
            wandb_run.log({"reward_analysis/grid": wandb.Image(reward_grid_path)})
        else:
            print(f"Warning: Could not find reward grid image at {reward_grid_path}")

    print("Reward analysis completed successfully")


@hydra.main(config_path="config", config_name="reward_model", version_base=None)
def main(cfg: DictConfig):
    # Get the dataset name
    dataset_name = Path(cfg.data.data_path).stem

    # Replace only the dataset name placeholder in the template strings
    if hasattr(cfg.output, "model_dir_name"):
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace(
            "DATASET_NAME", dataset_name
        )

    if hasattr(cfg.output, "artifact_name"):
        cfg.output.artifact_name = cfg.output.artifact_name.replace(
            "DATASET_NAME", dataset_name
        )

    print("\n" + "=" * 50)
    print("Training reward model with Bradley-Terry preference learning")
    print("=" * 50)

    # Print config for visibility
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    random_seed = cfg.get("random_seed", 42)
    set_seed(random_seed)
    print(f"Global random seed set to {random_seed}")

    # Initialize wandb
    if cfg.wandb.use_wandb:
        # Generate experiment name based on data path
        dataset_name = Path(cfg.data.data_path).stem

        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"reward_{dataset_name}_{cfg.data.num_pairs}_{time.strftime('%Y%m%d_%H%M%S')}"

        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
        )

        print(f"Wandb initialized: {wandb.run.name}")

    output_dir = cfg.output.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    effective_num_workers = cfg.training.num_workers
    effective_pin_memory = cfg.training.pin_memory
    
    
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

    trajectories = process_data_trajectories(data_cpu)
    segments = segment_trajectory(trajectories, cfg.data.segment_length)

    print(
        f"Final data stats - Observation dimension: {state_dim}, Action dimension: {action_dim}"
    )
    print(
        f"Working with {len(segment_pairs) if segment_pairs is not None else 0} preference pairs across {len(segment_indices) if segment_indices is not None else 0} segments"
    )

    # Create dataset
    preference_dataset = PreferenceDataset(
        data_cpu, segment_pairs, segment_indices, preferences
    )

    # Create data loaders
    dataloaders = create_data_loaders(
        preference_dataset,
        train_ratio=0.8,  # 80% for training
        val_ratio=0.1,  # 10% for validation
        batch_size=cfg.training.batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        seed=random_seed,
        normalize_obs=cfg.data.normalize_obs,
        norm_method=cfg.data.norm_method
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Log dataset information to wandb
    if cfg.wandb.use_wandb:
        wandb.config.update(
            {
                "dataset": {
                    "name": Path(cfg.data.data_path).stem,
                    "total_pairs": len(preference_dataset),
                    "train_size": dataloaders["train_size"],
                    "val_size": dataloaders["val_size"],
                    "test_size": dataloaders["test_size"],
                    "observation_dim": state_dim,
                    "action_dim": action_dim,
                }
            }
        )

    # Initialize reward model
    model = RewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)

    # Log model info to wandb
    if cfg.wandb.use_wandb:
        wandb.config.update(
            {
                "model_info": {
                    "hidden_dims": cfg.model.hidden_dims,
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                }
            }
        )

        # Log model graph if possible
        if hasattr(wandb, "watch"):
            wandb.watch(model, log="all")

    print(f"Reward model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Start timing the training
    start_time = time.time()

    # Logging
    os.makedirs(output_dir, exist_ok=True)
    model_dir = output_dir
    dataset_name = Path(cfg.data.data_path).stem
    hidden_dims_str = "_".join(map(str, cfg.model.hidden_dims))

    sub_dir = f"{dataset_name}{pref_dataset_info}_model_seg{cfg.data.segment_length}_hidden{hidden_dims_str}_epochs{cfg.training.num_epochs}_pairs{cfg.data.num_pairs}"
    os.makedirs(os.path.join(model_dir, sub_dir), exist_ok=True)
    model_path = os.path.join(model_dir, sub_dir, "model.pt")

    training_curve_path = os.path.join(model_dir, sub_dir, "training_curve.png")

    # Train the model
    print("\nTraining reward model...")
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.model.lr,
        wandb=wandb if cfg.wandb.use_wandb else None,
        output_path=training_curve_path,
    )

    # Calculate and print training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Evaluate on test set
    test_metrics = evaluate_model_on_test_set(model, test_loader, device)

    # Log final test results to wandb
    if cfg.wandb.use_wandb:
        log_to_wandb(test_metrics, prefix="test")

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved with detailed name: {model_path}")

    # Run reward analysis
    analysis_dir = os.path.join(model_dir, sub_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Check if reward analysis is enabled (default to True)
    run_analysis = cfg.get("run_reward_analysis", True)

    if run_analysis:
        try:
            run_reward_analysis(
                model_path=model_path,
                data_path=cfg.data.data_path,
                output_dir=analysis_dir,
                num_episodes=cfg.get("analysis_episodes", 9),
                device=device,
                random_seed=random_seed,
                wandb_run=wandb if cfg.wandb.use_wandb else None,
            )
        except Exception as e:
            print(f"Warning: Error during reward analysis: {e}")

    # Finish wandb run
    if cfg.wandb.use_wandb and wandb.run:
        wandb.finish()

    print("\nReward model training complete!")


if __name__ == "__main__":
    main()
