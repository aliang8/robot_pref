import glob
import os
import time
from pathlib import Path

# from d3rlpy.metrics.scorer import evaluate_on_environment
import hydra
import numpy as np
import torch
from d3rlpy.algos import BCConfig, IQLConfig

# Import d3rlpy components
from d3rlpy.dataset import MDPDataset
from d3rlpy.datasets import MDPDataset
from d3rlpy.logging import WanDBAdapterFactory
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from models import RewardModel

# Import utility functions
from utils.data import AttrDict, load_tensordict
from utils.env import MetaWorldEnvCreator, RobomimicEnvCreator
from utils.eval import evaluate_policy_manual
from utils.seed import set_seed
from utils.viz import create_video_grid
from utils.wandb import log_to_wandb


def is_valid_video_file(file_path):
    """Simple check if a video file exists and is valid."""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def get_d3rlpy_experiment_path(base_logdir, experiment_name, with_timestamp=True):
    """Find the experiment directory in d3rlpy's logs.

    Args:
        base_logdir: Base directory for d3rlpy logs
        experiment_name: Name of the experiment
        with_timestamp: Whether the experiment directories include timestamps

    Returns:
        Path: Path to the experiment directory, or None if not found
    """
    base_path = Path(base_logdir)
    if not base_path.exists():
        return None

    # Try to find the experiment directory
    experiment_dirs = []

    if with_timestamp:
        # Match with timestamp format: {experiment_name}_{timestamp}
        for item in base_path.glob(f"{experiment_name}_*"):
            if item.is_dir():
                experiment_dirs.append(item)

        # If multiple matches, take the most recent one
        if experiment_dirs:
            # Sort by timestamp in directory name (assuming format is name_YYYYMMDD_HHMMSS)
            experiment_dirs.sort(key=lambda x: str(x), reverse=True)
            return experiment_dirs[0]
    else:
        # Look for exact match without timestamp
        experiment_dir = base_path / experiment_name
        if experiment_dir.exists() and experiment_dir.is_dir():
            return experiment_dir

    return None


def load_dataset(
    data,
    reward_model=None,
    device=None,
    use_ground_truth=False,
    max_segments=None,
    reward_batch_size=32,
    scale_rewards=False,
    reward_min=None,
    reward_max=None,
    use_zero_rewards=False,
):
    """Load and process dataset for either IQL or BC training.

    Args:
        data: TensorDict with observations, actions, rewards, and episode IDs
        reward_model: Trained reward model (required for IQL, None for BC)
        device: Device to run the reward model on (required for IQL)
        use_ground_truth: If True, use ground truth rewards for IQL instead of reward model predictions
        max_segments: Maximum number of segments to process (optional)
        reward_batch_size: Batch size for reward computation (for IQL)
        scale_rewards: If True, scales rewards to specified min/max range
        reward_min: Minimum value for scaled rewards (default: -1)
        reward_max: Maximum value for scaled rewards (default: 1)
        use_zero_rewards: If True, replace all rewards with zeros (sanity check)

    Returns:
        d3rlpy MDPDataset with observations, actions, rewards, and terminals
    """
    # Set default scaling values if not provided
    if scale_rewards:
        reward_min = reward_min if reward_min is not None else -1.0
        reward_max = reward_max if reward_max is not None else 1.0
        print(f"Scaling rewards to range [{reward_min}, {reward_max}]")

    # Extract necessary data
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    episode_ids = data["episode"]

    # For BC or ground truth rewards, extract original rewards
    if reward_model is None or use_ground_truth:
        if "reward" not in data:
            raise ValueError(
                "Ground truth rewards requested but 'reward' not found in data."
            )
        rewards = data["reward"].cpu()
        if use_ground_truth:
            print(
                "Using ground truth rewards from data instead of reward model predictions."
            )

    # Make sure data is on CPU for preprocessing
    observations = observations.cpu()
    actions = actions.cpu()
    episode_ids = episode_ids.cpu()

    # Filter out observations with NaN values
    valid_mask = ~torch.isnan(observations).any(dim=1) & ~torch.isnan(actions).any(
        dim=1
    )
    if reward_model is None or use_ground_truth:
        valid_mask = valid_mask & ~torch.isnan(rewards)

    if not valid_mask.any():
        raise ValueError("No valid observations found in the dataset.")

    # Extract valid data
    valid_obs = observations[valid_mask]
    valid_actions = actions[valid_mask]
    valid_episodes = episode_ids[valid_mask]

    if reward_model is None or use_ground_truth:
        valid_rewards = rewards[valid_mask].numpy()

    print(
        f"Using {valid_obs.shape[0]} valid observations out of {observations.shape[0]} total"
    )

    # Process rewards based on algorithm and options
    if reward_model is not None and not use_ground_truth:
        # IQL with reward model - Process in manageable batches
        process_batch_size = reward_batch_size or 1024
        all_rewards = []

        # Compute rewards using the trained reward model
        reward_model.eval()  # Ensure model is in evaluation mode

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, len(valid_obs), process_batch_size), desc="Computing rewards"
            ):
                end_idx = min(start_idx + process_batch_size, len(valid_obs))

                # Move batch to device
                batch_obs = valid_obs[start_idx:end_idx].to(device)
                batch_actions = valid_actions[start_idx:end_idx].to(device)

                # Compute rewards, need the per step reward not the summed reward
                batch_rewards = reward_model(batch_obs, batch_actions).cpu().numpy()
                all_rewards.append(batch_rewards)

        # Combine all rewards
        if len(all_rewards) == 1:
            rewards_np = all_rewards[0]
        else:
            rewards_np = np.concatenate(all_rewards)
    else:
        # BC or IQL with ground truth - use the extracted rewards
        rewards_np = valid_rewards

    # Apply zero rewards if requested (sanity check)
    if use_zero_rewards:
        print("\n" + "=" * 60)
        print("⚠️ SANITY CHECK MODE: USING ZERO REWARDS FOR ALL TRANSITIONS ⚠️")
        print(
            "This mode replaces all rewards with zeros to test if policy learning depends on rewards."
        )
        print("=" * 60 + "\n")
        original_rewards = rewards_np.copy()
        rewards_np = np.zeros_like(rewards_np)
        print(
            f"Reward stats before zeroing - Mean: {np.mean(original_rewards):.4f}, Min: {np.min(original_rewards):.4f}, Max: {np.max(original_rewards):.4f}"
        )
        print(
            f"Reward stats after zeroing - Mean: {np.mean(rewards_np):.4f}, Min: {np.min(rewards_np):.4f}, Max: {np.max(rewards_np):.4f}"
        )

    # Scale rewards if requested
    if scale_rewards:
        original_min = np.min(rewards_np)
        original_max = np.max(rewards_np)

        # Avoid division by zero
        if original_max - original_min > 1e-8:
            # Scale to [0, 1] first, then to target range
            rewards_np = (rewards_np - original_min) / (original_max - original_min)
            rewards_np = rewards_np * (reward_max - reward_min) + reward_min
            print(
                f"Scaled rewards from [{original_min:.4f}, {original_max:.4f}] to [{reward_min:.4f}, {reward_max:.4f}]"
            )
        else:
            # If all rewards are the same, set to the middle of the target range
            middle_value = (reward_max + reward_min) / 2
            rewards_np = np.ones_like(rewards_np) * middle_value
            print(
                f"All rewards have the same value ({original_min:.4f}), setting to {middle_value:.4f}"
            )

    # Create terminals array (True at the end of each episode)
    episode_ends = torch.cat(
        [
            valid_episodes[1:] != valid_episodes[:-1],
            torch.tensor([True]),  # Last observation is always an episode end
        ]
    )
    terminals_np = episode_ends.numpy()

    # Convert to numpy for d3rlpy
    observations_np = valid_obs.numpy()
    actions_np = valid_actions.numpy()

    # Create MDPDataset with the rewards
    dataset = MDPDataset(
        observations=observations_np,
        actions=actions_np,
        rewards=rewards_np,
        terminals=terminals_np,
    )

    # Print final dataset statistics
    print(
        f"Final dataset size: {dataset.size()} transitions with {dataset.size() - np.sum(terminals_np)} non-terminal transitions"
    )
    reward_stats = {
        "mean": np.mean(rewards_np),
        "std": np.std(rewards_np),
        "min": np.min(rewards_np),
        "max": np.max(rewards_np),
    }
    print(
        f"Reward statistics: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}, min={reward_stats['min']:.4f}, max={reward_stats['max']:.4f}"
    )

    return dataset


# Add a helper function to print model architecture information
def print_model_architecture(algo):
    """Print the architecture details of a d3rlpy algorithm.

    Args:
        algo: A d3rlpy algorithm instance (IQL, BC, etc.)
    """
    print("\nModel Architecture Details:")
    print("=" * 50)


@hydra.main(config_path="config", config_name="iql")
def main(cfg: DictConfig):
    """Train a policy using specified algorithm with Hydra config."""
    # Register custom resolvers for path operations
    OmegaConf.register_resolver("basename", lambda path: Path(path).stem)

    # Convert OmegaConf config to AttrDict for easier access and serialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = AttrDict.from_nested_dict(cfg_dict)

    if cfg.debug:
        cfg.training.n_epochs = 10
        cfg.training.n_steps_per_epoch = 100

    # Get algorithm name
    algorithm_name = cfg.algorithm

    # Get the dataset name and update templates
    dataset_name = Path(cfg.data.data_path).stem

    # Replace dataset name placeholder in template strings
    if hasattr(cfg.output, "model_dir_name"):
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace(
            "DATASET_NAME", dataset_name
        )

    print("\n" + "=" * 50)
    print(f"Training {algorithm_name.upper()} policy")
    print("=" * 50)

    # Print config for visibility (using original OmegaConf for pretty printing)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))

    # Set random seed for reproducibility
    random_seed = cfg.get("random_seed", 42)
    set_seed(random_seed)
    print(f"Global random seed set to {random_seed}")

    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get experiment name based on data path
    experiment_name = f"{algorithm_name.upper()}_{dataset_name}"

    # Set up d3rlpy log directory
    d3rlpy_logdir = f"{cfg.output.output_dir}/logs"

    # Load data
    print(f"Loading data from {cfg.data.data_path}")
    data = load_tensordict(cfg.data.data_path)

    # Get observation and action dimensions
    observations = data["obs"] if "obs" in data else data["state"]
    state_dim = observations.shape[1]
    action_dim = data["action"].shape[1]
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")

    # Create MDP dataset based on the algorithm
    if algorithm_name.lower() == "iql":
        # For IQL, we need a reward model
        # Load reward model
        if not cfg.data.use_zero_rewards and not cfg.data.use_ground_truth:
            reward_model = RewardModel(
                state_dim, action_dim, hidden_dims=cfg.model.hidden_dims
            )
            reward_model.load_state_dict(torch.load(cfg.data.reward_model_path))
            reward_model = reward_model.to(device)
            reward_model.eval()
            print(f"Loaded reward model from {cfg.data.reward_model_path}")

        else:
            reward_model = None

        # Check if we should use ground truth rewards
        use_ground_truth = cfg.data.get("use_ground_truth", False)
        if use_ground_truth:
            print("Using ground truth rewards instead of reward model predictions.")

        # Get reward scaling options
        scale_rewards = cfg.data.get("scale_rewards", False)
        reward_min = cfg.data.get("reward_min", -1.0)
        reward_max = cfg.data.get("reward_max", 1.0)
        if scale_rewards:
            print(f"Will scale rewards to range [{reward_min}, {reward_max}]")

        # Create MDP dataset
        print("Creating MDP dataset with rewards...")
        dataset = load_dataset(
            data,
            reward_model=reward_model,
            device=device,
            use_ground_truth=use_ground_truth,
            max_segments=cfg.data.max_segments,
            reward_batch_size=cfg.data.reward_batch_size,
            scale_rewards=scale_rewards,
            reward_min=reward_min,
            reward_max=reward_max,
            use_zero_rewards=cfg.data.get("use_zero_rewards", False),
        )
    else:  # BC or other algorithms that don't need a reward model
        # For BC, we can directly use the demonstrations
        print("Creating MDP dataset from demonstrations...")

        # Get reward scaling options (also apply to BC for consistency)
        scale_rewards = cfg.data.get("scale_rewards", False)
        reward_min = cfg.data.get("reward_min", -1.0)
        reward_max = cfg.data.get("reward_max", 1.0)
        if scale_rewards:
            print(f"Will scale rewards to range [{reward_min}, {reward_max}]")

        dataset = load_dataset(
            data,
            scale_rewards=scale_rewards,
            reward_min=reward_min,
            reward_max=reward_max,
            use_zero_rewards=cfg.data.get("use_zero_rewards", False),
        )

    # Initialize algorithm based on the algorithm_name
    print(f"Initializing {algorithm_name.upper()} algorithm...")

    if algorithm_name.lower() == "iql":
        # Initialize IQL algorithm
        iql_config = IQLConfig(**cfg.iql)
        algo = iql_config.create()
        
        # This is for wandb logging
        algo.create_impl(observation_shape=[state_dim], action_size=action_dim)

    elif algorithm_name.lower() == "bc":
        # Initialize BC algorithm
        bc_config = BCConfig(**cfg.bc)
        algo = bc_config.create()

        # This is for wandb logging
        algo.create_impl(observation_shape=[state_dim], action_size=action_dim)

        # For BC with weight decay
        if hasattr(cfg.model, "use_weight_decay") and cfg.model.use_weight_decay:
            if hasattr(algo, "create_impl"):
                impl = algo.create_impl(
                    state_dim, action_dim, algo._encoder_factory, algo._optim_factory
                )
                # Set weight decay if it's used
                if hasattr(impl.optim, "param_groups"):
                    for param_group in impl.optim.param_groups:
                        param_group["weight_decay"] = cfg.model.weight_decay

            # Fallback for older d3rlpy versions
            if hasattr(algo, "_impl") and hasattr(algo._impl, "optim"):
                for param_group in algo._impl.optim.param_groups:
                    param_group["weight_decay"] = cfg.model.weight_decay

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    
    # Initialize WanDBAdapterFactory for logging
    if cfg.wandb.use_wandb:
        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"{algorithm_name.upper()}_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"

        wandb_adapter_factory = WanDBAdapterFactory(project=cfg.wandb.project)
        print("WanDBAdapterFactory initialized")
    else:
        wandb_adapter_factory = None

    # Print model architecture details
    print_model_architecture(algo)

    # Get number of training epochs
    n_epochs = cfg.training.n_epochs
    
    # Train the model
    print(f"Training {algorithm_name.upper()} for {n_epochs} epochs...")

    # Create environment for evaluation
    env = None
    if not cfg.evaluation.skip_env_creation:
        # Use the environment name specified in the config
        if hasattr(cfg.data, "env_name") and cfg.data.env_name:
            env_name = cfg.data.env_name
            print(f"Creating environment: {env_name}")
        else:
            # Fallback to a default environment name if not specified
            env_name = "assembly-v2-goal-observable"
            print(f"No environment name specified in config. Using default: {env_name}")

        if "metaworld" in cfg.data.data_path:
            env_creator = MetaWorldEnvCreator(env_name)
        elif "robomimic" in cfg.data.data_path:
            env_creator = RobomimicEnvCreator(env_name)
        else:
            raise ValueError(
                f"No environment creator found for dataset: {cfg.data.data_path}"
            )

        # Create one environment to verify it works
        try:
            test_env = env_creator()

            # Print environment information
            print(
                f"Successfully created environment with observation space: {test_env.observation_space.shape}, action space: {test_env.action_space.shape}"
            )

            # Use the environment creator for evaluation
            env = env_creator
        except Exception as e:
            print(f"Error creating environment: {e}")
            print("Evaluation will be skipped.")
            env = None

    # Training loop
    for epoch, metrics in algo.fitter(
        dataset=dataset,
        n_steps=n_epochs * cfg.training.n_steps_per_epoch,
        n_steps_per_epoch=cfg.training.n_steps_per_epoch,
        experiment_name=experiment_name,
        with_timestamp=True,
        logger_adapter=wandb_adapter_factory,
        show_progress=True,
        save_interval=cfg.training.save_interval,
    ):
        # if first epoch, log cfgs
        if epoch == 0:
            wandb.run.config.update(cfg_dict)

        if env is not None:
            eval_model(env=env, algo=algo, cfg=cfg, epoch=epoch)
        
    # Print the training metrics summary
    print("\nTraining metrics summary:")

    # Get the model directory name from config if available, or fall back to default
    if hasattr(cfg.output, "model_dir_name"):
        model_dir_name = cfg.output.model_dir_name
        # Add zero reward indicator if using that mode
        if cfg.data.get("use_zero_rewards", False):
            model_dir_name += "_zero_rewards"

        # Create subdirectory based on the name template
        model_dir = os.path.join(cfg.output.output_dir, model_dir_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{algorithm_name.lower()}.pt")
    else:
        # Fall back to the original naming scheme
        zero_suffix = "_zero_rewards" if cfg.data.get("use_zero_rewards", False) else ""
        model_path = f"{cfg.output.output_dir}/{algorithm_name.lower()}_{Path(cfg.data.data_path).stem}{zero_suffix}.pt"

    print(f"Model saved to {model_path}")
    print("\nTraining complete!")

def eval_model(env, algo, cfg, epoch):
    """
    Run an evaluation of the trained policy in the environment,
    optionally recording videos and logging results to wandb.
    """

    # Set up video directory for final evaluation
    video_recording = getattr(cfg.evaluation, "record_video", False)
    video_path = None

    if video_recording:
        video_path = Path(cfg.output.output_dir) / f"evaluation_epoch{epoch}" / "videos"
        os.makedirs(video_path, exist_ok=True)
        print(f"Epoch {epoch} evaluation videos will be saved to: {video_path}")

    # Run evaluation
    evaluation_metrics = evaluate_policy_manual(
        env,
        algo,
        n_episodes=cfg.training.eval_episodes,
        verbose=True,
        parallel=getattr(cfg.evaluation, "parallel_eval", False),
        num_workers=getattr(cfg.evaluation, "eval_workers", 1),
        record_video=video_recording,
        video_path=video_path,
        video_fps=getattr(cfg.evaluation, "video_fps", 30),
    )

    # Print summary
    print(
        f"Final evaluation results: Mean return = {evaluation_metrics.get('mean_return', float('nan')):.2f}, "
        f"Success rate = {evaluation_metrics.get('success_rate', float('nan')):.2f}, "
        f"Episodes = {evaluation_metrics.get('num_episodes', 0)}"
    )

    # Log results to wandb
    if wandb.run is not None:
        log_to_wandb(evaluation_metrics, prefix="eval")

        # Log videos if available
        if video_recording and video_path:
            video_files = glob.glob(f"{video_path}*.mp4")
            print(f"Found {len(video_files)} final evaluation video files")

            if video_files:
                # Create a grid of videos if we have multiple
                if len(video_files) > 1:
                    print("Creating video grid from final evaluation videos...")
                    grid_path = (
                        f"{os.path.dirname(video_path)}/eval_grid_epoch{epoch}.mp4"
                    )
                    grid_path = create_video_grid(
                        video_files,
                        grid_path,
                        max_videos=6,
                        fps=getattr(cfg.evaluation, "video_fps", 30),
                    )
                    # Log the grid video with epoch as step for slider
                    wandb.log({
                        "rollouts/video_grid": wandb.Video(
                            grid_path,
                            fps=getattr(cfg.evaluation, "video_fps", 30),
                            format="mp4"
                        )
                    }, step=epoch)
                else:
                    # Log the single video with epoch as step for slider
                    wandb.log({
                        "rollouts/video": wandb.Video(
                            video_files[0],
                            fps=getattr(cfg.evaluation, "video_fps", 30),
                            format="mp4"
                        )
                    }, step=epoch)
                        

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
