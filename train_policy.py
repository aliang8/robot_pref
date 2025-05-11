import os
import time
import json
import torch
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
import glob

# Import d3rlpy components
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQL, BC
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory

# Import utility functions
from trajectory_utils import load_tensordict
from utils.env_utils import MetaWorldEnvCreator
from utils.callbacks import WandbCallback, CompositeCallback
from utils.wandb_utils import log_to_wandb
from utils.eval_utils import evaluate_policy_manual, custom_evaluate_on_environment
from utils.data_utils import AttrDict
from utils.viz import create_video_grid
from train_reward_model import SegmentRewardModel

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

def load_dataset(data, reward_model=None, device=None, use_ground_truth=False, max_segments=None, reward_batch_size=32, 
               scale_rewards=False, reward_min=None, reward_max=None):
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
            raise ValueError("Ground truth rewards requested but 'reward' not found in data.")
        rewards = data["reward"].cpu()
        if use_ground_truth:
            print("Using ground truth rewards from data instead of reward model predictions.")
    
    # Make sure data is on CPU for preprocessing
    observations = observations.cpu()
    actions = actions.cpu()
    episode_ids = episode_ids.cpu()
    
    # Filter out observations with NaN values
    valid_mask = ~torch.isnan(observations).any(dim=1) & ~torch.isnan(actions).any(dim=1)
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
    
    print(f"Using {valid_obs.shape[0]} valid observations out of {observations.shape[0]} total")
    
    # Process rewards based on algorithm and options
    if reward_model is not None and not use_ground_truth:
        # IQL with reward model - Process in manageable batches
        process_batch_size = reward_batch_size or 1024
        all_rewards = []
        
        # Compute rewards using the trained reward model
        reward_model.eval()  # Ensure model is in evaluation mode
        
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(valid_obs), process_batch_size), desc="Computing rewards"):
                end_idx = min(start_idx + process_batch_size, len(valid_obs))
                
                # Move batch to device
                batch_obs = valid_obs[start_idx:end_idx].to(device)
                batch_actions = valid_actions[start_idx:end_idx].to(device)
                
                # Compute rewards, need the per step reward not the summed reward
                batch_rewards = reward_model.reward_model(batch_obs, batch_actions).cpu().numpy()
                
                # Ensure proper shape for concatenation
                if np.isscalar(batch_rewards) or (hasattr(batch_rewards, 'shape') and batch_rewards.shape == ()):
                    batch_rewards = np.array([batch_rewards])
                    
                all_rewards.append(batch_rewards)
        
        # Combine all rewards
        if len(all_rewards) == 1:
            rewards_np = all_rewards[0]
        else:
            rewards_np = np.concatenate(all_rewards)
    else:
        # BC or IQL with ground truth - use the extracted rewards
        rewards_np = valid_rewards
    
    # Scale rewards if requested
    if scale_rewards:
        original_min = np.min(rewards_np)
        original_max = np.max(rewards_np)
        
        # Avoid division by zero
        if original_max - original_min > 1e-8:
            # Scale to [0, 1] first, then to target range
            rewards_np = (rewards_np - original_min) / (original_max - original_min)
            rewards_np = rewards_np * (reward_max - reward_min) + reward_min
            print(f"Scaled rewards from [{original_min:.4f}, {original_max:.4f}] to [{reward_min:.4f}, {reward_max:.4f}]")
        else:
            # If all rewards are the same, set to the middle of the target range
            middle_value = (reward_max + reward_min) / 2
            rewards_np = np.ones_like(rewards_np) * middle_value
            print(f"All rewards have the same value ({original_min:.4f}), setting to {middle_value:.4f}")
    
    # Create terminals array (True at the end of each episode)
    episode_ends = torch.cat([
        valid_episodes[1:] != valid_episodes[:-1],
        torch.tensor([True])  # Last observation is always an episode end
    ])
    terminals_np = episode_ends.numpy()
    
    # Convert to numpy for d3rlpy
    observations_np = valid_obs.numpy()
    actions_np = valid_actions.numpy()
    
    # Create MDPDataset with the rewards
    dataset = MDPDataset(
        observations=observations_np,
        actions=actions_np,
        rewards=rewards_np,
        terminals=terminals_np
    )
    
    # Print final dataset statistics
    print(f"Final dataset size: {dataset.size()} transitions with {dataset.size() - np.sum(terminals_np)} non-terminal transitions")
    reward_stats = {
        'mean': np.mean(rewards_np),
        'std': np.std(rewards_np),
        'min': np.min(rewards_np),
        'max': np.max(rewards_np)
    }
    print(f"Reward statistics: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}, min={reward_stats['min']:.4f}, max={reward_stats['max']:.4f}")
    
    return dataset

@hydra.main(config_path="config", config_name="iql")
def main(cfg: DictConfig):
    """Train a policy using specified algorithm with Hydra config."""
    # Convert OmegaConf config to AttrDict for easier access and serialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = AttrDict.from_nested_dict(cfg_dict)
    
    # Get algorithm name
    algorithm_name = cfg.algorithm
    
    print("\n" + "=" * 50)
    print(f"Training {algorithm_name.upper()} policy")
    print("=" * 50)
    
    # Print config for visibility (using original OmegaConf for pretty printing)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))
    
    # Set random seed for reproducibility
    random_seed = cfg.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")
    
    # Initialize wandb
    if cfg.wandb.use_wandb:
        # Generate experiment name based on data path
        dataset_name = Path(cfg.data.data_path).stem
        
        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"{algorithm_name.upper()}_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=cfg_dict,  # Use plain dict for wandb config
            tags=cfg.wandb.tags if hasattr(cfg.wandb, 'tags') else [algorithm_name],
            notes=cfg.wandb.notes
        )
        
        print(f"Wandb initialized: {wandb.run.name}")
    
    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get experiment name based on data path
    dataset_name = Path(cfg.data.data_path).stem
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
        reward_model = SegmentRewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
        reward_model.load_state_dict(torch.load(cfg.data.reward_model_path))
        reward_model = reward_model.to(device)
        reward_model.eval()
        print(f"Loaded reward model from {cfg.data.reward_model_path}")
        
        # Check if we should use ground truth rewards
        use_ground_truth = cfg.data.get('use_ground_truth', False)
        if use_ground_truth:
            print("Using ground truth rewards instead of reward model predictions.")
        
        # Get reward scaling options
        scale_rewards = cfg.data.get('scale_rewards', False)
        reward_min = cfg.data.get('reward_min', -1.0)
        reward_max = cfg.data.get('reward_max', 1.0)
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
            reward_max=reward_max
        )
    else:  # BC or other algorithms that don't need a reward model
        # For BC, we can directly use the demonstrations
        print("Creating MDP dataset from demonstrations...")
        
        # Get reward scaling options (also apply to BC for consistency)
        scale_rewards = cfg.data.get('scale_rewards', False)
        reward_min = cfg.data.get('reward_min', -1.0)
        reward_max = cfg.data.get('reward_max', 1.0)
        if scale_rewards:
            print(f"Will scale rewards to range [{reward_min}, {reward_max}]")
            
        dataset = load_dataset(
            data,
            scale_rewards=scale_rewards,
            reward_min=reward_min,
            reward_max=reward_max
        )
    
    # Create environment for evaluation
    env = None
    if not cfg.evaluation.skip_env_creation:
        # Use the environment name specified in the config
        if hasattr(cfg.data, 'env_name') and cfg.data.env_name:
            env_name = cfg.data.env_name
            print(f"Creating environment: {env_name}")
        else:
            # Fallback to a default environment name if not specified
            env_name = "assembly-v2-goal-observable"
            print(f"No environment name specified in config. Using default: {env_name}")
        
        # Create an environment creator
        env_creator = MetaWorldEnvCreator(env_name)
        
        # Create one environment to verify it works
        try:
            test_env = env_creator()
            
            # Print environment information
            print(f"Successfully created environment with observation space: {test_env.observation_space.shape}, action space: {test_env.action_space.shape}")
            
            # Use the environment creator for evaluation
            env = env_creator
        except Exception as e:
            print(f"Error creating environment: {e}")
            print("Evaluation will be skipped.")
            env = None
            
    # Create d3rlpy algorithm
    print(f"Creating {algorithm_name.upper()} algorithm...")
    if algorithm_name.lower() == "iql":
        # Create IQL
        algo = IQL(
            actor_learning_rate=cfg.model.actor_learning_rate,
            critic_learning_rate=cfg.model.critic_learning_rate,
            batch_size=cfg.model.batch_size,
            gamma=cfg.model.gamma,
            tau=cfg.model.tau,
            n_critics=cfg.model.n_critics,
            expectile=cfg.model.expectile,
            weight_temp=cfg.model.weight_temp,
            encoder_factory=VectorEncoderFactory(cfg.model.encoder_dims),
            use_gpu=torch.cuda.is_available()
        )
        
    elif algorithm_name.lower() == "bc":
        # Initialize BC algorithm
        algo = BC(
            learning_rate=cfg.model.learning_rate,
            batch_size=cfg.model.batch_size,
            encoder_factory=VectorEncoderFactory(cfg.model.encoder_dims),
            use_gpu=torch.cuda.is_available()
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    
    # Get number of training epochs
    n_epochs = cfg.training.n_epochs
    
    print(f"Training for {n_epochs} epochs")
    
    # Initialize wandb callback
    wandb_callback = WandbCallback(use_wandb=cfg.wandb.use_wandb)
    
    # For tracking evaluation metrics
    evaluation_results = []
    last_eval_epoch = -1
    
    # Define callback function for evaluation
    def evaluation_callback(algo, epoch, total_step):
        nonlocal last_eval_epoch
            
        # Only evaluate at specified intervals
        if epoch <= last_eval_epoch or (epoch % cfg.training.eval_interval != 0 and epoch != n_epochs - 1):
            return
            
        # Update last evaluated epoch
        last_eval_epoch = epoch
            
        # Check if environment is available
        if env is None:
            print(f"Epoch {epoch}: Skipping evaluation (no environment available)")
            return
            
        # Evaluate policy
        print(f"Evaluating policy at epoch {epoch}...")
        
        # Set up video recording directory in the d3rlpy results folder
        video_recording = cfg.evaluation.record_video
        video_path = None
        
        if video_recording:
            # Get the current experiment directory
            experiment_dir = get_d3rlpy_experiment_path(d3rlpy_logdir, experiment_name, with_timestamp=True)
            
            if experiment_dir:
                # Create a videos directory inside the experiment directory
                video_dir = experiment_dir / "videos" / f"epoch_{epoch}"
                os.makedirs(video_dir, exist_ok=True)
                video_path = str(video_dir / "eval")
                print(f"Videos will be saved to: {video_dir}")
            else:
                print("Warning: Could not find experiment directory for video recording")
                # Fall back to a general videos directory
                video_dir = Path(cfg.output.output_dir) / "videos" / f"epoch_{epoch}"
                os.makedirs(video_dir, exist_ok=True)
                video_path = str(video_dir / "eval")
            
        # Evaluate policy with video recording if enabled
        metrics = evaluate_policy_manual(
            env, 
            algo, 
            n_episodes=cfg.training.eval_episodes, 
            verbose=False,
            parallel=cfg.evaluation.parallel_eval,
            num_workers=cfg.evaluation.eval_workers,
            record_video=video_recording,
            video_path=video_path,
            video_fps=cfg.evaluation.video_fps
        )
        print(f"Epoch {epoch} evaluation: Return={metrics['mean_return']:.2f}, Success={metrics['success_rate']:.2f}")
        
        # Track best metrics
        eval_metrics_with_best = wandb_callback.update_eval_metrics(metrics, epoch)
        evaluation_results.append((epoch, eval_metrics_with_best))
        
        # Log metrics to wandb if enabled
        if cfg.wandb.use_wandb:
            log_to_wandb(eval_metrics_with_best, prefix="eval", epoch=epoch)
            
            # Log video grid if videos were recorded
            if video_recording and video_path and wandb.run:
                try:
                    video_files = glob.glob(f"{video_path}*.mp4")
                    # Filter out invalid video files
                    valid_video_files = [f for f in video_files if is_valid_video_file(f)]
                    print(f"Found {len(valid_video_files)} valid video files out of {len(video_files)}")
                    
                    # Create a grid of videos if we have multiple
                    if len(valid_video_files) > 1:
                        print("Creating video grid from evaluation videos...")
                        grid_path = f"{os.path.dirname(video_path)}/eval_grid_epoch_{epoch}.mp4"
                        try:
                            grid_video = create_video_grid(
                                valid_video_files, 
                                grid_path, 
                                max_videos=6, 
                                fps=cfg.evaluation.video_fps
                            )
                            if grid_video and is_valid_video_file(grid_video):
                                # Log the grid video
                                video_obj = wandb.Video(grid_video, fps=cfg.evaluation.video_fps, format="mp4")
                                log_to_wandb({"video_grid": video_obj}, prefix="eval")
                        except Exception as e:
                            print(f"Error creating video grid: {e}")
                except Exception as e:
                    print(f"Error handling videos: {e}")

    # Create a combined callback that handles both wandb logging and evaluation
    composite_callback = CompositeCallback([
        wandb_callback,
        evaluation_callback
    ])
    
    # Train the model
    print(f"Training {algorithm_name.upper()} for {n_epochs} epochs...")
    
    # Define scorers based on environment availability
    if env is not None:
        print("Using environment for evaluation during training")
        scorers = {
            'environment': custom_evaluate_on_environment(env)
        }
    else:
        print("Training without environment evaluation")
        scorers = {}

    # Train the model
    training_metrics = algo.fit(
        dataset,
        n_epochs=n_epochs,
        eval_episodes=None,  # Don't use the built-in eval which expects episodes format
        save_interval=10,
        scorers=scorers,
        experiment_name=experiment_name,
        with_timestamp=True,
        logdir=d3rlpy_logdir,
        verbose=True,
        callback=composite_callback  # Use the composite callback instead of a list
    )
    
    # Print the training metrics summary
    print("\nTraining metrics summary:")
    
    # Try to use algorithm's training metrics if available
    try:
        if training_metrics and isinstance(training_metrics, list) and len(training_metrics) > 0:
            # Check if metrics are in expected format
            if isinstance(training_metrics[0], tuple) and len(training_metrics[0]) == 2:
                for epoch, metrics in training_metrics:
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() 
                                           if isinstance(v, (int, float, np.number))])
                    print(f"Epoch {epoch}: {metrics_str}")
            else:
                print("Training metrics available but in unexpected format")
        else:
            print("No training metrics available from algorithm")
    except Exception as e:
        print(f"Error printing metrics: {e}")
    
    # Log final training metrics to wandb
    if cfg.wandb.use_wandb:
        # Log training metrics if available
        if training_metrics:
            try:
                log_to_wandb(training_metrics, prefix="train_final")
            except:
                print("Warning: Could not log algorithm's training metrics to wandb")
        
        # Get comprehensive training summary from our callback
        training_summary = wandb_callback.get_training_summary()
        
        # Create a complete summary with training and evaluation metrics
        summary_metrics = {
            "completed": True, 
            "total_epochs": n_epochs,
            "best_eval_epoch": wandb_callback.best_eval_epoch
        }
        
        # Add training loss metrics
        for key, val in training_summary.items():
            if isinstance(val, (int, float, np.int64, np.float32, np.float64, np.number)):
                summary_metrics[key] = val
        
        # Add best metrics if available
        if wandb_callback.best_eval_metrics:
            for k, v in wandb_callback.best_eval_metrics.items():
                if isinstance(v, (int, float, np.int64, np.float32, np.float64, np.number)):
                    summary_metrics[f"best_{k}"] = v
                    
        # Log the combined summary
        log_to_wandb(summary_metrics, prefix="summary")
        
        # Also log a final plot of training losses if available
        if wandb.run and wandb_callback.training_losses:
            try:
                # Create plot of training losses
                plt.figure(figsize=(10, 6))
                for loss_name, values in wandb_callback.training_losses.items():
                    if len(values) > 1:  # Only plot if we have multiple values
                        plt.plot(values, label=loss_name)
                
                plt.xlabel('Updates')
                plt.ylabel('Loss Value')
                plt.title('Training Losses')
                plt.legend()
                plt.tight_layout()
                
                # Log to wandb
                log_to_wandb({"media/plots/training_losses": wandb.Image(plt)})
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create training loss plot: {e}")

    # Save the model
    model_path = f"{cfg.output.output_dir}/{algorithm_name.lower()}_{Path(cfg.data.data_path).stem}.pt"
    algo.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Log model as wandb artifact if enabled
    if cfg.wandb.use_wandb and wandb.run:
        try:
            # Create metadata about the model
            model_metadata = {
                "algorithm": algorithm_name,
                "dataset": Path(cfg.data.data_path).stem,
                "epochs": n_epochs,
                "observation_dim": state_dim,
                "action_dim": action_dim
            }
            
            # Add best metrics if available
            if wandb_callback.best_eval_metrics:
                for k, v in wandb_callback.best_eval_metrics.items():
                    if isinstance(v, (int, float, np.int64, np.float32, np.float64, np.number)):
                        model_metadata[f"best_{k}"] = v
            
            # Log the artifact with metadata
            artifact = wandb_callback.log_model_artifact(model_path, metadata=model_metadata)
            if artifact:
                print(f"Model logged to wandb as artifact: {artifact.name}")
        except Exception as e:
            print(f"Warning: Could not log model as wandb artifact: {e}")

    # Run final comprehensive evaluation
    print("\nRunning final comprehensive evaluation...")
    if env is not None:
        # Set up video directory for final evaluation
        video_recording = cfg.evaluation.record_video
        video_path = None
        
        if video_recording:
            # Create a final evaluation video directory
            video_dir = Path(cfg.output.output_dir) / "videos" / "final_evaluation"
            os.makedirs(video_dir, exist_ok=True)
            video_path = str(video_dir / "final_eval")
            print(f"Final evaluation videos will be saved to: {video_dir}")
            
        evaluation_metrics = evaluate_policy_manual(
            env, 
            algo, 
            n_episodes=cfg.training.eval_episodes, 
            verbose=True,
            parallel=cfg.evaluation.parallel_eval,
            num_workers=cfg.evaluation.eval_workers,
            record_video=video_recording,
            video_path=video_path,
            video_fps=cfg.evaluation.video_fps
        )
        
        # Print summary
        print(f"Final evaluation results: Mean return = {evaluation_metrics['mean_return']:.2f}, " + 
              f"Success rate = {evaluation_metrics['success_rate']:.2f}, " +
              f"Episodes = {evaluation_metrics['num_episodes']}")
        
        # Log final evaluation results to wandb and final video grid only
        if cfg.wandb.use_wandb:
            log_to_wandb(evaluation_metrics, prefix="final_eval")
            
            # Log video grid if available
            if video_recording and video_path and wandb.run:
                try:
                    video_files = glob.glob(f"{video_path}*.mp4")
                    # Filter out invalid video files
                    valid_video_files = [f for f in video_files if is_valid_video_file(f)]
                    print(f"Found {len(valid_video_files)} valid video files out of {len(video_files)}")
                    
                    # Create a grid of videos if we have multiple
                    if len(valid_video_files) > 1:
                        print("Creating video grid from final evaluation videos...")
                        grid_path = f"{os.path.dirname(video_path)}/final_eval_grid.mp4"
                        try:
                            grid_video = create_video_grid(
                                valid_video_files, 
                                grid_path, 
                                max_videos=6, 
                                fps=cfg.evaluation.video_fps
                            )
                            if grid_video and is_valid_video_file(grid_video):
                                # Log the grid video
                                video_obj = wandb.Video(grid_video, fps=cfg.evaluation.video_fps, format="mp4")
                                log_to_wandb({"video_grid": video_obj}, prefix="final_eval")
                        except Exception as e:
                            print(f"Error creating final video grid: {e}")
                except Exception as e:
                    print(f"Error handling final videos: {e}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 