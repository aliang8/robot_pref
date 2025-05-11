import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pickle
import metaworld
import d3rlpy
from d3rlpy.algos import IQL, BC
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory
from pathlib import Path
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
import glob

# Import utility functions
from trajectory_utils import (
    RANDOM_SEED,
    load_tensordict
)

# Import reward models
from train_reward_model import SegmentRewardModel

# Import evaluation and rendering utilities
from utils.eval_utils import (
    evaluate_policy_manual,
    custom_evaluate_on_environment
)

# Import environment utilities
from utils.env_utils import (
    get_metaworld_env,
    MetaWorldEnvCreator
)

# Import visualization utilities
from utils.viz import create_video_grid

# Import data utilities
from utils.data_utils import AttrDict

# Import callback utilities
from utils.callbacks import WandbCallback, CompositeCallback

# Import wandb utilities
from utils.wandb_utils import log_to_wandb

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def create_mdp_dataset_with_sa_reward(data, reward_model, device, max_segments=1000, batch_size=32):
    """Create an MDP dataset with rewards predicted by a state-action reward model.
    
    Args:
        data: TensorDict with observations, actions, and episode IDs
        reward_model: Trained reward model to predict rewards
        device: Device to run the reward model on
        max_segments: Maximum number of segments to process (None for all)
        batch_size: Batch size for processing
        
    Returns:
        d3rlpy MDPDataset with predicted rewards
    """
    # Extract necessary data
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    episode_ids = data["episode"]
    
    # Make sure data is on CPU for preprocessing
    observations = observations.cpu()
    actions = actions.cpu()
    episode_ids = episode_ids.cpu()
    
    # Filter out observations with NaN values
    valid_mask = ~torch.isnan(observations).any(dim=1) & ~torch.isnan(actions).any(dim=1)
    
    if not valid_mask.any():
        raise ValueError("No valid observations found in the dataset.")
    
    # Extract valid data
    valid_obs = observations[valid_mask]
    valid_actions = actions[valid_mask]
    valid_episodes = episode_ids[valid_mask]
    
    print(f"Using {valid_obs.shape[0]} valid observations out of {observations.shape[0]} total")
    
    # Process in manageable batches to avoid memory issues
    process_batch_size = 1024  # Batch size for processing through reward model
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
        rewards = all_rewards[0]
    else:
        rewards = np.concatenate(all_rewards)
    
    # Create terminals array (True at the end of each episode)
    episode_ends = torch.cat([
        valid_episodes[1:] != valid_episodes[:-1],
        torch.tensor([True])  # Last observation is always an episode end
    ])
    terminals = episode_ends.numpy()
    
    # Convert to numpy for d3rlpy
    observations_np = valid_obs.numpy()
    actions_np = valid_actions.numpy()
    
    # Create MDPDataset with learned rewards
    dataset = MDPDataset(
        observations=observations_np,
        actions=actions_np,
        rewards=rewards,
        terminals=terminals
    )
    
    # Print final dataset statistics
    print(f"Final dataset size: {dataset.size()} transitions with {dataset.size() - np.sum(terminals)} non-terminal transitions")
    reward_stats = {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards)
    }
    print(f"Reward statistics: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}, min={reward_stats['min']:.4f}, max={reward_stats['max']:.4f}")
    
    return dataset

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
        
        # Create MDP dataset with learned rewards
        print("Creating MDP dataset with learned rewards...")
        dataset = create_mdp_dataset_with_sa_reward(
            data, 
            reward_model, 
            device, 
            max_segments=cfg.data.max_segments,
            batch_size=cfg.data.reward_batch_size
        )
    else:  # BC or other algorithms that don't need a reward model
        # For BC, we can directly use the demonstrations without modifying rewards
        print("Creating MDP dataset from demonstrations...")
        # Extract data ensuring no NaN values
        observations = data["obs"] if "obs" in data else data["state"]
        actions = data["action"]
        rewards = data["reward"] if "reward" in data else torch.zeros_like(actions[:, 0])
        episode_ids = data["episode"] if "episode" in data else torch.zeros_like(rewards, dtype=torch.long)
        
        # Convert to numpy and handle NaN values
        observations = observations.cpu().numpy()
        actions = actions.cpu().numpy()
        rewards = rewards.cpu().numpy()
        
        # Use episode_ids to create terminals
        episode_ids = episode_ids.cpu().numpy()
        
        # Create terminals array (1 at the end of each episode)
        terminals = np.zeros_like(rewards)
        episode_ends = np.where(np.diff(episode_ids, prepend=-1) != 0)[0]
        if len(episode_ends) > 0:
            terminals[episode_ends - 1] = 1
        
        # Filter out NaN values
        valid_mask = (~np.isnan(observations).any(axis=1)) & (~np.isnan(actions).any(axis=1)) & (~np.isnan(rewards))
        if not np.all(valid_mask):
            print(f"Filtering out {np.sum(~valid_mask)} transitions with NaN values")
            observations = observations[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            terminals = terminals[valid_mask]
        
        # Create the dataset
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )
    
    print(f"Created dataset with {dataset.size()} transitions")
    
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

    # Initialize algorithm based on the algorithm_name
    print(f"Initializing {algorithm_name.upper()} algorithm...")
    
    if algorithm_name.lower() == "iql":
        # Initialize IQL algorithm
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
        
        # Log to wandb if enabled
        if cfg.wandb.use_wandb:
            log_to_wandb(eval_metrics_with_best, epoch=epoch, prefix="eval")
            
            # Log video paths if videos were recorded
            if video_recording and video_path and wandb.run:
                # Try to find and upload videos
                try:
                    video_files = glob.glob(f"{video_path}*.mp4")
                    if video_files:
                        # Only log a few videos to save space
                        video_data = [wandb.Video(video_file, fps=cfg.evaluation.video_fps, format="mp4") 
                                    for video_file in video_files[:3]]
                        wandb.log({f"eval/videos_epoch_{epoch}": video_data})
                        print(f"Logged {len(video_data)} videos to wandb")
                except Exception as e:
                    print(f"Error logging videos to wandb: {e}")

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
                wandb.log({"media/plots/training_losses": wandb.Image(plt)})
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create training loss plot: {e}")

    # Save the model
    model_path = f"{cfg.output.output_dir}/{algorithm_name.lower()}_{Path(cfg.data.data_path).stem}.pt"
    algo.save_model(model_path)
    print(f"Model saved to {model_path}")
    
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
        
        # Log final results to wandb
        if cfg.wandb.use_wandb:
            log_to_wandb(evaluation_metrics, prefix="final_eval")
            
            # Log videos if available
            if video_recording and video_path and wandb.run:
                try:
                    video_files = glob.glob(f"{video_path}*.mp4")
                    if video_files:
                        # Upload up to 3 videos to save space
                        video_data = [wandb.Video(video_file, fps=cfg.evaluation.video_fps, format="mp4") 
                                     for video_file in video_files[:3]]
                        wandb.log({"final_eval/videos": video_data})
                except Exception as e:
                    print(f"Error logging final videos to wandb: {e}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 