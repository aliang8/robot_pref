import os
import time
import json
import torch
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import pickle
import metaworld
import d3rlpy
from d3rlpy.algos import IQL, BC
from d3rlpy.datasets import MDPDataset
# from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory
from pathlib import Path
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler

# Import d3rlpy components
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQL, DiscreteBC, BC, BCConfig
# from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory

# Import utility functions
from trajectory_utils import load_tensordict, RANDOM_SEED
from utils.env_utils import MetaWorldEnvCreator, RobomimicEnvCreator
from utils.callbacks import WandbCallback, CompositeCallback
from utils.wandb_utils import log_to_wandb, reset_global_step
from utils.eval_utils import evaluate_policy_manual, custom_evaluate_on_environment
from utils.data_utils import AttrDict
from utils.viz import create_video_grid
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
    """Create MDPDataset from tensordict data using a state-action reward model.
    
    Args:
        data: TensorDict containing trajectories
        reward_model: Trained reward model
        device: Device to use for computation
        max_segments: Maximum number of segments to sample (0 for all)
        batch_size: Batch size for reward prediction to speed up processing
    """
    # Extract relevant info from data
    observations = data["obs"] if "obs" in data else data["state"]
    actions = data["action"]
    rewards = data["reward"]  # Original rewards (will be replaced with learned rewards)
    episode_ids = data["episode"]
    
    # Ensure tensors are on CPU for numpy conversion
    observations = observations.cpu()
    actions = actions.cpu()
    rewards = rewards.cpu()
    episode_ids = episode_ids.cpu().numpy()
    
    # Create clean masks to handle NaN values - use vectorized operations
    print("Creating masks for NaN values...")
    obs_mask = ~torch.isnan(observations).any(dim=1)
    action_mask = ~torch.isnan(actions).any(dim=1)
    reward_mask = ~torch.isnan(rewards)
    
    # Combined mask for valid transitions
    valid_mask = obs_mask & action_mask & reward_mask
    
    # Count NaNs to report
    total_transitions = len(observations)
    valid_transitions = valid_mask.sum().item()
    print(f"Found {total_transitions - valid_transitions} NaN transitions out of {total_transitions} total transitions")
    
    # Split data into episodes
    unique_episodes = np.unique(episode_ids)
    
    all_valid_segments = []
    
    # Process each episode to find valid segments
    for episode_id in tqdm(unique_episodes, desc="Finding valid segments"):
        # Get episode data
        episode_mask = (episode_ids == episode_id)
        # Apply valid mask within this episode
        episode_valid_mask = valid_mask[episode_mask].numpy()
        
        # Skip if no valid transitions in this episode
        if not np.any(episode_valid_mask):
            continue
            
        # Get valid episode data
        episode_indices = np.where(episode_mask)[0][episode_valid_mask]
        
        # Skip if less than 2 valid transitions (need at least obs-action-obs)
        if len(episode_indices) < 2:
            continue
            
        # Check for consecutive indices - multiple segments may exist in one episode if NaNs break it up
        diffs = np.diff(episode_indices)
        break_points = np.where(diffs > 1)[0]
        
        # Process each consecutive segment in this episode
        start_idx = 0
        for bp in list(break_points) + [len(episode_indices) - 1]:
            # Get consecutive segment
            segment_indices = episode_indices[start_idx:bp+1]
            
            # Skip if too short
            if len(segment_indices) < 2:
                start_idx = bp + 1
                continue
            
            # Store this valid segment
            all_valid_segments.append(segment_indices)
            
            start_idx = bp + 1
    
    print(f"Found {len(all_valid_segments)} valid segments across all episodes")
    
    # Subsample segments if we have more than max_segments
    if max_segments > 0 and len(all_valid_segments) > max_segments:
        print(f"Subsampling {max_segments} segments from the {len(all_valid_segments)} available")
        selected_segments = random.sample(all_valid_segments, max_segments)
    else:
        selected_segments = all_valid_segments
    
    # Prepare for batch processing of rewards
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    
    # Process segments in batches for more efficient reward computation
    reward_model.eval()  # Ensure model is in eval mode
    
    # Group segments by similar length to minimize padding
    segment_lengths = [len(seg) for seg in selected_segments]
    length_sorted_indices = np.argsort(segment_lengths)
    
    # Process in sorted order for better batching efficiency
    processed_segments = 0
    segment_batches = []
    current_batch = []
    
    # Group segments with similar lengths together
    for idx in length_sorted_indices:
        current_batch.append(selected_segments[idx])
        if len(current_batch) >= batch_size:
            segment_batches.append(current_batch)
            current_batch = []
    
    # Add any remaining segments
    if current_batch:
        segment_batches.append(current_batch)
    
    # Process each batch of segments
    for batch_idx, segment_batch in enumerate(tqdm(segment_batches, desc="Processing segment batches")):
        batch_observations = []
        batch_actions = []
        batch_terminals = []
        batch_learned_rewards = []
        
        # First extract all observations and actions for current batch
        for segment_indices in segment_batch:
            # Get observations and actions for this segment
            segment_obs = observations[segment_indices]
            segment_actions = actions[segment_indices[:-1]]  # Last observation has no action
            
            # Check for NaN values
            if torch.isnan(segment_obs[:-1]).any() or torch.isnan(segment_actions).any():
                continue
                
            # Add observation and action to batch
            batch_observations.append(segment_obs[:-1])  # Excluding last observation
            batch_actions.append(segment_actions)
            
            # Create terminal flags (1 for last state in segment)
            terminals = np.zeros(len(segment_obs) - 1)
            terminals[-1] = 1
            batch_terminals.append(terminals)
        
        # Skip batch if all segments had NaNs
        if not batch_observations:
            continue
            
        # Compute rewards for all segments in batch at once
        with torch.no_grad():
            # Process each segment separately
            for i in range(len(batch_observations)):
                segment_obs = batch_observations[i].to(device)
                segment_actions = batch_actions[i].to(device)
                
                # Get reward predictions from reward model
                segment_rewards = []
                
                # Process in smaller sub-batches if the segment is very large
                sub_batch_size = 1024  # Memory efficient sub-batch size
                
                for j in range(0, len(segment_obs), sub_batch_size):
                    sub_obs = segment_obs[j:j+sub_batch_size]
                    sub_actions = segment_actions[j:j+sub_batch_size]
                    
                    sub_rewards = reward_model(sub_obs, sub_actions).cpu().numpy()
                    segment_rewards.append(sub_rewards)
                
                segment_rewards = np.concatenate(segment_rewards)
                batch_learned_rewards.append(segment_rewards)
        
        # Add batch data to dataset
        for i in range(len(batch_observations)):
            all_observations.append(batch_observations[i].cpu().numpy())
            all_actions.append(batch_actions[i].cpu().numpy())
            all_rewards.append(batch_learned_rewards[i])
            all_terminals.append(batch_terminals[i])
    
    # Check if we have any valid segments after filtering
    if not all_observations:
        raise ValueError("No valid segments found after processing.")
    
    # Concatenate all data
    all_observations = np.concatenate(all_observations)
    all_actions = np.concatenate(all_actions)
    all_rewards = np.concatenate(all_rewards)
    all_terminals = np.concatenate(all_terminals)
    
    # Create the D3RL dataset with the learned rewards
    dataset = MDPDataset(
        observations=all_observations,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals
    )
    
    # Print final dataset statistics
    print(f"Final dataset size: {dataset.size()} transitions with {dataset.size() - np.sum(all_terminals)} non-terminal transitions")
    reward_stats = {
        'mean': np.mean(all_rewards),
        'std': np.std(all_rewards),
        'min': np.min(all_rewards),
        'max': np.max(all_rewards)
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

    if cfg.debug:
        cfg.training.n_epochs = 10
        cfg.training.n_steps_per_epoch = 100
    
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
        
        # Reset global step counter to ensure clean start
        reset_global_step(0)
        
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
        
        if "metaworld" in cfg.data.data_path:
            # Create an environment creator that will generate different seeds for each call
            env_creator = MetaWorldEnvCreator(dataset_name)
        # elif "robomimic" in cfg.data.data_path:
        else:
            # Create an environment creator that will generate different seeds for each call
            env_creator = RobomimicEnvCreator(cfg.data.data_path)
        # else:
        #     raise ValueError(f"No environment creator found for dataset: {cfg.data.data_path}")
        
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
        # TODO: This doesn't work because of version mismatch I think (using d3rlpy 2.8.1)
        # algo = BC(
        #     learning_rate=cfg.model.learning_rate,
        #     batch_size=cfg.model.batch_size,
        #     encoder_factory=VectorEncoderFactory(cfg.model.encoder_dims),
        #     use_gpu=torch.cuda.is_available()
        # )
        bc_config = BCConfig(**cfg.bc)
        algo = bc_config.create()
        
        # For BC with weight decay
        if hasattr(cfg.model, 'use_weight_decay') and cfg.model.use_weight_decay:
            if hasattr(algo, 'create_impl'):
                impl = algo.create_impl(
                    state_dim, 
                    action_dim, 
                    algo._encoder_factory, 
                    algo._optim_factory
                )
                # Set weight decay if it's used
                if hasattr(impl.optim, 'param_groups'):
                    for param_group in impl.optim.param_groups:
                        param_group['weight_decay'] = cfg.model.weight_decay
                        
            # Fallback for older d3rlpy versions
            if hasattr(algo, '_impl') and hasattr(algo._impl, 'optim'):
                for param_group in algo._impl.optim.param_groups:
                    param_group['weight_decay'] = cfg.model.weight_decay
        
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
                    print(f"Found {len(video_files)} video files: {video_files}")
                    
                    if video_files:
                        # Log individual videos (maximum 3)
                        for i, video_file in enumerate(video_files[:3]):
                            wandb_callback.log_video(
                                video_file,
                                name=f"videos_epoch_{epoch}_{i+1}",
                                fps=cfg.evaluation.video_fps,
                                prefix="eval_rollout"
                            )
                            
                        # Create a grid of videos if we have multiple
                        if len(video_files) > 1:
                            print("Creating video grid from evaluation videos...")
                            grid_path = f"{os.path.dirname(video_path)}/eval_grid_epoch_{epoch}.mp4"
                            try:
                                grid_video = create_video_grid(
                                    video_files, 
                                    grid_path, 
                                    max_videos=6, 
                                    fps=cfg.evaluation.video_fps
                                )
                                if grid_video:
                                    # Log the grid video
                                    wandb_callback.log_video(
                                        grid_video,
                                        name=f"video_grid_epoch_{epoch}",
                                        fps=cfg.evaluation.video_fps,
                                        prefix="eval_rollout"
                                    )
                            except Exception as e:
                                print(f"Error creating video grid: {e}")
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
    # TODO: This doesn't work because of version mismatch I think (using d3rlpy 2.8.1)
    # training_metrics = algo.fit(
    #     dataset,
    #     n_epochs=n_epochs,
    #     eval_episodes=None,  # Don't use the built-in eval which expects episodes format
    #     save_interval=10,
    #     scorers=scorers,
    #     experiment_name=experiment_name,
    #     with_timestamp=True,
    #     logdir=d3rlpy_logdir,
    #     verbose=True,
    #     callback=composite_callback  # Use the composite callback instead of a list
    # )
    training_metrics = algo.fit(
        dataset,
        n_steps=n_epochs * cfg.training.n_steps_per_epoch,
        n_steps_per_epoch=cfg.training.n_steps_per_epoch,
        save_interval=10,
        evaluators=scorers,
        experiment_name=experiment_name,
        with_timestamp=True,
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
        
        # Log final results to wandb
        if cfg.wandb.use_wandb:
            log_to_wandb(evaluation_metrics, prefix="final_eval")
            
            # Log videos if available
            if video_recording and video_path and wandb.run:
                try:
                    video_files = glob.glob(f"{video_path}*.mp4")
                    print(f"Found {len(video_files)} final evaluation video files")
                    
                    if video_files:
                        # Upload up to 3 videos
                        for i, video_file in enumerate(video_files[:3]):
                            wandb_callback.log_video(
                                video_file,
                                name=f"video_{i+1}",
                                fps=cfg.evaluation.video_fps,
                                prefix="final_rollout"
                            )
                            
                        # Create a grid of videos if we have multiple
                        if len(video_files) > 1:
                            print("Creating video grid from final evaluation videos...")
                            grid_path = f"{os.path.dirname(video_path)}/final_eval_grid.mp4"
                            try:
                                grid_video = create_video_grid(
                                    video_files, 
                                    grid_path, 
                                    max_videos=6, 
                                    fps=cfg.evaluation.video_fps
                                )
                                if grid_video:
                                    # Log the grid video
                                    wandb_callback.log_video(
                                        grid_video,
                                        name="video_grid",
                                        fps=cfg.evaluation.video_fps,
                                        prefix="final_rollout"
                                    )
                            except Exception as e:
                                print(f"Error creating final video grid: {e}")
                except Exception as e:
                    print(f"Error logging final videos to wandb: {e}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 