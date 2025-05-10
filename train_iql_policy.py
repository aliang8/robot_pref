import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pickle
import metaworld
import d3rlpy
from d3rlpy.algos import IQL
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory
from pathlib import Path
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

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
            for i in range(len(batch_observations)):
                obs = batch_observations[i].to(device)
                act = batch_actions[i].to(device)
                
                # Process each observation-action pair for this segment
                segment_rewards = []
                
                # Process in mini-batches if segment is very long
                obs_actions = torch.cat([obs, act], dim=1)
                mini_batch_size = 64  # Adjust based on your GPU memory
                
                for j in range(0, len(obs_actions), mini_batch_size):
                    mini_batch = obs_actions[j:j+mini_batch_size]
                    rewards_chunk = reward_model.reward_model.model(mini_batch).cpu().numpy().flatten()
                    segment_rewards.extend(rewards_chunk)
                
                batch_learned_rewards.append(np.array(segment_rewards))
        
        # Add processed segments to overall dataset
        for i in range(len(batch_observations)):
            # Convert all to numpy for consistency
            all_observations.append(batch_observations[i].numpy())
            all_actions.append(batch_actions[i].numpy())
            all_rewards.append(batch_learned_rewards[i])
            all_terminals.append(batch_terminals[i])
        
        processed_segments += len(batch_observations)
    
    # Check if we have any valid data
    if len(all_observations) == 0:
        raise ValueError("No valid segments found after NaN filtering. Check your data.")
        
    # Concatenate data
    observations = np.concatenate(all_observations)
    actions = np.concatenate(all_actions)
    rewards = np.concatenate(all_rewards)
    terminals = np.concatenate(all_terminals)
    
    # Perform final NaN check
    if np.isnan(observations).any() or np.isnan(actions).any() or np.isnan(rewards).any():
        print("WARNING: NaN values still present after filtering!")
        # Replace remaining NaNs with zeros
        observations = np.nan_to_num(observations, nan=0.0)
        actions = np.nan_to_num(actions, nan=0.0)
        rewards = np.nan_to_num(rewards, nan=0.0)
    
    print(f"Final dataset: {observations.shape[0]} transitions from {len(all_observations)} valid segments")
    
    # Create MDPDataset
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )
    
    return dataset

def get_metaworld_env(task_name, seed=42):
    """Create a MetaWorld environment for the given task.
    
    Args:
        task_name: Name of the MetaWorld task to create
        seed: Random seed for the environment
    
    Returns:
        MetaWorld environment instance
    """
    
    # Clean up task name from file path or other formats
    original_task_name = task_name
    
    if '/' in task_name:
        task_name = task_name.split('/')[-1]
    
    # Remove prefixes/suffixes often found in filenames
    prefixes = ['buffer_', 'data_']
    for prefix in prefixes:
        if task_name.startswith(prefix):
            task_name = task_name[len(prefix):]
    
    # Remove file extensions
    if task_name.endswith('.pt') or task_name.endswith('.pkl'):
        task_name = task_name.rsplit('.', 1)[0]
    
    print(f"Creating MetaWorld environment for task: {task_name} (from {original_task_name})")
    
    # Method 1: Direct access to environment constructors (preferred method)
    from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                              ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
    
    # Prepare task name formats to try
    task_name_base = task_name
    if task_name.endswith('-v2') or task_name.endswith('-v1'):
        task_name_base = task_name[:-3]
    
    # Try different variations of the environment name
    env_variations = [
        f"{task_name}-goal-observable",                # If already has version suffix
        f"{task_name}-v2-goal-observable",             # Add v2 if not there
        f"{task_name_base}-v2-goal-observable",        # Clean base name with v2
        f"{task_name}-goal-hidden",                    # Hidden goal versions
        f"{task_name}-v2-goal-hidden",
        f"{task_name_base}-v2-goal-hidden",
    ]
    
    # Try to find and use the environment constructor
    env_constructor = None
    found_env_name = None
    
    for env_name in env_variations:
        if env_name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
            env_constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
            found_env_name = env_name
            print(f"Found goal-observable environment: {env_name}")
            break
        elif env_name in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN:
            env_constructor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name]
            found_env_name = env_name
            print(f"Found goal-hidden environment: {env_name}")
            break
    
    # If we found a constructor directly, use it
    if env_constructor is not None:
        env = env_constructor(seed=seed)  # Use provided seed
        print(f"Successfully created environment: {found_env_name}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        return env
        
    # If no direct match, list available environments for debugging
    print("Available MetaWorld environments:")
    print("\nGoal Observable environments:")
    for name in sorted(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()):
        print(f"  - {name}")
    print("\nGoal Hidden environments:")
    for name in sorted(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys()):
        print(f"  - {name}")
        
    # Try to find a similar environment name
    best_match = None
    best_score = 0
    
    # Check for partial matches in environment names
    for env_dict in [ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN]:
        for env_name in env_dict.keys():
            # Simple string similarity - check if task name is in env name
            if task_name_base in env_name:
                score = len(task_name_base)
                if score > best_score:
                    best_score = score
                    best_match = (env_name, env_dict[env_name])
    
    if best_match:
        env_name, constructor = best_match
        print(f"Found closest matching environment: {env_name}")
        env = constructor(seed=seed)  # Use provided seed
        print(f"Successfully created environment: {env_name}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        return env
    
    # If we can't find any matching environment
    raise ValueError(f"Could not create environment for task: {original_task_name}")

def get_d3rlpy_experiment_path(base_logdir, experiment_name, with_timestamp=True):
    """Get the path to the d3rlpy experiment directory.
    
    Args:
        base_logdir: Base logging directory
        experiment_name: Name of the experiment
        with_timestamp: Whether the directory has a timestamp
        
    Returns:
        Path to the experiment directory (or None if not found)
    """
    # If timestamp is used, the format is {experiment_name}_{timestamp}
    # If no timestamp, just use the experiment name
    
    experiment_dir = Path(base_logdir)
    if not experiment_dir.exists():
        return None
        
    # If with_timestamp is False, just use the experiment name directly
    if not with_timestamp:
        return experiment_dir / experiment_name
    
    # Look for directories that start with the experiment name
    matching_dirs = list(experiment_dir.glob(f"{experiment_name}_*"))
    
    # Sort by creation time (most recent first)
    matching_dirs.sort(key=lambda p: p.stat().st_ctime, reverse=True)
    
    # Return the most recent one if any exists
    if matching_dirs:
        return matching_dirs[0]
        
    return None

def log_evaluation_to_wandb(metrics, epoch=None, prefix=""):
    """Log evaluation metrics to wandb."""
    if not wandb.run:
        return
        
    log_dict = {}
    
    # Add prefix to metric names if provided
    pre = f"{prefix}_" if prefix else ""
    
    if epoch is not None:
        log_dict[f"{pre}epoch"] = epoch
        
    # Log numerical metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.int64, np.float32, np.float64)):
            log_dict[f"{pre}{key}"] = value
            
    # If returns array is available, log a histogram
    if "returns" in metrics and isinstance(metrics["returns"], (list, np.ndarray)):
        if wandb.run:
            wandb.log({f"{pre}returns_histogram": wandb.Histogram(metrics["returns"])}, step=epoch if epoch is not None else None)
    
    # Log to wandb if any metrics to log
    if log_dict and wandb.run:
        wandb.log(log_dict, step=epoch if epoch is not None else None)

class MetaWorldEnvCreator:
    """A picklable environment creator for MetaWorld environments."""
    
    def __init__(self, dataset_name):
        """Initialize the creator with the dataset name."""
        self.dataset_name = dataset_name
    
    def __call__(self):
        """Create a new environment with a random seed."""
        # Generate a unique seed each time this function is called
        unique_seed = int(time.time() * 1000) % 100000 + random.randint(0, 10000)
        return get_metaworld_env(self.dataset_name, seed=unique_seed)

@hydra.main(config_path="config/train_iql", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Train an IQL policy using learned reward model with Hydra config."""
    print("\n" + "=" * 50)
    print("Training IQL policy with learned rewards")
    print("=" * 50)
    
    # Print config for visibility
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize wandb
    if cfg.wandb.use_wandb:
        # Generate experiment name based on data path
        dataset_name = Path(cfg.data.data_path).stem
        
        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"IQL_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
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
    experiment_name = f"IQL_{dataset_name}_SA"
    
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
    print(f"Created dataset with {dataset.size()} transitions")
    
    # Create environment for evaluation
    env = None
    if not cfg.evaluation.skip_env_creation:
        dataset_name = Path(cfg.data.data_path).stem
        print(f"Creating environment for evaluation using dataset: {dataset_name}")
        
        # Create an environment creator that will generate different seeds for each call
        env_creator = MetaWorldEnvCreator(dataset_name)
        
        # Create one environment to verify it works
        test_env = env_creator()
        print("Successfully created environment for evaluation")
        
        # Use the environment creator for evaluation
        env = env_creator

    # Initialize IQL algorithm
    print("Initializing IQL algorithm...")
    iql = IQL(
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
    
    # For tracking evaluation metrics
    evaluation_results = []
    last_eval_epoch = -1
    
    # Define callback function for evaluation
    def evaluation_callback(algo, epoch, total_step):
        nonlocal last_eval_epoch
            
        # Only evaluate at specified intervals
        if epoch <= last_eval_epoch or (epoch % cfg.training.eval_interval != 0 and epoch != cfg.training.iql_epochs - 1):
            return
            
        # Update last evaluated epoch
        last_eval_epoch = epoch
            
        # Check if environment is available
        if env is None:
            print(f"Epoch {epoch}: Skipping evaluation (no environment available)")
            return
            
        # Evaluate policy
        print(f"\nEvaluating policy at epoch {epoch}...")
            
        metrics = evaluate_policy_manual(
            env, 
            algo, 
            n_episodes=cfg.training.eval_episodes, 
            verbose=False,
            parallel=cfg.evaluation.parallel_eval,
            num_workers=cfg.evaluation.eval_workers
        )
        print(f"Mean return: {metrics['mean_return']:.2f}, Success rate: {metrics['success_rate']:.2f}")
        evaluation_results.append((epoch, metrics))
        
        # Log to wandb if enabled
        if cfg.wandb.use_wandb:
            log_evaluation_to_wandb(metrics, epoch=epoch, prefix="eval")
    
    # Train IQL
    print(f"Training IQL for {cfg.training.iql_epochs} epochs...")
    
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
    training_metrics = iql.fit(
        dataset,
        n_epochs=cfg.training.iql_epochs,
        eval_episodes=None,  # Don't use the built-in eval which expects episodes format
        save_interval=10,
        scorers=scorers,
        experiment_name=experiment_name,
        with_timestamp=True,
        logdir=d3rlpy_logdir,
        verbose=True,
        callback=evaluation_callback
    )
    
    # Print the training metrics summary
    print("\nTraining metrics summary:")
    for epoch, metrics in training_metrics:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch}: {metrics_str}")
    
    # Save the model
    model_path = f"{cfg.output.output_dir}/iql_{Path(cfg.data.data_path).stem}_sa.pt"
    iql.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Run final comprehensive evaluation
    print("\nRunning final comprehensive evaluation...")
    if env is not None:
        evaluation_metrics = evaluate_policy_manual(
            env, 
            iql, 
            n_episodes=cfg.training.eval_episodes, 
            verbose=True,
            parallel=cfg.evaluation.parallel_eval,
            num_workers=cfg.evaluation.eval_workers
        )
        
        # Log final evaluation to wandb
        if cfg.wandb.use_wandb:
            log_evaluation_to_wandb(evaluation_metrics, prefix="final")
    else:
        print("\nSkipping evaluation (no environment available)")
        evaluation_metrics = {
            'mean_return': 0.0,
            'std_return': 0.0,
            'success_rate': 0.0
        }
    
    # Save results
    results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'final_evaluation': evaluation_metrics,
        'training_metrics': training_metrics,
        'evaluation_during_training': evaluation_results,
        'dataset_size': dataset.size()
    }
    
    results_path = f"{cfg.output.output_dir}/sa_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {results_path}")
    print("\nTraining complete!")
    
    # Finish wandb run
    if cfg.wandb.use_wandb and wandb.run:
        # Upload the model file if it exists
        if os.path.exists(model_path):
            artifact = wandb.Artifact(f"iql_model_{wandb.run.id}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 