import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pickle
import metaworld
import d3rlpy
from d3rlpy.algos import IQL, BC, IQLConfig, BCConfig
from d3rlpy.datasets import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory
from pathlib import Path
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt

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

from env.robomimic_lowdim import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Define a simple AttrDict class that provides dot access to dictionaries
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        """Create nested AttrDict from nested dict."""
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dict(data[key]) for key in data})

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
            break
        elif env_name in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN:
            env_constructor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name]
            found_env_name = env_name
            break
    
    # If we found a constructor directly, use it
    if env_constructor is not None:
        env = env_constructor(seed=seed)  # Use provided seed
        return env
        
    # If no direct match, list available environments for debugging
    print("\nCould not find exact environment match. Available MetaWorld environments:")
    print("Goal Observable environments (first 5):")
    for name in sorted(list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()))[:5]:
        print(f"  - {name}")
    print("Goal Hidden environments (first 5):")
    for name in sorted(list(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys()))[:5]:
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
        return env
    
    # If we can't find any matching environment
    raise ValueError(f"Could not create environment for task: {original_task_name}")

def get_robomimic_env(
    data_path,
    render=True,
    render_offscreen=True,
    use_image_obs=True,
    base_path="/scr/matthewh6/robomimic/robomimic/datasets",
    seed=42,
):
    dataset_name = Path(data_path).stem
    type, hdf5_type = dataset_name.split("_", 1)
    task = Path(data_path).parent.stem

    dataset_path = f"{base_path}/{task}/{type}/{hdf5_type}_v15.hdf5"
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    obs_modality_dict = {
        "low_dim": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
            "object",
        ],
        "rgb": ["agentview_image"],
    }

    if render_offscreen or use_image_obs:
        os.environ["MUJOCO_GL"] = "egl"

    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=render,
        # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
    )

    env.env.hard_reset = False

    env = RobomimicLowdimWrapper(env)
    env.seed(seed)

    return env


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

def log_to_wandb(metrics, epoch=None, prefix="", step=None):
    """Log any metrics to wandb with proper prefixing.
    
    Args:
        metrics: Dict of metrics or list of (epoch, metrics_dict) tuples from d3rlpy
        epoch: Current epoch (optional)
        prefix: Prefix to add to metric names (e.g., "train", "eval")
        step: Step to use for wandb logging (defaults to epoch if provided)
    
    Returns:
        bool: True if metrics were logged, False otherwise
    """
    if not wandb.run:
        return False
    
    # Use epoch as step if step not specified
    if step is None and epoch is not None:
        step = epoch
        
    # Ensure prefix ends with / if it's not empty
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    
    # Handle d3rlpy training_metrics format (list of tuples)
    if isinstance(metrics, list) and len(metrics) > 0 and isinstance(metrics[0], tuple) and len(metrics[0]) == 2:
        # Log each epoch's metrics separately
        for epoch, epoch_metrics in metrics:
            # Create metrics dict with prefix
            log_dict = {f"{prefix}{k}": v for k, v in epoch_metrics.items() 
                       if isinstance(v, (int, float, np.int64, np.float32, np.float64, np.number))}
            
            # Add epoch
            log_dict["epoch"] = epoch
            
            # Log to wandb
            if log_dict:
                wandb.log(log_dict, step=epoch)
        
        print(f"Logged {len(metrics)} epochs of {prefix.rstrip('/')} metrics to wandb")
        return True
    
    # Handle single metrics dict
    elif isinstance(metrics, dict):
        log_dict = {}
        
        # Add epoch if provided
        if epoch is not None:
            log_dict[f"{prefix}epoch"] = epoch
        
        # Add all numerical metrics with prefix
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.int64, np.float32, np.float64, np.number)):
                log_dict[f"{prefix}{key}"] = value
        
        # Log histogram for returns if available
        if "returns" in metrics and isinstance(metrics["returns"], (list, np.ndarray)):
            wandb.log({f"{prefix}returns_histogram": wandb.Histogram(metrics["returns"])}, step=step)
        
        # Log to wandb
        if log_dict:
            wandb.log(log_dict, step=step)
            return True
    
    return False

# Keep these functions for backward compatibility but implement them using the unified function
def log_evaluation_to_wandb(metrics, epoch=None, prefix=""):
    """Log evaluation metrics to wandb."""
    return log_to_wandb(metrics, epoch=epoch, prefix=prefix)

def log_training_metrics_to_wandb(training_metrics, prefix="train"):
    """Log d3rlpy training metrics to wandb."""
    return log_to_wandb(training_metrics, prefix=prefix)

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
    
class RobomimicEnvCreator:
    """A picklable environment creator for Robomimic environments."""
    
    def __init__(self, data_path):
        """Initialize the creator with the dataset name."""
        self.data_path = data_path

    def __call__(self):
        """Create a new environment with a random seed."""
        # Generate a unique seed each time this function is called
        unique_seed = int(time.time() * 1000) % 100000 + random.randint(0, 10000)
        return get_robomimic_env(self.data_path, seed=unique_seed)


class WandbCallback:
    """Callback for d3rlpy to log metrics to wandb.
    
    This callback is designed to capture training metrics that d3rlpy logs
    during training, including loss values and evaluation scores.
    """
    
    def __init__(self, use_wandb=True, prefix="train"):
        self.use_wandb = use_wandb
        self.prefix = prefix
        self.epoch = 0
        self.best_eval_metrics = None
        self.best_eval_epoch = -1
        # Track metrics across epochs
        self.training_losses = {}
        self.current_epoch_metrics = {}
        self.evaluation_scores = []
        
    def __call__(self, algo, epoch, total_step):
        """Called by d3rlpy at the end of each epoch or update step."""
        self.epoch = epoch
        
        # Basic metrics to track
        metrics = {
            "epoch": epoch,
            "total_step": total_step
        }

        # Handle both old and new d3rlpy versions
        logger = getattr(algo, '_active_logger', None)
        if logger is None and hasattr(algo, '_impl'):
            logger = getattr(algo._impl, '_active_logger', None)
        if logger is None:
            return metrics  # Return early if no logger found
        
        # Get metrics from the metrics_buffer
        if hasattr(logger, '_metrics_buffer'):
            for name, buffer in logger._metrics_buffer.items():
                if buffer:  # Check if there are values
                    # Calculate the mean of accumulated values
                    mean_value = np.mean(buffer)
                    metrics[name] = mean_value
                    
                    # Store loss metrics separately for tracking over time
                    if name.endswith('_loss') or name.startswith('loss'):
                        self.training_losses[name] = self.training_losses.get(name, []) + [mean_value]
            
            # Store evaluation scores if present
            if 'evaluation' in logger._metrics_buffer and logger._metrics_buffer['evaluation']:
                eval_score = np.mean(logger._metrics_buffer['evaluation'])
                self.evaluation_scores.append((epoch, eval_score))
                metrics['evaluation_score'] = eval_score
                    
            # Store metrics for this epoch
            self.current_epoch_metrics = metrics.copy()

        # Log to wandb if enabled
        if self.use_wandb and wandb.run:
            log_to_wandb(metrics, epoch=epoch, prefix=self.prefix)
        
        return metrics
    
    def update_eval_metrics(self, eval_metrics, epoch):
        """Track the best evaluation metrics so far."""
        # Check if these are the best metrics so far
        is_best = False
        if self.best_eval_metrics is None:
            is_best = True
        elif 'mean_return' in eval_metrics and 'mean_return' in self.best_eval_metrics:
            if eval_metrics['mean_return'] > self.best_eval_metrics['mean_return']:
                is_best = True
        
        # Update best metrics if applicable
        if is_best:
            self.best_eval_metrics = eval_metrics.copy()
            self.best_eval_epoch = epoch
            
        # Add a flag for best metrics
        eval_metrics_with_best = eval_metrics.copy()
        eval_metrics_with_best['is_best'] = is_best
        
        return eval_metrics_with_best
    
    def get_training_summary(self):
        """Get a summary of training losses and metrics."""
        summary = {
            "epoch": self.epoch,
            "best_eval_epoch": self.best_eval_epoch,
        }
        
        # Add latest values of each loss
        for loss_name, values in self.training_losses.items():
            if values:
                summary[f"final_{loss_name}"] = values[-1]
                summary[f"mean_{loss_name}"] = np.mean(values)
        
        # Add latest evaluation score if available
        if self.evaluation_scores:
            latest_eval = self.evaluation_scores[-1]
            summary["final_evaluation_score"] = latest_eval[1]
        
        return summary

class CompositeCallback:
    """A callback that combines multiple callbacks into one."""
    
    def __init__(self, callbacks):
        """Initialize with a list of callbacks.
        
        Args:
            callbacks: List of callback functions/objects to call
        """
        self.callbacks = callbacks
    
    def __call__(self, algo, epoch, total_step):
        """Call all callbacks in order."""
        results = []
        for callback in self.callbacks:
            try:
                result = callback(algo, epoch, total_step)
                results.append(result)
            except Exception as e:
                print(f"Error in callback {callback}: {e}")
        return results

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
        dataset_name = Path(cfg.data.data_path).stem
        print(f"Creating environment for evaluation using dataset: {dataset_name}")
        
        if "metaworld" in cfg.data.data_path:
            # Create an environment creator that will generate different seeds for each call
            env_creator = MetaWorldEnvCreator(dataset_name)
        elif "robomimic" in cfg.data.data_path:
            # Create an environment creator that will generate different seeds for each call
            env_creator = RobomimicEnvCreator(cfg.data.data_path)
        else:
            raise ValueError(f"No environment creator found for dataset: {cfg.data.data_path}")
        
        # Create one environment to verify it works
        test_env = env_creator()
        
        if test_env is not None:
            # Print environment information once
            print(f"Successfully created environment with observation space: {test_env.observation_space.shape}, action space: {test_env.action_space.shape}")
        
        # Use the environment creator for evaluation
        env = env_creator

    # Initialize algorithm based on the algorithm_name
    print(f"Initializing {algorithm_name.upper()} algorithm...")
    algo = None
    if algorithm_name.lower() == "iql":
        # Initialize IQL algorithm
        # TODO: This doesn't work because of version mismatch I think (using d3rlpy 2.8.1)
        # algo = IQL(
        #     actor_learning_rate=cfg.model.actor_learning_rate,
        #     critic_learning_rate=cfg.model.critic_learning_rate,
        #     batch_size=cfg.model.batch_size,
        #     gamma=cfg.model.gamma,
        #     tau=cfg.model.tau,
        #     n_critics=cfg.model.n_critics,
        #     expectile=cfg.model.expectile,
        #     weight_temp=cfg.model.weight_temp,
        #     encoder_factory=VectorEncoderFactory(cfg.model.encoder_dims),
        #     use_gpu=torch.cuda.is_available()
        # )
        iql_config = IQLConfig(**cfg.iql)
        algo = iql_config.create()
        
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
                    import glob
                    video_files = glob.glob(f"{video_path}_episode_*.mp4")
                    if video_files:
                        video_artifacts = [wandb.Video(video_file, fps=cfg.evaluation.video_fps, format="mp4") 
                                           for video_file in video_files[:3]]  # Upload up to 3 videos
                        
                        wandb.log({f"media/videos/eval_epoch_{epoch}": video_artifacts}, step=epoch)
                except Exception as e:
                    print(f"Warning: Could not upload videos to wandb: {e}")

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
        
        # Log final evaluation to wandb
        if cfg.wandb.use_wandb:
            log_to_wandb(evaluation_metrics, prefix="final")
            
            # Upload final evaluation videos if available
            if video_recording and video_path and wandb.run:
                try:
                    import glob
                    video_files = glob.glob(f"{video_path}_episode_*.mp4")
                    if video_files:
                        video_artifacts = [wandb.Video(video_file, fps=cfg.evaluation.video_fps, format="mp4") 
                                           for video_file in video_files[:3]]  # Upload up to 3 videos
                        
                        wandb.log({f"media/videos/final_eval": video_artifacts})
                except Exception as e:
                    print(f"Warning: Could not upload final videos to wandb: {e}")
    else:
        print("\nSkipping evaluation (no environment available)")
        evaluation_metrics = {
            'mean_return': 0.0,
            'std_return': 0.0,
            'success_rate': 0.0
        }
    
    # Save results
    results = {
        'config': cfg_dict,  # Use plain dict for serialization
        'final_evaluation': evaluation_metrics,
        'training_metrics': training_metrics,
        'evaluation_during_training': evaluation_results,
        'dataset_size': dataset.size(),
        # Add best evaluation metrics
        'best_eval': {
            'epoch': wandb_callback.best_eval_epoch,
            'metrics': wandb_callback.best_eval_metrics
        },
        # Add training summary metrics from callback
        'training_summary': wandb_callback.get_training_summary(),
        'training_losses': wandb_callback.training_losses
    }
    
    results_path = f"{cfg.output.output_dir}/{algorithm_name.lower()}_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {results_path}")
    print("\nTraining complete!")
    
    # Finish wandb run
    if cfg.wandb.use_wandb and wandb.run:
        # Upload the model file if it exists
        if os.path.exists(model_path):
            artifact = wandb.Artifact(f"{algorithm_name.lower()}_model_{wandb.run.id}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 