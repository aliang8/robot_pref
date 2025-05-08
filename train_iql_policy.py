import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
import gym
import metaworld
import d3rlpy
from d3rlpy.algos import IQL
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory
from pathlib import Path

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict
)

# Import reward models
from train_reward_model import SegmentRewardModel, StateActionRewardModel

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

def get_metaworld_env(task_name):
    """Create a MetaWorld environment for the given task."""
    
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
    
    try:
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
            env = env_constructor(seed=42)  # Use consistent seed
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
            env = constructor(seed=42)
            print(f"Successfully created environment: {env_name}")
            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            return env
                
        # Fall back to ML1 method if direct environment access fails
        raise ValueError(f"Could not find a direct environment constructor for {task_name}")
        
    except ImportError as e:
        print(f"Could not import ALL_V2_ENVIRONMENTS: {e}")
        print("Falling back to ML1 method...")
    except Exception as e:
        print(f"Error using direct environment constructors: {e}")
        print("Falling back to ML1 method...")
    
    # Method 2: ML1 method (fallback if direct environment access fails)
    try:
        # Try to create an ML1 environment with the given task
        if not task_name.endswith('-v2') and not task_name.endswith('-v1'):
            task_name = f"{task_name}-v2"  # Add version if not present
            
        print(f"Trying ML1 with task: {task_name}")
        ml1 = metaworld.ML1(task_name)
        
        # Get base name without version
        base_name = task_name
        if '-v' in base_name:
            base_name = base_name.split('-v')[0]
            
        env = ml1.train_classes[base_name]()
        task = ml1.train_tasks[0]
        env.set_task(task)
        
        print(f"Successfully created environment with ML1: {task_name}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        return env
        
    except Exception as e:
        print(f"Error creating environment with ML1: {e}")
        
        # Try to get list of available ML1 environments
        try:
            ml1_envs = [attr for attr in dir(metaworld.ML1) if attr.startswith('ENV_')]
            available_tasks = []
            for env_set in ml1_envs:
                env_names = getattr(metaworld.ML1, env_set)
                available_tasks.extend(env_names)
                
            print("Available ML1 environments:")
            for task in sorted(available_tasks):
                print(f"  - {task}")
                
        except Exception:
            print("Could not list available ML1 environments")
            
        raise ValueError(f"Could not create environment for task: {original_task_name}")

def custom_evaluate_on_environment(env):
    """Custom environment evaluation function that handles observation conversion properly."""
    def scorer(algo, *args, **kwargs):
        # Check environment compatibility with the model
        env_obs_dim = env.observation_space.shape[0]
        
        # Get model's expected observation dimension through various ways
        model_obs_dim = None
        if hasattr(algo, 'observation_shape'):
            model_obs_dim = algo.observation_shape[0]
        elif hasattr(algo, '_impl') and hasattr(algo._impl, 'observation_shape'):
            model_obs_dim = algo._impl.observation_shape[0]
        
        if model_obs_dim is not None and env_obs_dim != model_obs_dim:
            print(f"Warning: Environment observation dimension ({env_obs_dim}) doesn't match model's expected dimension ({model_obs_dim})")
            print("Skipping evaluation with incompatible environment")
            return 0.0
        
        total_reward = 0.0
        n_episodes = 5  # Number of episodes to evaluate on for each call
        
        for episode in range(n_episodes):
            observation = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                # Extract observation if it's a tuple
                if isinstance(observation, tuple):
                    obs_array = observation[0]
                else:
                    obs_array = observation
                    
                # Ensure observation is a numpy array with batch dimension
                if not isinstance(obs_array, np.ndarray):
                    obs_array = np.array(obs_array, dtype=np.float32)
                
                if len(obs_array.shape) == 1:
                    obs_array = np.expand_dims(obs_array, axis=0)
                
                action = algo.predict(obs_array)[0]
                
                # Handle different step return formats
                step_result = env.step(action)
                
                # MetaWorld environments return 4 values: obs, reward, done, info
                if len(step_result) == 4:
                    observation, reward, done, _ = step_result
                # Some environments might return 5 values including truncated flag (gym>=0.26)
                elif len(step_result) == 5:
                    observation, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    print(f"Warning: Unexpected step result format: {step_result}")
                    break
                    
                episode_reward += reward
            
            total_reward += episode_reward
            
        # Return average reward across episodes
        avg_reward = total_reward / n_episodes
        print(f"Evaluation during training: {avg_reward:.2f} average reward over {n_episodes} episodes")
        return avg_reward
    
    return scorer

def evaluate_policy_manual(env, algo, n_episodes=10, verbose=True):
    """Manually evaluate policy on environment and return detailed metrics."""
    returns = []
    success_rate = 0
    steps_to_complete = []
    
    # Check if we can even run evaluation
    if env is None:
        print("No environment available for evaluation.")
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'success_rate': 0.0,
            'num_episodes': 0
        }
    
    # Print environment info
    if verbose:
        print(f"Evaluating on environment with:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
    
    for episode in tqdm(range(n_episodes), desc="Evaluating policy"):
        observation = env.reset()
        done = False
        episode_return = 0
        steps = 0
        
        while not done and steps < 1000:  # Add step limit to prevent infinite loops
            # Extract observation if it's a tuple (some environments return additional info)
            if isinstance(observation, tuple):
                obs_array = observation[0]  # Use only the observation part
            else:
                obs_array = observation
                
            # Ensure observation is a numpy array with batch dimension
            if not isinstance(obs_array, np.ndarray):
                obs_array = np.array(obs_array, dtype=np.float32)
            
            if len(obs_array.shape) == 1:
                obs_array = np.expand_dims(obs_array, axis=0)
                
            # Run prediction
            action = algo.predict(obs_array)[0]
            
            # Handle different step return formats
            step_result = env.step(action)
            
            # MetaWorld environments return 4 values: obs, reward, done, info
            if len(step_result) == 4:
                observation, reward, done, info = step_result
            # Some environments might return 5 values including truncated flag (gym>=0.26)
            elif len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                if verbose:
                    print(f"Warning: Unexpected step result format: {len(step_result)} values")
                break
                
            episode_return += reward
            steps += 1
            
            # Check for success
            if isinstance(info, dict) and 'success' in info and info['success']:
                success_rate += 1
                break
        
        returns.append(episode_return)
        steps_to_complete.append(steps)
    
    if len(returns) > 0:
        success_rate_pct = 100.0 * success_rate / n_episodes
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_steps = np.mean(steps_to_complete)
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")
            print(f"  Success rate: {success_rate_pct:.1f}%")
            print(f"  Average steps per episode: {mean_steps:.1f}")
            print(f"  Evaluated over {n_episodes} episodes")
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'success_rate': success_rate / n_episodes,
            'mean_steps': mean_steps,
            'num_episodes': n_episodes,
            'returns': returns,
            'steps': steps_to_complete
        }
    else:
        if verbose:
            print("No complete episodes during evaluation.")
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'success_rate': 0.0,
            'num_episodes': 0
        }

def main():
    parser = argparse.ArgumentParser(description="Train an IQL policy using learned state-action reward model")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATHS[0],
                        help="Path to the PT file containing trajectory data")
    parser.add_argument("--reward_model_path", type=str, default="reward_model/state_action_reward_model.pt",
                        help="Path to the trained state-action reward model")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256],
                        help="Hidden dimensions of the reward model")
    parser.add_argument("--iql_epochs", type=int, default=100,
                        help="Number of epochs for IQL training")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for IQL training")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    parser.add_argument("--output_dir", type=str, default="iql_model",
                        help="Directory to save model and results")
    parser.add_argument("--max_segments", type=int, default=1000,
                        help="Maximum number of segments to use for dataset creation")
    parser.add_argument("--reward_batch_size", type=int, default=32,
                        help="Batch size for reward prediction (higher values speed up processing)")
    parser.add_argument("--skip_env_creation", action="store_true",
                        help="Skip environment creation and evaluation (useful if MetaWorld is not available)")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Interval (in epochs) between policy evaluations during training")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_tensordict(args.data_path)
    
    # Get observation and action dimensions
    observations = data["obs"] if "obs" in data else data["state"]
    state_dim = observations.shape[1]
    action_dim = data["action"].shape[1]
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Load reward model
    reward_model = SegmentRewardModel(state_dim, action_dim, hidden_dims=args.hidden_dims)
    reward_model.load_state_dict(torch.load(args.reward_model_path))
    reward_model = reward_model.to(device)
    reward_model.eval()
    print(f"Loaded reward model from {args.reward_model_path}")
    
    # Create MDP dataset with learned rewards
    print("Creating MDP dataset with learned rewards...")
    dataset = create_mdp_dataset_with_sa_reward(
        data, 
        reward_model, 
        device, 
        max_segments=args.max_segments,
        batch_size=args.reward_batch_size
    )
    print(f"Created dataset with {dataset.size()} transitions")
    
    # Create environment for evaluation
    env = None
    if not args.skip_env_creation:
        dataset_name = Path(args.data_path).stem
        try:
            env = get_metaworld_env(dataset_name)
            print("Successfully created environment for evaluation")
        except Exception as e:
            print(f"Warning: Could not create environment for evaluation: {str(e)}")
            print("Will train without environment evaluation.")
    
    # Initialize IQL algorithm
    print("Initializing IQL algorithm...")
    iql = IQL(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=args.batch_size,
        gamma=0.99,
        tau=0.005,
        n_critics=2,
        expectile=0.7,
        weight_temp=3.0,
        encoder_factory=VectorEncoderFactory([256, 256, 256]),
        use_gpu=torch.cuda.is_available()
    )
    
    # Create a custom callback to evaluate the policy at regular intervals
    evaluation_results = []
    
    def evaluation_callback(algo, epoch, total_step):
        if epoch % args.eval_interval == 0 and env is not None:
            print(f"\n--- Evaluating policy at epoch {epoch} ---")
            metrics = evaluate_policy_manual(env, algo, n_episodes=args.eval_episodes, verbose=False)
            print(f"Mean return: {metrics['mean_return']:.2f}, Success rate: {metrics['success_rate']:.2f}")
            evaluation_results.append((epoch, metrics))
    
    # Train IQL
    print(f"Training IQL for {args.iql_epochs} epochs...")
    training_metrics = []
    try:
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
            n_epochs=args.iql_epochs,
            eval_episodes=None,  # Don't use the built-in eval which expects episodes format
            save_interval=10,
            scorers=scorers,
            experiment_name=f"IQL_{Path(args.data_path).stem}_SA",
            with_timestamp=True,
            logdir=f"{args.output_dir}/logs",
            verbose=True,
            callback=evaluation_callback
        )
        
        # Print the training metrics summary
        print("\nTraining metrics summary:")
        for epoch, metrics in training_metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Epoch {epoch}: {metrics_str}")
            
    except Exception as e:
        print(f"Error during IQL training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Continuing with trained model so far...")
    
    # Save the model
    model_path = f"{args.output_dir}/iql_{Path(args.data_path).stem}_sa.pt"
    iql.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Run final comprehensive evaluation
    print("\nRunning final comprehensive evaluation...")
    if env is not None:
        evaluation_metrics = evaluate_policy_manual(env, iql, n_episodes=args.eval_episodes, verbose=True)
    else:
        print("\nSkipping evaluation (no environment available)")
        evaluation_metrics = {
            'mean_return': 0.0,
            'std_return': 0.0,
            'success_rate': 0.0
        }
    
    # Save results
    results = {
        'args': vars(args),
        'final_evaluation': evaluation_metrics,
        'training_metrics': training_metrics,
        'evaluation_during_training': evaluation_results,
        'dataset_size': dataset.size()
    }
    
    results_path = f"{args.output_dir}/sa_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {results_path}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 