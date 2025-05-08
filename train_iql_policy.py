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

def create_mdp_dataset_with_sa_reward(data, reward_model, device):
    """Create MDPDataset from tensordict data using a state-action reward model."""
    # Extract relevant info from data
    observations = data["obs"] if "obs" in data else data["state"]  # Use obs key if available
    actions = data["action"]
    rewards = data["reward"]  # Original rewards (will be replaced with learned rewards)
    episode_ids = data["episode"]
    
    # Ensure tensors are on CPU for numpy conversion
    observations = observations.cpu()
    actions = actions.cpu()
    rewards = rewards.cpu()
    episode_ids = episode_ids.cpu().numpy()
    
    # Split data into episodes
    unique_episodes = np.unique(episode_ids)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    
    # Process each episode
    for episode_id in tqdm(unique_episodes, desc="Processing episodes for MDP dataset"):
        # Get episode data
        episode_mask = (episode_ids == episode_id)
        episode_obs = observations[episode_mask]
        episode_actions = actions[episode_mask]
        
        # Skip very short episodes
        if len(episode_obs) < 2:
            continue
        
        # Compute rewards for each observation-action pair
        learned_rewards = []
        
        for i in range(len(episode_obs) - 1):  # -1 because last observation has no action
            obs = episode_obs[i].unsqueeze(0).to(device)
            action = episode_actions[i].unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Get reward from the model
                reward = reward_model.reward_model(obs, action).item()
                learned_rewards.append(reward)
        
        # Add terminal flag (1 for last state in episode)
        terminals = np.zeros(len(episode_obs) - 1)
        terminals[-1] = 1
        
        # Append to the dataset (excluding last observation)
        all_observations.append(episode_obs[:-1].numpy())
        all_actions.append(episode_actions[:-1].numpy())
        all_rewards.append(np.array(learned_rewards))
        all_terminals.append(terminals)
    
    # Concatenate data
    observations = np.concatenate(all_observations)
    actions = np.concatenate(all_actions)
    rewards = np.concatenate(all_rewards)
    terminals = np.concatenate(all_terminals)
    
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
    # Extract task name from file path
    if '/' in task_name:
        task_name = task_name.split('/')[-1].replace('buffer_', '').replace('.pt', '')
    
    # Special handling for specific task names
    if task_name == 'bin-picking-v2':
        task_name = 'bin-picking'
    elif task_name == 'peg-insert-side-v2':
        task_name = 'peg-insert-side'

    # Create MetaWorld environment
    try:
        ml1 = metaworld.ML1(task_name + '-v2')  # Create single-task environment
        env = ml1.train_classes[task_name]()
        task = ml1.train_tasks[0]
        env.set_task(task)
        
        print(f"Created MetaWorld environment: {task_name}-v2")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        return env
    except Exception as e:
        print(f"Error creating environment for {task_name}: {e}")
        print("Available tasks:")
        for task in metaworld.ML1.available_tasks():
            print(f"  - {task}")
        raise

def evaluate_policy(env, algo, n_episodes=10):
    """Evaluate policy on the environment."""
    returns = []
    success_rate = 0
    
    for _ in tqdm(range(n_episodes), desc="Evaluating policy"):
        observation = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action = algo.predict([observation])[0]
            observation, reward, done, info = env.step(action)
            episode_return += reward
            
            # Check for success
            if 'success' in info and info['success']:
                success_rate += 1
                break
        
        returns.append(episode_return)
    
    success_rate /= n_episodes
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Success rate: {success_rate:.2f}")
    
    return mean_return, success_rate

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
    dataset = create_mdp_dataset_with_sa_reward(data, reward_model, device)
    print(f"Created dataset with {dataset.size()} transitions")
    
    # Create environment for evaluation
    dataset_name = Path(args.data_path).stem
    env = get_metaworld_env(dataset_name)
    
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
    
    # Train IQL
    print(f"Training IQL for {args.iql_epochs} epochs...")
    iql.fit(
        dataset,
        n_epochs=args.iql_epochs,
        eval_episodes=args.eval_episodes,
        save_interval=10,
        scorers={
            'environment': evaluate_on_environment(env)
        },
        experiment_name=f"IQL_{dataset_name}_SA",
        with_timestamp=True,
        logdir=f"{args.output_dir}/logs",
        verbose=True
    )
    
    # Save the model
    iql.save_model(f"{args.output_dir}/iql_{dataset_name}_sa.pt")
    
    # Evaluate the trained policy
    print("\nEvaluating trained policy...")
    mean_return, success_rate = evaluate_policy(env, iql, n_episodes=args.eval_episodes)
    
    # Save results
    results = {
        'args': vars(args),
        'mean_return': mean_return,
        'success_rate': success_rate,
        'dataset_size': dataset.size()
    }
    
    with open(f"{args.output_dir}/sa_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Model saved to {args.output_dir}/iql_{dataset_name}_sa.pt")
    print(f"Results saved to {args.output_dir}/sa_results.pkl")
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 