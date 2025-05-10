import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Import the reward model class
from train_reward_model import SegmentRewardModel, PreferenceDataset

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_reward_model(model_path, state_dim, action_dim, hidden_dims=[256, 256], device="cpu"):
    """Load a trained reward model."""
    model = SegmentRewardModel(state_dim, action_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model_on_dataset(model, dataset, device="cpu", num_samples=1000):
    """Evaluate the reward model on a dataset.

    Args:
        model: SegmentRewardModel
        dataset: PreferenceDataset
        device: Device to use for evaluation
        num_samples: Number of samples to evaluate on

    Returns:
        dict: Evaluation metrics
    """
    # Create subset of data for evaluation
    indices = list(range(len(dataset)))
    if num_samples < len(indices):
        sample_indices = random.sample(indices, num_samples)
    else:
        sample_indices = indices

    correct = 0
    total = 0
    reward_diffs = []

    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Evaluating model"):
            obs1, actions1, obs2, actions2, pref = dataset[idx]

            # Move to device
            obs1, actions1 = obs1.to(device), actions1.to(device)
            obs2, actions2 = obs2.to(device), actions2.to(device)

            # Get rewards
            reward1 = model(obs1, actions1).item()
            reward2 = model(obs2, actions2).item()

            # Determine predicted preference
            if reward1 > reward2:
                pred = 1  # Prefer segment1
            else:
                pred = 0  # Prefer segment2

            # Check accuracy
            if pred == pref.item():
                correct += 1
            total += 1

            # Record reward difference
            reward_diffs.append(abs(reward1 - reward2))

    # Compute metrics
    accuracy = correct / total
    avg_reward_diff = sum(reward_diffs) / len(reward_diffs)

    metrics = {
        "accuracy": accuracy,
        "num_samples": total,
        "avg_reward_diff": avg_reward_diff,
        "reward_diffs": reward_diffs,
    }

    return metrics


def predict_reward_for_segments(model, test_segments, device="cpu"):
    """Predict rewards for test segments.

    Args:
        model: SegmentRewardModel
        test_segments: List of test segments (obs, actions)
        device: Device to use for prediction

    Returns:
        list: Predicted rewards for each segment
    """
    rewards = []

    with torch.no_grad():
        for obs, actions in tqdm(test_segments, desc="Predicting rewards"):
            # Move to device
            obs, actions = obs.to(device), actions.to(device)

            # Get reward
            reward = model(obs, actions).item()
            rewards.append(reward)

    return rewards


def plot_reward_distribution(rewards, segment_labels=None, output_path=None):
    """Plot the distribution of predicted rewards.

    Args:
        rewards: List of predicted rewards
        segment_labels: Optional list of labels for each segment
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    if segment_labels is not None:
        # Group rewards by label
        label_rewards = {}
        for reward, label in zip(rewards, segment_labels):
            if label not in label_rewards:
                label_rewards[label] = []
            label_rewards[label].append(reward)

        # Plot histogram for each label
        for label, label_rewards in label_rewards.items():
            plt.hist(label_rewards, bins=20, alpha=0.7, label=f"Cluster {label}")

        plt.legend()
    else:
        # Plot single histogram
        plt.hist(rewards, bins=20)

    plt.title("Distribution of Predicted Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)

    plt.show()


def create_reward_heatmap(model, device="cpu", output_path=None):
    """Create a heatmap of predicted rewards for varying inputs.

    Args:
        model: SegmentRewardModel
        device: Device to use for prediction
        output_path: Path to save the plot

    This function can be customized based on the input parameters that are most
    relevant for the specific application.
    """
    # Define the range of values to test
    # For example, varying the x, y positions of an end-effector
    x_values = np.linspace(-0.5, 0.5, 20)
    y_values = np.linspace(-0.5, 0.5, 20)

    # Initialize reward grid
    reward_grid = np.zeros((len(x_values), len(y_values)))

    # Generate a template trajectory (all zeros)
    obs_template = torch.zeros(1, model.state_dim)
    action_template = torch.zeros(1, model.action_dim)

    # Fill reward grid
    with torch.no_grad():
        for i, x in enumerate(tqdm(x_values, desc="Computing reward heatmap")):
            for j, y in enumerate(y_values):
                # Create a synthetic observation and action
                obs = obs_template.clone()
                action = action_template.clone()

                # Set x, y values for observation
                obs[0, 0] = x
                obs[0, 1] = y

                # Predict reward
                obs, action = obs.to(device), action.to(device)
                reward = model(obs, action).item()

                # Update grid
                reward_grid[i, j] = reward

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        reward_grid,
        origin="lower",
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
    )
    plt.colorbar(label="Predicted Reward")
    plt.title("Reward Heatmap by Position")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    if output_path:
        plt.savefig(output_path)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test a trained reward model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained reward model"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to the test preference dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reward_model_evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to evaluate on"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer dimensions (must match trained model)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    test_data = torch.load(args.test_dataset)

    test_dataset = PreferenceDataset(test_data)
    print(f"Test dataset loaded with {len(test_dataset)} preference pairs")

    # Get state and action dimensions from a sample
    obs1, actions1, _, _, _ = test_dataset[0]
    state_dim = obs1.shape[1]
    action_dim = actions1.shape[1]

    # Load model
    model = load_reward_model(
        args.model_path, state_dim, action_dim, hidden_dims=args.hidden_dims, device=device
    )
    print(f"Loaded model from {args.model_path}")

    # Evaluate model
    metrics = evaluate_model_on_dataset(
        model, test_dataset, device=device, num_samples=args.num_samples
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average reward difference: {metrics['avg_reward_diff']:.4f}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "eval_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    # Plot reward differences
    plt.figure(figsize=(10, 6))
    plt.hist(metrics["reward_diffs"], bins=20)
    plt.title("Distribution of Reward Differences")
    plt.xlabel("Absolute Reward Difference")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "reward_diffs.png"))

    # Get segments from test data
    test_segments = list(zip(test_data["obs"], test_data["action"]))

    # If clusters are available in the data, get those too
    segment_labels = test_data.get("clusters", None)

    # Predict rewards for test segments
    rewards = predict_reward_for_segments(model, test_segments, device=device)

    # Plot reward distribution
    plot_reward_distribution(
        rewards,
        segment_labels=segment_labels,
        output_path=os.path.join(args.output_dir, "reward_distribution.png"),
    )

    # Create reward heatmap
    create_reward_heatmap(
        model,
        device=device,
        output_path=os.path.join(args.output_dir, "reward_heatmap.png"),
    )

    print(f"Evaluation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
