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
from train_preference_reward_model import RewardModel, PreferenceDataset

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_reward_model(model_path, input_dim, hidden_dims=[256, 128, 64], device="cpu"):
    """Load a trained reward model."""
    model = RewardModel(input_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model_on_dataset(model, dataset, device="cpu", num_samples=1000):
    """Evaluate the reward model on a dataset.

    Args:
        model: RewardModel
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
            segment1, segment2, target = dataset[idx]

            # Move to device
            segment1 = segment1.to(device)
            segment2 = segment2.to(device)

            # Get rewards
            reward1 = model(segment1.unsqueeze(0)).item()
            reward2 = model(segment2.unsqueeze(0)).item()

            # Determine predicted preference
            if reward1 > reward2:
                pred = 1  # Prefer segment1
            else:
                pred = 2  # Prefer segment2

            # Get ground truth preference
            if target[0] > target[1]:
                true_pref = 1
            else:
                true_pref = 2

            # Check accuracy
            if pred == true_pref:
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
        model: RewardModel
        test_segments: List of test segments
        device: Device to use for prediction

    Returns:
        list: Predicted rewards for each segment
    """
    # Preprocess segments
    processed_segments = []

    for segment in tqdm(test_segments, desc="Preprocessing test segments"):
        # Use only the EEF positions (first 3 dimensions)
        if segment.shape[1] >= 3:
            eef_trajectory = segment[:, :3]
        else:
            eef_trajectory = segment  # Use all dims if less than 3

        # Get fixed-length representation (match training data)
        target_length = 64

        if len(eef_trajectory) >= target_length:
            # Truncate to target length
            fixed_length = eef_trajectory[:target_length]
        else:
            # Pad with the last frame
            padding_length = target_length - len(eef_trajectory)
            padding = eef_trajectory[-1:].repeat(padding_length, 1)
            fixed_length = torch.cat([eef_trajectory, padding], dim=0)

        # Flatten the fixed-length trajectory
        flattened = fixed_length.reshape(-1)
        processed_segments.append(flattened)

    # Predict rewards
    rewards = []

    with torch.no_grad():
        for segment in tqdm(processed_segments, desc="Predicting rewards"):
            # Move to device
            segment = segment.to(device)

            # Get reward
            reward = model(segment.unsqueeze(0)).item()
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
        model: RewardModel
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

    # Get input dimension from model
    input_dim = model.network[0].in_features
    # Assuming 64 timesteps with 3D positions
    timesteps = 64
    features_per_timestep = 3

    # Generate a template trajectory (all zeros)
    template = torch.zeros(input_dim)

    # Fill reward grid
    with torch.no_grad():
        for i, x in enumerate(tqdm(x_values, desc="Computing reward heatmap")):
            for j, y in enumerate(y_values):
                # Create a synthetic trajectory where (x,y) values are set
                trajectory = template.clone()

                # Set x values for all timesteps
                for t in range(timesteps):
                    idx_x = t * features_per_timestep + 0  # X coordinate
                    idx_y = t * features_per_timestep + 1  # Y coordinate

                    # Set trajectory at this position - can be customized
                    trajectory[idx_x] = x
                    trajectory[idx_y] = y

                # Predict reward
                trajectory = trajectory.to(device)
                reward = model(trajectory.unsqueeze(0)).item()

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
        default=[256, 128, 64],
        help="Hidden layer dimensions (must match trained model)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    with open(args.test_dataset, "rb") as f:
        test_data = pickle.load(f)

    test_dataset = PreferenceDataset(test_data)
    print(f"Test dataset loaded with {len(test_dataset)} preference pairs")

    # Get input dimension from a sample
    sample_segment, _, _ = test_dataset[0]
    input_dim = sample_segment.shape[0]

    # Load model
    model = load_reward_model(
        args.model_path, input_dim, hidden_dims=args.hidden_dims, device=device
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
    test_segments = test_data["segments"]

    # If clusters are available in the data, get those too
    if "clusters" in test_data:
        segment_labels = test_data["clusters"]
    else:
        segment_labels = None

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
