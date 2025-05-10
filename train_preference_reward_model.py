import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class PreferenceDataset(Dataset):
    """Dataset for training a reward model from preference data."""

    def __init__(self, preference_data):
        """Initialize the dataset.

        Args:
            preference_data: Dict with preference data
        """
        self.segments = preference_data["segments"]
        self.segment_pairs = preference_data["segment_pairs"]
        self.preference_labels = preference_data["preference_labels"]

        # Preprocess segments - get fixed-size representation
        self.processed_segments = self._preprocess_segments()

    def _preprocess_segments(self):
        """Preprocess segments to get fixed-size representations."""
        processed = []

        for segment in tqdm(self.segments, desc="Preprocessing segments"):
            # Use only the EEF positions (first 3 dimensions)
            if segment.shape[1] >= 3:
                eef_trajectory = segment[:, :3]
            else:
                eef_trajectory = segment  # Use all dims if less than 3

            # Get fixed-length representation
            # Strategy 1: Use the first 64 frames, pad if needed
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
            processed.append(flattened)

        return processed

    def __len__(self):
        return len(self.segment_pairs)

    def __getitem__(self, idx):
        # Get indices of the segments in the pair
        i, j = self.segment_pairs[idx]

        # Get segments
        segment1 = self.processed_segments[i]
        segment2 = self.processed_segments[j]

        # Get preference label
        label = self.preference_labels[idx]

        # Convert label to regression target
        if label == 1:  # Prefer segment1
            target = torch.tensor([1.0, 0.0], dtype=torch.float32)
        elif label == 2:  # Prefer segment2
            target = torch.tensor([0.0, 1.0], dtype=torch.float32)
        else:  # Equal/Undecided
            target = torch.tensor([0.5, 0.5], dtype=torch.float32)

        return segment1, segment2, target


class RewardModel(nn.Module):
    """Neural network for predicting rewards from trajectory segments."""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        """Initialize the reward model.

        Args:
            input_dim: Dimension of the input (flattened trajectory)
            hidden_dims: List of hidden layer dimensions
        """
        super(RewardModel, self).__init__()

        # Create layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        # Combine layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

    def predict_preference(self, segment1, segment2):
        """Predict preference between two segments.

        Args:
            segment1: First segment
            segment2: Second segment

        Returns:
            tuple: (probability_prefer_segment1, probability_prefer_segment2)
        """
        # Get rewards
        reward1 = self.forward(segment1)
        reward2 = self.forward(segment2)

        # Convert to preference probabilities using softmax
        logits = torch.cat([reward1, reward2], dim=1)
        probs = F.softmax(logits, dim=1)

        return probs


def load_preference_dataset(dataset_file):
    """Load preference dataset from file."""
    print(f"Loading preference dataset from {dataset_file}")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def train_reward_model(
    dataset, model, optimizer, device, num_epochs=100, batch_size=32, eval_interval=5
):
    """Train the reward model on preference data.

    Args:
        dataset: PreferenceDataset
        model: RewardModel
        optimizer: Optimizer
        device: Device to use for training
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        eval_interval: Evaluate every N epochs

    Returns:
        tuple: (trained_model, training_losses, eval_accuracies)
    """
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Track losses and accuracies
    training_losses = []
    eval_accuracies = []

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        # Process batches
        for batch_idx, (segment1, segment2, targets) in enumerate(
            tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        ):
            # Move to device
            segment1 = segment1.to(device)
            segment2 = segment2.to(device)
            targets = targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            reward1 = model(segment1)
            reward2 = model(segment2)
            rewards = torch.cat([reward1, reward2], dim=1)

            # Compute loss
            loss = criterion(rewards, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Track loss
            epoch_losses.append(loss.item())

        # Compute average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        training_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Evaluate model
        if (epoch + 1) % eval_interval == 0:
            accuracy = evaluate_model(model, dataset, device)
            eval_accuracies.append((epoch + 1, accuracy))
            print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.4f}")

    return model, training_losses, eval_accuracies


def evaluate_model(model, dataset, device, num_samples=1000):
    """Evaluate the model on a subset of the dataset.

    Args:
        model: RewardModel
        dataset: PreferenceDataset
        device: Device to use for evaluation
        num_samples: Number of samples to evaluate on

    Returns:
        float: Accuracy
    """
    model.eval()

    # Create subset of data for evaluation
    indices = list(range(len(dataset)))
    if num_samples < len(indices):
        sample_indices = random.sample(indices, num_samples)
    else:
        sample_indices = indices

    correct = 0
    total = 0

    with torch.no_grad():
        for idx in sample_indices:
            segment1, segment2, target = dataset[idx]

            # Move to device
            segment1 = segment1.to(device)
            segment2 = segment2.to(device)

            # Get rewards
            reward1 = model(segment1.unsqueeze(0))
            reward2 = model(segment2.unsqueeze(0))

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

    accuracy = correct / total
    return accuracy


def plot_training_results(training_losses, eval_accuracies, output_dir):
    """Plot training losses and evaluation accuracies."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    # Plot accuracies
    if eval_accuracies:
        epochs, accuracies = zip(*eval_accuracies)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, accuracies, marker="o")
        plt.title("Evaluation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "eval_accuracy.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train a reward model from preference data"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the preference dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reward_model",
        help="Directory to save the model and results",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=5, help="Evaluate every N epochs"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    preference_data = load_preference_dataset(args.dataset)
    dataset = PreferenceDataset(preference_data)

    print(f"Dataset loaded with {len(dataset)} preference pairs")

    # Get input dimension from a sample
    sample_segment, _, _ = dataset[0]
    input_dim = sample_segment.shape[0]

    # Create model
    model = RewardModel(input_dim, hidden_dims=args.hidden_dims).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    print("Training reward model...")
    model, training_losses, eval_accuracies = train_reward_model(
        dataset,
        model,
        optimizer,
        device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
    )

    # Plot training results
    plot_training_results(training_losses, eval_accuracies, args.output_dir)

    # Save model
    model_path = os.path.join(args.output_dir, "reward_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save training metrics
    metrics = {
        "training_losses": training_losses,
        "eval_accuracies": eval_accuracies,
        "args": vars(args),
    }
    metrics_path = os.path.join(args.output_dir, "training_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"Training metrics saved to {metrics_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
