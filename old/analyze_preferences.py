import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
import argparse
from PIL import Image
from collections import Counter


def load_data(
    processed_data_path="processed_data.pkl", preference_data_path="preference_data.pkl"
):
    """Load processed data and preference data."""
    print(f"Loading processed data from {processed_data_path}...")
    with open(processed_data_path, "rb") as f:
        processed_data = pickle.load(f)

    print(f"Loading preference data from {preference_data_path}...")
    with open(preference_data_path, "rb") as f:
        preference_data = pickle.load(f)

    return processed_data, preference_data


def load_original_data(data_paths):
    """Load the original tensordict data from multiple paths."""
    all_data = []

    for i, path in enumerate(data_paths):
        print(f"Loading dataset {i + 1}/{len(data_paths)}: {path}")
        try:
            data = torch.load(path)
            all_data.append((i, path, data))
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return all_data


def plot_cluster_distribution(cluster_assignments, n_clusters):
    """Plot the distribution of segments across clusters."""
    # Count segments in each cluster
    counts = np.bincount(cluster_assignments, minlength=n_clusters + 1)[1:]

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_clusters + 1), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Number of segments")
    plt.title("Distribution of segments across clusters")
    plt.xticks(range(1, n_clusters + 1))
    plt.grid(axis="y", alpha=0.3)

    # Add percentage labels
    total = len(cluster_assignments)
    for i, count in enumerate(counts):
        percentage = count / total * 100
        plt.text(i + 1, count + 0.1, f"{percentage:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig("cluster_distribution.png", dpi=300)
    plt.close()
    print(f"Cluster distribution plot saved to cluster_distribution.png")


def plot_dataset_distribution_by_cluster(
    cluster_assignments, dataset_indicators, n_clusters, data_paths
):
    """Plot the distribution of datasets across clusters."""
    # Create a more readable label for each dataset
    dataset_labels = [
        f"DS{i + 1}: {os.path.basename(path)}" for i, path in enumerate(data_paths)
    ]

    # Get unique datasets
    unique_datasets = np.unique(dataset_indicators)
    n_datasets = len(unique_datasets)

    # Ensure arrays are numpy arrays
    cluster_assignments = np.asarray(cluster_assignments)
    dataset_indicators = np.asarray(dataset_indicators)

    # Check if dataset_indicators and cluster_assignments have the same length
    if len(dataset_indicators) != len(cluster_assignments):
        print(
            f"Warning: dataset_indicators length ({len(dataset_indicators)}) doesn't match cluster_assignments length ({len(cluster_assignments)})"
        )
        print("Creating segment-level dataset indicators")

        # This means dataset_indicators has frame-level annotations, but we need segment-level
        # We'll use the start index of each segment to determine its dataset
        segment_dataset_indicators = np.zeros(len(cluster_assignments), dtype=int)

        for i, (start, _) in enumerate(segment_indices):
            if start < len(dataset_indicators):
                segment_dataset_indicators[i] = dataset_indicators[start]

        # Use the mapped dataset indicators instead
        dataset_indicators = segment_dataset_indicators
        unique_datasets = np.unique(dataset_indicators)
        n_datasets = len(unique_datasets)

    # Count occurrences of each dataset in each cluster
    cluster_dataset_counts = {}
    for cluster in range(1, n_clusters + 1):
        # Get indices of segments in this cluster
        indices = np.where(cluster_assignments == cluster)[0]
        # Count datasets in these indices
        dataset_counts = Counter(dataset_indicators[i] for i in indices)
        cluster_dataset_counts[cluster] = [
            dataset_counts.get(d, 0) for d in unique_datasets
        ]

    # Convert to numpy array
    counts_array = np.zeros((n_clusters, n_datasets))
    for c in range(1, n_clusters + 1):
        counts_array[c - 1] = cluster_dataset_counts[c]

    # Calculate percentages within each cluster
    sums = counts_array.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1
    percentages = counts_array / sums * 100

    # Create a stacked bar chart
    plt.figure(figsize=(12, 8))

    bottom = np.zeros(n_clusters)
    for d in range(n_datasets):
        plt.bar(
            range(1, n_clusters + 1),
            percentages[:, d],
            bottom=bottom,
            label=dataset_labels[d] if d < len(dataset_labels) else f"Dataset {d + 1}",
            alpha=0.7,
        )
        bottom += percentages[:, d]

    plt.xlabel("Cluster")
    plt.ylabel("Percentage of Segments")
    plt.title("Dataset Distribution by Cluster")
    plt.xticks(range(1, n_clusters + 1))
    plt.yticks(range(0, 101, 10))
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize="small")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("dataset_distribution_by_cluster.png", dpi=300)
    plt.close()

    # Also create a count-based plot
    plt.figure(figsize=(12, 8))

    bottom = np.zeros(n_clusters)
    for d in range(n_datasets):
        plt.bar(
            range(1, n_clusters + 1),
            counts_array[:, d],
            bottom=bottom,
            label=dataset_labels[d] if d < len(dataset_labels) else f"Dataset {d + 1}",
            alpha=0.7,
        )
        bottom += counts_array[:, d]

    plt.xlabel("Cluster")
    plt.ylabel("Number of Segments")
    plt.title("Dataset Distribution by Cluster (Counts)")
    plt.xticks(range(1, n_clusters + 1))
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize="small")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("dataset_counts_by_cluster.png", dpi=300)
    plt.close()

    print(
        f"Dataset distribution plots saved to dataset_distribution_by_cluster.png and dataset_counts_by_cluster.png"
    )


def plot_cluster_distribution_by_dataset(
    cluster_assignments, dataset_indicators, n_clusters, data_paths
):
    """Plot the distribution of clusters for each dataset."""
    # Create a more readable label for each dataset
    dataset_labels = [
        f"DS{i + 1}: {os.path.basename(path)}" for i, path in enumerate(data_paths)
    ]

    # Get unique datasets
    unique_datasets = np.unique(dataset_indicators)

    # Create subplots for each dataset
    fig, axes = plt.subplots(
        len(unique_datasets), 1, figsize=(12, 5 * len(unique_datasets))
    )
    if len(unique_datasets) == 1:
        axes = [axes]

    # Ensure arrays are numpy arrays
    cluster_assignments = np.asarray(cluster_assignments)
    dataset_indicators = np.asarray(dataset_indicators)

    for i, dataset_idx in enumerate(unique_datasets):
        ax = axes[i]

        # Get indices of segments from this dataset
        # Fix: ensure dataset_indicators has the same length as cluster_assignments
        if len(dataset_indicators) != len(cluster_assignments):
            print(
                f"Warning: dataset_indicators length ({len(dataset_indicators)}) doesn't match cluster_assignments length ({len(cluster_assignments)})"
            )
            print("Using segment indices instead of the full dataset indicators array")

            # Create a mapping from segment indices to dataset indicators
            segment_to_dataset = {}
            for seg_idx, (start, end) in enumerate(segment_indices):
                if start < len(dataset_indicators):
                    segment_to_dataset[seg_idx] = dataset_indicators[start]

            # Create a mask for this dataset
            dataset_mask = np.array(
                [
                    segment_to_dataset.get(i, -1) == dataset_idx
                    for i in range(len(cluster_assignments))
                ]
            )
        else:
            dataset_mask = dataset_indicators == dataset_idx

        dataset_clusters = cluster_assignments[dataset_mask]

        # Count clusters
        counts = np.bincount(dataset_clusters, minlength=n_clusters + 1)[1:]

        # Create bar plot
        ax.bar(range(1, n_clusters + 1), counts)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of segments")
        ax.set_title(f"Cluster Distribution for {dataset_labels[dataset_idx]}")
        ax.set_xticks(range(1, n_clusters + 1))
        ax.grid(axis="y", alpha=0.3)

        # Add percentage labels
        total = len(dataset_clusters)
        for j, count in enumerate(counts):
            percentage = count / total * 100 if total > 0 else 0
            ax.text(j + 1, count + 0.1, f"{percentage:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig("cluster_distribution_by_dataset.png", dpi=300)
    plt.close()
    print(
        f"Cluster distribution by dataset plot saved to cluster_distribution_by_dataset.png"
    )


def plot_reward_by_cluster(
    cluster_assignments, segment_indices, rewards, dataset_indicators=None
):
    """Plot the distribution of rewards by cluster."""
    # Get unique clusters
    unique_clusters = np.unique(cluster_assignments)

    # Calculate average reward per segment
    segment_rewards = []
    for idx, (start, end) in enumerate(segment_indices):
        avg_reward = rewards[start : end + 1].mean().item()
        dataset_idx = dataset_indicators[start] if dataset_indicators is not None else 0
        segment_rewards.append((cluster_assignments[idx], avg_reward, dataset_idx))

    # Group by cluster
    cluster_rewards = {c: [] for c in unique_clusters}
    for cluster, reward, _ in segment_rewards:
        cluster_rewards[cluster].append(reward)

    # Create box plot
    plt.figure(figsize=(12, 6))

    # Prepare data for boxplot
    data = [cluster_rewards[c] for c in sorted(unique_clusters)]
    labels = [f"Cluster {c}" for c in sorted(unique_clusters)]

    plt.boxplot(data, labels=labels, showfliers=False)
    plt.violinplot(data, showextrema=False)

    # Add mean markers
    for i, d in enumerate(data):
        plt.scatter(i + 1, np.mean(d), marker="o", color="red", s=50, zorder=3)

    plt.xlabel("Cluster")
    plt.ylabel("Average Reward")
    plt.title("Distribution of Rewards by Cluster")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_by_cluster.png", dpi=300)
    plt.close()
    print(f"Reward distribution plot saved to reward_by_cluster.png")

    # If dataset indicators are available, create a plot comparing rewards by dataset for each cluster
    if dataset_indicators is not None:
        unique_datasets = np.unique([d for _, _, d in segment_rewards])

        # Group by cluster and dataset
        cluster_dataset_rewards = {
            c: {d: [] for d in unique_datasets} for c in unique_clusters
        }
        for cluster, reward, dataset in segment_rewards:
            cluster_dataset_rewards[cluster][dataset].append(reward)

        # Create a plot for each cluster
        for cluster in sorted(unique_clusters):
            plt.figure(figsize=(10, 6))

            # Prepare data for boxplot
            data = [
                cluster_dataset_rewards[cluster][d]
                for d in sorted(unique_datasets)
                if len(cluster_dataset_rewards[cluster][d]) > 0
            ]
            labels = [
                f"Dataset {d + 1}"
                for d in sorted(unique_datasets)
                if len(cluster_dataset_rewards[cluster][d]) > 0
            ]

            if len(data) > 0:  # Only create plots if there's data
                plt.boxplot(data, labels=labels, showfliers=False)
                plt.violinplot(data, showextrema=False)

                # Add mean markers
                for i, d in enumerate(data):
                    plt.scatter(
                        i + 1, np.mean(d), marker="o", color="red", s=50, zorder=3
                    )

                plt.xlabel("Dataset")
                plt.ylabel("Average Reward")
                plt.title(f"Distribution of Rewards for Cluster {cluster} by Dataset")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"reward_cluster{cluster}_by_dataset.png", dpi=300)
            plt.close()

        print(f"Reward distribution by dataset plots saved")


def create_cluster_summary_video(
    images,
    cluster_assignments,
    segment_indices,
    dataset_indicators=None,
    output_file="cluster_summary.mp4",
):
    """Create a video showing examples from each cluster."""
    # Get unique clusters
    unique_clusters = np.sort(np.unique(cluster_assignments))
    n_clusters = len(unique_clusters)

    # For each cluster, select a representative segment from each dataset if available
    exemplars = {}
    dataset_sources = {}

    for cluster in unique_clusters:
        # Find segments in this cluster
        segments_in_cluster = np.where(cluster_assignments == cluster)[0]

        # If we have dataset indicators, try to get one exemplar from each dataset
        if dataset_indicators is not None:
            # Group by dataset
            dataset_segments = {}
            for idx in segments_in_cluster:
                start, end = segment_indices[idx]
                dataset_idx = dataset_indicators[start]
                if dataset_idx not in dataset_segments:
                    dataset_segments[dataset_idx] = []
                dataset_segments[dataset_idx].append(idx)

            # Initialize exemplars for this cluster
            exemplars[cluster] = []
            dataset_sources[cluster] = []

            # Get one exemplar from each dataset
            for dataset_idx, indices in dataset_segments.items():
                if indices:
                    exemplar_idx = np.random.choice(indices)
                    start, end = segment_indices[exemplar_idx]
                    exemplars[cluster].append(images[start : end + 1].cpu().numpy())
                    dataset_sources[cluster].append(dataset_idx)

        # If no dataset indicators or no exemplars yet, just pick one randomly
        if dataset_indicators is None or not exemplars.get(cluster, []):
            if len(segments_in_cluster) > 0:
                exemplar_idx = np.random.choice(segments_in_cluster)
                start, end = segment_indices[exemplar_idx]
                exemplars[cluster] = [images[start : end + 1].cpu().numpy()]
                dataset_sources[cluster] = [0]  # Default dataset index

    # Set up figure for animation
    n_rows = n_clusters
    max_cols = max(len(ex) for ex in exemplars.values())

    fig = plt.figure(figsize=(4 * max_cols, 3 * n_rows))
    gs = GridSpec(n_rows, max_cols + 1, width_ratios=[3] * max_cols + [1], figure=fig)

    # Calculate the maximum segment length across all exemplars
    max_length = max([ex.shape[0] for exs in exemplars.values() for ex in exs])

    # Setup for each cluster
    axes = []
    image_plots = []
    titles = []

    for i, cluster in enumerate(unique_clusters):
        cluster_exemplars = exemplars[cluster]
        cluster_datasets = dataset_sources[cluster]

        # Add title for this cluster
        cluster_title = fig.text(
            0.02,
            1 - (i + 0.5) / n_rows,
            f"Cluster {cluster}",
            fontsize=14,
            ha="left",
            va="center",
        )
        titles.append(cluster_title)

        # Create a subplot for each exemplar
        row_images = []
        row_axes = []

        for j, (exemplar, dataset_idx) in enumerate(
            zip(cluster_exemplars, cluster_datasets)
        ):
            if j < max_cols:  # Only show up to max_cols exemplars
                ax = fig.add_subplot(gs[i, j])
                ax.set_title(f"Dataset {dataset_idx + 1}")
                ax.axis("off")

                # Initial empty image
                img = ax.imshow(np.zeros_like(exemplar[0]), animated=True)

                row_axes.append(ax)
                row_images.append(img)

        axes.append(row_axes)
        image_plots.append(row_images)

    # Add step counter
    step_text = fig.text(0.5, 0.98, "Step: 0", fontsize=14, ha="center")

    # Animation function
    def animate(frame):
        # Update step counter
        step_text.set_text(f"Step: {frame}")

        # Update each exemplar
        all_images = [step_text]

        for i, cluster in enumerate(unique_clusters):
            cluster_exemplars = exemplars[cluster]

            for j, exemplar in enumerate(cluster_exemplars):
                if j < len(image_plots[i]):  # Check we have a plot for this exemplar
                    # If we're past the end of this exemplar, show the last frame
                    frame_idx = min(frame, len(exemplar) - 1)
                    image_plots[i][j].set_array(exemplar[frame_idx])
                    all_images.append(image_plots[i][j])

        return all_images

    # Create animation
    print(f"Creating cluster summary animation with {max_length} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=max_length, interval=200, blit=True
    )

    # Save animation
    print(f"Saving video to {output_file}...")
    writer = animation.FFMpegWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close(fig)

    print(f"Cluster summary video saved to {output_file}")
    return output_file


def analyze_preferences(segment_pairs, segment_indices, synthetic_prefs, rewards):
    """Analyze the synthetic preferences in relation to rewards."""
    print("\nAnalyzing synthetic preferences...")

    # Calculate total reward for each segment in the pairs
    pair_rewards = []
    for (idx1, idx2), pref in zip(segment_pairs, synthetic_prefs):
        start1, end1 = segment_indices[idx1]
        start2, end2 = segment_indices[idx2]

        reward1 = rewards[start1 : end1 + 1].sum().item()
        reward2 = rewards[start2 : end2 + 1].sum().item()

        pair_rewards.append((reward1, reward2, pref))

    # Calculate preference statistics
    correct_pref_count = 0
    for reward1, reward2, pref in pair_rewards:
        # Calculate expected preference based on reward
        expected_pref = 1 if reward1 > reward2 else 2

        # Check if synthetic preference matches expected
        if pref == expected_pref:
            correct_pref_count += 1

    print(
        f"Synthetic preference accuracy: {correct_pref_count}/{len(pair_rewards)} "
        f"({correct_pref_count / len(pair_rewards) * 100:.1f}%)"
    )

    # Plot reward differences vs. preferences
    plt.figure(figsize=(10, 6))

    # Extract reward differences
    reward_diffs = [
        r1 - r2 if pref == 1 else r2 - r1 for (r1, r2, pref) in pair_rewards
    ]

    plt.bar(range(len(reward_diffs)), reward_diffs)
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Preference Pair")
    plt.ylabel("Reward Difference (Preferred - Non-preferred)")
    plt.title("Reward Differences in Preference Pairs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("preference_reward_differences.png", dpi=300)
    plt.close()

    print(f"Preference analysis plot saved to preference_reward_differences.png")


def create_t_sne_visualization(segments, cluster_assignments, perplexity=30):
    """Create a t-SNE visualization of the segments colored by cluster."""
    from sklearn.manifold import TSNE

    print("\nCreating t-SNE visualization of segments...")

    # Prepare data for t-SNE
    # Average each segment embedding over time dimension
    segment_features = np.array([seg.mean(0).cpu().numpy() for seg in segments])

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    segment_tsne = tsne.fit_transform(segment_features)

    # Plot results
    plt.figure(figsize=(12, 10))

    # Get unique clusters and assign colors
    unique_clusters = np.unique(cluster_assignments)

    # Create scatter plot
    for cluster in unique_clusters:
        # Get indices of segments in this cluster
        indices = np.where(cluster_assignments == cluster)[0]

        # Plot these points
        plt.scatter(
            segment_tsne[indices, 0],
            segment_tsne[indices, 1],
            label=f"Cluster {cluster}",
            alpha=0.7,
        )

    plt.title("t-SNE Visualization of Segments by Cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig("segment_tsne.png", dpi=300)
    plt.close()

    print(f"t-SNE visualization saved to segment_tsne.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze saved preference data")
    parser.add_argument(
        "--processed-data",
        type=str,
        default="processed_data.pkl",
        help="Path to processed data pickle file",
    )
    parser.add_argument(
        "--preference-data",
        type=str,
        default="preference_data.pkl",
        help="Path to preference data pickle file",
    )
    parser.add_argument(
        "--create-videos", action="store_true", help="Create videos (may be slow)"
    )
    parser.add_argument(
        "--tsne", action="store_true", help="Create t-SNE visualization (may be slow)"
    )

    args = parser.parse_args()

    # Make segment_indices global for other functions to access
    global segment_indices

    # Load data
    processed_data, preference_data = load_data(
        args.processed_data, args.preference_data
    )

    # Extract needed data
    segments = processed_data["segments"]
    segment_indices = processed_data.get("segment_indices", [])
    cluster_assignments = processed_data["cluster_assignments"]
    segment_pairs = preference_data["segment_pairs"]
    synthetic_prefs = preference_data["synthetic_preferences"]

    # Get parameters
    parameters = processed_data.get("parameters", {})
    n_clusters = parameters.get("n_clusters", 8)
    data_paths = parameters.get("data_paths", [])

    # Get dataset indicators if available
    dataset_indicators = processed_data.get("dataset_indicators", None)

    # Load original data if needed and available
    original_data = None
    if data_paths:
        try:
            original_data_list = load_original_data(data_paths)
            if original_data_list:
                # Just use the first dataset for reward analysis if needed
                _, _, original_data = original_data_list[0]
        except Exception as e:
            print(f"Warning: Could not load original data: {e}")
            # Continue without original data

    # Basic information
    print("\nData summary:")
    print(f"Number of segments: {len(segments)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of preference pairs: {len(segment_pairs)}")
    print(f"Number of segment indices: {len(segment_indices)}")

    if dataset_indicators is not None:
        print(f"Dataset indicators shape: {np.shape(dataset_indicators)}")
        dataset_counts = Counter(dataset_indicators)
        print("Dataset distribution:")
        for dataset_idx, count in sorted(dataset_counts.items()):
            print(
                f"  Dataset {dataset_idx + 1}: {count} samples "
                + (
                    f"({data_paths[dataset_idx]})"
                    if dataset_idx < len(data_paths)
                    else ""
                )
            )

    # Plot cluster distribution
    plot_cluster_distribution(cluster_assignments, n_clusters)

    # Plot dataset-specific distributions if we have that information
    if dataset_indicators is not None and data_paths:
        # Convert to numpy array if needed
        if not isinstance(dataset_indicators, np.ndarray):
            dataset_indicators = np.array(dataset_indicators)

        plot_dataset_distribution_by_cluster(
            cluster_assignments, dataset_indicators, n_clusters, data_paths
        )
        plot_cluster_distribution_by_dataset(
            cluster_assignments, dataset_indicators, n_clusters, data_paths
        )

    # Plot reward by cluster if we have the original data
    if original_data is not None and "reward" in original_data:
        plot_reward_by_cluster(
            cluster_assignments,
            segment_indices,
            original_data["reward"],
            dataset_indicators,
        )

    # Analyze preferences if we have original data for rewards
    if original_data is not None and "reward" in original_data:
        analyze_preferences(
            segment_pairs, segment_indices, synthetic_prefs, original_data["reward"]
        )

    # Optional: Create cluster summary video
    if args.create_videos and original_data is not None and "image" in original_data:
        create_cluster_summary_video(
            original_data["image"],
            cluster_assignments,
            segment_indices,
            dataset_indicators,
            output_file="cluster_summary.mp4",
        )

    # Optional: Create t-SNE visualization
    if args.tsne:
        create_t_sne_visualization(segments, cluster_assignments)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
