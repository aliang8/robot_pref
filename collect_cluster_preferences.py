import torch
import numpy as np
import os
import random
import pickle
import argparse
import itertools
import sys
import subprocess
import platform
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from IPython.display import HTML, display
import time

# For running in interactive environments
is_notebook = "ipykernel" in sys.modules
if is_notebook:
    from IPython.display import display, HTML
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    load_preprocessed_segments,
)

# Import clustering functions
from eef_clustering import extract_eef_trajectories

# Import visualization functions from eef_segment_matching instead
from eef_segment_matching import (
    create_eef_trajectory_animation,
    create_comparison_video,
)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_clustering_results(results_file):
    """Load clustering results from a file."""
    print(f"Loading clustering results from {results_file}")
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    return results


def get_representative_segments(data, clusters, segment_indices, n_per_cluster=1):
    """Extract representative segments for each cluster.

    Args:
        data: TensorDict with observations
        clusters: Cluster assignments
        segment_indices: List of (start_idx, end_idx) for each segment
        n_per_cluster: Number of representative segments per cluster

    Returns:
        dict: Mapping from cluster_id to list of representative segments
              Each segment is (segment_idx, start_idx, end_idx)
    """
    unique_clusters = np.unique(clusters)
    print(f"Found {len(unique_clusters)} unique clusters")

    # Get cluster centroids by finding segment closest to mean
    cluster_representatives = {}

    # For each cluster, find representative segments
    for cluster_id in unique_clusters:
        # Get indices of segments in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]

        # Skip if cluster is empty
        if len(cluster_indices) == 0:
            print(f"Cluster {cluster_id} is empty")
            continue

        print(f"Cluster {cluster_id}: {len(cluster_indices)} segments")

        # Choose representative segments
        # Strategy 1: Random sampling
        n_samples = min(n_per_cluster, len(cluster_indices))
        representative_indices = random.sample(list(cluster_indices), n_samples)

        # Save representative segments with metadata
        representatives = []
        for idx in representative_indices:
            start_idx, end_idx = segment_indices[idx]
            representatives.append((idx, start_idx, end_idx))

        cluster_representatives[cluster_id] = representatives

    return cluster_representatives


def display_segment(data, start_idx, end_idx, title=None, cluster_id=None):
    """Display a single segment."""
    # Create animation for EEF trajectory
    eef_positions = data["obs"][:, :3]

    if title is None:
        title = f"Segment from Cluster {cluster_id}"

    anim = create_eef_trajectory_animation(
        eef_positions, start_idx, end_idx, title=title
    )

    # Display animation
    if is_notebook:
        display(HTML(anim.to_jshtml()))

    return anim

def present_preference_query(data, segment1, segment2, query_id=None, skip_videos=False, no_auto_open=False, ground_truth=None):
    """Present a preference query to the user.

    Args:
        data: TensorDict with observations
        segment1: (idx1, start_idx1, end_idx1) for first segment
        segment2: (idx2, start_idx2, end_idx2) for second segment
        query_id: Optional ID for the query
        skip_videos: If True, skip generating visualizations to save time
        no_auto_open: If True, don't automatically open videos
        ground_truth: Optional ground truth preference (1 if segment1 is preferred, 2 if segment2 is preferred, 0 if equal)
        
    Returns:
        int: 1 if segment1 is preferred, 2 if segment2 is preferred, 0 if equal
    """
    idx1, start_idx1, end_idx1 = segment1
    idx2, start_idx2, end_idx2 = segment2

    # Clear output in notebook environments
    if is_notebook:
        from IPython.display import clear_output

        clear_output(wait=True)

    query_title = f"Query {query_id}" if query_id is not None else "Preference Query"
    print(f"\n{query_title}")
    print("=" * 40)

    # Display the two segments textually
    print(f"Segment 1: {start_idx1}-{end_idx1} (Length: {end_idx1 - start_idx1 + 1})")
    print(f"Segment 2: {start_idx2}-{end_idx2} (Length: {end_idx2 - start_idx2 + 1})")

    # Track all generated files
    generated_files = []

    # Display visualizations only if not skipping
    if not skip_videos:
        print("\nGenerating trajectory comparisons...")

        # Create an integrated video with both trajectories and observations
        output_file = (
            f"comparison_query_{query_id}.mp4"
            if query_id is not None
            else "comparison.mp4"
        )

        # Try to create integrated video (which returns either video path or fallback image path)
        viz_file = create_integrated_comparison_video(
            data, segment1, segment2, output_file=output_file
        )

        if viz_file:
            generated_files.append(viz_file)
        
        # Automatically open the visualization if not in notebook and auto-open is enabled
        if not is_notebook and not no_auto_open:
            for file_path in generated_files:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    print(f"Opening visualization: {file_path}")
                    open_video_file(file_path)
    else:
        print("\n[Visualizations skipped to save time]")

    # Get user preference
    while True:
        print("\nWhich segment do you prefer?")
        print("1: Segment 1")
        print("2: Segment 2")
        print("0: Equal/Cannot decide")

        try:
            preference = input("Enter preference (0/1/2): ").strip()
            preference = int(preference)
            if preference in [0, 1, 2]:
                break
            else:
                print("Invalid input. Please enter 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Check if user preference matches ground truth for statistics
    if ground_truth is not None and preference != 0:  # Only count non-equal user preferences
        if preference == ground_truth:
            print(f"✓ Your preference matches ground truth!")
        else:
            print(f"✗ Your preference differs from ground truth.")
    
    return preference

def get_reward_based_preference(data, segment1, segment2):
    """Determine ground truth preference based on cumulative reward.
    
    Args:
        data: TensorDict with observations and rewards
        segment1: (idx1, start_idx1, end_idx1) for first segment
        segment2: (idx2, start_idx2, end_idx2) for second segment
        
    Returns:
        int: 1 if segment1 has higher reward, 2 if segment2 has higher reward, 0 if equal
    """
    if "reward" not in data:
        print("WARNING: No reward data found, cannot determine ground truth preference")
        return None
    
    # Extract segment information
    _, start_idx1, end_idx1 = segment1
    _, start_idx2, end_idx2 = segment2
    
    # Get rewards - ensure they're on CPU
    rewards = data["reward"].cpu()
    
    # Calculate cumulative reward for each segment
    reward1 = rewards[start_idx1:end_idx1+1].sum().item()
    reward2 = rewards[start_idx2:end_idx2+1].sum().item()
    
    # Determine preference
    if abs(reward1 - reward2) < 1e-6:  # Small epsilon for float comparison
        return 0  # Equal rewards
    elif reward1 > reward2:
        return 1  # First segment preferred
    else:
        return 2  # Second segment preferred

def collect_cluster_preferences(data, cluster_representatives, num_comparisons=None, skip_videos=False, no_auto_open=False):
    """Collect user preferences between cluster representatives.

    Args:
        data: TensorDict with observations
        cluster_representatives: Dict mapping cluster_id to representative segments
        num_comparisons: Number of comparisons to conduct (default: all pairs)
        skip_videos: Skip generating videos for preference collection
        no_auto_open: If True, don't automatically open videos
        
    Returns:
        list: User preferences and ground truth in format 
             [(cluster_id1, cluster_id2, user_preference, gt_preference), ...]
              where preference is 1 if cluster_id1 is preferred, 2 if cluster_id2 is preferred
    """
    # Get all cluster IDs
    cluster_ids = sorted(cluster_representatives.keys())
    print(f"Collecting preferences for {len(cluster_ids)} clusters")

    # Generate all possible pairs of cluster IDs
    cluster_pairs = list(itertools.combinations(cluster_ids, 2))

    # Shuffle pairs for randomness
    random.shuffle(cluster_pairs)

    # If num_comparisons is specified, limit the number of comparisons
    if num_comparisons and num_comparisons < len(cluster_pairs):
        print(
            f"Limiting to {num_comparisons} comparisons out of {len(cluster_pairs)} possible pairs"
        )
        cluster_pairs = cluster_pairs[:num_comparisons]

    # Collect preferences
    preferences = []
    user_correct_count = 0
    total_comparable_queries = 0
    
    for i, (cluster_id1, cluster_id2) in enumerate(cluster_pairs):
        print(
            f"\nComparison {i + 1}/{len(cluster_pairs)}: Cluster {cluster_id1} vs Cluster {cluster_id2}"
        )

        # Select a random representative from each cluster
        segment1 = random.choice(cluster_representatives[cluster_id1])
        segment2 = random.choice(cluster_representatives[cluster_id2])
        
        # Get ground truth preference based on reward
        gt_preference = get_reward_based_preference(data, segment1, segment2)
        if gt_preference is not None:
            print(f"Ground truth preference (based on reward): {gt_preference}")
        
        # Present preference query to the user
        user_preference = present_preference_query(
            data, segment1, segment2, 
            query_id=i+1, 
            skip_videos=skip_videos,
            no_auto_open=no_auto_open,
            ground_truth=gt_preference  # Pass ground truth to show it after user's choice
        )
        
        # Record preference with ground truth
        preferences.append((cluster_id1, cluster_id2, user_preference, gt_preference))
        
        # Check if user preference matches ground truth for statistics
        if gt_preference is not None and user_preference != 0:  # Only count non-equal user preferences
            total_comparable_queries += 1
            if user_preference == gt_preference:
                user_correct_count += 1
                print(f"✓ Your preference matches ground truth!")
            else:
                print(f"✗ Your preference differs from ground truth.")
    
    # Calculate overall user accuracy
    if total_comparable_queries > 0:
        accuracy = user_correct_count / total_comparable_queries
        print(f"\nOverall accuracy: {user_correct_count}/{total_comparable_queries} = {accuracy:.2%}")
    
    return preferences


def derive_cluster_ranking(preferences):
    """Derive an approximate ranking of clusters from pairwise preferences.

    Args:
        preferences: List of (cluster_id1, cluster_id2, user_preference, gt_preference) tuples
        
    Returns:
        tuple: (user_ranking, gt_ranking) ordered lists of cluster IDs from most to least preferred
    """
    print("Deriving cluster rankings from preferences...")
    
    # Extract all unique cluster IDs
    cluster_ids = set()
    for c1, c2, _, _ in preferences:
        cluster_ids.add(c1)
        cluster_ids.add(c2)

    cluster_ids = sorted(list(cluster_ids))
    
    # Initialize win counts for user preferences and ground truth
    user_win_counts = {cluster_id: 0 for cluster_id in cluster_ids}
    gt_win_counts = {cluster_id: 0 for cluster_id in cluster_ids}
    
    # Count wins for each cluster based on user preferences and ground truth
    valid_user_comparisons = 0
    valid_gt_comparisons = 0
    
    for c1, c2, user_pref, gt_pref in preferences:
        # Count user preference wins (only for non-equal preferences)
        if user_pref == 1:
            user_win_counts[c1] += 1
            valid_user_comparisons += 1
        elif user_pref == 2:
            user_win_counts[c2] += 1
            valid_user_comparisons += 1
            
        # Count ground truth preference wins (only for non-equal and non-None preferences)
        if gt_pref is not None:
            if gt_pref == 1:
                gt_win_counts[c1] += 1
                valid_gt_comparisons += 1
            elif gt_pref == 2:
                gt_win_counts[c2] += 1
                valid_gt_comparisons += 1
    
    # Sort clusters by win count (descending)
    user_ranked_clusters = sorted(user_win_counts.keys(), key=lambda c: user_win_counts[c], reverse=True)
    gt_ranked_clusters = sorted(gt_win_counts.keys(), key=lambda c: gt_win_counts[c], reverse=True)
    
    # Print user-based ranking
    print("\nUser-based cluster ranking (most to least preferred):")
    for i, cluster_id in enumerate(user_ranked_clusters):
        print(f"  {i+1}. Cluster {cluster_id} (wins: {user_win_counts[cluster_id]})")
    
    # Print ground truth-based ranking
    print("\nGround truth-based cluster ranking (most to least preferred):")
    for i, cluster_id in enumerate(gt_ranked_clusters):
        print(f"  {i+1}. Cluster {cluster_id} (wins: {gt_win_counts[cluster_id]})")
    
    # Calculate rank correlation between user and ground truth rankings
    if valid_user_comparisons > 0 and valid_gt_comparisons > 0:
        # Create maps from cluster ID to rank position
        user_rank_map = {cluster: rank for rank, cluster in enumerate(user_ranked_clusters)}
        gt_rank_map = {cluster: rank for rank, cluster in enumerate(gt_ranked_clusters)}
        
        # Calculate Spearman's rank correlation coefficient (simplified)
        # Using sum of squared differences in ranks
        squared_diff_sum = 0
        n = len(cluster_ids)
        
        for cluster in cluster_ids:
            user_rank = user_rank_map[cluster]
            gt_rank = gt_rank_map[cluster]
            squared_diff_sum += (user_rank - gt_rank)**2
        
        # Spearman's formula: rho = 1 - (6 * sum(d²) / (n³ - n))
        if n > 1:  # Need at least 2 clusters to calculate correlation
            rho = 1 - (6 * squared_diff_sum) / (n**3 - n)
            print(f"\nRank correlation between user and ground truth rankings: {rho:.2f}")
            print(f"  (1.0 = perfect agreement, -1.0 = perfect disagreement, 0.0 = no relationship)")
            
            # Print clusters with biggest rank differences
            rank_diffs = [(c, abs(user_rank_map[c] - gt_rank_map[c])) for c in cluster_ids]
            rank_diffs.sort(key=lambda x: x[1], reverse=True)
            
            if len(rank_diffs) > 0 and rank_diffs[0][1] > 0:
                print("\nClusters with largest ranking disagreement:")
                for cluster, diff in rank_diffs[:3]:  # Show top 3 disagreements
                    if diff > 0:
                        print(f"  Cluster {cluster}: User rank: {user_rank_map[cluster]+1}, GT rank: {gt_rank_map[cluster]+1}, Diff: {diff}")
    
    return user_ranked_clusters, gt_ranked_clusters


def generate_augmented_preferences(data, clusters, segment_indices, cluster_ranking):
    """Generate augmented preferences based on cluster ranking.

    Args:
        data: TensorDict with observations
        clusters: Cluster assignments
        segment_indices: List of (start_idx, end_idx) for each segment
        cluster_ranking: Ordered list of cluster IDs from most to least preferred

    Returns:
        tuple: (augmented_preferences, accuracy_stats)
            augmented_preferences: list of (i, j, pref) preferences
            accuracy_stats: dict with accuracy statistics vs ground truth
    """
    print("Generating augmented preferences...")

    # Convert cluster ranking to a mapping of cluster to rank (0 is best)
    cluster_ranks = {
        cluster_id: rank for rank, cluster_id in enumerate(cluster_ranking)
    }

    # Generate augmented preferences
    augmented_preferences = []
    
    # Statistics to track accuracy
    stats = {
        'total_pairs': 0,
        'ground_truth_available': 0,
        'matching_ground_truth': 0,
        'accuracy': 0.0
    }
    
    # Iterate through all possible segment pairs
    n_segments = len(segment_indices)
    segments_by_cluster = {c: [] for c in cluster_ranking}

    # Group segments by cluster
    for i, (seg_idx, cluster) in enumerate(zip(range(n_segments), clusters)):
        segments_by_cluster[cluster].append(seg_idx)

    # Print distribution
    print("Segments per cluster:")
    for cluster_id, seg_indices in segments_by_cluster.items():
        print(f"  Cluster {cluster_id}: {len(seg_indices)} segments")

    # Generate preferences from cluster ranking
    # For each pair of clusters where one has higher rank than the other
    augmented_count = 0

    # Create a progress bar for the outer loop
    for i, cluster1 in enumerate(tqdm(cluster_ranking, desc="Processing clusters")):
        for cluster2 in cluster_ranking[i + 1 :]:  # Lower ranks (less preferred)
            # Get segments from both clusters
            segments1 = segments_by_cluster[cluster1]
            segments2 = segments_by_cluster[cluster2]

            # If both clusters have segments
            if segments1 and segments2:
                # Generate a subset of all possible pairs to avoid excessive augmentation
                # Determine sampling ratio based on dataset size
                total_pairs = len(segments1) * len(segments2)

                # Adjust sampling based on total pairs
                if total_pairs > 1000:
                    # Sample a fixed number or percentage
                    num_samples = min(1000, int(total_pairs * 0.1))
                else:
                    num_samples = total_pairs

                # Sample segment pairs
                pairs = []
                for _ in range(num_samples):
                    idx1 = random.choice(segments1)
                    idx2 = random.choice(segments2)
                    pairs.append((idx1, idx2))

                # Generate preferences
                for idx1, idx2 in pairs:
                    # Cluster1 has higher rank (lower rank number) than Cluster2
                    augmented_preference = 1  # Prefer segment from cluster1
                    
                    # Get ground truth preference based on reward
                    gt_preference = None
                    if "reward" in data:
                        start1, end1 = segment_indices[idx1]
                        start2, end2 = segment_indices[idx2]
                        
                        # Calculate cumulative reward for each segment
                        reward1 = data["reward"][start1:end1+1].sum().item()
                        reward2 = data["reward"][start2:end2+1].sum().item()
                        
                        if abs(reward1 - reward2) < 1e-6:  # Equal (within epsilon)
                            gt_preference = 0
                        elif reward1 > reward2:
                            gt_preference = 1
                        else:
                            gt_preference = 2
                    
                    # Update statistics
                    stats['total_pairs'] += 1
                    if gt_preference is not None and gt_preference != 0:
                        stats['ground_truth_available'] += 1
                        if augmented_preference == gt_preference:
                            stats['matching_ground_truth'] += 1
                    
                    # Add to augmented preferences
                    augmented_preferences.append((idx1, idx2, augmented_preference))
                    augmented_count += 1
    
    # Calculate accuracy if possible
    if stats['ground_truth_available'] > 0:
        stats['accuracy'] = stats['matching_ground_truth'] / stats['ground_truth_available']
    
    print(f"Generated {augmented_count} augmented preferences")
    print(f"Augmented preference accuracy vs ground truth: {stats['accuracy']:.2%} ({stats['matching_ground_truth']}/{stats['ground_truth_available']} comparable pairs)")
    
    return augmented_preferences, stats

def create_preference_dataset(data, segment_indices, augmented_preferences, output_file, preference_stats=None):
    """Create a dataset for training a reward model from augmented preferences.

    Args:
        data: TensorDict with observations
        segment_indices: List of (start_idx, end_idx) for each segment
        augmented_preferences: List of (i, j, pref) preference tuples
        output_file: Path to save the dataset
        preference_stats: Optional dictionary with preference comparison statistics
        
    Returns:
        dict: Preference dataset with segments and labels
    """
    print(f"Creating preference dataset at {output_file}")
    
    # Extract segment pairs and preferences
    segment_pairs = []
    preference_labels = []

    for i, j, pref in augmented_preferences:
        segment_pairs.append((i, j))
        preference_labels.append(pref)
    
    # Convert lists to tensors for better compatibility
    if isinstance(segment_indices, torch.Tensor):
        segment_indices_tensor = segment_indices.detach().clone()
    else:
        segment_indices_tensor = torch.tensor(segment_indices)
    
    # For segment pairs, we need a special approach since it's a list of tuples
    if isinstance(segment_pairs, torch.Tensor):
        segment_pairs_tensor = segment_pairs.detach().clone()
    else:
        segment_pairs_tensor = torch.tensor(segment_pairs)
    
    # Convert preference labels to tensor
    if isinstance(preference_labels, torch.Tensor):
        preference_labels_tensor = preference_labels.detach().clone()
    else:
        preference_labels_tensor = torch.tensor(preference_labels)
    
    # Create compact data representation with only necessary fields
    compact_data = {}
    essential_fields = ['obs', 'action', 'episode', 'reward']
    
    print("Creating compact data copy with only essential fields...")
    for field in essential_fields:
        if field in data:
            print(f"Including field: {field}")
            # Clone to avoid modifying original data and ensure it's on CPU
            compact_data[field] = data[field].clone().cpu() if isinstance(data[field], torch.Tensor) else data[field]
    
    # Create metadata including preference statistics if available
    metadata = {
        'source_file': data.get('_source_path', 'unknown'),
        'n_segments': len(segment_indices),
        'n_pairs': len(preference_labels),
        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'included_fields': list(compact_data.keys())
    }
    
    # Add preference statistics if available
    if preference_stats:
        metadata['preference_stats'] = preference_stats
    
    # Create standardized dataset structure
    preference_dataset = {
        'data': compact_data,                        # Essential tensor data
        'segment_indices': segment_indices_tensor,   # Indices for segments
        'segment_pairs': segment_pairs_tensor,       # Pairs for preference learning 
        'preference_labels': preference_labels_tensor,  # Preference labels (1=first preferred, 2=second)
        'metadata': metadata
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save dataset directly
    print(f"Saving preference dataset to {output_file} with {len(preference_labels)} preferences")
    torch.save(preference_dataset, output_file)
    print(f"Successfully saved preference dataset with included fields: {list(compact_data.keys())}")
    
    return preference_dataset


def open_video_file(video_path):
    """Open a video file using the system's default video player.

    Args:
        video_path: Path to the video file
    """
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False

        # Get the operating system
        system = platform.system()

        if system == "Windows":
            # Windows
            os.startfile(video_path)
        elif system == "Darwin":
            # macOS
            subprocess.call(["open", video_path])
        else:
            # Linux or other Unix-like systems
            subprocess.call(["xdg-open", video_path])

        print(f"Opened video: {video_path}")
        return True
    except Exception as e:
        print(f"Error opening video: {e}")
        return False


def create_integrated_comparison_video(
    data, segment1, segment2, output_file="integrated_comparison.mp4"
):
    """Create an integrated video showing both camera observations and EEF trajectories side by side.

    Args:
        data: TensorDict with observations and images
        segment1: (idx1, start_idx1, end_idx1) for first segment
        segment2: (idx2, start_idx2, end_idx2) for second segment
        output_file: Path to save the output video

    Returns:
        Path to the created video file, or None if creation failed
    """
    print("Creating integrated comparison video...")

    # Extract segment information
    idx1, start_idx1, end_idx1 = segment1
    idx2, start_idx2, end_idx2 = segment2

    # Get end-effector trajectories
    eef_positions = data["obs"][:, :3] if "obs" in data else data["state"][:, :3]
    traj1 = eef_positions[start_idx1 : end_idx1 + 1].cpu().numpy()
    traj2 = eef_positions[start_idx2 : end_idx2 + 1].cpu().numpy()

    # Get observation images if available
    has_images = "image" in data
    if has_images:
        img_seq1 = data["image"][start_idx1 : end_idx1 + 1].cpu().numpy()
        img_seq2 = data["image"][start_idx2 : end_idx2 + 1].cpu().numpy()

    # Set up figure with 2x2 grid:
    # [Trajectory 1, Trajectory 2]
    # [Camera 1,    Camera 2   ]
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2)

    # Top row: 3D trajectory plots
    ax_traj1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_traj1.set_title("Segment 1 Trajectory")

    ax_traj2 = fig.add_subplot(gs[0, 1], projection="3d")
    ax_traj2.set_title("Segment 2 Trajectory")

    # Bottom row: Camera observations
    if has_images:
        ax_img1 = fig.add_subplot(gs[1, 0])
        ax_img1.set_title("Segment 1 Observation")
        ax_img1.axis("off")

        ax_img2 = fig.add_subplot(gs[1, 1])
        ax_img2.set_title("Segment 2 Observation")
        ax_img2.axis("off")

    # Set consistent scale for trajectories
    all_points = np.vstack([traj1, traj2])
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    # Add padding
    padding = 0.1
    x_range = max(x_max - x_min, 0.01)
    y_range = max(y_max - y_min, 0.01)
    z_range = max(z_max - z_min, 0.01)

    x_min -= padding * x_range
    y_min -= padding * y_range
    z_min -= padding * z_range
    x_max += padding * x_range
    y_max += padding * y_range
    z_max += padding * z_range

    # Set axis limits for both trajectories
    for ax in [ax_traj1, ax_traj2]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Initialize plot elements
    (line1,) = ax_traj1.plot([], [], [], "r-", linewidth=2)
    point1 = ax_traj1.scatter([], [], [], color="blue", s=50)

    (line2,) = ax_traj2.plot([], [], [], "r-", linewidth=2)
    point2 = ax_traj2.scatter([], [], [], color="blue", s=50)

    # Initialize observation images if available
    if has_images:
        img1 = ax_img1.imshow(np.zeros_like(img_seq1[0]))
        img2 = ax_img2.imshow(np.zeros_like(img_seq2[0]))

    # Add frame counter
    frame_text = fig.text(0.5, 0.95, "Frame: 0", ha="center", fontsize=14)

    # Animation update function
    def update(frame):
        # Update frame counter
        frame_text.set_text(f"Frame: {frame}")

        # Update trajectories
        idx1 = min(frame, len(traj1) - 1)
        idx2 = min(frame, len(traj2) - 1)

        # Update first trajectory
        line1.set_data(traj1[: idx1 + 1, 0], traj1[: idx1 + 1, 1])
        line1.set_3d_properties(traj1[: idx1 + 1, 2])
        point1._offsets3d = ([traj1[idx1, 0]], [traj1[idx1, 1]], [traj1[idx1, 2]])

        # Update second trajectory
        line2.set_data(traj2[: idx2 + 1, 0], traj2[: idx2 + 1, 1])
        line2.set_3d_properties(traj2[: idx2 + 1, 2])
        point2._offsets3d = ([traj2[idx2, 0]], [traj2[idx2, 1]], [traj2[idx2, 2]])

        # Update camera images if available
        if has_images:
            img1.set_array(img_seq1[idx1])
            img2.set_array(img_seq2[idx2])

        # Rotate views for better visualization
        ax_traj1.view_init(elev=30, azim=(frame % 360))
        ax_traj2.view_init(elev=30, azim=(frame % 360))

        # Return all updated elements
        artists = [frame_text, line1, point1, line2, point2]
        if has_images:
            artists.extend([img1, img2])
        return artists

    # Create animation
    max_frames = max(len(traj1), len(traj2))
    ani = animation.FuncAnimation(
        fig, update, frames=max_frames, interval=100, blit=True
    )

    # Save animation as video or fallback to PNG frames
    try:
        # Try different codec configurations
        codec_configs = [
            # Try with default settings first
            {"fps": 10, "bitrate": 1800},
            # Try with H.264 explicitly
            {
                "fps": 10,
                "bitrate": 1800,
                "codec": "h264",
                "extra_args": ["-pix_fmt", "yuv420p"],
            },
            # Try with MPEG4 codec
            {"fps": 10, "bitrate": 1800, "codec": "mpeg4"},
        ]

        success = False
        for config in codec_configs:
            try:
                print(f"Trying video encoding with config: {config}")
                writer = animation.FFMpegWriter(**config)
                ani.save(output_file, writer=writer)
                success = True
                print(f"Successfully saved integrated video to {output_file}")
                break
            except Exception as e:
                print(f"Failed with config {config}: {e}")

        if not success:
            raise Exception("All video encoding attempts failed")

        plt.close(fig)
        return output_file

    except Exception as e:
        print(f"Error creating integrated video: {e}")
        # Fallback to saving key frames as images
        frames_dir = "integrated_frames"
        os.makedirs(frames_dir, exist_ok=True)

        # Save key frames
        key_frames = [
            0,
            max_frames // 4,
            max_frames // 2,
            3 * max_frames // 4,
            max_frames - 1,
        ]
        key_frames = [
            min(f, max_frames - 1) for f in key_frames
        ]  # Ensure within bounds

        all_frame_paths = []
        for i, frame in enumerate(key_frames):
            # Update the figure
            update(frame)

            # Save the frame
            frame_path = os.path.join(frames_dir, f"integrated_frame_{i}.png")
            plt.savefig(frame_path, dpi=100)
            print(f"Saved frame {i} to {frame_path}")
            all_frame_paths.append(frame_path)

        # Create a combined image showing all frames
        combined_path = os.path.join(frames_dir, "all_integrated_frames.png")
        plt.figure(figsize=(15, 10))

        for i, frame_path in enumerate(all_frame_paths):
            plt.subplot(len(all_frame_paths), 1, i + 1)
            img = plt.imread(frame_path)
            plt.imshow(img)
            plt.title(f"Frame {i}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(combined_path, dpi=100)
        plt.close("all")
        print(f"Saved combined frames to {combined_path}")

        return combined_path


def main():
    parser = argparse.ArgumentParser(description="Collect and augment preferences based on cluster representatives")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATHS[0],
                        help="Path to the PT file containing trajectory data")
    parser.add_argument("--preprocessed_data", type=str, default=None,
                        help="Path to preprocessed data file (used instead of data_path if provided)")
    parser.add_argument("--clustering_results", type=str, required=True,
                        help="Path to the pickle file with clustering results")
    parser.add_argument("--output_dir", type=str, default="preference_data",
                        help="Directory to save preference data and dataset")
    parser.add_argument("--n_representatives", type=int, default=3,
                        help="Number of representative segments per cluster")
    parser.add_argument("--max_comparisons", type=int, default=None,
                        help="Maximum number of pairwise comparisons to perform")
    parser.add_argument("--skip_videos", action="store_true",
                        help="Skip generating videos for preference collection to speed things up")
    parser.add_argument("--no_auto_open", action="store_true",
                        help="Don't automatically open visualization videos")
    parser.add_argument("--use_gt_ranking", action="store_true",
                        help="Use ground truth ranking for augmented preferences instead of user ranking")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if args.preprocessed_data:
        print(f"Loading preprocessed data from {args.preprocessed_data}")
        preproc_data = load_preprocessed_segments(args.preprocessed_data)

        # Create data structure expected by preference collection functions
        data = {}

        # Check which fields are available in preprocessed data
        essential_fields = ['obs', 'state', 'image', 'episode', 'reward', 'action']
        missing_fields = []

        for field in essential_fields:
            if field in preproc_data:
                data[field] = preproc_data[field]
            else:
                missing_fields.append(field)

        # Handle missing fields
        if "obs" not in data and "state" not in data:
            print(
                f"WARNING: Preprocessed data does not contain observation data (obs or state)."
            )
            print(f"Loading original data from {args.data_path} for observations.")
            orig_data = load_tensordict(args.data_path)

            # Copy missing essential fields
            for field in missing_fields:
                if field in orig_data:
                    data[field] = orig_data[field]
                    print(f"Loaded field '{field}' from original data")

        # Add source path for reference
        data["_source_path"] = args.preprocessed_data
    else:
        print(f"Loading data from {args.data_path}")
        data = load_tensordict(args.data_path)
        data["_source_path"] = args.data_path

    # Load clustering results
    clustering_results = load_clustering_results(args.clustering_results)
    clusters = clustering_results["clusters"]
    segment_indices = clustering_results["segment_indices"]

    print(f"Loaded clustering results with {len(clusters)} clusters")

    # Get representative segments for each cluster
    cluster_representatives = get_representative_segments(
        data, clusters, segment_indices, n_per_cluster=args.n_representatives
    )

    # Collect user preferences between clusters
    user_preferences = collect_cluster_preferences(
        data,
        cluster_representatives,
        num_comparisons=args.max_comparisons,
        skip_videos=args.skip_videos,
        no_auto_open=args.no_auto_open
    )
    
    # Calculate preference accuracy statistics
    preference_stats = {
        'user_vs_ground_truth': {
            'total_comparisons': len(user_preferences),
            'comparable_pairs': 0,
            'matching_preferences': 0,
            'accuracy': 0.0
        }
    }
    
    # Count how many user preferences match ground truth
    for _, _, user_pref, gt_pref in user_preferences:
        if gt_pref is not None and user_pref != 0:  # Skip equal or None preferences
            preference_stats['user_vs_ground_truth']['comparable_pairs'] += 1
            if user_pref == gt_pref:
                preference_stats['user_vs_ground_truth']['matching_preferences'] += 1
    
    # Calculate accuracy if there were any comparable pairs
    if preference_stats['user_vs_ground_truth']['comparable_pairs'] > 0:
        accuracy = (preference_stats['user_vs_ground_truth']['matching_preferences'] / 
                   preference_stats['user_vs_ground_truth']['comparable_pairs'])
        preference_stats['user_vs_ground_truth']['accuracy'] = accuracy
    
    # Save raw preferences
    preferences_file = os.path.join(args.output_dir, "raw_preferences.pkl")
    
    raw_data = {
        'user_preferences': user_preferences,  # Now includes ground truth
        'cluster_representatives': cluster_representatives,
        'preference_stats': preference_stats
    }
    
    torch.save(raw_data, preferences_file)
    print(f"Saved raw preferences to {preferences_file}")

    # Derive cluster ranking
    user_ranking, gt_ranking = derive_cluster_ranking(user_preferences)
    
    # Store ranking comparison in preference stats
    preference_stats['rankings'] = {
        'user_ranking': user_ranking,
        'ground_truth_ranking': gt_ranking
    }
    
    # Choose which ranking to use based on arguments
    if args.use_gt_ranking and gt_ranking:
        print("\nUsing ground truth ranking for augmented preferences generation")
        selected_ranking = gt_ranking
    else:
        print("\nUsing user-based ranking for augmented preferences generation")
        selected_ranking = user_ranking
    
    # Generate augmented preferences using the selected ranking
    augmented_preferences, augmented_stats = generate_augmented_preferences(
        data, clusters, segment_indices, selected_ranking
    )
    
    # Add augmented preference statistics to overall stats
    preference_stats['augmented_vs_ground_truth'] = augmented_stats
    
    # Create preference dataset
    dataset_file = os.path.join(args.output_dir, "preference_dataset.pkl")
    preference_dataset = create_preference_dataset(
        data, segment_indices, augmented_preferences, dataset_file, 
        preference_stats=preference_stats  # Include stats in metadata
    )
    
    print("\nProcess complete!")
    print(f"Collected {len(user_preferences)} direct user preferences")
    print(f"Generated {len(augmented_preferences)} augmented preferences")
    
    # Print user preference accuracy summary
    if preference_stats['user_vs_ground_truth']['comparable_pairs'] > 0:
        print("\nUser preference vs. ground truth summary:")
        print(f"  Total comparable preferences: {preference_stats['user_vs_ground_truth']['comparable_pairs']}")
        print(f"  Matching preferences: {preference_stats['user_vs_ground_truth']['matching_preferences']}")
        print(f"  Accuracy: {preference_stats['user_vs_ground_truth']['accuracy']:.2%}")

    # Print augmented preference accuracy summary
    print("\nAugmented preference vs. ground truth summary:")
    print(f"  Total comparable preferences: {augmented_stats['ground_truth_available']}")
    print(f"  Matching preferences: {augmented_stats['matching_ground_truth']}")
    print(f"  Accuracy: {augmented_stats['accuracy']:.2%}")
    
    # Print comparison between user and augmented preferences
    if preference_stats['user_vs_ground_truth']['comparable_pairs'] > 0 and augmented_stats['ground_truth_available'] > 0:
        user_acc = preference_stats['user_vs_ground_truth']['accuracy']
        aug_acc = augmented_stats['accuracy']
        
        print("\nComparison:")
        if abs(user_acc - aug_acc) < 0.05:  # Within 5%
            print(f"  User and augmented preferences have similar accuracy (difference: {abs(user_acc - aug_acc):.2%})")
        elif user_acc > aug_acc:
            print(f"  User preferences are more accurate than augmented preferences by {user_acc - aug_acc:.2%}")
        else:
            print(f"  Augmented preferences are more accurate than user preferences by {aug_acc - user_acc:.2%}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
