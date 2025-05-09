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

# For running in interactive environments
is_notebook = 'ipykernel' in sys.modules
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
    load_preprocessed_segments
)

# Import clustering functions
from eef_clustering import extract_eef_trajectories

# Import visualization functions from eef_segment_matching instead
from eef_segment_matching import (
    create_eef_trajectory_animation,
    create_comparison_video
)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_clustering_results(results_file):
    """Load clustering results from a file."""
    print(f"Loading clustering results from {results_file}")
    with open(results_file, 'rb') as f:
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
        
    anim = create_eef_trajectory_animation(eef_positions, start_idx, end_idx, title=title)
    
    # Display animation
    if is_notebook:
        display(HTML(anim.to_jshtml()))
    
    return anim

def present_preference_query(data, segment1, segment2, query_id=None, skip_videos=False):
    """Present a preference query to the user.
    
    Args:
        data: TensorDict with observations
        segment1: (idx1, start_idx1, end_idx1) for first segment
        segment2: (idx2, start_idx2, end_idx2) for second segment
        query_id: Optional ID for the query
        skip_videos: If True, skip generating videos to save time
        
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
    
    # Get end-effector trajectories
    eef_positions = data["obs"][:, :3] if "obs" in data else data["state"][:, :3]
    
    # Display the two segments textually
    print(f"Segment 1: {start_idx1}-{end_idx1} (Length: {end_idx1-start_idx1+1})")
    print(f"Segment 2: {start_idx2}-{end_idx2} (Length: {end_idx2-start_idx2+1})")
    
    # Display animations and video only if not skipping
    if not skip_videos:
        print("\nRendering trajectories...")
        
        # Create two EEF trajectory animations
        import matplotlib.pyplot as plt
        from matplotlib import animation
        import matplotlib.gridspec as gridspec
        
        # 3D animations of EEF positions
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        # First trajectory
        ax1 = fig.add_subplot(gs[0], projection='3d')
        ax1.set_title("Segment 1")
        
        # Second trajectory
        ax2 = fig.add_subplot(gs[1], projection='3d')
        ax2.set_title("Segment 2")
        
        # Get trajectory data
        traj1 = eef_positions[start_idx1:end_idx1+1].cpu().numpy()
        traj2 = eef_positions[start_idx2:end_idx2+1].cpu().numpy()
        
        # Set the same scale for both plots to make comparison fair
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
        
        # Set limits for both axes
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # Initialize lines
        line1, = ax1.plot([], [], [], 'r-', linewidth=2)
        point1 = ax1.scatter([], [], [], color='blue', s=50)
        
        line2, = ax2.plot([], [], [], 'r-', linewidth=2)
        point2 = ax2.scatter([], [], [], color='blue', s=50)
        
        # Function to initialize animation
        def init():
            line1.set_data([], [])
            line1.set_3d_properties([])
            point1._offsets3d = ([], [], [])
            
            line2.set_data([], [])
            line2.set_3d_properties([])
            point2._offsets3d = ([], [], [])
            
            return line1, point1, line2, point2
        
        # Animation function
        def animate(i):
            # Update first trajectory
            frame_idx1 = min(i, len(traj1) - 1)
            x1, y1, z1 = traj1[:frame_idx1+1, 0], traj1[:frame_idx1+1, 1], traj1[:frame_idx1+1, 2]
            line1.set_data(x1, y1)
            line1.set_3d_properties(z1)
            point1._offsets3d = ([traj1[frame_idx1, 0]], [traj1[frame_idx1, 1]], [traj1[frame_idx1, 2]])
            
            # Update second trajectory
            frame_idx2 = min(i, len(traj2) - 1)
            x2, y2, z2 = traj2[:frame_idx2+1, 0], traj2[:frame_idx2+1, 1], traj2[:frame_idx2+1, 2]
            line2.set_data(x2, y2)
            line2.set_3d_properties(z2)
            point2._offsets3d = ([traj2[frame_idx2, 0]], [traj2[frame_idx2, 1]], [traj2[frame_idx2, 2]])
            
            # Rotate the view for better visualization
            for ax in [ax1, ax2]:
                ax.view_init(elev=30, azim=i % 360)
            
            return line1, point1, line2, point2
        
        # Create animation
        max_frames = max(len(traj1), len(traj2))
        ani = animation.FuncAnimation(
            fig, animate, init_func=init, frames=max_frames,
            interval=100, blit=True
        )
        
        plt.tight_layout()
        
        # Try saving animation using multiple codec options
        temp_anim_path = "temp_comparison_3d.mp4"
        success = False
        
        # Try a sequence of codec configurations, from more advanced to simpler ones
        codec_configs = [
            # Try with default settings first
            {'fps': 15, 'bitrate': 1800},
            # Try with H.264 explicitly
            {'fps': 15, 'bitrate': 1800, 'codec': 'h264', 'extra_args': ['-pix_fmt', 'yuv420p']},
            # Try with MPEG4 codec
            {'fps': 15, 'bitrate': 1800, 'codec': 'mpeg4'},
            # Try a lower quality setting that might be more compatible
            {'fps': 10, 'bitrate': 1000}
        ]
        
        for config in codec_configs:
            try:
                print(f"Trying video encoding with config: {config}")
                writer = animation.FFMpegWriter(**config)
                ani.save(temp_anim_path, writer=writer)
                success = True
                print(f"Successfully saved 3D trajectory animation to {temp_anim_path}")
                break
            except Exception as e:
                print(f"Failed with config {config}: {e}")
        
        # If all video attempts fail, save as a sequence of PNG frames
        if not success:
            print("Video encoding failed. Saving key frames as PNG images instead...")
            frames_dir = "comparison_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Save key frames as PNGs
            num_frames = max(len(traj1), len(traj2))
            key_frames = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
            key_frames = [min(f, num_frames - 1) for f in key_frames]  # Ensure within bounds
            
            for i, frame in enumerate(key_frames):
                # Update the figure for this frame
                animate(frame)
                
                # Save the frame
                frame_path = os.path.join(frames_dir, f"frame_{i}_step_{frame}.png")
                plt.savefig(frame_path, dpi=100)
                print(f"Saved frame {i} (step {frame}) to {frame_path}")
                
                # Add to generated videos list so it can be opened
                generated_videos.append(frame_path)
            
            # Create a combined image showing all key frames
            combined_fig = plt.figure(figsize=(15, 10))
            for i, frame in enumerate(key_frames):
                ax = combined_fig.add_subplot(len(key_frames), 1, i+1)
                frame_path = os.path.join(frames_dir, f"frame_{i}_step_{frame}.png")
                img = plt.imread(frame_path)
                ax.imshow(img)
                ax.set_title(f"Step {frame}")
                ax.axis('off')
            
            combined_path = os.path.join(frames_dir, "all_frames.png")
            plt.tight_layout()
            plt.savefig(combined_path, dpi=100)
            print(f"Saved combined frames to {combined_path}")
            generated_videos.append(combined_path)
            
            # Replace the failed video path with the combined image
            if temp_anim_path in generated_videos:
                generated_videos.remove(temp_anim_path)
        
        plt.close(fig)
        
        # Track generated videos
        generated_videos = [temp_anim_path]
        
        # Also create observation video if images are available
        if "image" in data:
            print("Creating observation video...")
            
            # Create side-by-side video with observations
            temp_video = "temp_comparison_obs.mp4"
            try:
                create_comparison_video(
                    eef_positions,
                    (start_idx1, end_idx1),
                    [(start_idx2, end_idx2)], 
                    [0],  # Placeholder for distance
                    dataset_indicators=None,
                    output_file=temp_video,
                    data=data
                )
                print(f"Saved observation video to {temp_video}")
                generated_videos.append(temp_video)
            except Exception as e:
                print(f"Error creating observation video: {e}")
                print("Saving observation frames as images instead...")
                
                # Create a directory for the observation frames
                frames_dir = "observation_frames"
                os.makedirs(frames_dir, exist_ok=True)
                
                # Get image sequences
                img_seq1 = data["image"][start_idx1:end_idx1+1].cpu().numpy()
                img_seq2 = data["image"][start_idx2:end_idx2+1].cpu().numpy()
                
                # Determine key frames to save
                max_frames = max(len(img_seq1), len(img_seq2))
                key_frames = [0, max_frames // 4, max_frames // 2, 3 * max_frames // 4, min(max_frames - 1, max_frames - 1)]
                
                # Create a combined figure for each key frame
                for i, frame_idx in enumerate(key_frames):
                    # Get the appropriate frame index for each sequence
                    idx1 = min(frame_idx, len(img_seq1) - 1)
                    idx2 = min(frame_idx, len(img_seq2) - 1)
                    
                    # Create a figure with the two images side by side
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Show the images
                    ax1.imshow(img_seq1[idx1])
                    ax1.set_title(f"Segment 1 - Frame {idx1}")
                    ax1.axis('off')
                    
                    ax2.imshow(img_seq2[idx2])
                    ax2.set_title(f"Segment 2 - Frame {idx2}")
                    ax2.axis('off')
                    
                    # Save the figure
                    frame_path = os.path.join(frames_dir, f"obs_frame_{i}_step_{frame_idx}.png")
                    plt.savefig(frame_path, dpi=100)
                    plt.close(fig)
                    print(f"Saved observation frame {i} (step {frame_idx}) to {frame_path}")
                    
                    # Add to generated videos list
                    generated_videos.append(frame_path)
                
                # Create a combined figure showing all key frames
                fig = plt.figure(figsize=(15, 10))
                gs = gridspec.GridSpec(len(key_frames), 2)
                
                for i, frame_idx in enumerate(key_frames):
                    # Get the appropriate frame index for each sequence
                    idx1 = min(frame_idx, len(img_seq1) - 1)
                    idx2 = min(frame_idx, len(img_seq2) - 1)
                    
                    # Add subplots for this frame
                    ax1 = fig.add_subplot(gs[i, 0])
                    ax2 = fig.add_subplot(gs[i, 1])
                    
                    # Show the images
                    ax1.imshow(img_seq1[idx1])
                    ax1.set_title(f"Segment 1 - Frame {idx1}")
                    ax1.axis('off')
                    
                    ax2.imshow(img_seq2[idx2])
                    ax2.set_title(f"Segment 2 - Frame {idx2}")
                    ax2.axis('off')
                
                # Save the combined figure
                combined_path = os.path.join(frames_dir, "all_observation_frames.png")
                plt.tight_layout()
                plt.savefig(combined_path, dpi=100)
                plt.close(fig)
                print(f"Saved combined observation frames to {combined_path}")
                generated_videos.append(combined_path)
    else:
        print("\n[Videos skipped to save time]")
    
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
    
    return preference

def collect_cluster_preferences(data, cluster_representatives, num_comparisons=None, skip_videos=False):
    """Collect user preferences between cluster representatives.
    
    Args:
        data: TensorDict with observations
        cluster_representatives: Dict mapping cluster_id to representative segments
        num_comparisons: Number of comparisons to conduct (default: all pairs)
        skip_videos: Skip generating videos for preference collection
        
    Returns:
        list: User preferences in format [(cluster_id1, cluster_id2, preference), ...]
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
        print(f"Limiting to {num_comparisons} comparisons out of {len(cluster_pairs)} possible pairs")
        cluster_pairs = cluster_pairs[:num_comparisons]
    
    # Collect preferences
    preferences = []
    
    for i, (cluster_id1, cluster_id2) in enumerate(cluster_pairs):
        print(f"\nComparison {i+1}/{len(cluster_pairs)}: Cluster {cluster_id1} vs Cluster {cluster_id2}")
        
        # Select a random representative from each cluster
        segment1 = random.choice(cluster_representatives[cluster_id1])
        segment2 = random.choice(cluster_representatives[cluster_id2])
        
        # Present preference query
        preference = present_preference_query(data, segment1, segment2, query_id=i+1, skip_videos=skip_videos)
        
        # Record preference
        preferences.append((cluster_id1, cluster_id2, preference))
        print(f"Recorded preference: {preference}")
    
    return preferences

def derive_cluster_ranking(preferences):
    """Derive an approximate ranking of clusters from pairwise preferences.
    
    Args:
        preferences: List of (cluster_id1, cluster_id2, preference) tuples
        
    Returns:
        list: Ordered list of cluster IDs from most to least preferred
    """
    print("Deriving cluster ranking from preferences...")
    
    # Extract all unique cluster IDs
    cluster_ids = set()
    for c1, c2, _ in preferences:
        cluster_ids.add(c1)
        cluster_ids.add(c2)
    
    cluster_ids = sorted(list(cluster_ids))
    
    # Initialize win counts for each cluster
    win_counts = {cluster_id: 0 for cluster_id in cluster_ids}
    
    # Count wins for each cluster
    for c1, c2, pref in preferences:
        if pref == 1:
            win_counts[c1] += 1
        elif pref == 2:
            win_counts[c2] += 1
    
    # Sort clusters by win count (descending)
    ranked_clusters = sorted(win_counts.keys(), key=lambda c: win_counts[c], reverse=True)
    
    print("Cluster ranking (most to least preferred):")
    for i, cluster_id in enumerate(ranked_clusters):
        print(f"  {i+1}. Cluster {cluster_id} (wins: {win_counts[cluster_id]})")
    
    return ranked_clusters

def generate_augmented_preferences(data, clusters, segment_indices, cluster_ranking):
    """Generate augmented preferences based on cluster ranking.
    
    Args:
        data: TensorDict with observations
        clusters: Cluster assignments
        segment_indices: List of (start_idx, end_idx) for each segment
        cluster_ranking: Ordered list of cluster IDs from most to least preferred
        
    Returns:
        list: Augmented preferences [(i, j, pref), ...]
              where i, j are segment indices and pref is 1 if i preferred, 2 if j preferred
    """
    print("Generating augmented preferences...")
    
    # Convert cluster ranking to a mapping of cluster to rank (0 is best)
    cluster_ranks = {cluster_id: rank for rank, cluster_id in enumerate(cluster_ranking)}
    
    # Generate augmented preferences
    augmented_preferences = []
    
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
        for cluster2 in cluster_ranking[i+1:]:  # Lower ranks (less preferred)
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
                    augmented_preferences.append((idx1, idx2, 1))  # Prefer segment from cluster1
                    augmented_count += 1
    
    print(f"Generated {augmented_count} augmented preferences")
    
    return augmented_preferences

def create_preference_dataset(data, segment_indices, augmented_preferences, output_file):
    """Create a dataset for training a reward model from augmented preferences.
    
    Args:
        data: TensorDict with observations
        segment_indices: List of (start_idx, end_idx) for each segment
        augmented_preferences: List of (i, j, pref) preference tuples
        output_file: Path to save the dataset
        
    Returns:
        dict: Preference dataset with segments and labels
    """
    print(f"Creating preference dataset at {output_file}")
    
    # Extract segments from data
    all_segments = []
    for start_idx, end_idx in segment_indices:
        segment_obs = data["obs"][start_idx:end_idx+1]
        all_segments.append(segment_obs)
    
    # Create preference pairs
    segment_pairs = []
    preference_labels = []
    
    for i, j, pref in augmented_preferences:
        segment_pairs.append((i, j))
        preference_labels.append(pref)
    
    # Create dataset
    preference_dataset = {
        'segments': all_segments,
        'segment_indices': segment_indices,
        'segment_pairs': segment_pairs,
        'preference_labels': preference_labels,
        'metadata': {
            'n_segments': len(all_segments),
            'n_preferences': len(preference_labels),
            'data_path': data.get('_source_path', 'unknown')
        }
    }
    
    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(preference_dataset, f)
    
    print(f"Saved preference dataset with {len(preference_labels)} preferences")
    
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
        
        if system == 'Windows':
            # Windows
            os.startfile(video_path)
        elif system == 'Darwin':
            # macOS
            subprocess.call(['open', video_path])
        else:
            # Linux or other Unix-like systems
            subprocess.call(['xdg-open', video_path])
            
        print(f"Opened video: {video_path}")
        return True
    except Exception as e:
        print(f"Error opening video: {e}")
        return False

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
        essential_fields = ['obs', 'state', 'image', 'episode', 'reward']
        missing_fields = []
        
        for field in essential_fields:
            if field in preproc_data:
                data[field] = preproc_data[field]
            else:
                missing_fields.append(field)
        
        # Handle missing fields
        if 'obs' not in data and 'state' not in data:
            print(f"WARNING: Preprocessed data does not contain observation data (obs or state).")
            print(f"Loading original data from {args.data_path} for observations.")
            orig_data = load_tensordict(args.data_path)
            
            # Copy missing essential fields
            for field in missing_fields:
                if field in orig_data:
                    data[field] = orig_data[field]
                    print(f"Loaded field '{field}' from original data")
        
        # Add source path for reference
        data['_source_path'] = args.preprocessed_data
    else:
        print(f"Loading data from {args.data_path}")
        data = load_tensordict(args.data_path)
        data['_source_path'] = args.data_path
    
    # Load clustering results
    clustering_results = load_clustering_results(args.clustering_results)
    clusters = clustering_results['clusters']
    segment_indices = clustering_results['segment_indices']
    
    print(f"Loaded clustering results with {len(clusters)} clusters")
    
    # Get representative segments for each cluster
    cluster_representatives = get_representative_segments(
        data, clusters, segment_indices, n_per_cluster=args.n_representatives
    )
    
    # Collect user preferences between clusters
    user_preferences = collect_cluster_preferences(
        data, cluster_representatives, 
        num_comparisons=args.max_comparisons,
        skip_videos=args.skip_videos
    )
    
    # Save raw preferences
    preferences_file = os.path.join(args.output_dir, "raw_preferences.pkl")
    with open(preferences_file, 'wb') as f:
        pickle.dump({
            'user_preferences': user_preferences,
            'cluster_representatives': cluster_representatives
        }, f)
    
    print(f"Saved raw preferences to {preferences_file}")
    
    # Derive cluster ranking
    cluster_ranking = derive_cluster_ranking(user_preferences)
    
    # Generate augmented preferences
    augmented_preferences = generate_augmented_preferences(
        data, clusters, segment_indices, cluster_ranking
    )
    
    # Create preference dataset
    dataset_file = os.path.join(args.output_dir, "preference_dataset.pkl")
    preference_dataset = create_preference_dataset(
        data, segment_indices, augmented_preferences, dataset_file
    )
    
    print("Process complete!")
    print(f"Collected {len(user_preferences)} direct user preferences")
    print(f"Generated {len(augmented_preferences)} augmented preferences")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 