import torch
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pickle
import argparse
import dtw

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    compute_dtw_distance_matrix
)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def extract_eef_trajectories(data, segment_length=20, max_segments=None, use_relative_differences=False):
    """Extract end-effector trajectories or trajectory differences from the observation data.
    
    Args:
        data: TensorDict with observations and episode IDs
        segment_length: Length of segments to extract
        max_segments: Maximum number of segments to extract
        use_relative_differences: If True, extract frame-to-frame differences in EEF positions
                                 instead of absolute positions
        
    Returns:
        segments: List of EEF trajectory segments or difference segments
        segment_indices: List of (start_idx, end_idx) for each segment
        original_segments: List of original EEF position segments (for visualization)
                          Same as 'segments' if use_relative_differences=False
    """
    # Extract the observations and episode IDs
    observations = data["obs"] if "obs" in data else data["state"]
    episode_ids = data["episode"]
    
    print(f"Observation shape: {observations.shape}, device: {observations.device}")
    print(f"Episode IDs shape: {episode_ids.shape}, device: {episode_ids.device}")
    
    # Move data to CPU for processing
    observations_cpu = observations.cpu()
    episode_ids_cpu = episode_ids.cpu()
    
    # Extract only the first 3 dimensions (EEF position)
    eef_positions = observations_cpu[:, :3]
    
    print(f"Extracted EEF positions with shape: {eef_positions.shape}")
    print(f"First 5 EEF positions:")
    for i in range(min(5, len(eef_positions))):
        print(f"  {i}: {eef_positions[i]}")
        
    # Create segments
    if use_relative_differences:
        print(f"Creating relative difference segments with window size H={segment_length}")
        # Need segment_length + 1 for computing differences
        effective_length = segment_length + 1
    else:
        print(f"Creating absolute position segments with window size H={segment_length}")
        effective_length = segment_length
    
    segments = []
    original_segments = []
    segment_indices = []  # Store start and end indices of each segment
    
    # Create mask for NaN values in EEF positions
    eef_valid = ~torch.isnan(eef_positions).any(dim=1)
    
    # Get unique episodes
    unique_episodes = torch.unique(episode_ids_cpu).tolist()
    print(f"Found {len(unique_episodes)} unique episodes")
    
    # Collect all valid episode data for sampling
    all_valid_blocks = []
    
    print("Finding valid blocks in all episodes...")
    with tqdm(total=len(unique_episodes), desc="Processing episodes") as pbar:
        for episode_id in unique_episodes:
            # Get indices for this episode
            ep_indices = torch.where(episode_ids_cpu == episode_id)[0]
            
            # Apply valid mask to this episode
            ep_valid = eef_valid[ep_indices]
            
            # Skip if no valid positions
            if not ep_valid.any():
                pbar.update(1)
                continue
                
            # Get valid indices
            valid_indices = ep_indices[ep_valid]
            
            # Check if we have enough consecutive indices
            if len(valid_indices) < effective_length:
                pbar.update(1)
                continue
                
            # Find consecutive blocks using diff
            diffs = torch.diff(valid_indices)
            breaks = torch.where(diffs > 1)[0].tolist() + [len(valid_indices) - 1]
            
            # Process each block
            start_idx = 0
            for b in breaks:
                # Check if block is long enough
                if b - start_idx + 1 >= effective_length:
                    # Store this valid block
                    all_valid_blocks.append({
                        'episode_id': episode_id,
                        'start_block': start_idx,
                        'end_block': b,
                        'valid_indices': valid_indices,
                        'length': b - start_idx + 1
                    })
                start_idx = b + 1
            
            pbar.update(1)
    
    print(f"Found {len(all_valid_blocks)} valid blocks across all episodes")
    
    # Simple loop: just keep sampling blocks in sequence until we have enough segments
    total_attempts = 0
    max_attempts = 10000  # Safety limit
    
    # If no max_segments specified, set a reasonable default based on number of blocks
    if max_segments is None:
        max_segments = min(5000, len(all_valid_blocks) * 2)
        print(f"No max_segments specified, setting to {max_segments}")
    
    # Shuffle blocks once at the beginning for variety
    random.shuffle(all_valid_blocks)
    
    with tqdm(total=max_segments, desc="Sampling segments") as pbar:
        block_idx = 0
        while len(segments) < max_segments and total_attempts < max_attempts:
            total_attempts += 1
            
            # Get current block and move to next (circle back when reaching the end)
            block = all_valid_blocks[block_idx]
            block_idx = (block_idx + 1) % len(all_valid_blocks)
            
            valid_indices = block['valid_indices']
            start_block = block['start_block'] 
            end_block = block['end_block']
            
            # Calculate how many possible segments can be created from this block
            possible_segments = block['length'] - effective_length + 1
            if possible_segments <= 0:
                continue
                
            # Choose a position within this block (sequentially if multiple samples from same block)
            # Use the total_attempts to ensure we pick different positions for the same block
            segment_offset = total_attempts % possible_segments
            start_idx = start_block + segment_offset
            end_idx = start_idx + effective_length - 1
            
            # Get start and end indices in the original data
            start = valid_indices[start_idx].item()
            end = valid_indices[end_idx].item()
                        
            # Get EEF trajectory for this segment
            segment_eef = eef_positions[start:end+1]
            
            # Double check for NaN values
            if torch.isnan(segment_eef).any():
                continue
                
            # For relative differences, compute frame-to-frame differences
            if use_relative_differences:
                diff_segment = segment_eef[1:] - segment_eef[:-1]
                segments.append(diff_segment)
                original_segments.append(segment_eef)
            else:
                segments.append(segment_eef)
                original_segments.append(segment_eef)
                
            segment_indices.append((start, end))
            pbar.update(1)
            
            # Exit early if we're taking too many attempts with little progress
            if total_attempts >= 10 * max_segments and len(segments) < max_segments / 10:
                print(f"Warning: Made {total_attempts} attempts but only found {len(segments)} valid segments. Stopping early.")
                break
    
    print(f"Created {len(segments)} {'difference' if use_relative_differences else 'absolute position'} segments with {total_attempts} sampling attempts")
    if segments:
        print(f"Each segment has shape: {segments[0].shape}")
    
    return segments, segment_indices, original_segments

def compute_dtw_distance(query_segment, reference_segment):
    """Compute DTW distance between two segments.
    
    Args:
        query_segment: First segment (trajectory)
        reference_segment: Second segment (trajectory)
        
    Returns:
        float: DTW distance between segments
    """
    # Convert to numpy for dtw
    query = query_segment.cpu().numpy()
    reference = reference_segment.cpu().numpy()
    
    # Use the custom DTW implementation
    cost, _, _ = dtw.get_single_match(query, reference)
    
    # Check if the cost is finite
    if not np.isfinite(cost):
        # Fall back to a simpler distance metric
        cost = np.mean((query.mean(0) - reference.mean(0))**2)
    
    return cost

def perform_hierarchical_clustering(distance_matrix, n_clusters=5, method='average'):
    """Perform hierarchical clustering on the distance matrix.
    
    Args:
        distance_matrix: DTW distance matrix
        n_clusters: Number of clusters to extract
        method: Linkage method to use ('average', 'complete', 'single', or 'ward')
                'ward' is only appropriate for Euclidean distances, not for DTW
    
    Returns:
        Z: Linkage matrix
    """
    print(f"Performing hierarchical clustering with {method} linkage method...")
    
    # Check for non-finite values in the distance matrix
    non_finite_mask = ~np.isfinite(distance_matrix)
    if np.any(non_finite_mask):
        non_finite_count = np.sum(non_finite_mask)
        print(f"WARNING: Found {non_finite_count} non-finite values in distance matrix!")
        print("Replacing non-finite values with large finite values...")
        
        # Replace non-finite values with a large finite value
        max_finite = np.max(distance_matrix[np.isfinite(distance_matrix)])
        replacement_value = max_finite * 10.0 if max_finite > 0 else 1000.0
        distance_matrix[non_finite_mask] = replacement_value
    
    # Convert distance matrix to condensed form for linkage
    condensed_dist = []
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            condensed_dist.append(distance_matrix[i, j])
    
    condensed_dist = np.array(condensed_dist)
    print(f"Condensed distance matrix shape: {condensed_dist.shape}")
    
    # Perform clustering
    Z = linkage(condensed_dist, method=method)
    
    return Z

def get_clusters(Z, n_clusters=5):
    """Get cluster assignments from linkage matrix Z."""
    print(f"Getting {n_clusters} clusters from linkage matrix...")
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Print cluster distribution
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    print("Cluster distribution:")
    for cluster, count in zip(unique_clusters, counts):
        print(f"  Cluster {cluster}: {count} segments ({count/len(clusters)*100:.1f}%)")
    
    return clusters

def plot_clustering(Z, n_clusters=5, output_file="eef_clustering_dendrogram.png"):
    """Plot the hierarchical clustering dendrogram."""
    plt.figure(figsize=(12, 8))
    dendrogram(
        Z, 
        truncate_mode='level', 
        p=n_clusters,
        leaf_font_size=10, 
        color_threshold=0.7*max(Z[:,2])
    )
    plt.title('Hierarchical Clustering of End-Effector Trajectories')
    plt.xlabel('Segment Index')
    plt.ylabel('Distance')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram to {output_file}")

def compute_eef_position_ranges(data_path):
    """Compute the global min and max ranges for EEF positions.
    
    Args:
        data_path: Path to dataset file
        
    Returns:
        tuple: (x_min, x_max, y_min, y_max, z_min, z_max) for consistent visualization
    """
    print("Computing global EEF position ranges for consistent visualization...")
    
    # Load data
    data = load_tensordict(data_path)
    
    # Extract EEF positions
    observations = data["obs"] if "obs" in data else data["state"]
    observations_cpu = observations.cpu()
    eef_positions = observations_cpu[:, :3].numpy()
    
    # Remove NaN values
    eef_positions = eef_positions[~np.isnan(eef_positions).any(axis=1)]
    
    # Compute min and max
    global_min = np.min(eef_positions, axis=0)
    global_max = np.max(eef_positions, axis=0)
    
    # Add padding for better visualization
    padding = 0.1
    range_vals = global_max - global_min
    
    # Add padding
    global_min -= padding * range_vals
    global_max += padding * range_vals
    
    x_min, y_min, z_min = global_min
    x_max, y_max, z_max = global_max
    
    print(f"Global EEF position ranges:")
    print(f"  X range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"  Y range: [{y_min:.4f}, {y_max:.4f}]")
    print(f"  Z range: [{z_min:.4f}, {z_max:.4f}]")
    
    return x_min, x_max, y_min, y_max, z_min, z_max

def plot_representative_trajectories(segments, clusters, n_per_cluster=3, output_dir="eef_plots", global_ranges=None):
    """Plot representative trajectories for each cluster.
    
    Args:
        segments: List of EEF trajectory segments
        clusters: Cluster assignments for each segment
        n_per_cluster: Number of segments to plot per cluster
        output_dir: Directory to save plots
        global_ranges: Optional tuple (x_min, x_max, y_min, y_max, z_min, z_max) for consistent visualization
    """
    # Find unique clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    # Assign colors to clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine axis limits to use
    if global_ranges:
        x_min, x_max, y_min, y_max, z_min, z_max = global_ranges
    else:
        # Find global min and max for consistent scaling
        all_coords = np.vstack([seg.cpu().numpy() for seg in segments])
        x_min, y_min, z_min = np.min(all_coords, axis=0)
        x_max, y_max, z_max = np.max(all_coords, axis=0)
        
        # Add some padding
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_min -= padding * x_range
        y_min -= padding * y_range
        z_min -= padding * z_range
        x_max += padding * x_range
        y_max += padding * y_range
        z_max += padding * z_range
    
    # Plot representative trajectories for each cluster
    for i, cluster_id in enumerate(unique_clusters):
        # Get segments for this cluster
        cluster_segments = [segments[j] for j in range(len(segments)) if clusters[j] == cluster_id]
        
        # Skip if no segments in this cluster
        if not cluster_segments:
            print(f"No segments found for cluster {cluster_id}")
            continue
            
        # Sample n_per_cluster segments (or fewer if not enough)
        n_samples = min(n_per_cluster, len(cluster_segments))
        sample_idxs = random.sample(range(len(cluster_segments)), n_samples)
        
        # Create figure for this cluster
        fig = plt.figure(figsize=(6*n_samples, 5))
        
        # Create subplots for different views
        for j in range(n_samples):
            sample_idx = sample_idxs[j]
            segment = cluster_segments[sample_idx]
            traj = segment.cpu().numpy()
            
            # 3D plot
            ax = fig.add_subplot(1, n_samples, j+1, projection='3d')
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i])
            
            # Add scatter points for start and end
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=50, label='Start')
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=50, label='End')
            
            # Set consistent axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # Set title and labels
            ax.set_title(f"Cluster {cluster_id} - Sample {j+1}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Add legend for the first subplot only
            if j == 0:
                ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cluster_{cluster_id}_representative.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Saved representative trajectory plots to {output_dir}")

def create_combined_grid_video(data, segments, original_segments, segment_indices, clusters, n_per_cluster=5, output_file="combined_grid_video.mp4", global_ranges=None):
    """Create a grid video showing both EEF trajectories and camera observations for each cluster.
    
    Args:
        data: TensorDict with observation data
        segments: List of EEF trajectory segments 
        original_segments: List of original EEF position segments (for visualization)
        segment_indices: List of (start_idx, end_idx) for each segment
        clusters: Cluster assignments for each segment
        n_per_cluster: Number of segments to show per cluster
        output_file: Path to save the output video
        global_ranges: Optional tuple (x_min, x_max, y_min, y_max, z_min, z_max) for consistent visualization
    
    Returns:
        Path to the saved video file
    """
    # Check if we have image data
    has_image_data = "image" in data
    if not has_image_data:
        print("No image data found. Only showing EEF trajectories.")
    
    # Prepare images if available
    if has_image_data:
        images = data["image"].cpu()
    
    # Find unique clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    # Assign colors to clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_colors = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
    
    # Determine EEF axis limits
    if global_ranges:
        x_min, x_max, y_min, y_max, z_min, z_max = global_ranges
    else:
        # Find global min and max for consistent scaling
        all_coords = np.vstack([seg.cpu().numpy() for seg in original_segments])
        x_min, y_min, z_min = np.min(all_coords, axis=0)
        x_max, y_max, z_max = np.max(all_coords, axis=0)
        
        # Add some padding
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
    
    # Determine grid layout - 2 columns per segment (one for EEF, one for camera view)
    cols_per_segment = 2 if has_image_data else 1
    total_cols = n_per_cluster * cols_per_segment
    
    # Create figure
    fig = plt.figure(figsize=(total_cols * 3, n_clusters * 3))
    gs = gridspec.GridSpec(n_clusters, total_cols)
    
    # Prepare lists for animation data
    all_eef_plots = []  # Will hold (line, point) tuples
    all_image_plots = []  # Will hold image plots
    all_trajectories = []  # Will hold EEF trajectory data
    all_segment_frames = []  # Will hold image sequences
    
    print(f"Creating combined grid with {n_clusters} clusters, {n_per_cluster} segments per cluster")
    
    # Initialize subplots for each cluster and segment
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        # Get segments for this cluster
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        
        # Skip if no segments in this cluster
        if not cluster_indices:
            print(f"Skipping cluster {cluster_id} - no segments found")
            continue
        
        # Sample segments (or use all if less than requested)
        n_samples = min(n_per_cluster, len(cluster_indices))
        if n_samples < n_per_cluster:
            print(f"Warning: Only {n_samples} segments available for cluster {cluster_id}")
        
        # Sample indices
        sample_idxs = random.sample(cluster_indices, n_samples)
        
        # Create subplots for this cluster
        for j, sample_idx in enumerate(sample_idxs):
            # Get segment data
            segment = original_segments[sample_idx]
            start_idx, end_idx = segment_indices[sample_idx]
            
            # Get EEF trajectory
            traj = segment.cpu().numpy()
            
            # Base column index for this segment
            base_col = j * cols_per_segment
            
            # 1. Create 3D trajectory subplot
            ax_eef = fig.add_subplot(gs[cluster_idx, base_col], projection='3d')
            
            # Initialize empty plots
            line, = ax_eef.plot([], [], [], linewidth=2, color=cluster_colors[cluster_id])
            point = ax_eef.scatter([], [], [], color='red', s=50)
            
            # Set consistent axis limits
            ax_eef.set_xlim(x_min, x_max)
            ax_eef.set_ylim(y_min, y_max)
            ax_eef.set_zlim(z_min, z_max)
            
            # Set title and labels
            ax_eef.set_title(f"Cluster {cluster_id}")
            ax_eef.set_xlabel('X')
            ax_eef.set_ylabel('Y')
            ax_eef.set_zlabel('Z')
            
            # Add colored border for cluster identification
            for spine in ax_eef.spines.values():
                spine.set_edgecolor(cluster_colors[cluster_id])
                spine.set_linewidth(3)
            
            # Store plot and trajectory
            all_eef_plots.append((line, point))
            all_trajectories.append(traj)
            
            # 2. Create image subplot if we have image data
            if has_image_data:
                ax_img = fig.add_subplot(gs[cluster_idx, base_col + 1])
                
                # Get image sequence for this segment
                segment_images = images[start_idx:end_idx+1]
                
                # Initialize with first frame
                img_plot = ax_img.imshow(segment_images[0])
                
                # Add colored border to identify the cluster
                for spine in ax_img.spines.values():
                    spine.set_edgecolor(cluster_colors[cluster_id])
                    spine.set_linewidth(3)
                
                # Remove axis ticks for cleaner appearance
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                
                # Store image plot and sequence
                all_image_plots.append(img_plot)
                all_segment_frames.append(segment_images)
    
    # Add a global title
    fig.suptitle("Trajectory Clusters", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add a frame counter
    frame_text = fig.text(0.5, 0.01, "Frame: 0", ha='center', fontsize=12)
    
    # Animation update function
    def update(frame):
        updated_artists = [frame_text]
        frame_text.set_text(f"Frame: {frame}")
        
        # 1. Update EEF trajectories
        for i, ((line, point), traj) in enumerate(zip(all_eef_plots, all_trajectories)):
            # Update trajectory up to current frame
            frame_idx = min(frame, len(traj) - 1)
            
            # Update line data (growing trajectory)
            line.set_data(traj[:frame_idx+1, 0], traj[:frame_idx+1, 1])
            line.set_3d_properties(traj[:frame_idx+1, 2])
            
            # Update current position point
            point._offsets3d = ([traj[frame_idx, 0]], [traj[frame_idx, 1]], [traj[frame_idx, 2]])
            
            # Rotate the view
            ax = line.axes
            ax.view_init(elev=30, azim=(frame % 360))
            
            updated_artists.extend([line, point])
        
        # 2. Update images if we have image data
        if has_image_data:
            for i, (img_plot, segment) in enumerate(zip(all_image_plots, all_segment_frames)):
                # Get the correct frame, loop back to start if we reach the end
                frame_idx = frame % len(segment)
                
                # Update the image
                img_plot.set_array(segment[frame_idx])
                updated_artists.append(img_plot)
        
        return updated_artists
    
    # Find the maximum trajectory length to determine animation frames
    max_length = max(len(traj) for traj in all_trajectories)
    if has_image_data:
        # Adjust max length to handle different segment lengths
        image_max_length = max(len(frames) for frames in all_segment_frames)
        max_length = max(max_length, image_max_length)
    
    # Create animation
    print(f"Creating animation with {max_length} frames...")
    anim = animation.FuncAnimation(fig, update, frames=max_length, interval=100, blit=True)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close(fig)
    
    print(f"Saved combined visualization to {output_file}")
    return output_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cluster robot trajectories based on end-effector positions")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATHS[0],
                       help="Path to the PT file containing trajectory data")
    parser.add_argument('--segment_length', type=int, default=64,
                        help='Length of trajectory segments')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters to extract')
    parser.add_argument('--max_segments', type=int, default=1000,
                        help='Maximum number of segments to extract (default: all)')
    parser.add_argument('--max_dtw_segments', type=int, default=None,
                        help='Maximum number of segments for DTW computation (default: all)')
    parser.add_argument('--output_dir', type=str, default='eef_clustering',
                        help='Directory to save results')
    parser.add_argument('--use_relative_differences', action='store_true',
                        help='Use relative differences instead of absolute positions')
    parser.add_argument('--linkage_method', type=str, default='average', 
                        choices=['average', 'complete', 'single', 'ward'],
                        help='Linkage method for hierarchical clustering (default: average)')
    parser.add_argument("--use_shared_ranges", action="store_true",
                        help="Use shared ranges across all visualizations")
    parser.add_argument("--segments_per_cluster", type=int, default=5,
                        help="Number of representative segments to show per cluster in the grid video")
    parser.add_argument("--skip_videos", action="store_true",
                        help="Skip generating videos to speed up execution")
    parser.add_argument("--preprocessed_data", type=str, default=None,
                        help="Path to preprocessed data file (skips extraction if provided)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(f"Clustering robot trajectories based on end-effector {'relative differences' if args.use_relative_differences else 'positions'}")
    print(f"Data path: {args.data_path if args.preprocessed_data is None else args.preprocessed_data}")
    print(f"Segment length: {args.segment_length}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Linkage method: {args.linkage_method}")
    print(f"Use shared ranges: {args.use_shared_ranges}")
    print(f"Skip videos: {args.skip_videos}")
    print(f"Using preprocessed data: {args.preprocessed_data is not None}")
    print("="*50 + "\n")
    
    # Compute global EEF position ranges if needed
    global_ranges = None
    if args.use_shared_ranges:
        global_ranges = compute_eef_position_ranges(args.data_path)
    
    # Load data and extract segments
    segment_type = "relative" if args.use_relative_differences else "absolute"
    
    if args.preprocessed_data is not None:
        # Load preprocessed data
        from trajectory_utils import load_preprocessed_segments
        preproc_data = load_preprocessed_segments(args.preprocessed_data)
        
        # Check if the preprocessed data has the expected format and settings
        if 'use_relative_differences' in preproc_data and preproc_data['use_relative_differences'] != args.use_relative_differences:
            print(f"WARNING: Preprocessed data was created with use_relative_differences={preproc_data['use_relative_differences']}")
            print(f"Current setting is use_relative_differences={args.use_relative_differences}")
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Aborting.")
                return
        
        # Extract segments from preprocessed data
        segments = preproc_data['segments']
        segment_indices = preproc_data['segment_indices']
        original_segments = preproc_data['original_segments']
        
        # Load the original data if needed for visualization
        data = None
        if not args.skip_videos:
            # Try to use observation data from the preprocessed file first
            data = {}
            
            # Check for essential fields in preprocessed data
            essential_fields = ['obs', 'state', 'image', 'episode']
            has_essential_data = False
            
            for field in essential_fields:
                if field in preproc_data:
                    data[field] = preproc_data[field]
                    if field in ['obs', 'state']:
                        has_essential_data = True
            
            # If preprocessed data doesn't have the essential fields, try to load from original file
            if not has_essential_data:
                print(f"Preprocessed data doesn't contain necessary observation data for visualization.")
                print(f"Loading original data from {args.data_path}")
                data = load_tensordict(args.data_path)
            else:
                print("Using observation data from preprocessed file for visualization")
    else:
        # Load original data and extract segments
        print(f"Loading data from {args.data_path}")
        data = load_tensordict(args.data_path)

        # Extract EEF trajectories
        print("\nExtracting end-effector trajectories...")
        segments, segment_indices, original_segments = extract_eef_trajectories(
            data, 
            segment_length=args.segment_length, 
            max_segments=args.max_segments,
            use_relative_differences=args.use_relative_differences
        )
        
        print(f"Extracted {len(segments)} segments.")
    
    # Save segment visualization if not skipping videos
    if not args.skip_videos:
        output_file = os.path.join(args.output_dir, f"eef_{segment_type}_segments_3d.mp4")
        create_combined_grid_video(
            data,
            segments,
            original_segments,
            segment_indices,
            np.zeros(len(segments), dtype=int),
            n_per_cluster=args.segments_per_cluster,
            output_file=output_file,
            global_ranges=global_ranges
        )
    else:
        print("\nSkipping initial segment visualization (--skip_videos is set)")
    
    # Compute DTW distance matrix
    print("\nComputing DTW distance matrix...")
    distance_matrix, idx_mapping = compute_dtw_distance_matrix(segments, max_segments=args.max_dtw_segments)
    
    # If using a subset of segments for DTW, map back to original indices
    if args.max_dtw_segments is not None and args.max_dtw_segments < len(segments):
        segments_for_clustering = [segments[idx_mapping[i]] for i in range(len(idx_mapping))]
        original_segments_for_clustering = [original_segments[idx_mapping[i]] for i in range(len(idx_mapping))]
        indices_for_clustering = [segment_indices[idx_mapping[i]] for i in range(len(idx_mapping))]
    else:
        segments_for_clustering = segments
        original_segments_for_clustering = original_segments
        indices_for_clustering = segment_indices
    
    # Save distance matrix visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar(label='DTW Distance')
    plt.title('DTW Distance Matrix')
    plt.xlabel('Segment Index')
    plt.ylabel('Segment Index')
    output_file = os.path.join(args.output_dir, f"eef_{segment_type}_dtw_distance_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distance matrix visualization to {output_file}")
    
    # Perform hierarchical clustering
    Z = perform_hierarchical_clustering(distance_matrix, n_clusters=args.n_clusters, method=args.linkage_method)
    
    # Plot dendrogram
    plot_clustering(Z, n_clusters=args.n_clusters, 
                   output_file=os.path.join(args.output_dir, f"eef_{segment_type}_clustering_dendrogram.png"))
    
    # Get cluster assignments
    clusters = get_clusters(Z, n_clusters=args.n_clusters)
    
    # Plot representative trajectories (using original segments for visualization) if not skipping videos
    if not args.skip_videos:
        print("\nPlotting representative trajectories...")
        plot_representative_trajectories(
            original_segments_for_clustering, 
            clusters, 
            n_per_cluster=3,
            output_dir=args.output_dir,
            global_ranges=global_ranges
        )
        
        # Create combined grid video with EEF trajectories and camera observations
        print("\nCreating combined grid video...")
        create_combined_grid_video(
            data,
            segments_for_clustering,
            original_segments_for_clustering,
            indices_for_clustering,
            clusters,
            n_per_cluster=args.segments_per_cluster,
            output_file=f"{args.output_dir}/combined_grid_video.mp4",
            global_ranges=global_ranges
        )
    else:
        print("\nSkipping all video generation (--skip_videos is set)")
    
    # Save results
    print("\nSaving clustering results...")
    results = {
        'args': vars(args),
        'segment_indices': segment_indices,
        'distance_matrix': distance_matrix,
        'linkage_matrix': Z,
        'clusters': clusters,
        'idx_mapping': idx_mapping,
        'global_ranges': global_ranges,
        'use_relative_differences': args.use_relative_differences,
        'linkage_method': args.linkage_method
    }
    
    with open(f"{args.output_dir}/clustering_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {args.output_dir}/clustering_results.pkl")
    print(f"\nEEF trajectory {'relative difference' if args.use_relative_differences else 'position'} clustering complete!")
    
    return clusters

if __name__ == "__main__":
    main() 