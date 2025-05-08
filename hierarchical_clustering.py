import torch
import numpy as np
from tensordict import TensorDict
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pathlib import Path
import pickle
import dtw
from PIL import Image
import os
from tqdm import tqdm
import random
from matplotlib import animation
import matplotlib.gridspec as gridspec
from IPython.display import HTML, display

# Set torch hub cache directory
os.environ['TORCH_HOME'] = '/scr/aliang80/.cache'

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_tensordict(file_path):
    """Load tensordict data from file."""
    data = torch.load(file_path)
    print(f"Loaded TensorDict with shape: {data['image'].shape}, device: {data['image'].device}")
    print(f"Fields: {list(data.keys())}")
    return data

def extract_images_from_tensordict(data):
    """Extract images from tensordict and convert to proper format for DINOv2."""
    images = data["image"]
    print(f"Extracted images with shape: {images.shape}, type: {images.dtype}")
    return images

def compute_dinov2_embeddings(images, batch_size=32, cache_file=None):
    """Compute DINOv2 embeddings for the images with caching support."""
    # Check if cache exists
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        embeddings = torch.load(cache_file)
        print(f"Loaded embeddings with shape: {embeddings.shape}, device: {embeddings.device}")
        return embeddings
    
    print(f"Computing DINOv2 embeddings for {len(images)} images (this may take a while)...")
    
    # Initialize DINOv2 model - correct way to load it
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"DINOv2 model loaded on {device}")
    
    # Define image transformation for PIL images
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    # Create tqdm progress bar
    total_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), desc=f"Processing batches (batch size: {batch_size})", total=total_batches):
        batch_images = images[i:i+batch_size]
        
        # Convert to proper format and apply transforms
        processed_images = []
        for img in batch_images:
            # Convert torch tensor to PIL Image
            img_np = img.cpu().numpy().astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            # Apply transforms
            processed_img = transform(pil_img)
            processed_images.append(processed_img)
        
        # Stack into batch
        batch_tensor = torch.stack(processed_images)
        
        # Compute embeddings
        with torch.no_grad():
            batch_tensor = batch_tensor.to(device)
            batch_embeddings = model(batch_tensor)
            embeddings.append(batch_embeddings.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"Generated embeddings with shape: {all_embeddings.shape}, device: {all_embeddings.device}")
    
    # Cache the embeddings if cache_file is provided
    if cache_file:
        print(f"Caching embeddings to {cache_file}")
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        torch.save(all_embeddings, cache_file)
    
    return all_embeddings

def create_segments(embeddings, episode_ids, H):
    """Create H-step segments from embeddings, ensuring segments don't cross episode boundaries."""
    print(f"Creating segments with window size H={H} from embeddings of shape {embeddings.shape}")
    segments = []
    segment_indices = []  # Store start and end indices of each segment
    
    # Ensure episode_ids is on CPU for indexing operations
    episode_ids_cpu = episode_ids.cpu()
    
    unique_episodes = torch.unique(episode_ids_cpu).tolist()
    print(f"Found {len(unique_episodes)} unique episodes")
    
    for episode_id in tqdm(unique_episodes, desc="Processing episodes"):
        # Get indices for this episode (ensure indices are on CPU)
        ep_indices = torch.where(episode_ids_cpu == episode_id)[0]
        ep_embeddings = embeddings[ep_indices]
        
        num_segments_in_episode = max(0, len(ep_embeddings) - H + 1)
        
        # Create segments for this episode
        for i in range(0, len(ep_embeddings) - H + 1):
            segment = ep_embeddings[i:i+H]
            segments.append(segment)
            # Store indices as CPU tensors or integers
            segment_indices.append((int(ep_indices[i].item()), int(ep_indices[i+H-1].item())))
        
        if num_segments_in_episode == 0:
            print(f"Episode {episode_id} has {len(ep_embeddings)} frames (less than H={H}), skipping")
    
    print(f"Created {len(segments)} segments across {len(unique_episodes)} episodes")
    print(f"Each segment has shape: {segments[0].shape if segments else 'N/A'}")
    return segments, segment_indices

def compute_dtw_distance_matrix(segments, max_segments=None):
    """Compute DTW distance matrix between segments using the custom DTW implementation.
    
    Args:
        segments: List of segment embeddings
        max_segments: Maximum number of segments to use (randomly sampled if specified)
    """
    n_segments = len(segments)
    
    # If max_segments is specified and less than n_segments, sample a subset
    if max_segments is not None and max_segments < n_segments:
        print(f"Sampling {max_segments} segments out of {n_segments} for DTW calculation (random seed: {RANDOM_SEED})")
        # Use seeded random sampling
        segment_indices = random.sample(range(n_segments), max_segments)
        selected_segments = [segments[i] for i in segment_indices]
        # Create a mapping from new indices to original indices
        idx_mapping = {i: segment_indices[i] for i in range(max_segments)}
        n_segments = max_segments
    else:
        selected_segments = segments
        idx_mapping = {i: i for i in range(n_segments)}
        print(f"Using all {n_segments} segments for DTW calculation")
    
    print(f"This will compute {n_segments * (n_segments - 1) // 2} pairwise distances")
    
    distance_matrix = np.zeros((n_segments, n_segments))
    
    # Compute descriptive statistics for distances
    min_dist = float('inf')
    max_dist = float('-inf')
    sum_dist = 0
    count = 0
    non_finite_count = 0
    
    # Create tqdm for tracking progress
    total_comparisons = n_segments * (n_segments - 1) // 2
    with tqdm(total=total_comparisons, desc="Computing DTW distances") as pbar:
        for i in range(n_segments):
            for j in range(i+1, n_segments):
                # Convert to numpy for dtw
                query = selected_segments[i].numpy()
                reference = selected_segments[j].numpy()
                
                try:
                    # Use the custom DTW implementation
                    cost, _, _ = dtw.get_single_match(query, reference)
                    
                    # Check if the cost is finite
                    if not np.isfinite(cost):
                        print(f"WARNING: Non-finite cost ({cost}) obtained for segments {i} and {j}")
                        # Fall back to a simpler distance metric
                        cost = np.mean((query.mean(0) - reference.mean(0))**2)
                        non_finite_count += 1
                    
                    distance_matrix[i, j] = cost
                    distance_matrix[j, i] = cost
                    
                    # Update statistics only for finite values
                    if np.isfinite(cost):
                        min_dist = min(min_dist, cost)
                        max_dist = max(max_dist, cost)
                        sum_dist += cost
                        count += 1
                except Exception as e:
                    print(f"Error computing DTW for segments {i} and {j}: {e}")
                    # Fall back to a simpler distance metric
                    fallback_cost = np.mean((query.mean(0) - reference.mean(0))**2)
                    distance_matrix[i, j] = fallback_cost
                    distance_matrix[j, i] = fallback_cost
                    non_finite_count += 1
                
                pbar.update(1)
    
    # Print statistics about the distance matrix
    if count > 0:
        avg_dist = sum_dist / count
        print(f"Distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    if non_finite_count > 0:
        print(f"WARNING: {non_finite_count} distances were computed using fallback method due to non-finite DTW values")
    
    # Return the distance matrix and the mapping to original indices
    return distance_matrix, idx_mapping

def perform_hierarchical_clustering(distance_matrix, n_clusters=5):
    """Perform hierarchical clustering on the distance matrix."""
    print(f"Performing hierarchical clustering with target {n_clusters} clusters...")
    
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
        print(f"Replaced non-finite values with {replacement_value}")
    
    # Convert distance matrix to condensed form for linkage
    condensed_dist = []
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            dist_value = distance_matrix[i, j]
            # Double-check that the value is finite
            if not np.isfinite(dist_value):
                dist_value = replacement_value
            condensed_dist.append(dist_value)
    
    condensed_dist = np.array(condensed_dist)
    
    # Verify all values in condensed matrix are finite
    if not np.all(np.isfinite(condensed_dist)):
        raise ValueError("Condensed distance matrix contains non-finite values after replacement!")
    
    print(f"Condensed distance matrix shape: {condensed_dist.shape}")
    print(f"Condensed distance matrix range: [{np.min(condensed_dist):.2f}, {np.max(condensed_dist):.2f}]")
    
    # Perform clustering
    Z = linkage(condensed_dist, method='ward')
    print(f"Linkage matrix shape: {Z.shape}")
    
    return Z

def get_clusters(Z, n_clusters=5):
    """Get cluster assignments from linkage matrix Z."""
    print(f"Getting {n_clusters} clusters from linkage matrix...")
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Print cluster distribution
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    print("Cluster distribution:")
    for i, (cluster, count) in enumerate(zip(unique_clusters, counts)):
        print(f"  Cluster {cluster}: {count} segments ({count/len(clusters)*100:.1f}%)")
    
    return clusters

def sample_segment_pairs_from_clusters(segments, segment_indices, cluster_assignments, rewards, n_pairs=10):
    """Sample segment pairs from different clusters for preference labeling."""
    unique_clusters = np.unique(cluster_assignments)
    n_clusters = len(unique_clusters)
    print(f"Sampling {n_pairs} segment pairs from {n_clusters} clusters...")
    
    # For each cluster, get segments that belong to it
    cluster_to_segments = {c: [] for c in unique_clusters}
    for i, c in enumerate(cluster_assignments):
        cluster_to_segments[c].append(i)
    
    # Print number of segments in each cluster
    for c in unique_clusters:
        print(f"  Cluster {c}: {len(cluster_to_segments[c])} segments")
    
    # Sample pairs where each pair has segments from different clusters
    pairs = []
    cluster_combinations = []
    for _ in range(n_pairs):
        # Randomly select two different clusters
        c1, c2 = random.sample(list(unique_clusters), 2)
        
        # Randomly select one segment from each cluster
        idx1 = random.choice(cluster_to_segments[c1])
        idx2 = random.choice(cluster_to_segments[c2])
        
        pairs.append((idx1, idx2))
        cluster_combinations.append((c1, c2))
    
    print(f"Sampled {len(pairs)} segment pairs from clusters")
    for i, ((idx1, idx2), (c1, c2)) in enumerate(zip(pairs, cluster_combinations)):
        print(f"  Pair {i+1}: Segment {idx1} (Cluster {c1}) vs Segment {idx2} (Cluster {c2})")
    
    return pairs

def generate_synthetic_preferences(segment_pairs, segment_indices, rewards):
    """Generate synthetic preference labels based on cumulative rewards."""
    preference_labels = []
    
    # Ensure rewards is on CPU for indexing
    rewards_cpu = rewards.cpu()
    
    for idx1, idx2 in segment_pairs:
        # Get segment indices
        start1, end1 = segment_indices[idx1]
        start2, end2 = segment_indices[idx2]
        
        # Calculate cumulative reward for each segment
        reward1 = sum(rewards_cpu[start1:end1+1])
        reward2 = sum(rewards_cpu[start2:end2+1])
        
        # Assign preference label (1 if segment1 is preferred, 2 if segment2 is preferred)
        if reward1 > reward2:
            preference_labels.append(1)
        else:
            preference_labels.append(2)
    
    return preference_labels

def create_segment_animation(images, start_idx, end_idx, title=None):
    """Create an animation of a segment for visualization."""
    # Ensure images are on CPU for visualization
    segment_images = images[start_idx:end_idx+1].cpu()
    
    fig = plt.figure(figsize=(4, 4))
    if title:
        plt.title(title)
    plt.axis('off')
    
    im = plt.imshow(segment_images[0].numpy())
    
    def animate(i):
        im.set_array(segment_images[i].numpy())
        return [im]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(segment_images), 
        interval=200, blit=True
    )
    
    plt.close()
    return anim

def display_preference_query(images, segment_pairs, segment_indices, preference_labels=None):
    """Display a pair of segments and ask for preference."""
    # Ensure images are on CPU for visualization
    images_cpu = images.cpu()
    
    for i, (idx1, idx2) in enumerate(segment_pairs):
        # Get segment indices
        start1, end1 = segment_indices[idx1]
        start2, end2 = segment_indices[idx2]
        
        # Create animations for both segments
        anim1 = create_segment_animation(images_cpu, start1, end1, title="Segment 1")
        anim2 = create_segment_animation(images_cpu, start2, end2, title="Segment 2")
        
        # Display synthetic preference if available
        if preference_labels is not None:
            pref = preference_labels[i]
            print(f"Pair {i+1}/{len(segment_pairs)}: Synthetic preference: {'Segment 1' if pref == 1 else 'Segment 2'}")
        else:
            print(f"Pair {i+1}/{len(segment_pairs)}: Which segment do you prefer?")
        
        # Create a figure with two subplots side by side
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        ax1 = plt.subplot(gs[0])
        ax1.set_title('Segment 1')
        ax1.axis('off')
        ax1.imshow(images_cpu[start1].numpy())
        
        ax2 = plt.subplot(gs[1])
        ax2.set_title('Segment 2')
        ax2.axis('off')
        ax2.imshow(images_cpu[start2].numpy())
        
        plt.tight_layout()
        plt.show()
        
        # Display animations
        display(HTML(anim1.to_jshtml()))
        display(HTML(anim2.to_jshtml()))
        
        if preference_labels is None:
            # Ask for human preference
            while True:
                pref = input("Enter 1 for Segment 1, 2 for Segment 2, or 0 for Equal/Can't decide: ")
                if pref in ['0', '1', '2']:
                    break
                print("Invalid input, please try again.")
            
            result = "Segment 1" if pref == '1' else "Segment 2" if pref == '2' else "Equal/Can't decide"
            print(f"You preferred: {result}")
            print("\n---\n")

def collect_human_preferences(data, segments, segment_indices, cluster_assignments, n_pairs=10):
    """Collect human preferences for segment pairs."""
    # Sample segment pairs from clusters
    segment_pairs = sample_segment_pairs_from_clusters(
        segments, segment_indices, cluster_assignments, data["reward"], n_pairs
    )
    
    # Generate synthetic preferences for comparison
    synthetic_prefs = generate_synthetic_preferences(
        segment_pairs, segment_indices, data["reward"]
    )
    
    # Show synthetic preferences
    print("Showing synthetic preferences based on rewards:")
    display_preference_query(data["image"], segment_pairs, segment_indices, synthetic_prefs)
    
    # Collect human preferences
    print("\nNow collecting your preferences:")
    human_prefs = []
    pairs_with_prefs = display_preference_query(data["image"], segment_pairs, segment_indices)
    
    # Save preferences
    preference_data = {
        'segment_pairs': segment_pairs,
        'segment_indices': segment_indices,
        'synthetic_preferences': synthetic_prefs,
        'human_preferences': human_prefs
    }
    
    with open('preference_data.pkl', 'wb') as f:
        pickle.dump(preference_data, f)
    
    return preference_data

def plot_clustering(Z, n_clusters=5):
    """Plot the hierarchical clustering dendrogram."""
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='level', p=n_clusters, 
               leaf_font_size=10, color_threshold=0.7*max(Z[:,2]))
    plt.title('Hierarchical Clustering of Trajectory Segments')
    plt.xlabel('Segment Index')
    plt.ylabel('Distance')
    plt.savefig('clustering_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_preference_video(images, segment_pairs, segment_indices, preference_labels, output_file="preference_video.mp4", num_pairs=3):
    """Create a video showing multiple trajectory pairs with preference labels."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np
    from matplotlib.gridspec import GridSpec
    import random
    
    # Select a subset of pairs if there are more than num_pairs
    if len(segment_pairs) > num_pairs:
        selected_indices = random.sample(range(len(segment_pairs)), num_pairs)
        selected_pairs = [segment_pairs[i] for i in selected_indices]
        selected_prefs = [preference_labels[i] for i in selected_indices]
    else:
        selected_pairs = segment_pairs
        selected_prefs = preference_labels
    
    # Extract trajectory data
    trajectories = []
    
    for (idx1, idx2) in selected_pairs:
        start1, end1 = segment_indices[idx1]
        start2, end2 = segment_indices[idx2]
        
        # Ensure the segments are the same length
        length1 = end1 - start1 + 1
        length2 = end2 - start2 + 1
        length = min(length1, length2)
        
        traj1 = images[start1:start1+length].cpu().numpy()
        traj2 = images[start2:start2+length].cpu().numpy()
        
        trajectories.append((traj1, traj2))
    
    # Set up figure for animation
    fig = plt.figure(figsize=(12, 4*num_pairs))
    gs = GridSpec(num_pairs, 3, width_ratios=[1, 1, 0.2], figure=fig)
    
    # Setup for each trajectory pair
    axes = []
    titles = []
    images_plots = []
    
    for i in range(num_pairs):
        # Two axes for trajectories
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax_info = fig.add_subplot(gs[i, 2])
        
        ax1.set_title(f"Trajectory 1")
        ax2.set_title(f"Trajectory 2")
        
        # Hide axes
        ax1.axis('off')
        ax2.axis('off')
        ax_info.axis('off')
        
        # Initial empty images
        img1 = ax1.imshow(np.zeros_like(trajectories[i][0][0]), animated=True)
        img2 = ax2.imshow(np.zeros_like(trajectories[i][1][0]), animated=True)
        
        # Add preference information
        pref = selected_prefs[i]
        pref_text = ax_info.text(0.5, 0.5, f"Preferred: {'Traj 1' if pref == 1 else 'Traj 2'}", 
                                 fontsize=12, ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.5))
        
        axes.append((ax1, ax2, ax_info))
        images_plots.append((img1, img2))
        titles.append((pref_text))
    
    # Single global step counter
    step_text = fig.text(0.5, 0.98, "Step: 0", fontsize=14, ha='center')
    
    # Determine the maximum trajectory length
    max_length = max([traj1.shape[0] for traj1, traj2 in trajectories])
    
    # Animation function
    def animate(i):
        # Update step counter
        step_text.set_text(f"Step: {i}")
        
        # Update each trajectory pair
        for j in range(num_pairs):
            traj1, traj2 = trajectories[j]
            img1, img2 = images_plots[j]
            
            # If we're past the end of this trajectory, show the last frame
            frame_idx = min(i, len(traj1) - 1)
            
            img1.set_array(traj1[frame_idx])
            img2.set_array(traj2[frame_idx])
            
        return [step_text] + [item for sublist in images_plots for item in sublist]
    
    # Create animation
    print(f"Creating animation with {max_length} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=max_length, interval=200, blit=True)
    
    # Save animation
    print(f"Saving video to {output_file}...")
    writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close(fig)
    
    print(f"Video saved to {output_file}")
    return output_file

def main():
    # Parameters
    data_paths = [
        "/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt",
        "/scr/shared/clam/datasets/metaworld/bin-picking-v2/buffer_bin-picking-v2.pt",
        "/scr/shared/clam/datasets/metaworld/peg-insert-side-v2/buffer_peg-insert-side-v2.pt",
        # Add more paths as needed
    ]
    H = 20  # Number of steps in each segment
    n_clusters = 3  # Number of clusters for visualization
    batch_size = 64  # Batch size for DINOv2 embedding computation
    max_segments_for_dtw = None  # Maximum number of segments to use for DTW computation
    segments_per_dataset = 500  # Maximum number of segments to extract from each dataset
    
    # Cache file paths
    embeddings_cache = "cached_data/dinov2_embeddings.pt"
    
    print("\n" + "="*50)
    print(f"Processing data with parameters:")
    print(f"  Data paths: {data_paths}")
    print(f"  Segment size (H): {H}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max segments per dataset: {segments_per_dataset}")
    print(f"  Max segments for DTW: {max_segments_for_dtw}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Embeddings cache: {embeddings_cache}")
    print("="*50 + "\n")
    
    # Lists to store data from all datasets
    all_images = []
    all_episode_ids = []
    all_rewards = []
    dataset_indicators = []  # To keep track of which dataset each sample comes from
    
    # Load and process each dataset
    for dataset_idx, data_path in enumerate(data_paths):
        print(f"\nProcessing dataset {dataset_idx+1}/{len(data_paths)}: {data_path}")
        
        # Load data
        print("Loading tensordict data...")
        data = load_tensordict(data_path)
        
        # Extract images and episode IDs
        print("Extracting images and episode IDs...")
        images = extract_images_from_tensordict(data)
        episode_ids = data["episode"]
        rewards = data["reward"]
        
        # If segments_per_dataset is set, randomly sample episodes until we have enough segments
        if segments_per_dataset > 0:
            print(f"Sampling episodes to get approximately {segments_per_dataset} segments...")
            
            # Get unique episodes
            unique_episodes = torch.unique(episode_ids).cpu().numpy()
            np.random.shuffle(unique_episodes)
            
            selected_indices = []
            estimated_segments = 0
            
            for episode in unique_episodes:
                # Get indices for this episode
                ep_indices = torch.where(episode_ids.cpu() == episode)[0].numpy()
                
                # Add these indices to our selection
                selected_indices.extend(ep_indices)
                
                # Estimate number of segments this will create (ep_length - H + 1)
                ep_length = len(ep_indices)
                estimated_segments += max(0, ep_length - H + 1)
                
                # If we have enough segments, stop
                if estimated_segments >= segments_per_dataset:
                    break
            
            # Sort the indices to maintain order
            selected_indices = sorted(selected_indices)
            
            # Subsample the data
            print(f"Selected {len(selected_indices)} frames from {len(unique_episodes)} episodes, expected to yield ~{estimated_segments} segments")
            images = images[selected_indices]
            episode_ids = episode_ids[selected_indices]
            rewards = rewards[selected_indices]
            
            # Reset episode IDs to be consecutive within this dataset
            unique_eps = torch.unique(episode_ids)
            ep_mapping = {old_ep.item(): i + (dataset_idx * 10000) for i, old_ep in enumerate(unique_eps)}
            new_episode_ids = torch.tensor([ep_mapping[ep.item()] for ep in episode_ids])
            episode_ids = new_episode_ids
        
        # Append to combined data
        all_images.append(images)
        all_episode_ids.append(episode_ids)
        all_rewards.append(rewards)
        
        # Add dataset indicator (which dataset each sample came from)
        dataset_indicators.extend([dataset_idx] * len(images))
    
    # Combine data from all datasets
    images = torch.cat(all_images, dim=0)
    episode_ids = torch.cat(all_episode_ids, dim=0)
    rewards = torch.cat(all_rewards, dim=0)
    
    print(f"\nCombined data from {len(data_paths)} datasets:")
    print(f"  Total images: {len(images)}")
    print(f"  Total unique episodes: {len(torch.unique(episode_ids))}")
    
    # Compute DINOv2 embeddings with caching
    print("\nProcessing embeddings...")
    embeddings = compute_dinov2_embeddings(images, batch_size=batch_size, cache_file=embeddings_cache)
    
    # Create segments
    print("\nSegmenting data...")
    segments, segment_indices = create_segments(embeddings, episode_ids, H)
    
    # Compute DTW distance matrix on a subset of segments
    print("\nComputing DTW distances...")
    distance_matrix, idx_mapping = compute_dtw_distance_matrix(segments, max_segments=max_segments_for_dtw)
    
    # Perform hierarchical clustering
    print("\nClustering segments...")
    Z = perform_hierarchical_clustering(distance_matrix, n_clusters)
    
    # Get cluster assignments
    cluster_assignments = get_clusters(Z, n_clusters)
    
    # Map cluster assignments back to original indices if using a subset
    if max_segments_for_dtw < len(segments):
        print("\nMapping cluster assignments back to original segment indices...")
        full_cluster_assignments = np.zeros(len(segments), dtype=int)
        
        # First, set all to a default cluster (used for segments not in the DTW computation)
        full_cluster_assignments[:] = -1  # -1 indicates "not clustered"
        
        # Then, assign clusters to the segments that were included in the DTW computation
        for i, cluster in enumerate(cluster_assignments):
            orig_idx = idx_mapping[i]
            full_cluster_assignments[orig_idx] = cluster
            
        print(f"Cluster assignment coverage: {np.sum(full_cluster_assignments != -1)}/{len(segments)} segments")
        
        # For segments that weren't clustered, assign to nearest clustered segment
        if np.any(full_cluster_assignments == -1):
            print("Assigning unclustered segments to nearest clustered segment (optimized)...")
            unclustered_indices = np.where(full_cluster_assignments == -1)[0]
            clustered_indices = np.where(full_cluster_assignments != -1)[0]
            
            print(f"Need to assign {len(unclustered_indices)} unclustered segments using {len(clustered_indices)} reference segments")
            
            # Convert segments to numpy arrays for faster processing
            unclustered_segments = np.array([segments[i].mean(0).numpy() for i in unclustered_indices])
            clustered_segments = np.array([segments[i].mean(0).numpy() for i in clustered_indices])
            clustered_labels = full_cluster_assignments[clustered_indices]
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            num_batches = (len(unclustered_indices) + batch_size - 1) // batch_size
            
            print(f"Processing in {num_batches} batches of {batch_size} segments each")
            
            for batch_idx in tqdm(range(num_batches), desc="Assigning unclustered segments"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(unclustered_indices))
                batch_unclustered = unclustered_segments[start_idx:end_idx]
                
                # Compute pairwise distances between this batch and all clustered segments
                # Using broadcasting for efficient computation
                # Shape: [batch_size, num_clustered_segments, embedding_dim]
                differences = batch_unclustered[:, np.newaxis, :] - clustered_segments[np.newaxis, :, :]
                
                # Compute squared distances
                # Shape: [batch_size, num_clustered_segments]
                squared_distances = np.sum(differences ** 2, axis=2)
                
                # Find the index of the closest clustered segment for each unclustered segment
                closest_idx = np.argmin(squared_distances, axis=1)
                
                # Assign the corresponding cluster
                batch_cluster_assignments = clustered_labels[closest_idx]
                
                # Update the full assignments
                batch_unclustered_indices = unclustered_indices[start_idx:end_idx]
                full_cluster_assignments[batch_unclustered_indices] = batch_cluster_assignments
            
            print(f"All {len(unclustered_indices)} unclustered segments assigned to nearest clusters")
        
        # Use the full cluster assignments
        cluster_assignments = full_cluster_assignments
    
    # Plot results
    print("\nPlotting clustering results...")
    plot_clustering(Z, n_clusters)
    
    # Save processed data
    print("\nSaving processed data...")
    processed_data = {
        'embeddings': embeddings,
        'segments': segments,
        'segment_indices': segment_indices,
        'distance_matrix': distance_matrix,
        'idx_mapping': idx_mapping,
        'linkage_matrix': Z,
        'cluster_assignments': cluster_assignments,
        'dataset_indicators': dataset_indicators,
        'parameters': {
            'data_paths': data_paths,
            'H': H,
            'n_clusters': n_clusters,
            'segments_per_dataset': segments_per_dataset,
            'max_segments_for_dtw': max_segments_for_dtw,
            'random_seed': RANDOM_SEED
        }
    }
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Saved processed data to processed_data.pkl")
    
    # Generate synthetic preferences and interface for human labeling
    print("\nSampling segment pairs for preference labeling...")
    segment_pairs = sample_segment_pairs_from_clusters(
        segments, segment_indices, cluster_assignments, rewards, n_pairs=5
    )
    
    print("\nGenerating synthetic preferences based on rewards...")
    synthetic_prefs = generate_synthetic_preferences(
        segment_pairs, segment_indices, rewards
    )
    
    # Create and save preference video
    print("\nCreating preference comparison video...")
    video_file = create_preference_video(
        images, segment_pairs, segment_indices, synthetic_prefs, 
        output_file="preference_comparison.mp4", num_pairs=3
    )
    
    # Save preference data
    print("\nSaving preference data...")
    preference_data = {
        'segment_pairs': segment_pairs,
        'segment_indices': segment_indices,
        'synthetic_preferences': synthetic_prefs,
        'dataset_indicators': dataset_indicators,
    }
    with open('preference_data.pkl', 'wb') as f:
        pickle.dump(preference_data, f)
    print(f"Saved preference data to preference_data.pkl")
    
    # Display synthetic preferences
    print("\nDisplaying synthetic preferences (based on rewards):")
    display_preference_query(images, segment_pairs, segment_indices, synthetic_prefs)
    
    # Option to collect human preferences
    collect_human = input("\nWould you like to provide human preference labels? (y/n): ")
    if collect_human.lower() == 'y':
        preference_data = collect_human_preferences(
            data, segments, segment_indices, cluster_assignments, n_pairs=5
        )
        print("Human preference collection complete!")
    
    print("\nProcessing complete!")
    print("="*50)

if __name__ == "__main__":
    main()
