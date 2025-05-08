import torch
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import dtw

# Set torch hub cache directory
os.environ['TORCH_HOME'] = '/scr/aliang80/.cache'

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Default data paths for MetaWorld tasks
DEFAULT_DATA_PATHS = [
    "/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt",
    "/scr/shared/clam/datasets/metaworld/bin-picking-v2/buffer_bin-picking-v2.pt",
    "/scr/shared/clam/datasets/metaworld/peg-insert-side-v2/buffer_peg-insert-side-v2.pt",
]

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
    
    # Initialize DINOv2 model
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
    if segments:
        print(f"Each segment has shape: {segments[0].shape}")
    
    return segments, segment_indices

def compute_dtw_distance(query_segment, reference_segment):
    """Compute DTW distance between two segments."""
    try:
        # Convert to numpy for dtw
        query = query_segment.numpy()
        reference = reference_segment.numpy()
        
        # Use the custom DTW implementation
        cost, _, _ = dtw.get_single_match(query, reference)
        
        # Check if the cost is finite
        if not np.isfinite(cost):
            # Fall back to a simpler distance metric
            cost = np.mean((query.mean(0) - reference.mean(0))**2)
    except Exception as e:
        # Fall back to a simpler distance metric
        query = query_segment.numpy()
        reference = reference_segment.numpy()
        cost = np.mean((query.mean(0) - reference.mean(0))**2)
    
    return cost

def compute_dtw_distance_matrix(segments, max_segments=None):
    """Compute DTW distance matrix between segments using the custom DTW implementation."""
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

def find_top_matches(query_segment, all_segments, top_k=5, exclude_self=True):
    """Find top k segments with lowest DTW distance to the query segment."""
    distances = []
    
    for i, segment in enumerate(tqdm(all_segments, desc="Computing DTW distances")):
        # Skip self-comparison if needed
        if exclude_self and torch.all(query_segment == segment):
            distances.append(float('inf'))
            continue
        
        distance = compute_dtw_distance(query_segment, segment)
        distances.append(distance)
    
    # Get indices of top-k lowest distances
    distances = np.array(distances)
    top_indices = np.argsort(distances)[:top_k]
    top_distances = distances[top_indices]
    
    return top_indices, top_distances

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

def sample_segments(segments, segment_indices, n_samples=500):
    """Sample n segments from the full list."""
    if n_samples >= len(segments):
        return segments, segment_indices
    
    # Sample indices
    sample_idxs = random.sample(range(len(segments)), n_samples)
    
    # Extract sampled segments and indices
    sampled_segments = [segments[i] for i in sample_idxs]
    sampled_indices = [segment_indices[i] for i in sample_idxs]
    
    return sampled_segments, sampled_indices 