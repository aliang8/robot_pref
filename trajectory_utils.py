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
os.environ["TORCH_HOME"] = "/scr/aliang80/.cache"

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
    data = torch.load(file_path, weights_only=False)
    print(
        f"Loaded TensorDict with shape: {data['image'].shape}, device: {data['image'].device}"
    )
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
        print(
            f"Loaded embeddings with shape: {embeddings.shape}, device: {embeddings.device}"
        )
        return embeddings

    print(
        f"Computing DINOv2 embeddings for {len(images)} images (this may take a while)..."
    )

    # Initialize DINOv2 model
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"DINOv2 model loaded on {device}")

    # Define image transformation for PIL images
    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    embeddings = []
    # Create tqdm progress bar
    total_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(images), batch_size),
        desc=f"Processing batches (batch size: {batch_size})",
        total=total_batches,
    ):
        batch_images = images[i : i + batch_size]

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
    print(
        f"Generated embeddings with shape: {all_embeddings.shape}, device: {all_embeddings.device}"
    )

    # Cache the embeddings if cache_file is provided
    if cache_file:
        print(f"Caching embeddings to {cache_file}")
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        torch.save(all_embeddings, cache_file)

    return all_embeddings


def create_segments(data, segment_length=20, max_segments=None):
    """Create segments from data for reward model training.

    Args:
        data: TensorDict with observations/actions/rewards/episodes
        segment_length: Length of segments (H)
        max_segments: Maximum number of segments to return (None for all)

    Returns:
        segments: List of segment embeddings
        segment_indices: List of (start_idx, end_idx) for each segment
    """
    # Extract necessary data
    episode_ids = data["episode"]
    observations = data["obs"] if "obs" in data else data["state"]

    print(
        f"Creating segments with window size H={segment_length} from observations of shape {observations.shape}"
    )
    segments = []
    segment_indices = []  # Store start and end indices of each segment

    # Ensure episode_ids is on CPU for indexing operations
    episode_ids_cpu = episode_ids.cpu()

    # Create mask for NaN values
    obs_valid = ~torch.isnan(observations).any(dim=1)

    unique_episodes = torch.unique(episode_ids_cpu).tolist()
    print(f"Found {len(unique_episodes)} unique episodes")

    with tqdm(total=len(unique_episodes), desc="Processing episodes") as pbar:
        for episode_id in unique_episodes:
            # Get indices for this episode
            ep_indices = torch.where(episode_ids_cpu == episode_id)[0]

            # Apply valid mask to this episode
            ep_valid = obs_valid[ep_indices]

            # Skip if no valid observations
            if not ep_valid.any():
                continue

            # Get valid indices
            valid_indices = ep_indices[ep_valid]

            # Ensure valid_indices is on CPU for consistent processing
            valid_indices = valid_indices.cpu()

            # Check for consecutive indices
            if len(valid_indices) < segment_length:
                continue

            # Find consecutive blocks using diff
            diffs = torch.diff(valid_indices)
            breaks = torch.where(diffs > 1)[0]

            # Process each block of consecutive indices
            start_idx = 0
            for b in breaks:
                # Check if block is long enough
                if b - start_idx + 1 >= segment_length:
                    # Create segments from this block
                    for i in range(start_idx, b - segment_length + 2):
                        start = valid_indices[i].item()
                        end = valid_indices[i + segment_length - 1].item()

                        # Get observations for this segment
                        segment_obs = observations[start : end + 1]

                        # Check again for NaN values (redundancy check)
                        if not torch.isnan(segment_obs).any():
                            segments.append(segment_obs)
                            segment_indices.append((start, end))

                start_idx = b.item() + 1

            # Process final block
            if len(valid_indices) - start_idx >= segment_length:
                for i in range(start_idx, len(valid_indices) - segment_length + 1):
                    start = valid_indices[i].item()
                    end = valid_indices[i + segment_length - 1].item()

                    # Get observations for this segment
                    segment_obs = observations[start : end + 1]

                    # Check again for NaN values
                    if not torch.isnan(segment_obs).any():
                        segments.append(segment_obs)
                        segment_indices.append((start, end))

            pbar.update(1)

    # Subsample if needed
    if max_segments is not None and max_segments < len(segments) and max_segments > 0:
        # Sample segments
        indices = random.sample(range(len(segments)), max_segments)
        segments = [segments[i] for i in indices]
        segment_indices = [segment_indices[i] for i in indices]

    print(f"Created {len(segments)} segments across {len(unique_episodes)} episodes")
    if segments:
        print(f"Each segment has shape: {segments[0].shape}")

    return segments, segment_indices


def sample_segment_pairs(segments, segment_indices, rewards, n_pairs=5000):
    """Generate synthetic preference pairs based on cumulative rewards.

    Args:
        segments: List of segment data
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        rewards: Tensor of reward values for all transitions
        n_pairs: Number of preference pairs to generate

    Returns:
        pairs: List of (idx1, idx2) tuples indicating segment pairs
        preferences: List of preference labels (1 if first segment preferred, 2 if second)
    """
    n_segments = len(segments)
    print(f"Generating {n_pairs} preference pairs from {n_segments} segments")

    # Ensure rewards is on CPU for indexing
    rewards_cpu = rewards.cpu() if isinstance(rewards, torch.Tensor) else rewards

    # Sample random pairs of segment indices
    pairs = []
    preference_labels = []

    # Keep generating pairs until we have enough or max attempts reached
    max_attempts = n_pairs * 5  # Allow more attempts to handle cases with equal rewards

    with tqdm(total=n_pairs, desc="Generating preference pairs") as pbar:
        while len(pairs) < n_pairs and len(pairs) < max_attempts:
            # Sample two different segments
            idx1, idx2 = random.sample(range(n_segments), 2)

            # Get segment indices
            start1, end1 = segment_indices[idx1]
            start2, end2 = segment_indices[idx2]

            # Calculate cumulative reward for each segment
            reward1 = rewards_cpu[start1 : end1 + 1].sum().item()
            reward2 = rewards_cpu[start2 : end2 + 1].sum().item()

            # Skip if rewards are too close (avoid ambiguous preferences)
            if abs(reward1 - reward2) < 1e-6:
                continue

            # Add pair to the list
            pairs.append((idx1, idx2))

            # Assign preference label (1 if segment1 is preferred, 2 if segment2 is preferred)
            if reward1 > reward2:
                preference_labels.append(1)
            else:
                preference_labels.append(2)

            pbar.update(1)

    print(f"Generated {len(pairs)} preference pairs")
    return pairs, preference_labels


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
            cost = np.mean((query.mean(0) - reference.mean(0)) ** 2)
    except Exception as e:
        # Fall back to a simpler distance metric
        query = query_segment.numpy()
        reference = reference_segment.numpy()
        cost = np.mean((query.mean(0) - reference.mean(0)) ** 2)

    return cost


def compute_dtw_distance_matrix(segments, max_segments=None):
    """Compute DTW distance matrix between segments using the custom DTW implementation."""
    n_segments = len(segments)

    # If max_segments is specified and less than n_segments, sample a subset
    if max_segments is not None and max_segments < n_segments:
        print(
            f"Sampling {max_segments} segments out of {n_segments} for DTW calculation (random seed: {RANDOM_SEED})"
        )
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
    min_dist = float("inf")
    max_dist = float("-inf")
    sum_dist = 0
    count = 0
    non_finite_count = 0

    # Create tqdm for tracking progress
    total_comparisons = n_segments * (n_segments - 1) // 2
    with tqdm(total=total_comparisons, desc="Computing DTW distances") as pbar:
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                # Convert to numpy for dtw
                query = selected_segments[i].numpy()
                reference = selected_segments[j].numpy()

                try:
                    # Use the custom DTW implementation
                    cost, _, _ = dtw.get_single_match(query, reference)

                    # Check if the cost is finite
                    if not np.isfinite(cost):
                        print(
                            f"WARNING: Non-finite cost ({cost}) obtained for segments {i} and {j}"
                        )
                        # Fall back to a simpler distance metric
                        cost = np.mean((query.mean(0) - reference.mean(0)) ** 2)
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
                    fallback_cost = np.mean((query.mean(0) - reference.mean(0)) ** 2)
                    distance_matrix[i, j] = fallback_cost
                    distance_matrix[j, i] = fallback_cost
                    non_finite_count += 1

                pbar.update(1)

    # Print statistics about the distance matrix
    if count > 0:
        avg_dist = sum_dist / count
        print(
            f"Distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}"
        )
    print(f"Distance matrix shape: {distance_matrix.shape}")

    if non_finite_count > 0:
        print(
            f"WARNING: {non_finite_count} distances were computed using fallback method due to non-finite DTW values"
        )

    # Return the distance matrix and the mapping to original indices
    return distance_matrix, idx_mapping


def find_top_matches(query_segment, all_segments, top_k=5, exclude_self=True):
    """Find top k segments with lowest DTW distance to the query segment."""
    distances = []

    for i, segment in enumerate(tqdm(all_segments, desc="Computing DTW distances")):
        # Skip self-comparison if needed
        if exclude_self and torch.all(query_segment == segment):
            distances.append(float("inf"))
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
    segment_images = images[start_idx : end_idx + 1].cpu()

    fig = plt.figure(figsize=(4, 4))
    if title:
        plt.title(title)
    plt.axis("off")

    im = plt.imshow(segment_images[0].numpy())

    def animate(i):
        im.set_array(segment_images[i].numpy())
        return [im]

    anim = animation.FuncAnimation(
        fig, animate, frames=len(segment_images), interval=200, blit=True
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


def save_preprocessed_segments(data, output_file, compress=True):
    """Save preprocessed segments data to a file.

    Args:
        data: Dictionary containing segments, indices, clusters, etc.
        output_file: Path to save the data
        compress: Whether to use compression (default: True)
    """
    print(f"Saving preprocessed data to {output_file}")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Make sure all tensors are on CPU
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.cpu()
        elif (
            isinstance(value, list)
            and len(value) > 0
            and isinstance(value[0], torch.Tensor)
        ):
            data[key] = [v.cpu() for v in value]

    # Add timestamp
    import datetime

    data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save data
    if compress:
        torch.save(data, output_file, _use_new_zipfile_serialization=True)
    else:
        torch.save(data, output_file, _use_new_zipfile_serialization=False)

    print(f"Saved preprocessed data with keys: {list(data.keys())}")

    # Print some statistics about the data
    if "segments" in data:
        print(f"Number of segments: {len(data['segments'])}")
    if "clusters" in data and "segment_indices" in data:
        print(f"Number of clusters: {len(np.unique(data['clusters']))}")
        print(f"Number of segment indices: {len(data['segment_indices'])}")


def load_preprocessed_segments(file_path):
    """Load preprocessed segments data from file.

    Args:
        file_path: Path to the saved data file

    Returns:
        Dictionary containing the preprocessed data
    """
    print(f"Loading preprocessed data from {file_path}")
    data = torch.load(file_path, weights_only=False)

    # Print some statistics about the loaded data
    print(f"Loaded data with keys: {list(data.keys())}")
    if "segments" in data:
        print(f"Number of segments: {len(data['segments'])}")
    if "clusters" in data and "segment_indices" in data:
        print(f"Number of clusters: {len(np.unique(data['clusters']))}")
        print(f"Number of segment indices: {len(data['segment_indices'])}")
    if "timestamp" in data:
        print(f"Data timestamp: {data['timestamp']}")

    return data
