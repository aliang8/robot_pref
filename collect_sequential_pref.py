import torch
import numpy as np
import os
import random
import pickle
import argparse
import itertools
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.animation as animation

# For running in interactive environments
is_notebook = 'ipykernel' in sys.modules
if is_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    load_preprocessed_segments,
    compute_dtw_distance_matrix
)

# Import clustering functions
from eef_clustering import (
    extract_eef_trajectories,
    compute_dtw_distance,
    perform_hierarchical_clustering
)

def get_reward_based_preference(data, segment1, segment2):
    """Determine ground truth preference based on cumulative reward.
    
    Args:
        data: TensorDict with observations and rewards
        segment1: (start_idx1, end_idx1) for first segment
        segment2: (start_idx2, end_idx2) for second segment
        
    Returns:
        int: 1 if segment1 has higher reward, 2 if segment2 has higher reward, 0 if equal
    """
    if "reward" not in data:
        print("WARNING: No reward data found, cannot determine ground truth preference")
        return None
    
    # Extract segment information
    start_idx1, end_idx1 = segment1
    start_idx2, end_idx2 = segment2
    
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

def extract_segments(data, segment_length=20, max_segments=None, use_relative_differences=False):
    """Extract segments from the data.
    
    Args:
        data: TensorDict with observations and episode IDs
        segment_length: Length of segments to extract
        max_segments: Maximum number of segments to extract
        use_relative_differences: If True, extract frame-to-frame differences
        
    Returns:
        segments: List of trajectory segments
        segment_indices: List of (start_idx, end_idx) for each segment
        original_segments: List of original position segments
    """
    print(f"Extracting segments with length {segment_length}...")
    
    # Reuse the extract_eef_trajectories function from eef_clustering.py
    segments, segment_indices, original_segments = extract_eef_trajectories(
        data, 
        segment_length=segment_length, 
        max_segments=max_segments,
        use_relative_differences=use_relative_differences
    )
    
    print(f"Extracted {len(segments)} segments")
    return segments, segment_indices, original_segments

def find_similar_segments(segments, query_idx, k=5, distance_matrix=None):
    """Find the k most similar segments to the query segment.
    
    Args:
        segments: List of trajectory segments
        query_idx: Index of the query segment
        k: Number of similar segments to find
        distance_matrix: Pre-computed distance matrix (optional)
        
    Returns:
        list: Indices of the k most similar segments
    """
    n_segments = len(segments)
    
    # Compute distance matrix if not provided
    if distance_matrix is None:
        print("Computing distance matrix...")
        distance_matrix = np.zeros((n_segments, n_segments))
        
        for i in tqdm(range(n_segments)):
            for j in range(i+1, n_segments):
                # Use the DTW distance function from eef_clustering.py
                distance = compute_dtw_distance(segments[i], segments[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    # Get distances from query segment to all other segments
    distances = distance_matrix[query_idx].copy()
    
    # Set distance to self to infinity to exclude it
    distances[query_idx] = float('inf')
    
    # Find k segments with smallest distances
    similar_indices = np.argsort(distances)[:k]
    
    return similar_indices.tolist()

def collect_sequential_preferences(data, segments, segment_indices, n_queries=100, k_augment=5, use_ground_truth=True, distance_matrix=None):
    """Collect sequential preferences with similarity-based augmentation.
    
    Args:
        data: TensorDict with observations and rewards
        segments: List of trajectory segments
        segment_indices: List of (start_idx, end_idx) for each segment
        n_queries: Number of preference queries to collect
        k_augment: Number of similar segments to use for augmentation
        use_ground_truth: If True, use ground truth rewards for preferences
        distance_matrix: Pre-computed distance matrix (optional)
        
    Returns:
        tuple: (all_preferences, distance_matrix, similar_segments_info)
            all_preferences: List of collected preferences [(i, j, pref), ...]
            distance_matrix: Computed or provided distance matrix
            similar_segments_info: List of dictionaries with info about similar segments for each preference
    """
    n_segments = len(segments)
    print(f"Collecting {n_queries} sequential preferences with k={k_augment} augmentations each...")
    
    # Check if we have enough segments
    if n_segments < 2:
        raise ValueError(f"Need at least 2 segments, but only found {n_segments}")
    
    # Check if we have reward data for ground truth
    if use_ground_truth and "reward" not in data:
        raise ValueError("Ground truth preferences requested but no reward data found")
    
    # Compute distance matrix for finding similar segments if not provided
    if distance_matrix is None:
        print("Computing distance matrix for similarity search...")
        # Use the compute_dtw_distance_matrix function from trajectory_utils
        distance_matrix, _ = compute_dtw_distance_matrix(segments)
    else:
        print("Using provided distance matrix")
    
    # Collect preferences
    preferences = []
    augmented_preferences = []
    similar_segments_info = []  # Store info about similar segments for visualization
    
    # Keep track of compared pairs to avoid duplicates
    compared_pairs = set()
    
    with tqdm(total=n_queries, desc="Collecting preferences") as pbar:
        while len(preferences) < n_queries:
            # Sample two different random segments
            i, j = random.sample(range(n_segments), 2)
            
            # Skip if this pair has already been compared
            if (i, j) in compared_pairs or (j, i) in compared_pairs:
                continue
            
            # Mark this pair as compared
            compared_pairs.add((i, j))
            
            # Get segment indices
            seg_i = segment_indices[i]
            seg_j = segment_indices[j]
            
            # Determine preference (either from user or ground truth)
            if use_ground_truth:
                pref = get_reward_based_preference(data, seg_i, seg_j)
                
                # Skip if no clear preference
                if pref == 0 or pref is None:
                    continue
            else:
                # For future implementation: collect user preference here
                # For now, just use ground truth
                pref = get_reward_based_preference(data, seg_i, seg_j)
                
                # Skip if no clear preference
                if pref == 0 or pref is None:
                    continue
            
            # Add preference to collected preferences
            preferences.append((i, j, pref))
            pbar.update(1)
            
            # Initialize similar segments info for this preference
            similar_info = {
                'original_preference': (i, j, pref),
                'similar_to_i': [],
                'similar_to_j': [],
                'augmented_preferences': []  # Track which augmentations belong to this preference
            }
            
            # Augment preferences based on similarity
            if k_augment > 0:
                # If segment i is preferred
                if pref == 1:
                    # Find segments similar to segment i
                    similar_to_i = find_similar_segments(segments, i, k=k_augment, distance_matrix=distance_matrix)
                    similar_info['similar_to_i'] = similar_to_i
                    
                    # All segments similar to i are also preferred over j
                    for sim_idx in similar_to_i:
                        aug_pref = (sim_idx, j, 1)
                        augmented_preferences.append(aug_pref)
                        similar_info['augmented_preferences'].append(aug_pref)
                    
                    # Find segments similar to segment j
                    similar_to_j = find_similar_segments(segments, j, k=k_augment, distance_matrix=distance_matrix)
                    similar_info['similar_to_j'] = similar_to_j
                    
                    # Segment i is preferred over all segments similar to j
                    for sim_idx in similar_to_j:
                        aug_pref = (i, sim_idx, 1)
                        augmented_preferences.append(aug_pref)
                        similar_info['augmented_preferences'].append(aug_pref)
                
                # If segment j is preferred
                elif pref == 2:
                    # Find segments similar to segment j
                    similar_to_j = find_similar_segments(segments, j, k=k_augment, distance_matrix=distance_matrix)
                    similar_info['similar_to_j'] = similar_to_j
                    
                    # All segments similar to j are also preferred over i
                    for sim_idx in similar_to_j:
                        aug_pref = (i, sim_idx, 2)
                        augmented_preferences.append(aug_pref)
                        similar_info['augmented_preferences'].append(aug_pref)
                    
                    # Find segments similar to segment i
                    similar_to_i = find_similar_segments(segments, i, k=k_augment, distance_matrix=distance_matrix)
                    similar_info['similar_to_i'] = similar_to_i
                    
                    # Segment j is preferred over all segments similar to i
                    for sim_idx in similar_to_i:
                        aug_pref = (sim_idx, j, 2)
                        augmented_preferences.append(aug_pref)
                        similar_info['augmented_preferences'].append(aug_pref)
            
            # Store similar segments info
            similar_segments_info.append(similar_info)
    
    print(f"Collected {len(preferences)} direct preferences")
    print(f"Generated {len(augmented_preferences)} augmented preferences")
    
    # Combine direct and augmented preferences
    all_preferences = preferences + augmented_preferences
    
    return all_preferences, distance_matrix, similar_segments_info

def verify_augmented_preferences(data, segment_indices, preferences, augmented_preferences):
    """Verify augmented preferences against ground truth.
    
    Args:
        data: TensorDict with observations and rewards
        segment_indices: List of (start_idx, end_idx) for each segment
        preferences: List of direct preferences [(i, j, pref), ...]
        augmented_preferences: List of augmented preferences [(i, j, pref), ...]
        
    Returns:
        dict: Statistics about augmented preference accuracy
    """
    if "reward" not in data:
        print("WARNING: No reward data found, cannot verify augmented preferences")
        return None
    
    print("Verifying augmented preferences against ground truth...")
    
    # Statistics
    stats = {
        'total': len(augmented_preferences),
        'correct': 0,
        'incorrect': 0,
        'equal': 0,
        'accuracy': 0.0
    }
    
    # Check each augmented preference
    for i, j, pref in tqdm(augmented_preferences, desc="Verifying"):
        # Get segment indices
        seg_i = segment_indices[i]
        seg_j = segment_indices[j]
        
        # Get ground truth preference
        gt_pref = get_reward_based_preference(data, seg_i, seg_j)
        
        if gt_pref is None:
            continue
        
        if gt_pref == 0:
            stats['equal'] += 1
        elif gt_pref == pref:
            stats['correct'] += 1
        else:
            stats['incorrect'] += 1
    
    # Calculate accuracy
    comparable = stats['correct'] + stats['incorrect']
    if comparable > 0:
        stats['accuracy'] = stats['correct'] / comparable
    
    print(f"Augmentation accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{comparable} comparable pairs)")
    print(f"Equal preferences: {stats['equal']}")
    
    return stats

def create_preference_dataset(data, segment_indices, preferences, output_file):
    """Create a dataset for training a reward model from preferences.
    
    Args:
        data: TensorDict with observations
        segment_indices: List of (start_idx, end_idx) for each segment
        preferences: List of (i, j, pref) preference tuples
        output_file: Path to save the dataset
        
    Returns:
        dict: Preference dataset with segments and labels
    """
    print(f"Creating preference dataset at {output_file}")
    
    # Extract segment pairs and preferences
    segment_pairs = []
    preference_labels = []
    
    for i, j, pref in preferences:
        segment_pairs.append((i, j))
        preference_labels.append(pref)
    
    # Convert lists to tensors for better compatibility
    segment_indices_tensor = torch.tensor(segment_indices)
    segment_pairs_tensor = torch.tensor(segment_pairs)
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
    
    # Create metadata
    metadata = {
        'source_file': data.get('_source_path', 'unknown'),
        'n_segments': len(segment_indices),
        'n_pairs': len(preference_labels),
        'n_direct_pairs': len(preferences) - len(preference_labels),
        'n_augmented_pairs': len(preference_labels) - len(preferences),
        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'included_fields': list(compact_data.keys())
    }
    
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

def create_grid_video(images, segments_info, output_file, title=None, grid_size=None):
    """Create a grid video showing multiple segments at once.
    
    Args:
        images: Tensor of all observation images
        segments_info: List of dicts with keys 'indices', 'title', 'is_preferred', 'border_color', 'verdict'
        output_file: Path to save the output video
        title: Optional title for the video
        grid_size: Optional tuple (rows, cols) specifying the grid layout
        
    Returns:
        str: Path to the saved video
    """
    # Calculate grid dimensions
    n_segments = len(segments_info)
    
    if grid_size is None:
        # Auto-calculate grid size
        grid_size = int(np.ceil(np.sqrt(n_segments)))
        rows = grid_size
        cols = grid_size
    else:
        # Use provided grid size
        rows, cols = grid_size
    
    # Extract image sequences for all segments
    sequences = []
    max_length = 0
    
    for segment in segments_info:
        start, end = segment['indices']
        seq = images[start:end+1]
        sequences.append(seq)
        max_length = max(max_length, len(seq))
    
    # Create figure for video
    fig = plt.figure(figsize=(cols*4, rows*4))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    # Define colors for preferred and non-preferred segments
    preferred_color = 'green'
    non_preferred_color = 'red'
    
    # Set up subplots for each segment
    axes = []
    img_objects = []
    
    for i, (segment, seq) in enumerate(zip(segments_info, sequences)):
        row = i // cols
        col = i % cols
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        # Set title with verdict if available
        title_text = segment['title']
        if 'verdict' in segment and segment['verdict']:
            title_text = f"{title_text} {segment['verdict']}"
        ax.set_title(title_text)
        
        # Set up initial image
        img = ax.imshow(seq[0])
        img_objects.append(img)
        
        # Turn off axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set border color based on preference
        border_color = segment['border_color']
        if border_color is None:
            border_color = preferred_color if segment['is_preferred'] else non_preferred_color
            
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # Add a global title
    if title:
        suptitle = fig.suptitle(f"{title} - Frame: 0", fontsize=16)
    else:
        suptitle = fig.suptitle(f"Frame: 0", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Function to update the figure for animation
    def update(frame):
        updated_artists = [suptitle]
        
        # Update frame counter in title
        if title:
            suptitle.set_text(f"{title} - Frame: {frame}")
        else:
            suptitle.set_text(f"Frame: {frame}")
        
        # Update each image
        for i, (img, seq) in enumerate(zip(img_objects, sequences)):
            if frame < len(seq):
                img.set_array(seq[frame])
            updated_artists.append(img)
        
        return updated_artists
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=max_length, interval=100, blit=True)
    
    # Save as video file
    try:
        writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Robot Pref'))
        anim.save(output_file, writer=writer)
        print(f"Saved grid video to {output_file}")
    except Exception as e:
        print(f"Error saving video: {e}")
        # As a fallback, save a few key frames as images
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        base_path = os.path.splitext(output_file)[0]
        for frame in [0, max_length//4, max_length//2, 3*max_length//4, max_length-1]:
            if frame < max_length:
                update(frame)
                plt.savefig(f"{base_path}_frame{frame}.png")
                print(f"Saved frame {frame} to {base_path}_frame{frame}.png")
    
    plt.close(fig)
    return output_file

def create_augmented_grid_video(images, original_pair, augmented_pairs, segment_indices, output_file, data=None, distance_matrix=None, title=None, max_augmentations=10):
    """Create a grid video showing original preference pair and multiple augmented trajectories.
    
    Args:
        images: Tensor of all observation images
        original_pair: Tuple (i, j, pref) for the original preference
        augmented_pairs: List of (i, j, pref) tuples for augmented preferences
        segment_indices: List of (start_idx, end_idx) for each segment
        output_file: Path to save the output video
        data: TensorDict with observations and rewards
        distance_matrix: Optional distance matrix for showing distances in titles
        title: Optional title for the video
        max_augmentations: Maximum number of augmentations to show
        
    Returns:
        str: Path to the saved video
    """
    # Extract original pair info
    i, j, pref = original_pair
    
    # Limit the number of augmented pairs to show
    augmented_pairs = augmented_pairs[:min(max_augmentations, len(augmented_pairs))]
    
    # Calculate total number of rows (original pair + augmented pairs)
    n_rows = 1 + len(augmented_pairs)
    
    print(f"Creating grid video with {n_rows} rows (1 original + {len(augmented_pairs)} augmentations)")
    
    # Create output directory for videos
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Collect all segment indices and metadata
    all_segments = []
    
    # Get ground truth preference based on reward
    gt_pref = None
    if "reward" in data:
        gt_pref = get_reward_based_preference(data, segment_indices[i], segment_indices[j])
    
    # Create GT preference label
    gt_label = ""
    if gt_pref is not None:
        if gt_pref == 1:
            gt_label = "GT preference: Left > Right"
        elif gt_pref == 2:
            gt_label = "GT preference: Right > Left"
        else:
            gt_label = "GT preference: Equal"
    
    # Original preference pair (first row)
    all_segments.append({
        'indices': segment_indices[i],
        'title': f"Original Seg {i}\n{gt_label}",
        'is_preferred': pref == 1,
        'border_color': 'green' if pref == 1 else 'red',
        'verdict': ""
    })
    
    all_segments.append({
        'indices': segment_indices[j],
        'title': f"Original Seg {j}",
        'is_preferred': pref == 2,
        'border_color': 'green' if pref == 2 else 'red',
        'verdict': ""
    })
    
    # Add augmented pairs (each pair gets its own row)
    for aug_idx, (aug_i, aug_j, aug_pref) in enumerate(augmented_pairs):
        # Get augmented preference label
        aug_label = ""
        if aug_pref == 1:
            aug_label = "Augmented preference: Left > Right"
        elif aug_pref == 2:
            aug_label = "Augmented preference: Right > Left"
        
        # Get ground truth preference for this augmented pair
        aug_gt_pref = None
        aug_gt_label = ""
        if "reward" in data:
            aug_gt_pref = get_reward_based_preference(data, segment_indices[aug_i], segment_indices[aug_j])
            if aug_gt_pref == 1:
                aug_gt_label = "GT preference: Left > Right"
            elif aug_gt_pref == 2:
                aug_gt_label = "GT preference: Right > Left"
            elif aug_gt_pref == 0:
                aug_gt_label = "GT preference: Equal"
        
        # Combine labels
        combined_label = aug_label
        if aug_gt_label:
            combined_label += f"\n{aug_gt_label}"
            
        # Check if augmented preference matches ground truth
        matches_gt = (aug_pref == aug_gt_pref) if aug_gt_pref not in [None, 0] else None
        verdict = ""
        if matches_gt is not None:
            verdict = "✓" if matches_gt else "✗"
        
        # Check which segment is new/augmented
        is_i_augmented = (aug_i != i and aug_i != j)
        is_j_augmented = (aug_j != i and aug_j != j)
        
        # Get distances if distance matrix is available
        dist_i_str = ""
        dist_j_str = ""
        if distance_matrix is not None:
            # Only show distance for the augmented segment
            if is_i_augmented:
                # Get distance from original segment to augmented segment
                if aug_i != i:  # If this is an augmentation of i
                    dist_i = distance_matrix[i, aug_i]
                    dist_i_str = f" (d={dist_i:.2f})"
            if is_j_augmented:
                # Get distance from original segment to augmented segment
                if aug_j != j:  # If this is an augmentation of j
                    dist_j = distance_matrix[j, aug_j]
                    dist_j_str = f" (d={dist_j:.2f})"
        
        # Add first segment of augmented pair
        all_segments.append({
            'indices': segment_indices[aug_i],
            'title': f"Aug {aug_idx+1}: Seg {aug_i}{dist_i_str}\n{combined_label} {verdict}",
            'is_preferred': aug_pref == 1,
            'border_color': 'green' if aug_pref == 1 else 'red',
            'verdict': ""
        })
        
        # Add second segment of augmented pair (without aug_label)
        all_segments.append({
            'indices': segment_indices[aug_j],
            'title': f"Aug {aug_idx+1}: Seg {aug_j}{dist_j_str}",
            'is_preferred': aug_pref == 2,
            'border_color': 'green' if aug_pref == 2 else 'red',
            'verdict': ""
        })
    
    # Create the grid video
    video_path = create_grid_video(
        images,
        all_segments,
        output_file=output_file,
        title=title,
        grid_size=(n_rows, 2)  # Each row has 2 columns
    )
    
    print(f"Created grid video: {video_path}")
    return video_path

def visualize_all_augmentations(data, segments, segment_indices, direct_preferences, augmented_preferences, output_dir, similar_segments_info=None, distance_matrix=None, max_visualizations=3, max_augmentations=10):
    """Create visualizations showing original preference pairs and their augmentations.
    
    Args:
        data: TensorDict with observations
        segments: List of trajectory segments
        segment_indices: List of (start_idx, end_idx) for each segment
        direct_preferences: List of (i, j, pref) tuples for direct preferences
        augmented_preferences: List of (i, j, pref) tuples for augmented preferences
        output_dir: Directory to save the visualizations
        similar_segments_info: List of dictionaries with info about similar segments for each preference
        distance_matrix: Optional distance matrix for showing distances in titles
        max_visualizations: Maximum number of preference pairs to visualize
        max_augmentations: Maximum number of augmentations to show per preference pair
        
    Returns:
        list: Paths to the created visualizations
    """
    # Check if we have image data
    if "image" not in data:
        print("WARNING: No image data found in dataset. Cannot create grid visualizations.")
        return []
    
    # Get observation images
    images = data["image"].cpu()
    
    # Create output directory for videos
    vis_dir = os.path.join(output_dir, "augmentation_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select preference pairs to visualize
    if similar_segments_info:
        print(f"\nUsing similar_segments_info to group augmented preferences")
        # Use the similar_segments_info to get the grouping
        if len(similar_segments_info) > max_visualizations:
            # Randomly select some preference pairs
            selected_indices = random.sample(range(len(similar_segments_info)), max_visualizations)
        else:
            selected_indices = list(range(len(similar_segments_info)))
    else:
        print("\nWARNING: No similar_segments_info provided, falling back to heuristic grouping")
        # Group augmented preferences by original preference pair
        augmentation_groups = {}
        
        # First, identify which segments are used in direct preferences
        segments_in_direct = set()
        for i, j, _ in direct_preferences:
            segments_in_direct.add(i)
            segments_in_direct.add(j)
        
        print(f"DEBUG: Found {len(segments_in_direct)} segments in direct preferences")
        print(f"DEBUG: Total augmented preferences: {len(augmented_preferences)}")
        
        # Group augmented preferences by which direct preference they came from
        for aug_idx, (aug_i, aug_j, aug_pref) in enumerate(augmented_preferences):
            found_match = False
            # Try to find which direct preference this augmentation belongs to
            for dir_idx, (dir_i, dir_j, _) in enumerate(direct_preferences):
                # Check if this augmentation involves one of the segments from the direct preference
                if aug_i == dir_i or aug_j == dir_j:
                    if dir_idx not in augmentation_groups:
                        augmentation_groups[dir_idx] = []
                    augmentation_groups[dir_idx].append((aug_i, aug_j, aug_pref))
                    found_match = True
                    break
            
            if not found_match:
                print(f"DEBUG: Augmentation {aug_idx} ({aug_i}, {aug_j}) not matched to any direct preference")
        
        print(f"DEBUG: Created {len(augmentation_groups)} augmentation groups")
        for group_idx, augs in augmentation_groups.items():
            print(f"DEBUG: Group {group_idx} has {len(augs)} augmentations")
        
        # Choose random preference pairs to visualize
        if len(augmentation_groups) > max_visualizations:
            selected_indices = random.sample(list(augmentation_groups.keys()), max_visualizations)
        else:
            selected_indices = list(augmentation_groups.keys())
    
    print(f"Creating visualizations for {len(selected_indices)} randomly selected preference pairs")
    
    # Create visualizations
    visualization_paths = []
    
    for idx in selected_indices:
        if similar_segments_info:
            # Get info from similar_segments_info
            similar_info = similar_segments_info[idx]
            original_pair = similar_info['original_preference']
            augmented_pairs = similar_info.get('augmented_preferences', [])
        else:
            # Get original preference pair
            original_pair = direct_preferences[idx]
            
            # Get augmented preferences for this pair
            augmented_pairs = augmentation_groups.get(idx, [])
        
        if not augmented_pairs:
            print(f"DEBUG: No augmented pairs found for preference {idx}, skipping")
            continue
            
        # Create output file path
        output_file = os.path.join(vis_dir, f"augmentations_for_pref_{idx}.mp4")
        
        # Create title
        i, j, pref = original_pair
        title = f"Preference {idx+1}: {'Seg ' + str(i) + ' > ' + str(j) if pref == 1 else 'Seg ' + str(j) + ' > ' + str(i)}"
        
        print(f"DEBUG: Creating visualization for preference {idx} with {len(augmented_pairs)} augmentations")
        
        # Create grid visualization
        video_path = create_augmented_grid_video(
            images,
            original_pair,
            augmented_pairs,
            segment_indices,
            output_file,
            data=data,
            distance_matrix=distance_matrix,
            title=title,
            max_augmentations=max_augmentations
        )
        
        visualization_paths.append(video_path)
    
    return visualization_paths

@hydra.main(config_path="config", config_name="sequential_preferences", version_base=None)
def main(cfg: DictConfig):
    """Main function for collecting sequential preferences with similarity-based augmentation."""
    print("\n" + "="*50)
    print("Collecting sequential preferences with similarity-based augmentation")
    print("="*50)
    
    # Print config for visibility
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    random_seed = cfg.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print(f"Using random seed: {random_seed}")
    
    # Extract configuration parameters
    data_path = cfg.data.data_path
    preprocessed_data = cfg.data.preprocessed_data
    segment_length = cfg.data.segment_length
    max_segments = cfg.data.max_segments
    use_relative_differences = cfg.data.use_relative_differences
    
    n_queries = cfg.preferences.n_queries
    k_augment = cfg.preferences.k_augment
    use_ground_truth = cfg.preferences.use_ground_truth
    use_dtw_distance = cfg.preferences.use_dtw_distance
    max_dtw_segments = cfg.preferences.max_dtw_segments
    
    # Extract visualization parameters
    visualize = cfg.visualize
    visualize_augmented = cfg.visualize_augmented
    max_visualizations = cfg.max_visualizations
    max_augmentations = cfg.max_augmentations
    
    # Output parameters
    base_output_dir = cfg.output.output_dir
    
    # Create a more specific output directory with parameters
    dir_name = f"n{n_queries}_k{k_augment}_seed{random_seed}"
    if max_dtw_segments is not None:
        dir_name += f"_dtw{max_dtw_segments}"
    output_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize wandb if enabled
    if cfg.wandb.use_wandb:
        import wandb
        wandb.init(
            project="robot-preferences",
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Load data
    if preprocessed_data:
        print(f"Loading preprocessed data from {preprocessed_data}")
        preproc_data = load_preprocessed_segments(preprocessed_data)
        
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
        if 'obs' not in data and 'state' not in data:
            print(f"WARNING: Preprocessed data does not contain observation data (obs or state).")
            print(f"Loading original data from {data_path} for observations.")
            orig_data = load_tensordict(data_path)
            
            # Copy missing essential fields
            for field in missing_fields:
                if field in orig_data:
                    data[field] = orig_data[field]
                    print(f"Loaded field '{field}' from original data")
        
        # Add source path for reference
        data['_source_path'] = preprocessed_data
        
        # Extract segments if not already in preprocessed data
        if 'segments' in preproc_data and 'segment_indices' in preproc_data:
            print("Using segments from preprocessed data")
            segments = preproc_data['segments']
            segment_indices = preproc_data['segment_indices']
            original_segments = preproc_data.get('original_segments', segments)
            
            # Check if distance matrix is already computed
            distance_matrix = preproc_data.get('distance_matrix', None)
            if distance_matrix is not None:
                print("Using precomputed distance matrix from preprocessed data")
        else:
            print("Extracting segments from data...")
            segments, segment_indices, original_segments = extract_segments(
                data, 
                segment_length=segment_length,
                max_segments=max_segments,
                use_relative_differences=use_relative_differences
            )
            distance_matrix = None
    else:
        print(f"Loading data from {data_path}")
        data = load_tensordict(data_path)
        data['_source_path'] = data_path
        
        # Extract segments
        segments, segment_indices, original_segments = extract_segments(
            data, 
            segment_length=segment_length,
            max_segments=max_segments,
            use_relative_differences=use_relative_differences
        )
        distance_matrix = None
    
    # Check if we should use DTW distance matrix from clustering
    use_dtw_distance = cfg.preferences.get('use_dtw_distance', True)
    max_dtw_segments = cfg.preferences.get('max_dtw_segments', None)
    
    # Compute DTW distance matrix if needed and not already available
    if use_dtw_distance and distance_matrix is None:
        print("\nComputing DTW distance matrix...")
        distance_matrix, idx_mapping = compute_dtw_distance_matrix(
            segments, 
            max_segments=max_dtw_segments,
            random_seed=random_seed
        )
        
        # If using a subset of segments, adjust segments and indices
        if max_dtw_segments is not None and max_dtw_segments < len(segments):
            print(f"Using subset of {max_dtw_segments} segments for DTW calculation")
            segments = [segments[idx_mapping[i]] for i in range(len(idx_mapping))]
            segment_indices = [segment_indices[idx_mapping[i]] for i in range(len(idx_mapping))]
    
    # Collect sequential preferences with augmentation
    all_preferences, distance_matrix, similar_segments_info = collect_sequential_preferences(
        data, 
        segments, 
        segment_indices, 
        n_queries=n_queries, 
        k_augment=k_augment,
        use_ground_truth=use_ground_truth,
        distance_matrix=distance_matrix
    )
    
    # Separate direct and augmented preferences
    direct_preferences = all_preferences[:n_queries]
    augmented_preferences = all_preferences[n_queries:]
    
    # Verify augmented preferences against ground truth
    if use_ground_truth and "reward" in data:
        verification_stats = verify_augmented_preferences(
            data, 
            segment_indices, 
            direct_preferences, 
            augmented_preferences
        )
    else:
        verification_stats = None
    
    # Create visualizations for a few selected preference pairs
    visualization_paths = []
    if visualize and visualize_augmented and len(augmented_preferences) > 0:
        print("\nCreating augmentation grid visualizations...")
        # Create grid visualizations showing original preferences and their augmentations
        visualization_paths = visualize_all_augmentations(
            data, 
            segments, 
            segment_indices, 
            direct_preferences, 
            augmented_preferences, 
            output_dir,
            similar_segments_info=similar_segments_info,
            distance_matrix=distance_matrix,
            max_visualizations=max_visualizations,
            max_augmentations=max_augmentations
        )
        
        if visualization_paths:
            print(f"Created {len(visualization_paths)} augmentation grid videos")
    
    # Save raw preferences
    preferences_file = os.path.join(output_dir, "raw_preferences.pkl")
    
    raw_data = {
        'direct_preferences': direct_preferences,
        'augmented_preferences': augmented_preferences,
        'verification_stats': verification_stats,
        'distance_matrix': distance_matrix,
        'segments': segments,
        'segment_indices': segment_indices,
        'similar_segments_info': similar_segments_info,
        'visualization_paths': visualization_paths
    }
    
    torch.save(raw_data, preferences_file)
    print(f"Saved raw preferences to {preferences_file}")
    
    # Create preference dataset
    dataset_file = os.path.join(output_dir, "preference_dataset.pkl")
    preference_dataset = create_preference_dataset(
        data, 
        segment_indices, 
        all_preferences, 
        dataset_file
    )
    
    print("\nProcess complete!")
    print(f"Collected {len(direct_preferences)} direct preferences")
    print(f"Generated {len(augmented_preferences)} augmented preferences")
    
    # Print verification stats if available
    if verification_stats:
        print("\nAugmentation accuracy summary:")
        print(f"  Total augmented preferences: {verification_stats['total']}")
        print(f"  Correct: {verification_stats['correct']}")
        print(f"  Incorrect: {verification_stats['incorrect']}")
        print(f"  Equal: {verification_stats['equal']}")
        print(f"  Accuracy: {verification_stats['accuracy']:.2%}")
    
    # Print visualization summary if created
    if visualization_paths:
        print("\nCreated visualizations:")
        for path in visualization_paths:
            print(f"  {path}")
    
    return preference_dataset

if __name__ == "__main__":
    main() 