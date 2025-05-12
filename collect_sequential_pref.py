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
import time
import hydra
from omegaconf import DictConfig, OmegaConf

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
        tuple: (all_preferences, distance_matrix)
            all_preferences: List of collected preferences [(i, j, pref), ...]
            distance_matrix: Computed or provided distance matrix
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
            
            # Augment preferences based on similarity
            if k_augment > 0:
                # If segment i is preferred
                if pref == 1:
                    # Find segments similar to segment i
                    similar_to_i = find_similar_segments(segments, i, k=k_augment, distance_matrix=distance_matrix)
                    
                    # All segments similar to i are also preferred over j
                    for sim_idx in similar_to_i:
                        augmented_preferences.append((sim_idx, j, 1))
                    
                    # Find segments similar to segment j
                    similar_to_j = find_similar_segments(segments, j, k=k_augment, distance_matrix=distance_matrix)
                    
                    # Segment i is preferred over all segments similar to j
                    for sim_idx in similar_to_j:
                        augmented_preferences.append((i, sim_idx, 1))
                
                # If segment j is preferred
                elif pref == 2:
                    # Find segments similar to segment j
                    similar_to_j = find_similar_segments(segments, j, k=k_augment, distance_matrix=distance_matrix)
                    
                    # All segments similar to j are also preferred over i
                    for sim_idx in similar_to_j:
                        augmented_preferences.append((i, sim_idx, 2))
                    
                    # Find segments similar to segment i
                    similar_to_i = find_similar_segments(segments, i, k=k_augment, distance_matrix=distance_matrix)
                    
                    # Segment j is preferred over all segments similar to i
                    for sim_idx in similar_to_i:
                        augmented_preferences.append((sim_idx, j, 2))
    
    print(f"Collected {len(preferences)} direct preferences")
    print(f"Generated {len(augmented_preferences)} augmented preferences")
    
    # Combine direct and augmented preferences
    all_preferences = preferences + augmented_preferences
    
    return all_preferences, distance_matrix

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

@hydra.main(config_path="config", config_name="sequential_preferences", version_base=None)
def main(cfg: DictConfig):
    """Run sequential preference collection with Hydra configuration."""
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
    
    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    
    # Extract parameters from config
    data_path = cfg.data.data_path
    preprocessed_data = cfg.data.preprocessed_data
    segment_length = cfg.data.segment_length
    max_segments = cfg.data.max_segments
    use_relative_differences = cfg.data.use_relative_differences
    
    n_queries = cfg.preferences.n_queries
    k_augment = cfg.preferences.k_augment
    use_ground_truth = cfg.preferences.use_ground_truth
    
    output_dir = cfg.output.output_dir
    
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
    all_preferences, distance_matrix = collect_sequential_preferences(
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
    
    # Save raw preferences
    preferences_file = os.path.join(output_dir, "raw_preferences.pkl")
    
    raw_data = {
        'direct_preferences': direct_preferences,
        'augmented_preferences': augmented_preferences,
        'verification_stats': verification_stats,
        'distance_matrix': distance_matrix,
        'segments': segments,
        'segment_indices': segment_indices
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
    
    return preference_dataset

if __name__ == "__main__":
    main() 