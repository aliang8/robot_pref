import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
from tqdm import tqdm

from train_reward_model_old import load_preferences_from_directory


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def get_indices(traj_total, config):
    traj_idx = np.random.choice(traj_total, replace=False)
    idx_st = 500 * traj_idx + np.random.randint(0, 500 - config.segment_size)
    idx = [[j for j in range(idx_st, idx_st + config.segment_size)]]
    return idx


def consist_test_dataset(
    dataset, test_feedback_num, traj_total, segment_size, threshold
):
    test_traj_idx = np.random.choice(traj_total, 2 * test_feedback_num, replace=True)
    test_idx = [
        500 * i + np.random.randint(0, 500 - segment_size) for i in test_traj_idx
    ]
    test_idx_st_1 = test_idx[:test_feedback_num]
    test_idx_st_2 = test_idx[test_feedback_num:]
    test_idx_1 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_1]
    test_idx_2 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_2]
    test_labels = obtain_labels(
        dataset,
        test_idx_1,
        test_idx_2,
        segment_size=segment_size,
        threshold=threshold,
        noise=0.0,
    )
    test_binary_labels = obtain_labels(
        dataset,
        test_idx_1,
        test_idx_2,
        segment_size=segment_size,
        threshold=0,
        noise=0.0,
    )
    test_obs_act_1 = np.concatenate(
        (dataset["observations"][test_idx_1], dataset["actions"][test_idx_1]),
        axis=-1,
    )
    test_obs_act_2 = np.concatenate(
        (dataset["observations"][test_idx_2], dataset["actions"][test_idx_2]),
        axis=-1,
    )
    return test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels

def get_human_feedbacks(config, dataset):
    data_path = Path(config.data_path)
    seg_indices_path = data_path.parent / "segment_start_end_indices.npy"
    seg_pairs_path = data_path.parent / "segment_pairs.npy"
    prefs_path = data_path.parent / "preferences"

    # Load everything
    seg_indices = np.load(seg_indices_path, allow_pickle=True)
    seg_pairs = np.load(seg_pairs_path, allow_pickle=True)
    prefs, _ = load_preferences_from_directory(prefs_path)

    # Initialize lists to store processed data
    labels = []
    idx_st_1 = []
    idx_st_2 = []

    for pref in prefs:
        pair_ind = pref["pair_index"]
        preference = pref["preference"]  # Should be 'A', 'B', or 'equal'

        seg1, seg2 = seg_pairs[pair_ind]
        
        # Get the actual segment data using indices
        seg1_start, seg1_end = seg_indices[seg1]
        seg2_start, seg2_end = seg_indices[seg2]
        
        # Convert preference to numeric value
        # 'A' means first segment preferred, 'B' means second segment preferred, 'equal' means equal preference
        if preference == 'equal':
            binary_pref = 0.5
        else:
            binary_pref = 0 if preference == 'B' else 1
        
        # Store data in the format used by learn_reward.py
        labels.append(binary_pref)
        idx_st_1.append(seg1_start)
        idx_st_2.append(seg2_start)

    # Convert to numpy arrays
    labels = np.array(labels)
    idx_st_1 = np.array(idx_st_1)
    idx_st_2 = np.array(idx_st_2)

    return labels, idx_st_1, idx_st_2


def collect_simple_pairwise_feedback(dataset, traj_total, config):
    """
    Simplified version of collect_feedback for independent pairwise feedback only.
    No RLT (Ranking Learning with Ties) complexity - just simple pairwise comparisons.
    
    Args:
        dataset: The dataset containing observations, actions, rewards
        traj_total: Total number of trajectories 
        config: Configuration object with feedback_num, segment_size, threshold, noise
        
    Returns:
        tuple: (multiple_ranked_list, segment_indices)
            multiple_ranked_list: List of ranking lists, each containing the result of one pairwise comparison
            segment_indices: List of all segment starting indices used in feedback collection
    """
    # Set random seed for reproducible segment selection
    if hasattr(config, 'seed'):
        np.random.seed(config.seed)
        random.seed(config.seed)
        print(f"Set random seed to {config.seed} for reproducible segment selection")
    
    # Check if segment indices are cached
    segment_indices_path = getattr(config, 'segment_indices_path', 'segment_indices_cache.pkl')
    
    # Try to load existing segment indices
    segment_indices = load_segment_indices(segment_indices_path, config)
    
    if segment_indices is not None:
        print(f"Using cached segment indices from {segment_indices_path}")
        # Generate multiple_ranked_list from cached indices
        multiple_ranked_list = generate_feedback_from_indices(dataset, segment_indices, config)
        return multiple_ranked_list, segment_indices
    
    # Generate new segment indices and feedback
    multiple_ranked_list = []
    segment_indices = []  # Track all segment indices used
    used_pairs = set()  # Track used pairs to avoid duplicates
    print(f"Collecting {config.feedback_num} independent pairwise feedback samples")

    for i in range(config.feedback_num):
        max_attempts = 50  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            # Get two random trajectory segments
            idx_1 = get_indices(traj_total, config)
            idx_2 = get_indices(traj_total, config) 
            
            # Extract starting indices
            idx_st_1 = idx_1[0][0]
            idx_st_2 = idx_2[0][0]
            
            # Create a pair key (order-independent to catch (A,B) and (B,A) as same)
            pair_key = tuple(sorted([idx_st_1, idx_st_2]))
            
            # Check if this is a valid new pair
            if (idx_st_1 != idx_st_2 and  # Don't compare segment with itself
                pair_key not in used_pairs):  # Don't use duplicate pairs
                # Valid pair found
                used_pairs.add(pair_key)
                break
                
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not find unique pair after {max_attempts} attempts for sample {i+1}")
            # Use the last generated pair anyway to avoid infinite loop
        
        # Calculate segment rewards
        reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
        reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
        
        # Track segment indices
        segment_indices.extend([idx_st_1, idx_st_2])
        
        # Get preference label for this pair
        label = obtain_labels(
            dataset,
            idx_1,
            idx_2,
            segment_size=config.segment_size,
            threshold=config.threshold,
            noise=config.noise,
        )
        
        # Create a single ranking list for this comparison
        single_ranked_list = []
        
        if np.all(label[0] == [1, 0]):
            # Segment 1 is preferred over segment 2
            group_1 = [(idx_st_1, reward_1)]  # Higher ranked (preferred)
            group_2 = [(idx_st_2, reward_2)]  # Lower ranked  
            single_ranked_list.append(group_2)  # Lower rank first
            single_ranked_list.append(group_1)  # Higher rank second
            
        elif np.all(label[0] == [0.5, 0.5]):
            # Segments are considered equal
            group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
            single_ranked_list.append(group)
            
        elif np.all(label[0] == [0, 1]):
            # Segment 2 is preferred over segment 1
            group_1 = [(idx_st_1, reward_1)]  # Lower ranked
            group_2 = [(idx_st_2, reward_2)]  # Higher ranked (preferred)
            single_ranked_list.append(group_1)  # Lower rank first
            single_ranked_list.append(group_2)  # Higher rank second
        
        # Add this pairwise comparison to the list
        multiple_ranked_list.append(single_ranked_list)
        
        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1}/{config.feedback_num} pairwise comparisons")
    
    # Remove duplicates from segment indices
    segment_indices = list(set(segment_indices))

    # Save segment indices for future use
    save_segment_indices(segment_indices, segment_indices_path, config)
    
    print(f"Completed collection of {len(multiple_ranked_list)} pairwise feedback samples")
    print(f"Used {len(segment_indices)} unique segment indices")
    print(f"Generated {len(used_pairs)} unique segment pairs")
    return multiple_ranked_list, segment_indices


def load_segment_indices(segment_indices_path: str, config) -> Optional[List[int]]:
    """
    Load segment indices from cache file with validation.
    
    Args:
        segment_indices_path: Path to segment indices cache file
        config: Configuration object
        
    Returns:
        segment_indices: List of segment indices if cache is valid, None otherwise
    """
    if not os.path.exists(segment_indices_path):
        return None
        
    try:
        with open(segment_indices_path, 'rb') as f:
            cached_data = pickle.load(f)
            
        # Validate cache metadata
        cache_metadata = cached_data['metadata']
        current_metadata = {
            'feedback_num': config.feedback_num,
            'segment_size': config.segment_size,
            'seed': getattr(config, 'seed', None)
        }
        
        # Check if cache matches current config
        if (cache_metadata['feedback_num'] == current_metadata['feedback_num'] and
            cache_metadata['segment_size'] == current_metadata['segment_size'] and
            cache_metadata['seed'] == current_metadata['seed']):
            
            print("Segment indices cache validation successful")
            return cached_data['segment_indices']
        else:
            print("Segment indices cache validation failed - will regenerate")
            return None
            
    except Exception as e:
        print(f"Error loading segment indices cache: {e} - will regenerate")
        return None


def save_segment_indices(segment_indices: List[int], segment_indices_path: str, config):
    """
    Save segment indices to cache file with metadata.
    
    Args:
        segment_indices: List of segment indices to save
        segment_indices_path: Path to save segment indices cache
        config: Configuration object
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(segment_indices_path), exist_ok=True)
        
        cache_data = {
            'segment_indices': segment_indices,
            'metadata': {
                'feedback_num': config.feedback_num,
                'segment_size': config.segment_size,
                'seed': getattr(config, 'seed', None),
                'n_segments': len(segment_indices)
            }
        }
        
        with open(segment_indices_path, 'wb') as f:
            pickle.dump(cache_data, f)
            
        print(f"Saved {len(segment_indices)} segment indices to {segment_indices_path}")
        
    except Exception as e:
        print(f"Warning: Could not save segment indices cache: {e}")


def generate_feedback_from_indices(dataset, segment_indices: List[int], config):
    """
    Generate feedback comparisons from cached segment indices.
    
    Args:
        dataset: The dataset containing observations, actions, rewards
        segment_indices: List of segment starting indices
        config: Configuration object
        
    Returns:
        multiple_ranked_list: List of ranking lists for pairwise comparisons
    """
    multiple_ranked_list = []
    
    # Recreate pairwise comparisons from segment indices
    # Note: This assumes segment_indices contains pairs in sequence
    for i in range(0, len(segment_indices), 2):
        if i + 1 >= len(segment_indices):
            break
            
        idx_st_1 = segment_indices[i]
        idx_st_2 = segment_indices[i + 1]
        
        # Create segment index lists
        idx_1 = [[j for j in range(idx_st_1, idx_st_1 + config.segment_size)]]
        idx_2 = [[j for j in range(idx_st_2, idx_st_2 + config.segment_size)]]
        
        # Calculate segment rewards
        reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
        reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
        
        # Get preference label
        label = obtain_labels(
            dataset,
            idx_1,
            idx_2,
            segment_size=config.segment_size,
            threshold=config.threshold,
            noise=config.noise,
        )
        
        # Create ranking list for this comparison
        single_ranked_list = []
        
        if np.all(label[0] == [1, 0]):
            group_1 = [(idx_st_1, reward_1)]
            group_2 = [(idx_st_2, reward_2)]
            single_ranked_list.append(group_2)
            single_ranked_list.append(group_1)
        elif np.all(label[0] == [0.5, 0.5]):
            group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
            single_ranked_list.append(group)
        elif np.all(label[0] == [0, 1]):
            group_1 = [(idx_st_1, reward_1)]
            group_2 = [(idx_st_2, reward_2)]
            single_ranked_list.append(group_1)
            single_ranked_list.append(group_2)
        
        multiple_ranked_list.append(single_ranked_list)
    
    return multiple_ranked_list


def load_precomputed_dtw_matrix(dtw_matrix_path: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load a pre-computed DTW matrix and its corresponding segment indices.
    
    Args:
        dtw_matrix_path: Path to the pre-computed DTW matrix file
        
    Returns:
        tuple: (dtw_matrix, segment_indices)
            dtw_matrix: Matrix of DTW distances between segments
            segment_indices: List of segment start indices corresponding to matrix rows/columns
    """
    print(f"Loading pre-computed DTW matrix from: {dtw_matrix_path}")
    with open(dtw_matrix_path, 'rb') as f:
        dtw_matrix, segment_indices = pickle.load(f)
    print(f"Loaded DTW matrix of shape {dtw_matrix.shape} with {len(segment_indices)} segments")
    return dtw_matrix, segment_indices

def collect_dtw_augmentations(
    dataset, 
    traj_total, 
    config,
    original_pairs,
    original_preferences,
    use_relative_eef=True,
    use_goal_pos=True,
    augmentation_dataset=None
):
    """Collect DTW augmentations using original dataset for DTW matrix but augmentation dataset for segments."""
    print("\nCollecting DTW augmentations...")
    
    # Get segment indices for DTW matrix computation (from original dataset)
    segment_indices = load_segment_indices(config.segment_indices_path, config)
    # if segment_indices is None:
    #     print("Computing new segment indices...")
    #     segment_indices = get_indices(traj_total, config)
    #     save_segment_indices(segment_indices, config.segment_indices_path, config)
    
    # Prepare segments for DTW matrix computation (from original dataset)
    segments = []
    for idx in segment_indices:
        segment = {
            "observations": dataset["observations"][idx:idx + config.segment_size],
            "actions": dataset["actions"][idx:idx + config.segment_size]
        }
        segments.append(segment)
    
    # Compute DTW matrix using original dataset
    dtw_matrix = load_or_compute_dtw_matrix(
        segments, 
        use_relative_eef, 
        config.dtw_matrix_path, 
        config
    )
    
    # Prepare segments from augmentation dataset if available
    aug_segments = []
    if augmentation_dataset is not None:
        aug_N = augmentation_dataset["observations"].shape[0]
        aug_traj_total = aug_N // 500  # each trajectory has 500 steps
        
        # Get random segment indices from augmentation dataset
        aug_segment_indices = []
        for _ in range(config.dtw_subsample_size):
            traj_idx = np.random.randint(0, aug_traj_total)
            start_idx = 500 * traj_idx + np.random.randint(0, 500 - config.segment_size)
            aug_segment_indices.append(start_idx)
        
        # Create segments from augmentation dataset
        for idx in aug_segment_indices:
            segment = {
                "observations": augmentation_dataset["observations"][idx:idx + config.segment_size],
                "actions": augmentation_dataset["actions"][idx:idx + config.segment_size]
            }
            aug_segments.append(segment)
    
    # If no augmentation dataset, use original dataset for segments
    if not aug_segments:
        aug_segments = segments
    
    # Create mapping from segment indices to matrix indices
    segment_to_matrix_idx = {idx: i for i, idx in enumerate(segment_indices)}
    
    # Initialize DTW distances dictionary
    dtw_distances_dict = {}
    
    # For each original pair, find similar segments from augmentation dataset
    multiple_ranked_list = []
    candidate_segment_indices = []
    
    for pair_idx, (idx1, idx2) in enumerate(original_pairs):
        if pair_idx % 100 == 0:
            print(f"Processing pair {pair_idx}/{len(original_pairs)}")
        
        # Get DTW distances for this pair from the matrix
        if idx1 in segment_to_matrix_idx and idx2 in segment_to_matrix_idx:
            matrix_idx1 = segment_to_matrix_idx[idx1]
            matrix_idx2 = segment_to_matrix_idx[idx2]
            
            # Get distances for both segments
            distances1 = dtw_matrix[matrix_idx1]
            distances2 = dtw_matrix[matrix_idx2]
            
            # Store distances for analysis
            dtw_distances_dict[idx1] = list(zip(segment_indices, distances1))
            dtw_distances_dict[idx2] = list(zip(segment_indices, distances2))
            
            # Find similar segments from augmentation dataset
            similar_segments = []
            for i, aug_segment in enumerate(aug_segments):
                # Compute DTW distance to both original segments
                dist1 = compute_dtw_distance(segments[matrix_idx1], aug_segment, use_relative_eef, use_goal_pos)
                dist2 = compute_dtw_distance(segments[matrix_idx2], aug_segment, use_relative_eef, use_goal_pos)
                
                # Store segment and its distances
                similar_segments.append((i, dist1, dist2))
            
            # Sort by average distance to both original segments
            similar_segments.sort(key=lambda x: (x[1] + x[2]) / 2)
            
            # Take top K similar segments
            top_k = min(config.dtw_k_augment, len(similar_segments))
            top_segments = similar_segments[:top_k]
            
            # Create ranking list for this pair
            single_ranked_list = []
            
            # Add original segments first
            if original_preferences[pair_idx] == [1, 0]:  # First segment preferred
                group1 = [(idx1, 1.0)]  # Higher ranked
                group2 = [(idx2, 0.0)]  # Lower ranked
                single_ranked_list.append(group2)
                single_ranked_list.append(group1)
            elif original_preferences[pair_idx] == [0, 1]:  # Second segment preferred
                group1 = [(idx1, 0.0)]  # Lower ranked
                group2 = [(idx2, 1.0)]  # Higher ranked
                single_ranked_list.append(group1)
                single_ranked_list.append(group2)
            else:  # Equal preference
                group = [(idx1, 0.5), (idx2, 0.5)]
                single_ranked_list.append(group)
            
            # Add similar segments based on their distances
            for seg_idx, dist1, dist2 in top_segments:
                aug_idx = aug_segment_indices[seg_idx]
                candidate_segment_indices.append(aug_idx)
                
                # Determine preference based on distances
                if abs(dist1 - dist2) < 0.1:  # Similar distances
                    # Add to existing group with equal preference
                    single_ranked_list[0].append((aug_idx, 0.5))
                else:
                    # Create new group based on which original segment it's closer to
                    if dist1 < dist2:
                        single_ranked_list.append([(aug_idx, 1.0)])
                    else:
                        single_ranked_list.append([(aug_idx, 0.0)])
            
            multiple_ranked_list.append(single_ranked_list)
    
    print(f"Generated {len(multiple_ranked_list)} DTW augmentation pairs")
    return multiple_ranked_list, dtw_distances_dict, candidate_segment_indices


def load_or_compute_dtw_matrix(segments: List[Dict], use_relative_eef: bool, dtw_matrix_path: str, config) -> np.ndarray:
    """
    Load precomputed DTW matrix from cache or compute and save it.
    
    Args:
        segments: List of segment dictionaries with 'obs' field
        use_relative_eef: Whether to use relative EEF positions
        dtw_matrix_path: Path to save/load DTW matrix cache
        config: Configuration object
        
    Returns:
        dtw_matrix: Matrix of DTW distances between segments
    """
    # Create cache metadata for validation
    cache_metadata = {
        'n_segments': len(segments),
        'dtw_sample_size': getattr(config, 'dtw_subsample_size', 10000),
        'use_relative_eef': use_relative_eef,
        'segment_size': config.segment_size,
        'seed': getattr(config, 'seed', None)
    }
    
    # Check if cache file exists
    if os.path.exists(dtw_matrix_path):
        print(f"Loading DTW matrix from cache: {dtw_matrix_path}")
        try:
            with open(dtw_matrix_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Validate cache metadata
            cached_meta = cached_data['metadata']
            if (cached_meta['n_segments'] == cache_metadata['n_segments'] and
                cached_meta.get('dtw_sample_size', 10000) == cache_metadata['dtw_sample_size'] and
                cached_meta['use_relative_eef'] == cache_metadata['use_relative_eef'] and
                cached_meta['segment_size'] == cache_metadata['segment_size'] and
                cached_meta.get('seed', None) == cache_metadata['seed']):
                
                print("DTW matrix cache validation successful - using cached DTW matrix")
                return cached_data['dtw_matrix']
            else:
                print("DTW matrix cache validation failed - recomputing DTW matrix")
                print(f"  Cache meta: {cached_meta}")
                print(f"  Current meta: {cache_metadata}")
                
        except Exception as e:
            print(f"Error loading DTW matrix cache: {e} - recomputing DTW matrix")
    
    # Compute DTW matrix
    print(f"Computing DTW matrix for {len(segments)} segments...")
    
    # Import DTW module
    try:
        from utils import dtw
        print("Using custom DTW implementation")
    except ImportError:
        raise ImportError("DTW module not found. Make sure robot_pref.utils.dtw is available.")
    
    dtw_matrix = compute_full_dtw_matrix(segments, use_relative_eef, dtw)
    
    # Save to cache
    print(f"Saving DTW matrix to cache: {dtw_matrix_path}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dtw_matrix_path), exist_ok=True)
        
        cache_data = {
            'dtw_matrix': dtw_matrix,
            'metadata': cache_metadata
        }
        
        with open(dtw_matrix_path, 'wb') as f:
            pickle.dump(cache_data, f)
            
        print("DTW matrix saved successfully")
    except Exception as e:
        print(f"Warning: Could not save DTW matrix cache: {e}")
    
    return dtw_matrix


def collect_feedback(dataset, traj_total, config):
    multiple_ranked_list = []
    print("config.feedback_num", config.feedback_num)
    if config.feedback_type == "RLT":
        # If q_budget = 1, we collect independent pairwise feedback
        print("Construct RLT")
        cum_query = 0
        current_query_num = 0
        single_ranked_list = []
        while cum_query < config.feedback_num:
            if current_query_num >= config.q_budget:
                multiple_ranked_list.append(single_ranked_list)
                current_query_num = 0
                single_ranked_list = []
            # binary search
            group = []
            if len(single_ranked_list) == 0:
                idx_1 = get_indices(traj_total, config)
                idx_2 = get_indices(traj_total, config)
                idx_st_1 = idx_1[0][0]
                idx_st_2 = idx_2[0][0]
                reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
                reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
                label = obtain_labels(
                    dataset,
                    idx_1,
                    idx_2,
                    segment_size=config.segment_size,
                    threshold=config.threshold,
                    noise=config.noise,
                )
                cum_query += 1
                current_query_num += 1
                if np.all(label[0] == [1, 0]):
                    group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                elif np.all(label[0] == [0.5, 0.5]):
                    group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
                    single_ranked_list.append(group)
                elif np.all(label[0] == [0, 1]):
                    group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
            else:
                idx = get_indices(traj_total, config)
                idx_st = idx[0][0]
                reward = np.round(np.sum(dataset["rewards"][idx], axis=1), 2).item()
                low = 0
                high = len(single_ranked_list)
                pos = 0
                insert = False
                while low < high:
                    mid = (low + high) // 2
                    mid_num = (low + high) // 2
                    mid = (mid + mid_num) // 2
                    len_mid = len(single_ranked_list[mid])
                    # random select one element in the mid group
                    random_idx = np.random.randint(0, len_mid)
                    compare_idx = single_ranked_list[mid][random_idx][0]
                    compare_seg_idx = [
                        [
                            j
                            for j in range(
                                compare_idx, compare_idx + config.segment_size
                            )
                        ]
                    ]
                    label = obtain_labels(
                        dataset,
                        idx,
                        compare_seg_idx,
                        segment_size=config.segment_size,
                        threshold=config.threshold,
                        noise=config.noise,
                    )
                    cum_query += 1
                    current_query_num += 1
                    if np.all(label[0] == [0, 1]):
                        high = mid
                        pos = mid
                    elif np.all(label[0] == [1, 0]):
                        low = mid + 1
                        pos = mid + 1
                    else:
                        if cum_query <= config.feedback_num:
                            single_ranked_list[mid].append((idx_st, reward))
                        insert = True
                        break
                if insert == False and cum_query <= config.feedback_num:
                    single_ranked_list.insert(pos, [(idx_st, reward)])
        multiple_ranked_list.append(single_ranked_list)
        return multiple_ranked_list
    elif config.feedback_type == "SeqRank":
        print("Sequential Pairwise feedback (SeqRank)")
        single_ranked_list = []
        up = 0
        if len(single_ranked_list) == 0:
            idx_1 = get_indices(traj_total, config)
            idx_2 = get_indices(traj_total, config)
            idx_st_1 = idx_1[0][0]
            idx_st_2 = idx_2[0][0]
            reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
            reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
            label = obtain_labels(
                dataset,
                idx_1,
                idx_2,
                segment_size=config.segment_size,
                threshold=config.threshold,
                noise=config.noise,
            )
            if np.all(label[0] == [1, 0]):
                group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                single_ranked_list.append(group_2)
                single_ranked_list.append(group_1)
                up = -1
            elif np.all(label[0] == [0.5, 0.5]):
                group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
                single_ranked_list.append(group)
                up = 0
            elif np.all(label[0] == [0, 1]):
                group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                single_ranked_list.append(group_1)
                single_ranked_list.append(group_2)
        last_idx = idx_2
        last_idx_st = last_idx[0][0]
        for i in range(config.feedback_num - 1):
            idx = get_indices(traj_total, config)
            idx_st = idx[0][0]
            reward_last = np.round(
                np.sum(dataset["rewards"][last_idx], axis=1), 2
            ).item()
            reward = np.round(np.sum(dataset["rewards"][idx], axis=1), 2).item()
            label = obtain_labels(
                dataset,
                last_idx,
                idx,
                segment_size=config.segment_size,
                threshold=config.threshold,
                noise=config.noise,
            )
            if np.all(label[0] == [1, 0]):
                group_1, group_2 = [(last_idx_st, reward_last)], [(idx_st, reward)]
                curr_up = -1
                if up == curr_up or up == 0:
                    # insert front of single_ranked_list
                    single_ranked_list.insert(0, group_2)
                    up = -1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                    up = -1
            elif np.all(label[0] == [0.5, 0.5]):
                if up == -1:
                    single_ranked_list[0].append((idx_st, reward))
                else:
                    single_ranked_list[-1].append((idx_st, reward))
            elif np.all(label[0] == [0, 1]):
                group_1, group_2 = [(last_idx_st, reward_last)], [(idx_st, reward)]
                curr_up = 1
                if up == curr_up or up == 0:
                    single_ranked_list.append(group_2)
                    up = 1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
                    up = 1
            last_idx = idx
            last_idx_st = idx[0][0]
        multiple_ranked_list.append(single_ranked_list)
        return multiple_ranked_list


def collect_human_feedback(dataset, config):
    print("Human feedback")
    multiple_ranked_list = []
    if config.feedback_type == "RLT":
        # indepednent sampling
        if config.q_budget == 1:
            print("Independent pairwise feedback")
            path = f"./human_feedback/{config.env}/_Independent.txt"
            with open(path, "r") as f:
                for line in f:
                    single_ranked_list = []
                    line = line.split(" ")
                    idx_1 = int(line[0])
                    idx_2 = int(line[1])
                    label = int(line[2])
                    reward_1 = np.round(
                        np.sum(dataset["rewards"][idx_1 : idx_1 + config.segment_size]),
                        2,
                    ).item()
                    reward_2 = np.round(
                        np.sum(dataset["rewards"][idx_2 : idx_2 + config.segment_size]),
                        2,
                    ).item()
                    if label == 1:
                        group_1, group_2 = [(idx_1, reward_1)], [(idx_2, reward_2)]
                        single_ranked_list.append(group_2)
                        single_ranked_list.append(group_1)
                    elif label == 2:
                        group = [(idx_1, reward_1), (idx_2, reward_2)]
                        single_ranked_list.append(group)
                    elif label == 3:
                        group_1, group_2 = [(idx_1, reward_1)], [(idx_2, reward_2)]
                        single_ranked_list.append(group_1)
                        single_ranked_list.append(group_2)
                    multiple_ranked_list.append(single_ranked_list)
        # LiRE
        elif config.q_budget > 1:
            print("Construct RLT (LiRE)")
            for s in range(0, 2):
                path = f"./human_feedback/{config.env}/_RLT_{s}.txt"
                single_ranked_list = []
                with open(path, "r") as f:
                    for line in f:
                        line = line.split("[")[1].split("]")[0].split(", ")
                        group = []
                        for idx in line:
                            idx = int(idx)
                            group.append(
                                (
                                    idx,
                                    np.round(
                                        np.sum(
                                            dataset["rewards"][
                                                idx : idx + config.segment_size
                                            ]
                                        ),
                                        2,
                                    ),
                                )
                            )
                        single_ranked_list.append(group)
                multiple_ranked_list.append(single_ranked_list)

    elif config.feedback_type == "SeqRank":
        print("Sequential Pairwise feedback (SeqRank)")
        path = f"./human_feedback/{config.env}/_SeqRank.txt"
        single_ranked_list = []
        idx_st_1 = []
        idx_st_2 = []
        labels = []
        raw_labels = []
        reward_1 = []
        reward_2 = []
        with open(path, "r") as f:
            for line in f:
                line = line.split("\t")
                index_1 = int(line[0])
                index_2 = int(line[1])
                label = int(line[2])
                idx_st_1.append(index_1)
                idx_st_2.append(index_2)
                raw_labels.append(label)
            idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
            idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
            reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
            reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
            for i in range(len(raw_labels)):
                if raw_labels[i] == 1:
                    labels.append([1, 0])
                elif raw_labels[i] == 2:
                    labels.append([0.5, 0.5])
                elif raw_labels[i] == 3:
                    labels.append([0, 1])
            labels = np.array(labels)
        if np.all(labels[0] == [1, 0]):
            group_1, group_2 = [(idx_st_1[0], reward_1[0])], [
                (idx_st_2[0], reward_2[0])
            ]
            single_ranked_list.append(group_2)
            single_ranked_list.append(group_1)
            up = -1
        elif np.all(labels[0] == [0.5, 0.5]):
            group = [(idx_st_1[0], reward_1[0]), (idx_st_2[0], reward_2[0])]
            single_ranked_list.append(group)
            up = 0
        elif np.all(labels[0] == [0, 1]):
            group_1, group_2 = [(idx_st_1[0], reward_1[0])], [
                (idx_st_2[0], reward_2[0])
            ]
            single_ranked_list.append(group_1)
            single_ranked_list.append(group_2)
            up = 1
        for i in range(1, len(labels)):
            if np.all(labels[i] == [1, 0]):
                group_1, group_2 = [(idx_st_1[i], reward_1[i])], [
                    (idx_st_2[i], reward_2[i])
                ]
                curr_up = -1
                if up == curr_up or up == 0:
                    # insert front of single_ranked_list
                    single_ranked_list.insert(0, group_2)
                    up = -1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                    up = -1
            elif np.all(labels[i] == [0.5, 0.5]):
                if up == -1:
                    single_ranked_list[0].append((idx_st_2[i], reward_2[i]))
                else:
                    single_ranked_list[-1].append((idx_st_2[i], reward_2[i]))
            elif np.all(labels[i] == [0, 1]):
                group_1, group_2 = [(idx_st_1[i], reward_1[i])], [
                    (idx_st_2[i], reward_2[i])
                ]
                curr_up = 1
                if up == curr_up or up == 0:
                    single_ranked_list.append(group_2)
                    up = 1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
                    up = 1
        multiple_ranked_list.append(single_ranked_list)
    return multiple_ranked_list


def obtain_labels(dataset, idx_1, idx_2, segment_size=25, threshold=0.5, noise=0.0):
    idx_1 = np.array(idx_1)
    idx_2 = np.array(idx_2)
    labels = []
    reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
    reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
    labels = np.where(reward_1 < reward_2, 1, 0)
    labels = np.array([[1, 0] if i == 0 else [0, 1] for i in labels]).astype(float)
    gap = segment_size * threshold

    equal_labels = np.where(
        np.abs(reward_1 - reward_2) <= segment_size * threshold, 1, 0
    )
    labels = np.array(
        [labels[i] if equal_labels[i] == 0 else [0.5, 0.5] for i in range(len(labels))]
    )
    if noise != 0.0:
        p = noise
        for i in range(len(labels)):
            if labels[i][0] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            elif labels[i][1] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 1
                        labels[i][1] = 0
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            else:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 1
                        labels[i][1] = 0
    return labels


def get_reward_model_predictions(reward_model, seg1, seg2, return_ensemble_predictions=False, debug=False):
    """
    Unified function to get predictions from any type of reward model.
    
    Args:
        reward_model: RewardModel instance
        seg1: First segment tensor [segment_size, obs+act_dim]
        seg2: Second segment tensor [segment_size, obs+act_dim]
        return_ensemble_predictions: If True, return individual ensemble predictions
        debug: If True, print debug information
        
    Returns:
        If return_ensemble_predictions=False:
            tuple: (pred_return_1, pred_return_2) - scalar returns
        If return_ensemble_predictions=True:
            tuple: (ensemble_returns_1, ensemble_returns_2) - list of ensemble member returns
    """
    if debug:
        print(f"    Model type: {type(reward_model)}")
        print(f"    Has ensemble_model attr: {hasattr(reward_model, 'ensemble_model')}")
        if hasattr(reward_model, 'ensemble_model'):
            print(f"    ensemble_model value: {reward_model.ensemble_model}")
            print(f"    ensemble_model is None: {reward_model.ensemble_model is None}")
        print(f"    Has net attr: {hasattr(reward_model, 'net')}")
        if hasattr(reward_model, 'net'):
            print(f"    net value: {reward_model.net}")
        print(f"    Has single_model_forward: {hasattr(reward_model, 'single_model_forward')}")
        print(f"    Has ensemble_model_forward: {hasattr(reward_model, 'ensemble_model_forward')}")
    
    # Set model to eval mode
    if hasattr(reward_model, 'model') and reward_model.model is not None:
        reward_model.model.eval()
    elif hasattr(reward_model, 'ensemble_model') and reward_model.ensemble_model is not None:
        for member in reward_model.ensemble_model:
            member.eval()
    elif hasattr(reward_model, 'net') and reward_model.net is not None:
        reward_model.net.eval()
    
    # Move segments to device
    seg1 = seg1.to(reward_model.device)
    seg2 = seg2.to(reward_model.device)
    
    # Get predictions based on model type
    if hasattr(reward_model, 'ensemble_model') and reward_model.ensemble_model is not None and len(reward_model.ensemble_model) > 0:
        # Ensemble model - get predictions from each member
        ensemble_returns_1 = []
        ensemble_returns_2 = []
        
        for member_idx in range(len(reward_model.ensemble_model)):
            member_rewards_1 = reward_model.ensemble_model[member_idx](seg1)
            member_rewards_2 = reward_model.ensemble_model[member_idx](seg2)
            ensemble_returns_1.append(member_rewards_1.sum().item())
            ensemble_returns_2.append(member_rewards_2.sum().item())
        
        if return_ensemble_predictions:
            return ensemble_returns_1, ensemble_returns_2
        else:
            # Return averaged predictions
            pred_ret_1 = np.mean(ensemble_returns_1)
            pred_ret_2 = np.mean(ensemble_returns_2)
            return pred_ret_1, pred_ret_2
            
    else:
        # Single model - try different methods
        if hasattr(reward_model, 'ensemble_model_forward') and hasattr(reward_model, 'ensemble_model') and reward_model.ensemble_model is not None:
            # Use ensemble_model_forward if available and ensemble_model exists
            rewards_1 = reward_model.ensemble_model_forward(seg1)
            rewards_2 = reward_model.ensemble_model_forward(seg2)
        elif hasattr(reward_model, 'single_model_forward'):
            # Try single_model_forward method
            rewards_1 = reward_model.single_model_forward(seg1)
            rewards_2 = reward_model.single_model_forward(seg2)
        elif hasattr(reward_model, 'net') and reward_model.net is not None:
            # Use single net if available
            rewards_1 = reward_model.net(seg1)
            rewards_2 = reward_model.net(seg2)
        else:
            raise ValueError(f"Could not find a valid model to use for prediction in {reward_model}")
        
        pred_ret_1 = rewards_1.sum().item()
        pred_ret_2 = rewards_2.sum().item()
        
        if return_ensemble_predictions:
            # Return as single-element lists for consistency
            return [pred_ret_1], [pred_ret_2]
        else:
            return pred_ret_1, pred_ret_2


def compute_acquisition_scores(reward_model, obs_act_1, obs_act_2, labels, config):
    """
    Compute acquisition scores based on ensemble disagreement/entropy for preference pairs.
    
    Args:
        reward_model: Trained RewardModel with ensemble
        obs_act_1: First segments (observations + actions concatenated) [N, segment_size, obs+act_dim]
        obs_act_2: Second segments (observations + actions concatenated) [N, segment_size, obs+act_dim]
        labels: Preference labels [N, 2]
        config: Configuration object
        
    Returns:
        acquisition_scores: Array of acquisition scores for each preference pair [N]
    """
    print(f"Computing acquisition scores for {len(obs_act_1)} preference pairs...")
    
    # Convert data to tensors if needed
    if not isinstance(obs_act_1, torch.Tensor):
        obs_act_1 = torch.tensor(obs_act_1, dtype=torch.float32)
    if not isinstance(obs_act_2, torch.Tensor):
        obs_act_2 = torch.tensor(obs_act_2, dtype=torch.float32)
    
    acquisition_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(len(obs_act_1)), desc="Computing acquisition scores"):
            # Get segments for this pair
            seg1 = obs_act_1[i]  # [segment_size, obs+act_dim]
            seg2 = obs_act_2[i]  # [segment_size, obs+act_dim]
            
            # Get ensemble predictions
            ensemble_returns_1, ensemble_returns_2 = get_reward_model_predictions(
                reward_model, seg1, seg2, return_ensemble_predictions=True
            )
            
            # Convert to numpy arrays
            ensemble_returns_1 = np.array(ensemble_returns_1)
            ensemble_returns_2 = np.array(ensemble_returns_2)
            
            # Compute return deltas for each ensemble member
            ensemble_deltas = ensemble_returns_1 - ensemble_returns_2
            
            # Convert to preference probabilities
            ensemble_probs = []
            temperature = 1.0
            for delta in ensemble_deltas:
                prob_1 = 1.0 / (1.0 + np.exp(-delta / temperature))
                prob_2 = 1.0 - prob_1
                ensemble_probs.append([prob_1, prob_2])
            
            ensemble_probs = np.array(ensemble_probs)  # [ensemble_num, 2]
            mean_probs = np.mean(ensemble_probs, axis=0)  # [2]
            
            # Compute different uncertainty measures
            # 1. Variance in return deltas
            delta_variance = np.var(ensemble_deltas)
            
            # 2. Entropy of mean probabilities
            epsilon = 1e-8
            mean_probs_clipped = np.clip(mean_probs, epsilon, 1 - epsilon)
            entropy = -np.sum(mean_probs_clipped * np.log(mean_probs_clipped))
            
            # 3. Disagreement: variance in probabilities across ensemble members
            prob_disagreement = np.mean(np.var(ensemble_probs, axis=0))
            
            # Select uncertainty measure based on acquisition_method
            if config.acquisition_method == "entropy":
                acquisition_score = entropy
            elif config.acquisition_method == "disagreement":
                acquisition_score = prob_disagreement
            elif config.acquisition_method == "variance":
                acquisition_score = delta_variance
            elif config.acquisition_method == "combined":
                acquisition_score = entropy + prob_disagreement + 0.1 * delta_variance
            else:
                raise ValueError(f"Unknown acquisition_method: {config.acquisition_method}")
                
            acquisition_scores.append(acquisition_score)
    
    acquisition_scores = np.array(acquisition_scores)
    
    print("Acquisition score statistics:")
    print(f"  Method: {config.acquisition_method}")
    print(f"  Min: {acquisition_scores.min():.4f}")
    print(f"  Max: {acquisition_scores.max():.4f}")
    print(f"  Mean: {acquisition_scores.mean():.4f}")
    print(f"  Std: {acquisition_scores.std():.4f}")
    
    return acquisition_scores


def filter_augmentations_by_acquisition(
    multiple_ranked_list, 
    obs_act_1, 
    obs_act_2, 
    labels,
    acquisition_scores, 
    threshold_low=0.25, 
    threshold_high=0.75
):
    """
    Filter augmentations based on acquisition scores using percentile thresholds.
    
    Args:
        multiple_ranked_list: List of ranking lists
        obs_act_1: First segments
        obs_act_2: Second segments  
        labels: Preference labels
        acquisition_scores: Acquisition scores for each pair
        threshold_low: Lower percentile threshold (e.g., 0.25 for 25th percentile)
        threshold_high: Higher percentile threshold (e.g., 0.75 for 75th percentile)
        
    Returns:
        tuple: Filtered (multiple_ranked_list, obs_act_1, obs_act_2, labels)
    """
    # Compute percentile thresholds
    low_threshold = np.percentile(acquisition_scores, threshold_low * 100)
    high_threshold = np.percentile(acquisition_scores, threshold_high * 100)
    
    print(f"Filtering augmentations with acquisition scores between {low_threshold:.4f} and {high_threshold:.4f}")
    print(f"  (percentiles {threshold_low*100}% - {threshold_high*100}%)")
    
    # Filter based on thresholds
    mask = (acquisition_scores >= low_threshold) & (acquisition_scores <= high_threshold)
    
    filtered_multiple_ranked_list = [multiple_ranked_list[i] for i in range(len(multiple_ranked_list)) if mask[i]]
    filtered_obs_act_1 = obs_act_1[mask] if isinstance(obs_act_1, np.ndarray) else np.array(obs_act_1)[mask]
    filtered_obs_act_2 = obs_act_2[mask] if isinstance(obs_act_2, np.ndarray) else np.array(obs_act_2)[mask] 
    filtered_labels = labels[mask] if isinstance(labels, np.ndarray) else np.array(labels)[mask]
    
    print(f"Filtered {len(filtered_multiple_ranked_list)} augmentations from {len(multiple_ranked_list)} candidates")
    print(f"  Kept {mask.sum()}/{len(mask)} augmentations ({mask.mean()*100:.1f}%)")
    
    return filtered_multiple_ranked_list, filtered_obs_act_1, filtered_obs_act_2, filtered_labels


def display_preference_label_stats(labels, return_1, return_2, config, title="Preference Labels"):
    """
    Display comprehensive statistics about preference labels and return differences.
    
    Args:
        labels: Preference labels array [N, 2]
        return_1: Returns for first segments [N]
        return_2: Returns for second segments [N]
        config: Configuration object with threshold and segment_size
        title: Title for the statistics display
    """
    print(f"\n=== {title} Statistics ===")
    
    # Count different types of preferences
    seg1_better_mask = np.array([np.array_equal(label, [1, 0]) for label in labels])
    seg2_better_mask = np.array([np.array_equal(label, [0, 1]) for label in labels])
    equal_mask = np.array([np.array_equal(label, [0.5, 0.5]) for label in labels])
    
    num_seg1_better = np.sum(seg1_better_mask)
    num_seg2_better = np.sum(seg2_better_mask)
    num_equal = np.sum(equal_mask)
    total_pairs = len(labels)
    
    print(f"Total preference pairs: {total_pairs}")
    print(f"Segment 1 better: {num_seg1_better} ({num_seg1_better/total_pairs*100:.1f}%)")
    print(f"Segment 2 better: {num_seg2_better} ({num_seg2_better/total_pairs*100:.1f}%)")
    print(f"Equal preference: {num_equal} ({num_equal/total_pairs*100:.1f}%)")
    
    # Compute return deltas (return_1 - return_2)
    return_deltas = return_1 - return_2
    
    print("\n=== Return Delta Statistics ===")
    print(f"Overall return delta - Mean: {np.mean(return_deltas):.3f}, Std: {np.std(return_deltas):.3f}")
    print(f"Overall return delta - Min: {np.min(return_deltas):.3f}, Max: {np.max(return_deltas):.3f}")
    
    # Stats when segment 1 is better
    if num_seg1_better > 0:
        seg1_better_deltas = return_deltas[seg1_better_mask]
        print(f"\nWhen Segment 1 is better (n={num_seg1_better}):")
        print(f"  Return delta - Mean: {np.mean(seg1_better_deltas):.3f}, Std: {np.std(seg1_better_deltas):.3f}")
        print(f"  Return delta - Min: {np.min(seg1_better_deltas):.3f}, Max: {np.max(seg1_better_deltas):.3f}")
        print(f"  Avg return 1: {np.mean(return_1[seg1_better_mask]):.3f}, Avg return 2: {np.mean(return_2[seg1_better_mask]):.3f}")
    
    # Stats when segment 2 is better
    if num_seg2_better > 0:
        seg2_better_deltas = return_deltas[seg2_better_mask]
        print(f"\nWhen Segment 2 is better (n={num_seg2_better}):")
        print(f"  Return delta - Mean: {np.mean(seg2_better_deltas):.3f}, Std: {np.std(seg2_better_deltas):.3f}")
        print(f"  Return delta - Min: {np.min(seg2_better_deltas):.3f}, Max: {np.max(seg2_better_deltas):.3f}")
        print(f"  Avg return 1: {np.mean(return_1[seg2_better_mask]):.3f}, Avg return 2: {np.mean(return_2[seg2_better_mask]):.3f}")
    
    # Stats when segments are equal
    if num_equal > 0:
        equal_deltas = return_deltas[equal_mask]
        print(f"\nWhen Segments are equal (n={num_equal}):")
        print(f"  Return delta - Mean: {np.mean(equal_deltas):.3f}, Std: {np.std(equal_deltas):.3f}")
        print(f"  Return delta - Min: {np.min(equal_deltas):.3f}, Max: {np.max(equal_deltas):.3f}")
        print(f"  Avg return 1: {np.mean(return_1[equal_mask]):.3f}, Avg return 2: {np.mean(return_2[equal_mask]):.3f}")
    
    print("\n=== Threshold Analysis ===")
    print(f"Threshold used: {config.threshold}")
    print(f"Segment size: {config.segment_size}")
    print(f"Threshold gap: {config.segment_size * config.threshold}")
    
    # Check how many pairs are within threshold
    abs_deltas = np.abs(return_deltas)
    within_threshold = np.sum(abs_deltas <= config.segment_size * config.threshold)
    print(f"Pairs within threshold gap: {within_threshold} ({within_threshold/total_pairs*100:.1f}%)")
    print("=" * 50)
    
    return {
        "total_pairs": total_pairs,
        "num_seg1_better": num_seg1_better,
        "num_seg2_better": num_seg2_better,
        "num_equal": num_equal,
        "mean_delta": np.mean(return_deltas),
        "std_delta": np.std(return_deltas),
        "within_threshold": within_threshold
    }


def compute_full_dtw_matrix(segments: List[Dict], use_relative_eef: bool, use_goal_pos: bool, dtw) -> np.ndarray:
    """
    Compute full DTW distance matrix between segments.
    Slower but more accurate than fast approximation.
    
    Args:
        segments: List of segment dictionaries with 'obs' field
        use_relative_eef: Whether to use relative EEF positions
        dtw: DTW module for computing distances
        
    Returns:
        distance_matrix: Matrix of DTW distances between segments
    """
    n_segments = len(segments)
    
    distance_matrix = np.zeros((n_segments, n_segments))
    min_dist = float("inf")
    max_dist = float("-inf")
    sum_dist = 0
    count = 0
    non_finite_count = 0

    total_comparisons = n_segments * (n_segments - 1) // 2
    with tqdm(total=total_comparisons, desc="Computing DTW distances") as pbar:
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                # Extract EE positions (assuming first 3 dimensions are EE positions)
                query = segments[i]["obs"].numpy()[:, :3]
                reference = segments[j]["obs"].numpy()[:, :3]
                import ipdb; ipdb.set_trace()
                if use_goal_pos:
                    query = np.concatenate((query, segments[i]["obs"].numpy()[:, 36:]), axis=1)
                    reference = np.concatenate((reference, segments[j]["obs"].numpy()[:, 36:]), axis=1)

                # Use relative positions if requested
                if use_relative_eef:
                    query = query[1:] - query[:-1]
                    reference = reference[1:] - reference[:-1]

                try:
                    cost, _ = dtw.get_single_match(query, reference)
                except:
                    # Fallback: use Euclidean distance between segment means
                    cost = np.linalg.norm(np.mean(query, axis=0) - np.mean(reference, axis=0))

                distance_matrix[i, j] = cost
                distance_matrix[j, i] = cost

                # Update statistics
                if np.isfinite(cost):
                    min_dist = min(min_dist, cost)
                    max_dist = max(max_dist, cost)
                    sum_dist += cost
                    count += 1
                else:
                    non_finite_count += 1

                pbar.update(1)

    if count > 0:
        avg_dist = sum_dist / count
        print(f"DTW distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}")
    if non_finite_count > 0:
        print(f"WARNING: {non_finite_count} DTW distances were non-finite")
        
    return distance_matrix


def analyze_dtw_augmentation_quality(
    reward_model, 
    dtw_obs_act_1, 
    dtw_obs_act_2, 
    dtw_labels, 
    dataset,
    segment_start_indices_1,
    segment_start_indices_2,
    config,
    output_path=None,
    wandb_run=None
):
    """
    Analyze the quality of DTW augmentations by examining the relationship between
    reward model uncertainty and ground truth accuracy.
    
    Args:
        reward_model: Trained reward model with ensemble
        dtw_obs_act_1: First segments of DTW augmented pairs [N, segment_size, obs+act_dim] 
        dtw_obs_act_2: Second segments of DTW augmented pairs [N, segment_size, obs+act_dim]
        dtw_labels: Predicted preference labels from DTW propagation [N, 2]
        dataset: Original dataset to compute ground truth preferences
        segment_start_indices_1: Starting indices of first segments [N]
        segment_start_indices_2: Starting indices of second segments [N]
        config: Configuration object
        output_path: Path to save analysis plot
        wandb_run: W&B run for logging
        
    Returns:
        dict: Analysis metrics
    """
    print("Analyzing DTW augmentation quality...")
    
    if len(dtw_obs_act_1) == 0:
        print("No DTW augmentations to analyze")
        return {}
    
    # Set ensemble models to eval mode
    for member in range(reward_model.ensemble_num):
        reward_model.ensemble_model[member].eval()
    
    # Convert data to tensors if needed
    if not isinstance(dtw_obs_act_1, torch.Tensor):
        dtw_obs_act_1 = torch.tensor(dtw_obs_act_1, dtype=torch.float32)
    if not isinstance(dtw_obs_act_2, torch.Tensor):
        dtw_obs_act_2 = torch.tensor(dtw_obs_act_2, dtype=torch.float32)
    
    dtw_obs_act_1 = dtw_obs_act_1.to(reward_model.device)
    dtw_obs_act_2 = dtw_obs_act_2.to(reward_model.device)
    
    # Compute ground truth preferences for DTW augmented pairs
    print("Computing ground truth preferences for DTW augmentations...")
    gt_preferences = []
    
    for i in range(len(segment_start_indices_1)):
        idx_st_1 = segment_start_indices_1[i]
        idx_st_2 = segment_start_indices_2[i]
        
        # Calculate segment rewards
        idx_1 = [[j for j in range(idx_st_1, idx_st_1 + config.segment_size)]]
        idx_2 = [[j for j in range(idx_st_2, idx_st_2 + config.segment_size)]]
        
        gt_label = obtain_labels(
            dataset,
            idx_1,
            idx_2,
            segment_size=config.segment_size,
            threshold=config.threshold,
            noise=0.0,  # No noise for ground truth
        )[0]
        gt_preferences.append(gt_label)
    
    gt_preferences = np.array(gt_preferences)
    
    # Compute reward model uncertainty/disagreement for each pair
    print("Computing reward model uncertainty for DTW augmentations...")
    uncertainties = []
    predicted_preferences = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dtw_obs_act_1)), desc="Computing uncertainties"):
            seg1 = dtw_obs_act_1[i]  # [segment_size, obs+act_dim]
            seg2 = dtw_obs_act_2[i]  # [segment_size, obs+act_dim]
            
            # Get ensemble predictions using unified function
            ensemble_returns_1, ensemble_returns_2 = get_reward_model_predictions(
                reward_model, seg1, seg2, return_ensemble_predictions=True
            )
            
            # Convert to numpy arrays
            ensemble_returns_1 = np.array(ensemble_returns_1)
            ensemble_returns_2 = np.array(ensemble_returns_2)
            
            # Compute return deltas for each ensemble member
            ensemble_deltas = ensemble_returns_1 - ensemble_returns_2
            
            # Convert to preference probabilities
            ensemble_probs = []
            temperature = 1.0
            for delta in ensemble_deltas:
                prob_1 = 1.0 / (1.0 + np.exp(-delta / temperature))
                prob_2 = 1.0 - prob_1
                ensemble_probs.append([prob_1, prob_2])
            
            ensemble_probs = np.array(ensemble_probs)  # [ensemble_num, 2]
            mean_probs = np.mean(ensemble_probs, axis=0)  # [2]
            predicted_preferences.append(mean_probs)
            
            # Compute different uncertainty measures
            # 1. Variance in return deltas
            delta_variance = np.var(ensemble_deltas)
            
            # 2. Entropy of mean probabilities
            epsilon = 1e-8
            mean_probs_clipped = np.clip(mean_probs, epsilon, 1 - epsilon)
            entropy = -np.sum(mean_probs_clipped * np.log(mean_probs_clipped))
            
            # 3. Disagreement: variance in probabilities across ensemble members
            prob_disagreement = np.mean(np.var(ensemble_probs, axis=0))
            
            # Select uncertainty measure based on acquisition_method
            if config.acquisition_method == "entropy":
                acquisition_score = entropy
            elif config.acquisition_method == "disagreement":
                acquisition_score = prob_disagreement
            elif config.acquisition_method == "variance":
                acquisition_score = delta_variance
            elif config.acquisition_method == "combined":
                acquisition_score = entropy + prob_disagreement + 0.1 * delta_variance
            else:
                raise ValueError(f"Unknown acquisition_method: {config.acquisition_method}")
                
            uncertainties.append(acquisition_score)
    
    uncertainties = np.array(uncertainties)
    predicted_preferences = np.array(predicted_preferences)
    
    # Compute accuracy of DTW propagated labels vs ground truth
    print("Computing accuracy metrics...")
    
    # Convert predicted preferences to discrete labels
    predicted_labels = []
    for pred_probs in predicted_preferences:
        if pred_probs[0] > pred_probs[1] + 0.1:  # Threshold for preference
            predicted_labels.append([1, 0])
        elif pred_probs[1] > pred_probs[0] + 0.1:
            predicted_labels.append([0, 1])
        else:
            predicted_labels.append([0.5, 0.5])
    predicted_labels = np.array(predicted_labels)
    
    # Compute accuracy: DTW propagated vs ground truth
    dtw_vs_gt_accuracy = np.mean([
        np.array_equal(dtw_labels[i], gt_preferences[i]) 
        for i in range(len(dtw_labels))
    ])
    
    # Compute accuracy: Reward model prediction vs ground truth  
    rm_vs_gt_accuracy = np.mean([
        np.array_equal(predicted_labels[i], gt_preferences[i])
        for i in range(len(predicted_labels))
    ])
    
    # Compute accuracy: Reward model prediction vs DTW propagated
    rm_vs_dtw_accuracy = np.mean([
        np.array_equal(predicted_labels[i], dtw_labels[i])
        for i in range(len(predicted_labels))
    ])
    
    print("DTW Augmentation Quality Analysis:")
    print(f"  Acquisition method: {config.acquisition_method}")
    print(f"  DTW propagated vs Ground Truth accuracy: {dtw_vs_gt_accuracy:.3f}")
    print(f"  Reward Model vs Ground Truth accuracy: {rm_vs_gt_accuracy:.3f}") 
    print(f"  Reward Model vs DTW propagated accuracy: {rm_vs_dtw_accuracy:.3f}")
    print(f"  Mean uncertainty: {np.mean(uncertainties):.3f}")
    print(f"  Uncertainty std: {np.std(uncertainties):.3f}")
    
    # Create visualization
    if output_path is not None:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('DTW Augmentation Quality Analysis', fontsize=16)
        
        # 1. Uncertainty vs DTW-GT accuracy scatter plot
        ax1 = axes[0, 0]
        dtw_gt_correct = [np.array_equal(dtw_labels[i], gt_preferences[i]) for i in range(len(dtw_labels))]
        scatter1 = ax1.scatter(uncertainties, dtw_gt_correct, alpha=0.6, s=20)
        ax1.set_xlabel('Reward Model Uncertainty')
        ax1.set_ylabel('DTW vs GT Correct (0/1)')
        ax1.set_title('Uncertainty vs DTW Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # 2. Uncertainty vs RM-GT accuracy scatter plot  
        ax2 = axes[0, 1]
        rm_gt_correct = [np.array_equal(predicted_labels[i], gt_preferences[i]) for i in range(len(predicted_labels))]
        scatter2 = ax2.scatter(uncertainties, rm_gt_correct, alpha=0.6, s=20, color='orange')
        ax2.set_xlabel('Reward Model Uncertainty') 
        ax2.set_ylabel('RM vs GT Correct (0/1)')
        ax2.set_title('Uncertainty vs RM Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # 3. Uncertainty histogram
        ax3 = axes[1, 0]
        ax3.hist(uncertainties, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Uncertainty')
        ax3.set_ylabel('Count')
        ax3.set_title('Uncertainty Distribution')
        ax3.axvline(np.mean(uncertainties), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(uncertainties):.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Accuracy comparison bar chart
        ax4 = axes[1, 1]
        
        # Prepare data for grouped bar chart
        pref_types = ['Overall', 'Seg1 Better', 'Seg2 Better', 'Equal Pref']
        baseline_accs = [
            dtw_vs_gt_accuracy,
            dtw_vs_gt_accuracy,
            dtw_vs_gt_accuracy,
            dtw_vs_gt_accuracy
        ]
        augmented_accs = [
            rm_vs_gt_accuracy,
            rm_vs_gt_accuracy,
            rm_vs_gt_accuracy,
            rm_vs_gt_accuracy
        ]
        
        x = np.arange(len(pref_types))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, baseline_accs, width, label='DTW vs GT', alpha=0.7, color='blue')
        bars2 = ax4.bar(x + width/2, augmented_accs, width, label='RM vs GT', alpha=0.7, color='green')
        
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy Comparison by Preference Type')
        ax4.set_xticks(x)
        ax4.set_xticklabels(pref_types, rotation=15)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"DTW augmentation analysis saved to: {output_path}")
    
    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log({
            "dtw_analysis/dtw_vs_gt_accuracy": dtw_vs_gt_accuracy,
            "dtw_analysis/rm_vs_gt_accuracy": rm_vs_gt_accuracy,
            "dtw_analysis/rm_vs_dtw_accuracy": rm_vs_dtw_accuracy,
            "dtw_analysis/mean_uncertainty": np.mean(uncertainties),
            "dtw_analysis/uncertainty_std": np.std(uncertainties),
            "dtw_analysis/num_augmentations": len(dtw_labels)
        })
    
    return {
        "dtw_vs_gt_accuracy": dtw_vs_gt_accuracy,
        "rm_vs_gt_accuracy": rm_vs_gt_accuracy, 
        "rm_vs_dtw_accuracy": rm_vs_dtw_accuracy,
        "uncertainties": uncertainties,
        "mean_uncertainty": np.mean(uncertainties),
        "uncertainty_std": np.std(uncertainties),
        "num_augmentations": len(dtw_labels)
    }


def compare_baseline_vs_augmented_performance(
    baseline_reward_model,
    augmented_reward_model, 
    test_obs_act_1,
    test_obs_act_2,
    test_labels,
    test_binary_labels,
    test_gt_return_1,  # Ground truth returns for first segments
    test_gt_return_2,  # Ground truth returns for second segments
    config,
    output_path=None,
    wandb_run=None
):
    """
    Compare performance of baseline (no augmentation) vs augmented reward models on held-out test set.
    
    Args:
        baseline_reward_model: RewardModel trained on original data only
        augmented_reward_model: RewardModel trained on original + DTW augmented data
        test_obs_act_1: Test set first segments [N, segment_size, obs+act_dim]
        test_obs_act_2: Test set second segments [N, segment_size, obs+act_dim]
        test_labels: Test set preference labels [N, 2]
        test_binary_labels: Test set binary preference labels [N, 2]
        test_gt_return_1: Ground truth returns for first segments [N]
        test_gt_return_2: Ground truth returns for second segments [N]
        config: Configuration object
        output_path: Path to save comparison plot
        wandb_run: W&B run for logging
        
    Returns:
        dict: Comparison metrics
    """
    print("\n=== Baseline vs Augmented Model Comparison ===")
    
    # Convert data to tensors if needed
    if not isinstance(test_obs_act_1, torch.Tensor):
        test_obs_act_1 = torch.tensor(test_obs_act_1, dtype=torch.float32)
    if not isinstance(test_obs_act_2, torch.Tensor):
        test_obs_act_2 = torch.tensor(test_obs_act_2, dtype=torch.float32)
    
    # Set models to eval mode
    baseline_reward_model.model.eval() if hasattr(baseline_reward_model, 'model') else None
    augmented_reward_model.model.eval() if hasattr(augmented_reward_model, 'model') else None
    
    # Helper function to evaluate a single model
    def evaluate_model(reward_model, model_name):
        print(f"Evaluating {model_name} model...")
        
        predictions = []
        losses = []
        predicted_deltas = []  # Store predicted return deltas
        
        # Track accuracy by preference type
        seg1_better_correct = 0
        seg1_better_total = 0
        seg2_better_correct = 0
        seg2_better_total = 0
        equal_pref_correct = 0
        equal_pref_total = 0
        
        with torch.no_grad():
            for i in tqdm(range(len(test_obs_act_1)), desc=f"Evaluating {model_name}"):
                seg1 = test_obs_act_1[i]
                seg2 = test_obs_act_2[i]
                true_label = test_labels[i]
                
                # Check if ensemble model is available
                if hasattr(reward_model, 'ensemble_model') and reward_model.ensemble_model is not None and len(reward_model.ensemble_model) > 0:
                    # Ensemble model - get individual member predictions for ensemble averaging
                    ensemble_returns_1, ensemble_returns_2 = get_reward_model_predictions(
                        reward_model, seg1, seg2, return_ensemble_predictions=True
                    )
                    
                    # Convert to preference probabilities for each ensemble member
                    ensemble_probs = []
                    for ret1, ret2 in zip(ensemble_returns_1, ensemble_returns_2):
                        delta = ret1 - ret2
                        prob_1 = 1.0 / (1.0 + np.exp(-delta))
                        prob_2 = 1.0 - prob_1
                        ensemble_probs.append([prob_1, prob_2])
                    
                    # Average ensemble predictions
                    pred_probs = np.mean(ensemble_probs, axis=0)
                    pred_delta = np.mean(ensemble_returns_1) - np.mean(ensemble_returns_2)
                    
                else:
                    # Single model - get averaged prediction
                    pred_ret_1, pred_ret_2 = get_reward_model_predictions(
                        reward_model, seg1, seg2
                    )
                    
                    # Convert to preference probability
                    delta = pred_ret_1 - pred_ret_2
                    prob_1 = 1.0 / (1.0 + np.exp(-delta))
                    prob_2 = 1.0 - prob_1
                    pred_probs = [prob_1, prob_2]
                    pred_delta = delta
                
                predictions.append(pred_probs)
                predicted_deltas.append(pred_delta)
                
                # Convert prediction to discrete label
                if pred_probs[0] > pred_probs[1] + 0.1:
                    pred_label = [1, 0]
                elif pred_probs[1] > pred_probs[0] + 0.1:
                    pred_label = [0, 1]
                else:
                    pred_label = [0.5, 0.5]
                
                # Check accuracy by preference type
                is_correct = np.array_equal(pred_label, true_label)
                
                if np.array_equal(true_label, [1, 0]):  # Segment 1 better
                    seg1_better_total += 1
                    if is_correct:
                        seg1_better_correct += 1
                elif np.array_equal(true_label, [0, 1]):  # Segment 2 better
                    seg2_better_total += 1
                    if is_correct:
                        seg2_better_correct += 1
                elif np.array_equal(true_label, [0.5, 0.5]):  # Equal preference
                    equal_pref_total += 1
                    if is_correct:
                        equal_pref_correct += 1
                
                # Compute loss (binary cross entropy)
                pred_tensor = torch.tensor(pred_probs, dtype=torch.float32)
                true_tensor = torch.tensor(true_label, dtype=torch.float32)
                loss = torch.nn.functional.binary_cross_entropy(pred_tensor, true_tensor)
                losses.append(loss.item())
        
        predictions = np.array(predictions)
        losses = np.array(losses)
        
        # Convert predictions to discrete labels
        pred_labels = []
        for pred_probs in predictions:
            if pred_probs[0] > pred_probs[1] + 0.1:
                pred_labels.append([1, 0])
            elif pred_probs[1] > pred_probs[0] + 0.1:
                pred_labels.append([0, 1])
            else:
                pred_labels.append([0.5, 0.5])
        pred_labels = np.array(pred_labels)
        
        # Compute overall accuracy
        accuracy = np.mean([
            np.array_equal(pred_labels[i], test_labels[i])
            for i in range(len(pred_labels))
        ])
        
        # Compute binary accuracy (strict preferences only)
        binary_accuracy = np.mean([
            np.array_equal(pred_labels[i], test_binary_labels[i])
            for i in range(len(pred_labels))
        ])
        
        # Compute accuracy by preference type
        seg1_better_acc = seg1_better_correct / seg1_better_total if seg1_better_total > 0 else 0.0
        seg2_better_acc = seg2_better_correct / seg2_better_total if seg2_better_total > 0 else 0.0
        equal_pref_acc = equal_pref_correct / equal_pref_total if equal_pref_total > 0 else 0.0
        
        print(f"  {model_name} accuracy by preference type:")
        print(f"    Segment 1 better: {seg1_better_acc:.3f} ({seg1_better_correct}/{seg1_better_total})")
        print(f"    Segment 2 better: {seg2_better_acc:.3f} ({seg2_better_correct}/{seg2_better_total})")
        print(f"    Equal preference: {equal_pref_acc:.3f} ({equal_pref_correct}/{equal_pref_total})")
        
        return {
            'accuracy': accuracy,
            'binary_accuracy': binary_accuracy, 
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'predictions': predictions,
            'pred_labels': pred_labels,
            'predicted_deltas': predicted_deltas,
            'seg1_better_acc': seg1_better_acc,
            'seg2_better_acc': seg2_better_acc,
            'equal_pref_acc': equal_pref_acc,
            'seg1_better_total': seg1_better_total,
            'seg2_better_total': seg2_better_total,
            'equal_pref_total': equal_pref_total
        }
    
    # Evaluate both models
    baseline_results = evaluate_model(baseline_reward_model, "Baseline")
    augmented_results = evaluate_model(augmented_reward_model, "Augmented")
    
    # Compute ground truth deltas from test set
    print("Computing ground truth return deltas for test set...")
    gt_deltas = test_gt_return_1 - test_gt_return_2
    
    gt_deltas = np.array(gt_deltas)
    baseline_pred_deltas = np.array(baseline_results['predicted_deltas'])
    augmented_pred_deltas = np.array(augmented_results['predicted_deltas'])
    
    # Compute MSE for delta predictions
    baseline_delta_mse = np.mean((baseline_pred_deltas - gt_deltas) ** 2)
    augmented_delta_mse = np.mean((augmented_pred_deltas - gt_deltas) ** 2)
    
    # Normalize MSE by variance of ground truth deltas for interpretability
    gt_delta_var = np.var(gt_deltas)
    if gt_delta_var > 0:
        baseline_delta_mse_normalized = baseline_delta_mse / gt_delta_var
        augmented_delta_mse_normalized = augmented_delta_mse / gt_delta_var
        delta_mse_improvement = (baseline_delta_mse - augmented_delta_mse) / baseline_delta_mse * 100  # % improvement
    else:
        baseline_delta_mse_normalized = baseline_delta_mse
        augmented_delta_mse_normalized = augmented_delta_mse
        delta_mse_improvement = 0.0
    
    # Compute improvements
    accuracy_improvement = augmented_results['accuracy'] - baseline_results['accuracy']
    binary_accuracy_improvement = augmented_results['binary_accuracy'] - baseline_results['binary_accuracy']
    loss_improvement = baseline_results['mean_loss'] - augmented_results['mean_loss']  # Lower loss is better
    
    # Print results
    print("\nTest Set Performance Comparison:")
    print(f"  Test set size: {len(test_obs_act_1)} pairs")
    print("  Test set breakdown:")
    print(f"    Segment 1 better: {baseline_results['seg1_better_total']} pairs")
    print(f"    Segment 2 better: {baseline_results['seg2_better_total']} pairs") 
    print(f"    Equal preference: {baseline_results['equal_pref_total']} pairs")
    print("  Baseline model:")
    print(f"    Overall accuracy: {baseline_results['accuracy']:.4f}")
    print(f"    Binary accuracy: {baseline_results['binary_accuracy']:.4f}")
    print("    Accuracy by preference type:")
    print(f"      Segment 1 better: {baseline_results['seg1_better_acc']:.4f}")
    print(f"      Segment 2 better: {baseline_results['seg2_better_acc']:.4f}")
    print(f"      Equal preference: {baseline_results['equal_pref_acc']:.4f}")
    print(f"    Mean loss: {baseline_results['mean_loss']:.4f} ± {baseline_results['std_loss']:.4f}")
    print(f"    Delta MSE: {baseline_delta_mse:.4f} (normalized: {baseline_delta_mse_normalized:.4f})")
    print("  Augmented model:")
    print(f"    Overall accuracy: {augmented_results['accuracy']:.4f}")
    print(f"    Binary accuracy: {augmented_results['binary_accuracy']:.4f}")
    print("    Accuracy by preference type:")
    print(f"      Segment 1 better: {augmented_results['seg1_better_acc']:.4f}")
    print(f"      Segment 2 better: {augmented_results['seg2_better_acc']:.4f}")
    print(f"      Equal preference: {augmented_results['equal_pref_acc']:.4f}")
    print(f"    Mean loss: {augmented_results['mean_loss']:.4f} ± {augmented_results['std_loss']:.4f}")
    print(f"    Delta MSE: {augmented_delta_mse:.4f} (normalized: {augmented_delta_mse_normalized:.4f})")
    print("  Improvements:")
    print(f"    Overall accuracy: {accuracy_improvement:+.4f} ({accuracy_improvement/baseline_results['accuracy']*100:+.1f}%)")
    print(f"    Binary accuracy: {binary_accuracy_improvement:+.4f} ({binary_accuracy_improvement/baseline_results['binary_accuracy']*100:+.1f}%)")
    print(f"    Seg1 better accuracy: {augmented_results['seg1_better_acc'] - baseline_results['seg1_better_acc']:+.4f}")
    print(f"    Seg2 better accuracy: {augmented_results['seg2_better_acc'] - baseline_results['seg2_better_acc']:+.4f}")
    print(f"    Equal pref accuracy: {augmented_results['equal_pref_acc'] - baseline_results['equal_pref_acc']:+.4f}")
    print(f"    Loss: {loss_improvement:+.4f} ({loss_improvement/baseline_results['mean_loss']*100:+.1f}%)")
    print(f"    Delta MSE: {delta_mse_improvement:+.1f}% improvement")
    
    # Create visualization
    if output_path is not None:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Baseline vs Augmented Model Comparison', fontsize=16)
        
        # 1. Accuracy comparison bar chart
        ax1 = axes[0, 0]
        
        # Prepare data for grouped bar chart
        pref_types = ['Overall', 'Seg1 Better', 'Seg2 Better', 'Equal Pref']
        baseline_accs = [
            baseline_results['accuracy'],
            baseline_results['seg1_better_acc'],
            baseline_results['seg2_better_acc'], 
            baseline_results['equal_pref_acc']
        ]
        augmented_accs = [
            augmented_results['accuracy'],
            augmented_results['seg1_better_acc'],
            augmented_results['seg2_better_acc'],
            augmented_results['equal_pref_acc']
        ]
        
        x = np.arange(len(pref_types))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.7, color='orange')
        bars2 = ax1.bar(x + width/2, augmented_accs, width, label='Augmented', alpha=0.7, color='green')
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison by Preference Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pref_types, rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Loss comparison
        ax2 = axes[0, 1] 
        losses = [baseline_results['mean_loss'], augmented_results['mean_loss']]
        loss_stds = [baseline_results['std_loss'], augmented_results['std_loss']]
        
        bars = ax2.bar(['Baseline', 'Augmented'], losses, yerr=loss_stds, 
                      alpha=0.7, capsize=5, color=['orange', 'green'])
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Comparison (Lower is Better)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, loss, std in zip(bars, losses, loss_stds):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                    f'{loss:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # 3. Prediction confidence comparison (entropy)
        ax3 = axes[1, 0]
        
        def compute_entropy(probs):
            epsilon = 1e-8 
            probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
            return -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
        
        baseline_entropy = compute_entropy(baseline_results['predictions'])
        augmented_entropy = compute_entropy(augmented_results['predictions'])
        
        ax3.hist(baseline_entropy, bins=30, alpha=0.7, label='Baseline', color='orange')
        ax3.hist(augmented_entropy, bins=30, alpha=0.7, label='Augmented', color='green')
        ax3.set_xlabel('Prediction Entropy')
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Confidence Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Delta MSE comparison
        ax4 = axes[1, 1]
        
        delta_mses = [baseline_delta_mse_normalized, augmented_delta_mse_normalized]
        model_names = ['Baseline', 'Augmented']
        colors = ['orange', 'green']
        
        bars = ax4.bar(model_names, delta_mses, color=colors, alpha=0.7)
        ax4.set_ylabel('Normalized Delta MSE')
        ax4.set_title('Delta Prediction MSE (Lower is Better)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mse in zip(bars, delta_mses):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mse:.3f}', ha='center', va='bottom')
        
        # Add improvement percentage as text
        if delta_mse_improvement != 0:
            ax4.text(0.5, 0.95, f'{delta_mse_improvement:+.1f}% improvement', 
                    transform=ax4.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison analysis saved to: {output_path}")
    
    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log({
            "comparison/baseline_accuracy": baseline_results['accuracy'],
            "comparison/augmented_accuracy": augmented_results['accuracy'],
            "comparison/accuracy_improvement": accuracy_improvement,
            "comparison/baseline_binary_accuracy": baseline_results['binary_accuracy'],
            "comparison/augmented_binary_accuracy": augmented_results['binary_accuracy'], 
            "comparison/binary_accuracy_improvement": binary_accuracy_improvement,
            "comparison/baseline_seg1_better_acc": baseline_results['seg1_better_acc'],
            "comparison/augmented_seg1_better_acc": augmented_results['seg1_better_acc'],
            "comparison/seg1_better_acc_improvement": augmented_results['seg1_better_acc'] - baseline_results['seg1_better_acc'],
            "comparison/baseline_seg2_better_acc": baseline_results['seg2_better_acc'],
            "comparison/augmented_seg2_better_acc": augmented_results['seg2_better_acc'],
            "comparison/seg2_better_acc_improvement": augmented_results['seg2_better_acc'] - baseline_results['seg2_better_acc'],
            "comparison/baseline_equal_pref_acc": baseline_results['equal_pref_acc'],
            "comparison/augmented_equal_pref_acc": augmented_results['equal_pref_acc'],
            "comparison/equal_pref_acc_improvement": augmented_results['equal_pref_acc'] - baseline_results['equal_pref_acc'],
            "comparison/test_set_breakdown/seg1_better_total": baseline_results['seg1_better_total'],
            "comparison/test_set_breakdown/seg2_better_total": baseline_results['seg2_better_total'],
            "comparison/test_set_breakdown/equal_pref_total": baseline_results['equal_pref_total'],
            "comparison/baseline_loss": baseline_results['mean_loss'],
            "comparison/augmented_loss": augmented_results['mean_loss'],
            "comparison/loss_improvement": loss_improvement,
            "comparison/baseline_loss_std": baseline_results['std_loss'],
            "comparison/augmented_loss_std": augmented_results['std_loss'],
            "comparison/baseline_delta_mse": baseline_delta_mse,
            "comparison/augmented_delta_mse": augmented_delta_mse,
            "comparison/baseline_delta_mse_normalized": baseline_delta_mse_normalized,
            "comparison/augmented_delta_mse_normalized": augmented_delta_mse_normalized,
            "comparison/delta_mse_improvement": delta_mse_improvement,
            "comparison/gt_delta_variance": gt_delta_var,
        })
    
    return {
        "baseline_results": baseline_results,
        "augmented_results": augmented_results,
        "accuracy_improvement": accuracy_improvement,
        "binary_accuracy_improvement": binary_accuracy_improvement,
        "seg1_better_acc_improvement": augmented_results['seg1_better_acc'] - baseline_results['seg1_better_acc'],
        "seg2_better_acc_improvement": augmented_results['seg2_better_acc'] - baseline_results['seg2_better_acc'],
        "equal_pref_acc_improvement": augmented_results['equal_pref_acc'] - baseline_results['equal_pref_acc'],
        "loss_improvement": loss_improvement,
        "baseline_delta_mse": baseline_delta_mse,
        "augmented_delta_mse": augmented_delta_mse,
        "baseline_delta_mse_normalized": baseline_delta_mse_normalized,
        "augmented_delta_mse_normalized": augmented_delta_mse_normalized,
        "delta_mse_improvement": delta_mse_improvement,
        "gt_delta_variance": gt_delta_var,
        "test_set_size": len(test_obs_act_1),
        "test_set_breakdown": {
            "seg1_better_total": baseline_results['seg1_better_total'],
            "seg2_better_total": baseline_results['seg2_better_total'],
            "equal_pref_total": baseline_results['equal_pref_total']
        }
    }


def plot_baseline_vs_augmented_scatter_analysis(
    baseline_reward_model,
    augmented_reward_model,
    obs_act_segments,
    gt_returns,
    segment_size,
    output_file=None,
    max_samples=5000,
    wandb_run=None,
    random_seed=42
):
    """
    Create side-by-side scatter plots comparing baseline vs augmented model predictions
    against ground truth returns for individual segments.
    
    Args:
        baseline_reward_model: Baseline reward model
        augmented_reward_model: Augmented reward model
        obs_act_segments: Individual segments [N, segment_size, obs+act_dim]
        gt_returns: Ground truth returns for segments [N]
        segment_size: Size of segments
        output_file: Path to save plot
        max_samples: Maximum number of samples to plot
        wandb_run: W&B run for logging
        random_seed: Random seed for sampling
        
    Returns:
        dict: Analysis metrics for both models
    """
    print("Creating baseline vs augmented scatter analysis for individual segments...")
    
    # Set random seed for reproducible sampling
    np.random.seed(random_seed)
    
    # Sample data if too large
    n_samples = min(max_samples, len(obs_act_segments))
    if len(obs_act_segments) > max_samples:
        indices = np.random.choice(len(obs_act_segments), max_samples, replace=False)
        obs_act_segments_sample = obs_act_segments[indices]
        gt_returns_sample = gt_returns[indices]
    else:
        obs_act_segments_sample = obs_act_segments
        gt_returns_sample = gt_returns
    
    # Convert to tensors if needed
    if not isinstance(obs_act_segments_sample, torch.Tensor):
        obs_act_segments_sample = torch.tensor(obs_act_segments_sample, dtype=torch.float32)
    
    def get_model_predictions(reward_model, model_name):
        """Get predictions from a reward model for individual segments."""
        print(f"  Computing {model_name} predictions...")
        
        pred_returns = []
        
        with torch.no_grad():
            for i in tqdm(range(len(obs_act_segments_sample)), desc=f"Predicting {model_name}"):
                seg = obs_act_segments_sample[i]
                
                # Get single segment prediction (no pairs needed)
                if hasattr(reward_model, 'ensemble_model') and reward_model.ensemble_model is not None and len(reward_model.ensemble_model) > 0:
                    # Ensemble model - average predictions
                    ensemble_returns = []
                    for member_idx in range(len(reward_model.ensemble_model)):
                        member_rewards = reward_model.ensemble_model[member_idx](seg.to(reward_model.device))
                        ensemble_returns.append(member_rewards.sum().item())
                    pred_ret = np.mean(ensemble_returns)
                else:
                    # Single model - use unified prediction function approach
                    seg = seg.to(reward_model.device)
                    if hasattr(reward_model, 'ensemble_model_forward') and hasattr(reward_model, 'ensemble_model') and reward_model.ensemble_model is not None:
                        rewards = reward_model.ensemble_model_forward(seg)
                    elif hasattr(reward_model, 'single_model_forward'):
                        rewards = reward_model.single_model_forward(seg)
                    elif hasattr(reward_model, 'net') and reward_model.net is not None:
                        rewards = reward_model.net(seg)
                    else:
                        raise ValueError(f"Could not find a valid model to use for prediction in {reward_model}")
                    pred_ret = rewards.sum().item()
                
                pred_returns.append(pred_ret)
        
        return np.array(pred_returns)
    
    # Get predictions from both models
    baseline_pred = get_model_predictions(baseline_reward_model, "baseline")
    augmented_pred = get_model_predictions(augmented_reward_model, "augmented")
    
    # Compute metrics
    def compute_metrics(pred, gt, model_name):
        corr = np.corrcoef(pred, gt)[0, 1] if len(set(gt)) > 1 else 0.0
        mse = np.mean((pred - gt) ** 2)
        mae = np.mean(np.abs(pred - gt))
        
        print(f"  {model_name} metrics:")
        print(f"    Correlation: {corr:.3f}")
        print(f"    MSE: {mse:.3f}")
        print(f"    MAE: {mae:.3f}")
        
        return {
            'correlation': corr,
            'mse': mse,
            'mae': mae
        }
    
    baseline_metrics = compute_metrics(baseline_pred, gt_returns_sample, "Baseline")
    augmented_metrics = compute_metrics(augmented_pred, gt_returns_sample, "Augmented")
    
    # Create side-by-side plots
    if output_file is not None:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Baseline vs Augmented Model: Segment Return Prediction', fontsize=16)
        
        # Common plot settings
        def setup_scatter_plot(ax, pred, gt, title, metrics):
            ax.scatter(gt, pred, alpha=0.6, s=20)
            
            # Add perfect prediction line
            min_val = min(gt.min(), pred.min())
            max_val = max(gt.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
            
            ax.set_xlabel('Ground Truth Return')
            ax.set_ylabel('Predicted Return')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add metrics text
            ax.text(0.05, 0.95, f'r = {metrics["correlation"]:.3f}\nMSE = {metrics["mse"]:.3f}\nMAE = {metrics["mae"]:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Baseline model
        setup_scatter_plot(axes[0], baseline_pred, gt_returns_sample, 'Baseline Model', baseline_metrics)
        
        # Augmented model
        setup_scatter_plot(axes[1], augmented_pred, gt_returns_sample, 'Augmented Model', augmented_metrics)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Baseline vs augmented scatter analysis saved to: {output_file}")
    
    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log({
            "scatter_comparison/baseline_correlation": baseline_metrics['correlation'],
            "scatter_comparison/baseline_mse": baseline_metrics['mse'],
            "scatter_comparison/baseline_mae": baseline_metrics['mae'],
            "scatter_comparison/augmented_correlation": augmented_metrics['correlation'],
            "scatter_comparison/augmented_mse": augmented_metrics['mse'],
            "scatter_comparison/augmented_mae": augmented_metrics['mae'],
            "scatter_comparison/samples_used": n_samples,
            "scatter_comparison/correlation_improvement": augmented_metrics['correlation'] - baseline_metrics['correlation'],
            "scatter_comparison/mse_improvement": baseline_metrics['mse'] - augmented_metrics['mse'],  # Lower MSE is better
            "scatter_comparison/mae_improvement": baseline_metrics['mae'] - augmented_metrics['mae']   # Lower MAE is better
        })
    
    return {
        "baseline_metrics": baseline_metrics,
        "augmented_metrics": augmented_metrics,
        "n_samples": n_samples
    }


def plot_individual_test_example_deltas(
    baseline_reward_model,
    augmented_reward_model,
    test_obs_act_1,
    test_obs_act_2,
    test_gt_return_1,
    test_gt_return_2,
    output_file=None,
    max_examples=100,
    wandb_run=None,
    random_seed=42
):
    """
    Create a dedicated chart showing individual test example deltas for baseline, augmented, and ground truth.
    
    Args:
        baseline_reward_model: Baseline reward model
        augmented_reward_model: Augmented reward model
        test_obs_act_1: Test set first segments [N, segment_size, obs+act_dim]
        test_obs_act_2: Test set second segments [N, segment_size, obs+act_dim]
        test_gt_return_1: Ground truth returns for first segments [N]
        test_gt_return_2: Ground truth returns for second segments [N]
        output_file: Path to save plot
        max_examples: Maximum number of examples to show
        wandb_run: W&B run for logging
        random_seed: Random seed for sampling
        
    Returns:
        dict: Analysis metrics
    """
    print("Creating individual test example deltas chart...")
    
    # Set random seed for reproducible sampling
    np.random.seed(random_seed)
    
    # Convert data to tensors if needed
    if not isinstance(test_obs_act_1, torch.Tensor):
        test_obs_act_1 = torch.tensor(test_obs_act_1, dtype=torch.float32)
    if not isinstance(test_obs_act_2, torch.Tensor):
        test_obs_act_2 = torch.tensor(test_obs_act_2, dtype=torch.float32)
    
    # Compute ground truth deltas
    gt_deltas = test_gt_return_1 - test_gt_return_2
    
    # Get predictions from both models
    print("  Computing baseline predictions...")
    baseline_pred_deltas = []
    with torch.no_grad():
        for i in tqdm(range(len(test_obs_act_1)), desc="Baseline predictions"):
            seg1 = test_obs_act_1[i]
            seg2 = test_obs_act_2[i]
            pred_ret_1, pred_ret_2 = get_reward_model_predictions(baseline_reward_model, seg1, seg2)
            baseline_pred_deltas.append(pred_ret_1 - pred_ret_2)
    baseline_pred_deltas = np.array(baseline_pred_deltas)
    
    print("  Computing augmented predictions...")
    augmented_pred_deltas = []
    with torch.no_grad():
        for i in tqdm(range(len(test_obs_act_1)), desc="Augmented predictions"):
            seg1 = test_obs_act_1[i]
            seg2 = test_obs_act_2[i]
            pred_ret_1, pred_ret_2 = get_reward_model_predictions(augmented_reward_model, seg1, seg2)
            augmented_pred_deltas.append(pred_ret_1 - pred_ret_2)
    augmented_pred_deltas = np.array(augmented_pred_deltas)
    
    # Sample examples to display (sorted by GT delta magnitude for interesting cases)
    n_examples = min(max_examples, len(gt_deltas))
    sorted_indices = np.argsort(np.abs(gt_deltas))[-n_examples:]  # Take examples with largest |GT delta|
    
    sample_gt = gt_deltas[sorted_indices]
    sample_baseline = baseline_pred_deltas[sorted_indices]
    sample_augmented = augmented_pred_deltas[sorted_indices]
    
    # Normalize all predictions and ground truth to standardized scale (z-scores)
    print("  Normalizing all deltas to standardized scale (mean=0, std=1)...")
    
    # Compute statistics
    gt_mean = np.mean(gt_deltas)
    gt_std = np.std(gt_deltas)
    
    baseline_mean = np.mean(baseline_pred_deltas)
    baseline_std = np.std(baseline_pred_deltas)
    
    augmented_mean = np.mean(augmented_pred_deltas)
    augmented_std = np.std(augmented_pred_deltas)
    
    print("    Original scales:")
    print(f"      GT: mean={gt_mean:.3f}, std={gt_std:.3f}")
    print(f"      Baseline: mean={baseline_mean:.3f}, std={baseline_std:.3f}")
    print(f"      Augmented: mean={augmented_mean:.3f}, std={augmented_std:.3f}")
    
    # Normalize all to z-scores (mean=0, std=1)
    if gt_std > 0:
        gt_deltas_normalized = (gt_deltas - gt_mean) / gt_std
        sample_gt_normalized = (sample_gt - gt_mean) / gt_std
    else:
        gt_deltas_normalized = gt_deltas
        sample_gt_normalized = sample_gt
    
    if baseline_std > 0:
        baseline_pred_deltas_normalized = (baseline_pred_deltas - baseline_mean) / baseline_std
        sample_baseline_normalized = (sample_baseline - baseline_mean) / baseline_std
    else:
        baseline_pred_deltas_normalized = baseline_pred_deltas
        sample_baseline_normalized = sample_baseline
        
    if augmented_std > 0:
        augmented_pred_deltas_normalized = (augmented_pred_deltas - augmented_mean) / augmented_std
        sample_augmented_normalized = (sample_augmented - augmented_mean) / augmented_std
    else:
        augmented_pred_deltas_normalized = augmented_pred_deltas
        sample_augmented_normalized = sample_augmented
    
    print("    After z-score normalization:")
    print(f"      GT: mean={np.mean(gt_deltas_normalized):.3f}, std={np.std(gt_deltas_normalized):.3f}")
    print(f"      Baseline: mean={np.mean(baseline_pred_deltas_normalized):.3f}, std={np.std(baseline_pred_deltas_normalized):.3f}")
    print(f"      Augmented: mean={np.mean(augmented_pred_deltas_normalized):.3f}, std={np.std(augmented_pred_deltas_normalized):.3f}")
    
    # Create visualization
    if output_file is not None:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(max(12, n_examples * 0.3), 8))
        fig.suptitle(f'Individual Test Example Deltas - All Normalized to Z-Scores (Top {n_examples} by |GT Delta|)', fontsize=16)
        
        x = np.arange(n_examples)
        width = 0.25
        
        bars1 = ax.bar(x - width, sample_baseline_normalized, width, label='Baseline Pred (Normalized)', alpha=0.8, color='orange')
        bars2 = ax.bar(x, sample_augmented_normalized, width, label='Augmented Pred (Normalized)', alpha=0.8, color='green')
        bars3 = ax.bar(x + width, sample_gt_normalized, width, label='Ground Truth (Normalized)', alpha=0.8, color='blue')
        
        ax.set_xlabel('Test Example (sorted by |GT Delta|)')
        ax.set_ylabel('Normalized Return Delta (Z-Score)')
        ax.set_title('Baseline vs Augmented vs Ground Truth Delta Predictions (All Normalized to Z-Scores)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels (show every nth example for readability)
        step = max(1, n_examples // 20)  # Show at most 20 labels
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f'{i}' for i in range(0, n_examples, step)], rotation=45)
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Compute and display metrics
        baseline_mse = np.mean((sample_baseline_normalized - sample_gt_normalized) ** 2)
        augmented_mse = np.mean((sample_augmented_normalized - sample_gt_normalized) ** 2)
        baseline_mae = np.mean(np.abs(sample_baseline_normalized - sample_gt_normalized))
        augmented_mae = np.mean(np.abs(sample_augmented_normalized - sample_gt_normalized))
        baseline_corr = np.corrcoef(sample_baseline_normalized, sample_gt_normalized)[0, 1] if len(set(sample_gt_normalized)) > 1 else 0.0
        augmented_corr = np.corrcoef(sample_augmented_normalized, sample_gt_normalized)[0, 1] if len(set(sample_gt_normalized)) > 1 else 0.0
        
        # Add metrics text box
        metrics_text = f'Metrics (Top {n_examples} examples, all normalized to z-scores):\n' \
                      f'Baseline:  MSE={baseline_mse:.3f}, MAE={baseline_mae:.3f}, r={baseline_corr:.3f}\n' \
                      f'Augmented: MSE={augmented_mse:.3f}, MAE={augmented_mae:.3f}, r={augmented_corr:.3f}\n' \
                      f'Improvement: MSE={((baseline_mse-augmented_mse)/baseline_mse*100):+.1f}%, ' \
                      f'MAE={((baseline_mae-augmented_mae)/baseline_mae*100):+.1f}%, ' \
                      f'r={augmented_corr-baseline_corr:+.3f}\n\n' \
                      f'Normalization: All values converted to z-scores (mean=0, std=1)'
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Individual test example deltas chart saved to: {output_file}")
    
    # Compute full dataset metrics
    full_baseline_mse = np.mean((baseline_pred_deltas_normalized - gt_deltas_normalized) ** 2)
    full_augmented_mse = np.mean((augmented_pred_deltas_normalized - gt_deltas_normalized) ** 2) 
    full_baseline_mae = np.mean(np.abs(baseline_pred_deltas_normalized - gt_deltas_normalized))
    full_augmented_mae = np.mean(np.abs(augmented_pred_deltas_normalized - gt_deltas_normalized))
    full_baseline_corr = np.corrcoef(baseline_pred_deltas_normalized, gt_deltas_normalized)[0, 1] if len(set(gt_deltas_normalized)) > 1 else 0.0
    full_augmented_corr = np.corrcoef(augmented_pred_deltas_normalized, gt_deltas_normalized)[0, 1] if len(set(gt_deltas_normalized)) > 1 else 0.0
    
    print("  Full dataset metrics:")
    print(f"    Baseline:  MSE={full_baseline_mse:.3f}, MAE={full_baseline_mae:.3f}, r={full_baseline_corr:.3f}")
    print(f"    Augmented: MSE={full_augmented_mse:.3f}, MAE={full_augmented_mae:.3f}, r={full_augmented_corr:.3f}")
    
    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log({
            "individual_deltas/full_baseline_mse": full_baseline_mse,
            "individual_deltas/full_augmented_mse": full_augmented_mse,
            "individual_deltas/full_baseline_mae": full_baseline_mae,
            "individual_deltas/full_augmented_mae": full_augmented_mae,
            "individual_deltas/full_baseline_corr": full_baseline_corr,
            "individual_deltas/full_augmented_corr": full_augmented_corr,
            "individual_deltas/n_examples_shown": n_examples,
            "individual_deltas/n_total_examples": len(gt_deltas_normalized),
            "individual_deltas/mse_improvement": (full_baseline_mse - full_augmented_mse) / full_baseline_mse * 100,
            "individual_deltas/mae_improvement": (full_baseline_mae - full_augmented_mae) / full_baseline_mae * 100,
            "individual_deltas/corr_improvement": full_augmented_corr - full_baseline_corr
        })
    
    return {
        "baseline_mse": full_baseline_mse,
        "augmented_mse": full_augmented_mse,
        "baseline_mae": full_baseline_mae,
        "augmented_mae": full_augmented_mae,
        "baseline_corr": full_baseline_corr,
        "augmented_corr": full_augmented_corr,
        "n_examples_shown": n_examples,
        "n_total_examples": len(gt_deltas_normalized)
    }


def compute_dtw_matrix_cross(main_dataset, seg_indices, augmentation_dataset, target_seg_indices, config, use_relative_eef=True, use_goal_pos=False):
    """
    Compute DTW distance matrix between main dataset segments and target dataset segments.
    
    Args:
        main_dataset: Main dataset dictionary containing observations and actions
        seg_indices: Array of shape (N, 2) containing start and end indices for main segments
        augmentation_dataset: Augmentation dataset dictionary containing observations and actions
        target_seg_indices: Array of shape (N, 2) containing start and end indices for target segments
        config: Configuration object containing segment_size and other parameters
        use_relative_eef: Whether to use relative EEF positions
        use_goal_pos: Whether to include goal positions in DTW computation
        
    Returns:
        distance_matrix: Matrix of DTW distances between segments
        main_segment_indices: List of main dataset segment start indices
        target_segment_indices: List of target dataset segment start indices
    """
    print("\nComputing cross-dataset DTW matrix...")
    
    # Import DTW module
    try:
        from utils import dtw
        print("Using custom DTW implementation")
    except ImportError:
        raise ImportError("DTW module not found. Make sure robot_pref.utils.dtw is available.")

    seg_indices = seg_indices[:, 0]  # Use only start indices for segments
    target_seg_indices = target_seg_indices[:, 0]

    # Initialize distance matrix
    n_main = len(seg_indices)
    n_target = len(target_seg_indices)
    distance_matrix = np.zeros((n_main, n_target))
    
    # Compute statistics for progress tracking
    min_dist = float("inf")
    max_dist = float("-inf")
    sum_dist = 0
    count = 0
    non_finite_count = 0
    
    total_comparisons = n_main * n_target
    with tqdm(total=total_comparisons, desc="Computing cross-dataset DTW distances") as pbar:
        for i, main_idx in enumerate(seg_indices):
            for j, target_idx in enumerate(target_seg_indices):
                # Extract EE positions (assuming first 3 dimensions are EE positions)
                main_query = main_dataset["observations"][main_idx:main_idx + config.segment_size, :3]
                target_ref = augmentation_dataset["observations"][target_idx:target_idx + config.segment_size, :3]
                
                if use_goal_pos:
                    main_query = np.concatenate((main_query, main_dataset["observations"][main_idx:main_idx + config.segment_size, 36:]), axis=1)
                    target_ref = np.concatenate((target_ref, augmentation_dataset["observations"][target_idx:target_idx + config.segment_size, 36:]), axis=1)
                
                # Use relative positions if requested
                if use_relative_eef:
                    main_query = main_query[1:] - main_query[:-1]
                    target_ref = target_ref[1:] - target_ref[:-1]
                
                try:
                    cost, _ = dtw.get_single_match(main_query, target_ref)
                except:
                    assert False, "DTW failed"
                
                distance_matrix[i, j] = cost
                
                # Update statistics
                if np.isfinite(cost):
                    min_dist = min(min_dist, cost)
                    max_dist = max(max_dist, cost)
                    sum_dist += cost
                    count += 1
                else:
                    non_finite_count += 1
                
                pbar.update(1)
    
    if count > 0:
        avg_dist = sum_dist / count
        print(f"Cross-dataset DTW distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}")
    if non_finite_count > 0:
        print(f"WARNING: {non_finite_count} cross-dataset DTW distances were non-finite")
    
    return distance_matrix


def create_augmented_preferences_from_dtw(
    cross_dtw_matrix,
    labels,
    idx_st_1,
    idx_st_2,
    seg_indices,
    target_seg_indices,
    config,
):
    """
    Create augmented preferences based on the best DTW matches from the cross-dtw matrix.
    
    Args:
        cross_dtw_matrix: Matrix of DTW distances between main and target segments
        labels: Original preference labels
        idx_st_1: Start indices for first segments in original pairs (absolute indices in dataset)
        idx_st_2: Start indices for second segments in original pairs (absolute indices in dataset)
        seg_indices: Array of shape (N, 2) containing start and end indices for main dataset segments
        target_seg_indices: Array of shape (N, 2) containing start and end indices for target segments
        config: Configuration object containing parameters
        k_augment: Number of top matches to use for augmentation
        
    Returns:
        tuple: (augmented_labels, augmented_idx_st_1, augmented_idx_st_2)
            augmented_labels: New preference labels for augmented pairs
            augmented_idx_st_1: Start indices for first segments in augmented pairs
            augmented_idx_st_2: Start indices for second segments in augmented pairs
    """
    print("\nCreating augmented preferences from DTW matches...")
    
    # Build mapping from absolute start index to row index in seg_indices
    seg_start_to_row = {start: i for i, (start, end) in enumerate(seg_indices)}
    
    # Initialize lists for augmented data
    augmented_labels = []
    augmented_idx_st_1 = []
    augmented_idx_st_2 = []
    
    for label, abs_idx1, abs_idx2 in zip(labels, idx_st_1, idx_st_2):
        # Map absolute indices to row indices in seg_indices
        row1 = seg_start_to_row.get(abs_idx1, None)
        row2 = seg_start_to_row.get(abs_idx2, None)
        if row1 is None or row2 is None:
            print(f"Warning: Could not find segment for indices {abs_idx1}, {abs_idx2}")
            continue

        # First index corresponds to the main segment, second index corresponds to the augmentation segment
        distances1 = cross_dtw_matrix[row1]
        distances2 = cross_dtw_matrix[row2]
        
        # Find top k matches for each segment
        top_k_idx1 = np.argsort(distances1)[:config.dtw_k_augment]
        top_k_idx2 = np.argsort(distances2)[:config.dtw_k_augment]
        
        # Get corresponding target segment indices (start indices)
        top_k_target1 = target_seg_indices[top_k_idx1, 0]
        top_k_target2 = target_seg_indices[top_k_idx2, 0]
        
        # Create augmented pairs
        for t1 in top_k_target1:
            for t2 in top_k_target2:
                if t1 != t2:
                    augmented_labels.append(label)
                    augmented_idx_st_1.append(t1)
                    augmented_idx_st_2.append(t2)
    
    # Convert to numpy arrays
    augmented_labels = np.array(augmented_labels)
    augmented_idx_st_1 = np.array(augmented_idx_st_1)
    augmented_idx_st_2 = np.array(augmented_idx_st_2)
    
    print(f"Created {len(augmented_labels)} augmented preference pairs")
    print(f"Original preference distribution: {np.bincount(np.argmax(labels, axis=1))}")
    print(f"Augmented preference distribution: {np.bincount(np.argmax(augmented_labels, axis=1))}")
    
    return augmented_labels, augmented_idx_st_1, augmented_idx_st_2