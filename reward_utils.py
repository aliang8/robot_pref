import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym, random, torch, os, uuid
import rich
import wandb
from tqdm import tqdm
import pickle


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
    print(f"Collecting {config.feedback_num} independent pairwise feedback samples")
    
    for i in range(config.feedback_num):
        # Get two random trajectory segments
        idx_1 = get_indices(traj_total, config)
        idx_2 = get_indices(traj_total, config) 
        
        # Extract starting indices and calculate segment rewards
        idx_st_1 = idx_1[0][0]
        idx_st_2 = idx_2[0][0]
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


def collect_dtw_augmentations(dataset, traj_total, config, obs_act_1, obs_act_2, labels, segment_indices, use_relative_eef=True):
    """
    Collect DTW-guided augmentations using precomputed or cached DTW distance matrix.
    Uses segment indices from feedback collection to create segments and compute/load DTW matrix.
    
    Args:
        dataset: The dataset containing observations, actions, rewards
        traj_total: Total number of trajectories 
        config: Configuration object
        obs_act_1: Existing first segments [N, segment_size, obs+act_dim] (for compatibility)
        obs_act_2: Existing second segments [N, segment_size, obs+act_dim] (for compatibility)
        labels: Existing preference labels [N, 2] (for compatibility)
        segment_indices: List of segment starting indices from feedback collection
        use_relative_eef: Whether to use relative EEF positions for DTW
        
    Returns:
        tuple: (multiple_ranked_list, dtw_matrix, used_segment_indices)
    """
    print(f"Collecting DTW augmentations using segment indices from feedback collection")
    
    # Step 1: Create segments from provided segment indices
    print(f"Creating segments from {len(segment_indices)} indices...")
    segments = []
    
    for idx_st in tqdm(segment_indices, desc="Creating segments"):
        # Create segment indices
        segment_idx = [j for j in range(idx_st, idx_st + config.segment_size)]
        
        # Extract observations for this segment
        obs = dataset["observations"][segment_idx]  # [segment_size, obs_dim]
        
        # Convert to torch tensor and create segment dict
        segment = {
            "obs": torch.from_numpy(obs).float(),
            "start_idx": idx_st
        }
        segments.append(segment)

    # Step 2: Load or compute DTW distance matrix
    dtw_matrix_path = getattr(config, 'dtw_matrix_path', 'dtw_matrix_cache.pkl')
    dtw_matrix = load_or_compute_dtw_matrix(segments, use_relative_eef, dtw_matrix_path, config)
    
    # Step 3: Select diverse pairs from DTW matrix for augmentation
    print(f"Selecting {config.dtw_augmentation_size} pairs for augmentation...")
    
    n_segments = len(segments)
    
    # Get upper triangle indices and their DTW distances
    upper_triangle_indices = []
    upper_triangle_distances = []
    
    for i in range(n_segments):
        for j in range(i + 1, n_segments):
            upper_triangle_indices.append((i, j))
            upper_triangle_distances.append(dtw_matrix[i, j])
    
    upper_triangle_distances = np.array(upper_triangle_distances)
    
    # Strategy: Select pairs with diverse DTW distances
    # Sort by distance and select from different percentile ranges
    sorted_indices = np.argsort(upper_triangle_distances)
    n_pairs = min(config.dtw_augmentation_size, len(upper_triangle_indices))
    
    # Select pairs from different distance ranges for diversity
    # 30% from low distances, 40% from medium, 30% from high
    n_low = int(0.3 * n_pairs)
    n_medium = int(0.4 * n_pairs) 
    n_high = n_pairs - n_low - n_medium
    
    low_indices = sorted_indices[:len(sorted_indices)//3]
    medium_indices = sorted_indices[len(sorted_indices)//3:2*len(sorted_indices)//3]
    high_indices = sorted_indices[2*len(sorted_indices)//3:]
    
    # Sample from each range
    selected_pair_indices = []
    if len(low_indices) > 0:
        selected_pair_indices.extend(np.random.choice(low_indices, min(n_low, len(low_indices)), replace=False))
    if len(medium_indices) > 0:
        selected_pair_indices.extend(np.random.choice(medium_indices, min(n_medium, len(medium_indices)), replace=False))
    if len(high_indices) > 0:
        selected_pair_indices.extend(np.random.choice(high_indices, min(n_high, len(high_indices)), replace=False))
    
    selected_pairs = [upper_triangle_indices[idx] for idx in selected_pair_indices]
    selected_distances = [upper_triangle_distances[idx] for idx in selected_pair_indices]
    
    print(f"Selected {len(selected_pairs)} pairs with DTW distances ranging from "
          f"{min(selected_distances):.3f} to {max(selected_distances):.3f}")
    
    # Step 4: Create preference labels for selected pairs
    multiple_ranked_list = []
    
    for i, j in tqdm(selected_pairs, desc="Creating preference labels"):
        # Get the actual segment starting indices
        idx_st_1 = segment_indices[i]
        idx_st_2 = segment_indices[j]
        
        # Create segment index lists
        idx_1 = [[k for k in range(idx_st_1, idx_st_1 + config.segment_size)]]
        idx_2 = [[k for k in range(idx_st_2, idx_st_2 + config.segment_size)]]
        
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
        
        multiple_ranked_list.append(single_ranked_list)
    
    print(f"Completed DTW augmentation collection:")
    print(f"  - Used {len(segment_indices)} segments from feedback collection")
    print(f"  - DTW matrix shape: {dtw_matrix.shape[0]}x{dtw_matrix.shape[1]}")
    print(f"  - Created {len(multiple_ranked_list)} preference comparisons")
    
    return multiple_ranked_list, dtw_matrix, segment_indices


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
        'segment_indices': [seg['start_idx'] for seg in segments],
        'use_relative_eef': use_relative_eef,
        'segment_size': config.segment_size
    }
    
    # Check if cache file exists
    if os.path.exists(dtw_matrix_path):
        print(f"Loading DTW matrix from cache: {dtw_matrix_path}")
        try:
            with open(dtw_matrix_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Validate cache metadata
            if (cached_data['metadata']['n_segments'] == cache_metadata['n_segments'] and
                cached_data['metadata']['segment_indices'] == cache_metadata['segment_indices'] and
                cached_data['metadata']['use_relative_eef'] == cache_metadata['use_relative_eef'] and
                cached_data['metadata']['segment_size'] == cache_metadata['segment_size']):
                
                print("Cache validation successful - using cached DTW matrix")
                return cached_data['dtw_matrix']
            else:
                print("Cache validation failed - recomputing DTW matrix")
                
        except Exception as e:
            print(f"Error loading cache: {e} - recomputing DTW matrix")
    
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
    
    # Set ensemble models to eval mode
    for member in range(reward_model.ensemble_num):
        reward_model.ensemble_model[member].eval()
    
    # Convert data to tensors if needed
    if not isinstance(obs_act_1, torch.Tensor):
        obs_act_1 = torch.tensor(obs_act_1, dtype=torch.float32)
    if not isinstance(obs_act_2, torch.Tensor):
        obs_act_2 = torch.tensor(obs_act_2, dtype=torch.float32)
    
    obs_act_1 = obs_act_1.to(reward_model.device)
    obs_act_2 = obs_act_2.to(reward_model.device)
    
    acquisition_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(len(obs_act_1)), desc="Computing acquisition scores"):
            # Get segments for this pair
            seg1 = obs_act_1[i]  # [segment_size, obs+act_dim]
            seg2 = obs_act_2[i]  # [segment_size, obs+act_dim]
            
            # Get predictions from each ensemble member
            ensemble_returns_1 = []
            ensemble_returns_2 = []
            
            for member_idx in range(reward_model.ensemble_num):
                # Use individual ensemble member
                member_rewards_1 = reward_model.ensemble_model[member_idx](seg1)  # [segment_size, 1]
                member_rewards_2 = reward_model.ensemble_model[member_idx](seg2)  # [segment_size, 1]
                
                # Calculate returns (sum over time dimension)
                member_return_1 = member_rewards_1.sum().item()
                member_return_2 = member_rewards_2.sum().item()
                
                ensemble_returns_1.append(member_return_1)
                ensemble_returns_2.append(member_return_2)
            
            # Convert to numpy arrays
            ensemble_returns_1 = np.array(ensemble_returns_1)
            ensemble_returns_2 = np.array(ensemble_returns_2)
            
            # Compute return deltas for each ensemble member
            ensemble_deltas = ensemble_returns_1 - ensemble_returns_2
            
            # Method 1: Use variance in return deltas as disagreement measure
            delta_variance = np.var(ensemble_deltas)
            
            # Method 2: Convert to preference probabilities and compute entropy
            # Apply softmax to get preference probabilities for each ensemble member
            ensemble_probs = []
            for delta in ensemble_deltas:
                # Convert return delta to preference probability using temperature scaling
                temperature = 1.0
                prob_1 = 1.0 / (1.0 + np.exp(-delta / temperature))
                prob_2 = 1.0 - prob_1
                ensemble_probs.append([prob_1, prob_2])
            
            ensemble_probs = np.array(ensemble_probs)  # [ensemble_num, 2]
            
            # Compute mean probability across ensemble
            mean_probs = np.mean(ensemble_probs, axis=0)  # [2]
            
            # Compute entropy of mean probabilities
            epsilon = 1e-8  # For numerical stability
            mean_probs = np.clip(mean_probs, epsilon, 1 - epsilon)
            entropy = -np.sum(mean_probs * np.log(mean_probs))
            
            # Compute disagreement: variance in probabilities across ensemble members
            prob_disagreement = np.mean(np.var(ensemble_probs, axis=0))
            
            # Combine measures: use entropy + disagreement as acquisition score
            acquisition_score = entropy + prob_disagreement + 0.1 * delta_variance
            
            acquisition_scores.append(acquisition_score)
    
    acquisition_scores = np.array(acquisition_scores)
    
    print(f"Acquisition score statistics:")
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


def compute_full_dtw_matrix(segments: List[Dict], use_relative_eef: bool, dtw) -> np.ndarray:
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