import torch
import numpy as np
import random
from tqdm import tqdm


def compute_entropy(probs):
    """Compute entropy of preference probabilities."""
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs = torch.clamp(probs, min=eps, max=1-eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def compute_uncertainty_scores(model, segment_pairs, segment_indices, data, device, method="entropy"):
    """Compute uncertainty scores for segment pairs using specified method.
    
    Args:
        model: Reward model (either single model or ensemble)
        segment_pairs: List of segment pair indices
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        data: Data dictionary containing observations and actions
        device: Device to run computation on
        method: Uncertainty estimation method ("entropy", "disagreement", or "random")
        
    Returns:
        uncertainty_scores: List of uncertainty scores for each segment pair
    """
    # Extract observation and action fields
    obs_key = "obs" if "obs" in data else "state"
    action_key = "action"
    
    # Ensure data is on CPU for indexing operations
    data_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    uncertainty_scores = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    num_batches = (len(segment_pairs) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc=f"Computing {method} scores"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(segment_pairs))
            batch_pairs = segment_pairs[batch_start:batch_end]
            
            batch_scores = []
            
            for pair_idx, (seg_idx1, seg_idx2) in enumerate(batch_pairs):
                # Get segment indices
                start1, end1 = segment_indices[seg_idx1]
                start2, end2 = segment_indices[seg_idx2]
                
                # Extract observations and actions from CPU data
                obs1 = data_cpu[obs_key][start1:end1].clone()
                actions1 = data_cpu[action_key][start1:end1-1].clone()
                obs2 = data_cpu[obs_key][start2:end2].clone()
                actions2 = data_cpu[action_key][start2:end2-1].clone()
                
                # Ensure observations and actions have same length
                min_len1 = min(obs1.shape[0]-1, actions1.shape[0])
                min_len2 = min(obs2.shape[0]-1, actions2.shape[0])
                
                obs1 = obs1[:min_len1]
                actions1 = actions1[:min_len1]
                obs2 = obs2[:min_len2]
                actions2 = actions2[:min_len2]
                
                # Move to device only after slicing
                obs1 = obs1.to(device)
                actions1 = actions1.to(device)
                obs2 = obs2.to(device)
                actions2 = actions2.to(device)
                
                if method == "random":
                    # Random sampling - just assign random score
                    score = random.random()
                
                elif method == "entropy":
                    # Entropy-based uncertainty
                    if hasattr(model, 'mean_reward'):
                        # Use mean prediction from ensemble
                        reward1 = model.mean_reward(obs1, actions1)
                        reward2 = model.mean_reward(obs2, actions2)
                    else:
                        # Single model
                        reward1 = model(obs1, actions1)
                        reward2 = model(obs2, actions2)
                    
                    # Compute preference probability
                    logits = reward1 - reward2
                    probs = torch.sigmoid(logits)
                    
                    # Compute entropy of binary prediction
                    probs_2d = torch.stack([probs, 1 - probs], dim=0)
                    score = compute_entropy(probs_2d).item()
                
                elif method == "disagreement":
                    # Disagreement-based uncertainty (requires ensemble)
                    if not hasattr(model, 'models'):
                        raise ValueError("Disagreement method requires an ensemble model")
                    
                    # Get reward predictions from all models
                    rewards1 = model(obs1, actions1)  # Shape: [num_models]
                    rewards2 = model(obs2, actions2)  # Shape: [num_models]
                    
                    # Compute preference probability for each model
                    logits = rewards1 - rewards2  # Shape: [num_models]
                    probs = torch.sigmoid(logits)  # Shape: [num_models]
                    
                    # Disagreement is variance in preference probabilities
                    score = probs.var().item()
                
                batch_scores.append(score)
            
            uncertainty_scores.extend(batch_scores)
    
    return uncertainty_scores


def select_uncertain_pairs(uncertainty_scores, segment_pairs, k):
    """Select top-k most uncertain segment pairs.
    
    Args:
        uncertainty_scores: List of uncertainty scores for each segment pair
        segment_pairs: List of segment pair indices
        k: Number of pairs to select
        
    Returns:
        selected_pairs: List of selected segment pairs
        selected_indices: Indices of selected pairs in the original list
    """
    # Convert to numpy for easier manipulation
    scores = np.array(uncertainty_scores)
    
    # Ensure k is not larger than the number of available pairs
    k = min(k, len(segment_pairs))
    
    if k == 0:
        print("Warning: No pairs available to select")
        return [], []
    
    # Get indices of top-k highest uncertainty scores
    selected_indices = np.argsort(scores)[-k:]
    
    # Get corresponding segment pairs
    selected_pairs = [segment_pairs[i] for i in selected_indices]
    
    return selected_pairs, selected_indices.tolist()


def get_ground_truth_preferences(segment_pairs, segment_indices, rewards):
    """Generate ground truth preferences based on cumulative rewards.
    
    Args:
        segment_pairs: List of segment pair indices
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        rewards: Tensor of reward values for all transitions
        
    Returns:
        preferences: List of preferences (1 = first segment preferred, 2 = second segment preferred)
    """
    preferences = []
    
    # Ensure rewards is on CPU for indexing
    rewards_cpu = rewards.cpu() if isinstance(rewards, torch.Tensor) else rewards
    
    for seg_idx1, seg_idx2 in segment_pairs:
        # Get segment indices
        start1, end1 = segment_indices[seg_idx1]
        start2, end2 = segment_indices[seg_idx2]
        
        # Calculate cumulative rewards for each segment
        reward1 = rewards_cpu[start1:end1].sum().item()
        reward2 = rewards_cpu[start2:end2].sum().item()
        
        # Determine preference (1 = first segment preferred, 2 = second segment preferred)
        if reward1 > reward2:
            preferences.append(1)
        else:
            preferences.append(2)
    
    return preferences


def create_initial_dataset(segment_pairs, segment_indices, preferences, data, initial_size):
    """Create initial dataset with a small number of labeled pairs.
    
    Args:
        segment_pairs: List of segment pair indices
        segment_indices: List of (start_idx, end_idx) tuples for each segment
        preferences: List of preferences (1 = first segment preferred, 2 = second segment preferred)
        data: Data dictionary containing observations and actions
        initial_size: Initial number of labeled pairs
        
    Returns:
        labeled_pairs: List of initially labeled segment pairs
        labeled_preferences: List of preferences for initially labeled pairs
        unlabeled_pairs: List of remaining unlabeled segment pairs
        unlabeled_indices: Indices of unlabeled pairs in the original list
    """
    # Ensure initial_size is not larger than the number of available pairs
    initial_size = min(initial_size, len(segment_pairs))
    
    if initial_size == 0:
        print("Warning: No pairs available for initial dataset")
        return [], [], segment_pairs, list(range(len(segment_pairs)))
    
    # Randomly select initial pairs
    all_indices = list(range(len(segment_pairs)))
    
    # Handle case where initial_size is very close to total number of pairs
    if initial_size >= len(segment_pairs) - 1:
        print(f"Warning: Initial size {initial_size} is almost equal to total pairs {len(segment_pairs)}.")
        print("Using all pairs as labeled data.")
        labeled_indices = all_indices
        unlabeled_indices = []
    else:
        labeled_indices = random.sample(all_indices, initial_size)
        unlabeled_indices = [i for i in all_indices if i not in labeled_indices]
    
    # Get labeled pairs and preferences
    labeled_pairs = [segment_pairs[i] for i in labeled_indices]
    labeled_preferences = [preferences[i] for i in labeled_indices]
    
    # Get unlabeled pairs
    unlabeled_pairs = [segment_pairs[i] for i in unlabeled_indices]
    
    print(f"Created initial dataset with {len(labeled_pairs)} labeled pairs and {len(unlabeled_pairs)} unlabeled pairs")
    
    return labeled_pairs, labeled_preferences, unlabeled_pairs, unlabeled_indices


def select_active_preference_query(segments, segment_indices, data, reward_model, device, 
                                  n_candidates=100, uncertainty_method="entropy"):
    """Select the next preference query actively based on model uncertainty.
    
    Args:
        segments: List of trajectory segments
        segment_indices: List of (start_idx, end_idx) for each segment
        data: TensorDict with observations
        reward_model: Trained reward model for uncertainty estimation
        device: Device to run computation on
        n_candidates: Number of random candidate pairs to consider
        uncertainty_method: Method for uncertainty estimation ("entropy", "disagreement", "random")
        
    Returns:
        tuple: (i, j) indices of the selected segments for comparison
    """
    n_segments = len(segments)
    
    # Generate random candidate pairs
    candidate_pairs = []
    for _ in range(n_candidates):
        i, j = random.sample(range(n_segments), 2)
        candidate_pairs.append((i, j))
    
    print(f"Computing uncertainty for {len(candidate_pairs)} candidate pairs...")
    
    # Compute uncertainty scores
    uncertainty_scores = compute_uncertainty_scores(
        reward_model,
        candidate_pairs,
        segment_indices,
        data,
        device,
        method=uncertainty_method
    )
    
    # Select the most uncertain pair
    selected_pairs, _ = select_uncertain_pairs(uncertainty_scores, candidate_pairs, k=1)
    
    if not selected_pairs:
        # Fallback to random selection if no pairs were selected
        print("Warning: No pairs selected based on uncertainty. Falling back to random selection.")
        return random.sample(range(n_segments), 2)
    
    return selected_pairs[0]


def select_uncertain_pairs_comprehensive(reward_model, segments, segment_indices, data, device, 
                                       uncertainty_method="entropy", max_pairs=None,
                                       use_random_candidate_sampling=True, n_candidates=100):
    """Unified approach for uncertainty-based pair selection.
    
    This function provides two approaches:
    1. Compute uncertainty for all possible segment pairs (more thorough but expensive)
    2. Sample random candidates and compute uncertainty only for those (more efficient)
    
    Args:
        reward_model: Trained reward model for uncertainty estimation
        segments: List of trajectory segments
        segment_indices: List of (start_idx, end_idx) for each segment
        data: TensorDict with observations and actions
        device: Device to run computation on
        uncertainty_method: Method for uncertainty estimation ("entropy", "disagreement", "random")
        max_pairs: Maximum number of pairs to select (None = select all pairs)
        use_random_candidate_sampling: If True, sample random candidates; if False, evaluate all pairs
        n_candidates: Number of random candidate pairs to consider if using sampling
        
    Returns:
        all_pairs_ranked: List of all segment pairs ranked by uncertainty (highest to lowest)
    """
    n_segments = len(segments)
    
    # Determine approach based on parameters
    if use_random_candidate_sampling:
        # Approach 1: Sample random candidates (more efficient)
        print(f"Using random candidate sampling with {n_candidates} candidate pairs")
        candidate_pairs = []
        for _ in range(n_candidates):
            i, j = random.sample(range(n_segments), 2)
            candidate_pairs.append((i, j))
    else:
        # Approach 2: Generate all possible segment pairs (more thorough but expensive)
        print(f"Generating all possible pairs from {n_segments} segments")
        candidate_pairs = [(i, j) for i in range(n_segments) for j in range(i+1, n_segments)]
        
        # If too many pairs, warn and limit
        if len(candidate_pairs) > 10000:
            print(f"Warning: Generated {len(candidate_pairs)} pairs, which may be memory intensive")
    
    # Compute uncertainty scores
    print(f"Computing uncertainty for {len(candidate_pairs)} candidate pairs...")
    uncertainty_scores = compute_uncertainty_scores(
        reward_model,
        candidate_pairs,
        segment_indices,
        data,
        device,
        method=uncertainty_method
    )
    
    # Sort pairs by uncertainty (highest to lowest)
    sorted_indices = np.argsort(uncertainty_scores)[::-1]
    
    # Limit number of pairs if specified
    if max_pairs is not None and max_pairs < len(sorted_indices):
        sorted_indices = sorted_indices[:max_pairs]
    
    # Get pairs in order of uncertainty
    all_pairs_ranked = [candidate_pairs[i] for i in sorted_indices]
    
    return all_pairs_ranked 