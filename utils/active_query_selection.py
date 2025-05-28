import random

import numpy as np
import torch
from tqdm import tqdm


def compute_uncertainty_scores(
    model, segment_pairs, segment_start_end, data, method="entropy", device="cuda"
):
    """
    Compute uncertainty scores for segment pairs using the specified method.

    Args:
        model: Reward model (single or ensemble).
        segment_pairs: List of (seg_idx1, seg_idx2) tuples.
        segment_start_end: List of (start_idx, end_idx) tuples for each segment.
        data: Data dictionary with observations and actions.
        method: Uncertainty estimation method ("entropy", "disagreement", "random").
        device: Device to run computation on.

    Returns:
        List of uncertainty scores for each segment pair.
    """
    if method == "random":
        return [random.random() for _ in range(len(segment_pairs))]

    obs_key = "obs" if "obs" in data else "state"
    action_key = "action"

    batch_size = 256
    segment_rewards = {}

    # Precompute rewards for all segments
    for i in range(0, len(segment_start_end), batch_size):
        batch_inds = np.arange(i, min(i + batch_size, len(segment_start_end)))
        batch_obs = []
        batch_actions = []
        for seg_idx in batch_inds:
            start, end = segment_start_end[seg_idx]
            batch_obs.append(data[obs_key][start:end])
            batch_actions.append(data[action_key][start:end])
        stacked_obs = torch.stack(batch_obs).to(device)
        stacked_actions = torch.stack(batch_actions).to(device)
        with torch.no_grad():
            batch_rewards = model(stacked_obs, stacked_actions)
        for j, seg_idx in enumerate(batch_inds):
            segment_rewards[seg_idx] = batch_rewards[:, j]

    uncertainty_scores = []
    for seg_idx1, seg_idx2 in tqdm(segment_pairs, desc=f"Computing {method} scores"):
        reward1 = segment_rewards[seg_idx1]
        reward2 = segment_rewards[seg_idx2]

        ret1 = reward1.sum(dim=1)
        ret2 = reward2.sum(dim=1)

        logits = ret1 - ret2

        if method == "entropy":
            probs = torch.sigmoid(logits)
            p = torch.clamp(probs, min=1e-4, max=1 - 1e-4)
            mean_p = p.mean()  # Average over ensemble
            entropy = -mean_p * torch.log(mean_p) - (1 - mean_p) * torch.log(1 - mean_p)
            score = entropy.item()
        elif method == "disagreement":
            score = logits.var().item()
        else:
            raise ValueError(f"Invalid method: {method}")
        
        uncertainty_scores.append(score)

    return uncertainty_scores


def select_active_pref_query(
    reward_model, 
    segment_start_end, 
    data, 
    uncertainty_method="disagreement", 
    selection_pairs=1,
    candidate_pairs=None,
    dtw_matrix=None,
    uncertainty_subsample=50  # Number of top uncertain pairs to consider for DTW diversity
):
    """Select pairs with highest uncertainty for active learning.
    
    Args:
        reward_model: Trained ensemble reward model
        segment_start_end: List of segment start/end indices
        data: Dictionary of state-action data
        uncertainty_method: Method for computing uncertainty ("disagreement", "entropy", or "random")
        selection_pairs: Pairs to select based on uncertainty
        candidate_pairs: List of candidate segment pairs to consider (optional)
        dtw_matrix: Pre-computed DTW distance matrix (optional)
        uncertainty_subsample: Number of top uncertain pairs to consider for DTW-based diversity selection
        
    Returns:
        List of selected segment pairs
    """

    print(f"Computing uncertainty scores for {len(candidate_pairs)} candidate pairs...")
    uncertainty_scores = np.array(compute_uncertainty_scores(
        reward_model,
        candidate_pairs,
        segment_start_end,
        data,
        method=uncertainty_method,
    ))

    # Sort pairs by uncertainty scores
    sorted_indices = np.argsort(uncertainty_scores)[::-1]
    top_scores = uncertainty_scores[sorted_indices[:5]]
    print(f"Top uncertainty scores: {top_scores} indices: {sorted_indices[:5]}")

    # TODO: Implement DTW-based diversity selection if dtw_matrix is provided
    # Select top uncertain_subsample pairs based on uncertainty
    # top_uncertain_indices = sorted_indices[:uncertainty_subsample]
    # top_uncertain_pairs = [candidate_pairs[i] for i in top_uncertain_indices]
    # # If DTW matrix is provided, use it to prioritize diverse pairs among the uncertain ones
    # if dtw_matrix is not None and len(top_uncertain_pairs) > max_pairs:
    #     print(f"Using DTW distance to select diverse pairs from top {len(top_uncertain_pairs)} uncertain pairs")
        
    #     # Calculate DTW distances for each pair
    #     dtw_distances = []
    #     for pair_idx, (i, j) in enumerate(top_uncertain_pairs):
    #         dtw_dist = dtw_matrix[i, j]
    #         dtw_distances.append((pair_idx, dtw_dist))
        
    #     # Sort by DTW distance (higher = more dissimilar = better)
    #     dtw_distances.sort(key=lambda x: x[1], reverse=True)
        
    #     # Select the pairs with highest DTW distances
    #     top_diverse_indices = [dtw_distances[i][0] for i in range(min(max_pairs, len(dtw_distances)))]
    #     selected_pairs = [top_uncertain_pairs[i] for i in top_diverse_indices]
        
    #     # Print the selected pair information
    #     for idx, pair in enumerate(selected_pairs):
    #         i, j = pair
    #         print(f"Selected pair {idx+1}/{len(selected_pairs)}: {pair}")
    #         uncertainty_score = uncertainty_scores[sorted_indices[top_uncertain_indices[top_diverse_indices[idx]]]]
    #         print(f"  Uncertainty: {uncertainty_score:.4f}")
    #         print(f"  DTW distance: {dtw_matrix[i, j]:.4f}")
            
    #     return selected_pairs, uncertainty_score
    # else:

    # Select top pairs and their scores
    selected_pairs = [candidate_pairs[i] for i in sorted_indices[:selection_pairs]]
    selected_scores = uncertainty_scores[sorted_indices[:selection_pairs]]

    print(f"Selected index: {sorted_indices[:selection_pairs]}, Selected pair: {selected_pairs}, score: {selected_scores}")
    print(f"Selected {selection_pairs} pairs based on uncertainty method '{uncertainty_method}'")
    
    return selected_pairs, selected_scores
