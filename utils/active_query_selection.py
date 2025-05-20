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
        logits = reward1 - reward2

        if method == "entropy":
            probs = torch.sigmoid(logits)
            p = torch.clamp(probs, min=1e-8, max=1 - 1e-8)
            entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
            score = entropy.mean().item()
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
    uncertainty_method="entropy",
    max_pairs=None,
    candidate_pairs=None,
):
    """Compute uncertainty for all possible segment pairs

    Args:
        reward_model: Trained reward model for uncertainty estimation
        segment_start_end: List of (start_idx, end_idx) for each segment
        data: TensorDict with observations and actions
        device: Device to run computation on
        uncertainty_method: Method for uncertainty estimation ("entropy", "disagreement", "random")
        max_pairs: Maximum number of pairs to select (None = select all pairs)
        candidate_pairs: List of candidate pairs to evaluate

    Returns:
        all_pairs_ranked: List of all segment pairs ranked by uncertainty (highest to lowest)
    """
    # Compute uncertainty scores
    print(f"Computing uncertainty for {len(candidate_pairs)} candidate pairs...")
    uncertainty_scores = compute_uncertainty_scores(
        reward_model,
        candidate_pairs,
        segment_start_end,
        data,
        method=uncertainty_method,
    )

    sorted_scores = sorted(uncertainty_scores, reverse=True)
    print("Top 5 uncertainty scores: ", sorted_scores[:5])

    # Sort pairs by uncertainty (highest to lowest)
    sorted_indices = np.argsort(uncertainty_scores)[::-1]

    # Limit number of pairs if specified
    if max_pairs is not None and max_pairs < len(sorted_indices):
        sorted_indices = sorted_indices[:max_pairs]

    # Get pairs in order of uncertainty
    all_pairs_ranked = [candidate_pairs[i] for i in sorted_indices]

    return all_pairs_ranked
