import torch
import numpy as np
import random
from tqdm import tqdm


def compute_entropy(probs):
    """Compute entropy of preference probabilities."""
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs = torch.clamp(probs, min=eps, max=1 - eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def compute_uncertainty_scores(
    model, segment_pairs, segment_start_end, data, method="entropy", device="cuda"   
):
    """Compute uncertainty scores for segment pairs using specified method.

    Args:
        model: Reward model (either single model or ensemble)
        segment_pairs: List of segment pair indices
        segment_start_end: List of (start_idx, end_idx) or (ep, start, end) tuples for each segment
        data: Data dictionary containing observations and actions
        method: Uncertainty estimation method ("entropy", "disagreement", or "random")

    Returns:
        uncertainty_scores: List of uncertainty scores for each segment pair
    """
    # If using random method, we can just return random scores
    if method == "random":
        return [random.random() for _ in range(len(segment_pairs))]

    # Extract observation and action keys
    obs_key = "obs" if "obs" in data else "state"
    action_key = "action"

    print(f"Predicting rewards for {len(segment_start_end)} segment pairs")

    # Process segments in batches for more efficient forward passes
    batch_size = 256
    segment_rewards = {}

    with torch.no_grad():
        # Process segments in batches
        for i in range(0, len(segment_start_end), batch_size):
            # Collect observations and actions for this batch
            batch_obs = []
            batch_actions = []
            batch_inds = np.arange(i, min(i + batch_size, len(segment_start_end)))

            for seg_idx in batch_inds:
                start, end = segment_start_end[seg_idx]
                obs = data[obs_key][start:end]
                act = data[action_key][start:end]
                batch_obs.append(obs)
                batch_actions.append(act)

            # Stack into batch tensors
            stacked_obs = torch.stack(batch_obs).to(device)
            stacked_actions = torch.stack(batch_actions).to(device)

            # [N, B, T]
            batch_rewards = model(stacked_obs, stacked_actions)

            # Store rewards for each segment in the batch
            for j, seg_idx in enumerate(batch_inds):
                segment_rewards[seg_idx] = batch_rewards[:, j]

    # Now compute uncertainty scores for each pair using pre-computed rewards
    uncertainty_scores = []

    for seg_idx1, seg_idx2 in tqdm(segment_pairs, desc=f"Computing {method} scores"):
        # Get pre-computed rewards
        reward1 = segment_rewards[seg_idx1]
        reward2 = segment_rewards[seg_idx2]

        if method == "entropy":
            # Compute preference probability
            logits = reward1 - reward2
            probs = torch.sigmoid(logits)

            # Compute entropy of binary prediction
            eps = 1e-8
            p = torch.clamp(probs, min=eps, max=1 - eps)
            entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)

            # Take mean entropy over timesteps
            score = entropy.mean().item()

        elif method == "disagreement":
            # Compute preference probability for each model
            logits = reward1 - reward2  # Shape: [num_models]
            # probs = torch.sigmoid(logits)  # Shape: [num_models]

            # probs = probs.sum(dim=-1) # sum over timesteps
            # # Disagreement is variance in preference probabilities across models
            # score = probs.var().item()

            # better to compute variance over raw reward differences
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
