import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym
import h5py
import numpy as np
import torch
from tqdm import tqdm
import glob
import itertools
import json
import os
import random
import time
from pathlib import Path
from utils.seed import set_seed
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np
from utils import dtw


def load_preferences_from_directory(pref_dir):
    """
    Load all preference files from a directory and return unique preferences.

    Args:
        pref_dir (str): Path to directory containing preference JSON files

    Returns:
        list: List of unique preferences with pair indices and choices
        dict: Statistics about preferences
    """

    print("=" * 50)
    print(f"Loading preferences from {pref_dir}")
    print("=" * 50)

    all_preferences = []
    stats = {
        "total_files": 0,
        "total_preferences": 0,
        "unique_pairs": set(),
        "preference_counts": {"A": 0, "B": 0, "equal": 0},
    }

    # Load all JSON files in the directory
    for pref_file in glob.glob(os.path.join(pref_dir, "*.json")):
        try:
            with open(pref_file, "r") as f:
                data = json.load(f)
                stats["total_files"] += 1

                # Extract preferences from each file
                for pref in data.get("preferences", []):
                    pair_idx = pref.get("pair_index")
                    preference = pref.get("preference")
                    timestamp = pref.get("timestamp")

                    if pair_idx is not None and preference:
                        all_preferences.append(
                            {
                                "pair_index": pair_idx,
                                "preference": preference,
                                "timestamp": timestamp,
                            }
                        )
                        stats["unique_pairs"].add(pair_idx)
                        stats["preference_counts"][preference] += 1
                        stats["total_preferences"] += 1

        except Exception as e:
            print(f"Error loading preferences from {pref_file}: {e}")
            continue

    # Sort preferences by timestamp and get unique latest preference for each pair
    all_preferences.sort(key=lambda x: x["timestamp"])
    all_preferences = all_preferences[
        500:
    ]  # TODO: remove this, just hack to use the second collected prefs

    unique_preferences = {}
    for pref in all_preferences:
        unique_preferences[pref["pair_index"]] = pref

    # Convert to list of unique preferences
    unique_pref_list = list(unique_preferences.values())

    print(
        f"Loaded {len(unique_pref_list)} unique preferences from {stats['total_files']} files"
    )
    print(
        f"Preference distribution: A: {stats['preference_counts']['A']}, "
        f"B: {stats['preference_counts']['B']}, "
        f"Equal: {stats['preference_counts']['equal']}"
    )

    return unique_pref_list, stats


def get_feedbacks(data_path, num_prefs, human=False):
    # TODO: implement validation set for human preferences (use the human prefs or gt?)
    """Get ground truth or human feedbacks from the dataset.

    Args:
        data_path: Path to the dataset
        num_prefs: Number of preferences to return. If None, return all preferences.
                   If less than total available, randomly sample num_prefs.
    """
    data_path = Path(data_path)
    seg_indices_path = data_path.parent / "segment_start_end_indices.npy"
    seg_pairs_path = data_path.parent / "segment_pairs.npy"
    val_indices_path = data_path.parent / "val_segment_start_end_indices.npy"
    val_pairs_path = data_path.parent / "val_segment_pairs.npy"
    prefs_path = data_path.parent / "preferences"

    # Load everything
    seg_indices = np.load(seg_indices_path, allow_pickle=True)
    seg_pairs = np.load(seg_pairs_path, allow_pickle=True)
    val_seg_indices = np.load(val_indices_path, allow_pickle=True)
    val_seg_pairs = np.load(val_pairs_path, allow_pickle=True)

    # Initialize lists to store processed data
    labels = []
    idx_st_1 = []
    idx_st_2 = []
    val_labels = []
    val_idx_st_1 = []
    val_idx_st_2 = []

    # Get gt rewards from the dataset
    with h5py.File(data_path, "r") as f:
        sorted_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        # Collect all rewards in order
        all_rewards = []
        for demo_key in sorted_keys:
            rewards = f["data"][demo_key]["rewards"]
            all_rewards.extend(rewards)
    all_rewards = np.array(all_rewards)

    # Train set
    if human:
        prefs, _ = load_preferences_from_directory(prefs_path)
        print(f"Loaded {len(prefs)} human preferences")
    else:
        # Generate ground truth preferences for all segment pairs
        equal_threshold = 1e-5  # Threshold for considering equal preferences
        prefs = []

        for i, (seg1, seg2) in enumerate(
            tqdm(
                seg_pairs,
                desc="Generating ground truth preferences from rewards (TRAIN)",
            )
        ):
            seg1_start, seg1_end = seg_indices[seg1]
            seg2_start, seg2_end = seg_indices[seg2]

            # Calculate reward for each segment
            seg1_reward = np.sum(all_rewards[seg1_start:seg1_end])
            seg2_reward = np.sum(all_rewards[seg2_start:seg2_end])

            # Create preference based on rewards
            if np.abs(seg1_reward - seg2_reward) < equal_threshold:  # Equal rewards
                preference = "equal"
            elif seg1_reward > seg2_reward:
                preference = "A"  # First segment preferred
            else:
                preference = "B"  # Second segment preferred

            prefs.append({"pair_index": i, "preference": preference})
        print(f"Generated {len(prefs)} ground truth preferences from rewards (TRAIN)")

    # Load the val set as well TODO: for now, we are validating on gt prefs for human feedbacks
    val_prefs = []

    for i, (seg1, seg2) in enumerate(
        tqdm(
            val_seg_pairs, desc="Generating ground truth preferences from rewards (VAL)"
        )
    ):
        seg1_start, seg1_end = val_seg_indices[seg1]
        seg2_start, seg2_end = val_seg_indices[seg2]

        # Calculate reward for each segment
        seg1_reward = np.sum(all_rewards[seg1_start:seg1_end])
        seg2_reward = np.sum(all_rewards[seg2_start:seg2_end])

        # Create preference based on rewards (no equal prefs for validation)
        if seg1_reward > seg2_reward:
            preference = "A"  # First segment preferred
        else:
            preference = "B"  # Second segment preferred

        val_prefs.append({"pair_index": i, "preference": preference})
    print(f"Generated {len(val_prefs)} ground truth preferences from rewards (VAL)")

    # Randomly sample preferences if needed
    if num_prefs is not None and len(prefs) > num_prefs:
        print(
            f"Sampling {num_prefs} preferences from {len(prefs)} available preferences"
        )
        indices = np.random.choice(len(prefs), num_prefs, replace=False)
        prefs = [prefs[i] for i in indices]
    else:
        print(f"Using all {len(prefs)} preferences")

    for pref in prefs:
        pair_ind = pref["pair_index"]
        preference = pref["preference"]  # Should be 'A', 'B', or 'equal'

        seg1, seg2 = seg_pairs[pair_ind]

        # Get the actual segment data using indices
        seg1_start, seg1_end = seg_indices[seg1]
        seg2_start, seg2_end = seg_indices[seg2]

        # Convert preference to numeric value
        # 'A' means first segment preferred, 'B' means second segment preferred, 'equal' means equal preference
        if preference == "equal":
            binary_pref = 0.5
        else:
            binary_pref = 1 if preference == "A" else 0

        # Store data in the format used by learn_reward.py
        labels.append(binary_pref)
        idx_st_1.append(seg1_start)
        idx_st_2.append(seg2_start)

    # Sample val prefs as well
    if num_prefs is not None and len(val_prefs) > num_prefs:
        indices = np.random.choice(len(val_prefs), num_prefs, replace=False)
        val_prefs = [val_prefs[i] for i in indices]
    else:
        print(f"Using all {len(val_prefs)} validation preferences")

    for pref in val_prefs:  # use same amount of prefs for val
        pair_ind = pref["pair_index"]
        preference = pref["preference"]  # Should be 'A', 'B', or 'equal'

        seg1, seg2 = val_seg_pairs[pair_ind]

        # Get the actual segment data using indices
        seg1_start, seg1_end = val_seg_indices[seg1]
        seg2_start, seg2_end = val_seg_indices[seg2]

        # Convert preference to numeric value
        # 'A' means first segment preferred, 'B' means second segment preferred, 'equal' means equal preference
        if preference == "equal":
            binary_pref = 0.5
        else:
            binary_pref = 1 if preference == "A" else 0

        # Store data in the format used by learn_reward.py
        val_labels.append(binary_pref)
        val_idx_st_1.append(seg1_start)
        val_idx_st_2.append(seg2_start)

    # Convert to numpy arrays
    labels = np.array(labels)
    idx_st_1 = np.array(idx_st_1)
    idx_st_2 = np.array(idx_st_2)
    val_labels = np.array(val_labels)
    val_idx_st_1 = np.array(val_idx_st_1)
    val_idx_st_2 = np.array(val_idx_st_2)

    return labels, idx_st_1, idx_st_2, val_labels, val_idx_st_1, val_idx_st_2


def obtain_labels(
    dataset, idx_1, idx_2, segment_size=25, threshold=0.5, noise=0.0, labels=None
):
    idx_1 = np.array(idx_1)
    idx_2 = np.array(idx_2)

    if (
        labels is not None
    ):  # if human feedback labels are not provided rely on gt rewards
        labels = []
        reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
        reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
        # labels = np.where(reward_1 < reward_2, 1, 0)
        labels = np.where(reward_1 > reward_2, 1, 0)
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


def get_reward_model_predictions(
    reward_model, seg1, seg2, return_ensemble_predictions=False, debug=False
):
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
        if hasattr(reward_model, "ensemble_model"):
            print(f"    ensemble_model value: {reward_model.ensemble_model}")
            print(f"    ensemble_model is None: {reward_model.ensemble_model is None}")
        print(f"    Has net attr: {hasattr(reward_model, 'net')}")
        if hasattr(reward_model, "net"):
            print(f"    net value: {reward_model.net}")
        print(
            f"    Has single_model_forward: {hasattr(reward_model, 'single_model_forward')}"
        )
        print(
            f"    Has ensemble_model_forward: {hasattr(reward_model, 'ensemble_model_forward')}"
        )

    # Set model to eval mode
    if hasattr(reward_model, "model") and reward_model.model is not None:
        reward_model.model.eval()
    elif (
        hasattr(reward_model, "ensemble_model")
        and reward_model.ensemble_model is not None
    ):
        for member in reward_model.ensemble_model:
            member.eval()
    elif hasattr(reward_model, "net") and reward_model.net is not None:
        reward_model.net.eval()

    # Move segments to device
    seg1 = seg1.to(reward_model.device)
    seg2 = seg2.to(reward_model.device)

    # Get predictions based on model type
    if (
        hasattr(reward_model, "ensemble_model")
        and reward_model.ensemble_model is not None
        and len(reward_model.ensemble_model) > 0
    ):
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
        if (
            hasattr(reward_model, "ensemble_model_forward")
            and hasattr(reward_model, "ensemble_model")
            and reward_model.ensemble_model is not None
        ):
            # Use ensemble_model_forward if available and ensemble_model exists
            rewards_1 = reward_model.ensemble_model_forward(seg1)
            rewards_2 = reward_model.ensemble_model_forward(seg2)
        elif hasattr(reward_model, "single_model_forward"):
            # Try single_model_forward method
            rewards_1 = reward_model.single_model_forward(seg1)
            rewards_2 = reward_model.single_model_forward(seg2)
        elif hasattr(reward_model, "net") and reward_model.net is not None:
            # Use single net if available
            rewards_1 = reward_model.net(seg1)
            rewards_2 = reward_model.net(seg2)
        else:
            raise ValueError(
                f"Could not find a valid model to use for prediction in {reward_model}"
            )

        pred_ret_1 = rewards_1.sum().item()
        pred_ret_2 = rewards_2.sum().item()

        if return_ensemble_predictions:
            # Return as single-element lists for consistency
            return [pred_ret_1], [pred_ret_2]
        else:
            return pred_ret_1, pred_ret_2


def compute_dtw_matrix_cross(
    dataset, seg_indices, cross_dataset, cross_seg_indices, config
):
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

    seg_indices = seg_indices[:, 0]  # Use only start indices for segments
    cross_seg_indices = cross_seg_indices[:, 0]

    # Initialize distance matrix
    n = len(seg_indices)
    n_cross = len(cross_seg_indices)
    distance_matrix = np.full((n, n_cross), np.inf)  # default value is inf

    # Compute statistics for progress tracking
    min_dist = float("inf")
    min_goal_dist = float("inf")
    max_dist = float("-inf")
    max_goal_dist = float("-inf")
    sum_dist = 0
    sum_goal_dist = 0
    count = 0

    total_comparisons = n * n_cross
    with tqdm(
        total=total_comparisons, desc="Computing cross-dataset DTW distances"
    ) as pbar:
        for i, idx in enumerate(seg_indices):
            for j, cross_idx in enumerate(cross_seg_indices):
                # Extract EE positions (assuming first 3 dimensions are EE positions)
                main_query = dataset["observations"][
                    idx : idx + config.segment_size, :3
                ]
                target_ref = cross_dataset["observations"][
                    cross_idx : cross_idx + config.segment_size, :3
                ]

                if config.use_goal_pos:
                    goal_main_query = np.concatenate(
                        (
                            main_query,
                            dataset["goal_points"][idx : idx + config.segment_size, :],
                        ),
                        axis=1,
                    )
                    goal_target_ref = np.concatenate(
                        (
                            target_ref,
                            cross_dataset["goal_points"][
                                cross_idx : cross_idx + config.segment_size, :
                            ],
                        ),
                        axis=1,
                    )

                if config.use_relative_eef:
                    main_query = main_query[1:] - main_query[:-1]
                    target_ref = target_ref[1:] - target_ref[:-1]

                try:
                    cost, _ = dtw.get_single_match(main_query, target_ref)
                    goal_cost, _ = (
                        dtw.get_single_match(goal_main_query, goal_target_ref)
                        if config.use_goal_pos
                        else (cost, None)
                    )
                except:
                    print(
                        f"Error computing DTW for segments {idx} and {cross_idx}. Something is wrong."
                    )
                    import ipdb

                    ipdb.set_trace()

                # TODO: manually set values for now based on
                # Cross-dataset DTW distance statistics - Min: 4.41, Max: 267.66, Avg: 68.63
                # Cross-dataset DTW goal distance statistics - Min: 10.90, Max: 275.71, Avg: 93.83
                traj_norm = (cost - 4.41) / (267.66 - 4.41)
                goal_norm = (goal_cost - 10.90) / (275.71 - 10.90)

                final_cost = traj_norm + goal_norm

                distance_matrix[i, j] = final_cost

                # Update statistics
                min_dist = min(min_dist, cost)
                max_dist = max(max_dist, cost)
                sum_dist += cost
                if config.use_goal_pos:
                    min_goal_dist = min(min_goal_dist, goal_cost)
                    max_goal_dist = max(max_goal_dist, goal_cost)
                    sum_goal_dist += goal_cost

                count += 1

                pbar.update(1)

    avg_dist = sum_dist / count
    avg_goal_dist = sum_goal_dist / count if count > 0 else 0
    print(
        f"Cross-dataset DTW distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}"
    )
    print(
        f"Cross-dataset DTW goal distance statistics - Min: {min_goal_dist:.2f}, Max: {max_goal_dist:.2f}, Avg: {avg_goal_dist:.2f}"
    )
    import ipdb

    ipdb.set_trace()

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
        top_k_idx1 = np.argsort(distances1)[: config.dtw_k_augment]
        top_k_idx2 = np.argsort(distances2)[: config.dtw_k_augment]

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
    print(
        f"Augmented preference distribution: {np.bincount(np.argmax(augmented_labels, axis=1))}"
    )

    return augmented_labels, augmented_idx_st_1, augmented_idx_st_2
