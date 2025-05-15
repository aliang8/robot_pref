"""Compute the full DTW distance matrix for trajectory segments."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

import utils.dtw as dtw
from trajectory_utils import process_data_trajectories, segment_trajectory


def compute_dtw_distance_matrix(segments):
    """Compute DTW distance matrix between segments.

    Args:
        segments: List of segments to compute distances between

    Returns:
        distance_matrix: Matrix of DTW distances between segments
    """
    n_segments = len(segments)
    print(f"Computing DTW distance matrix for {n_segments} segments")

    distance_matrix = np.zeros((n_segments, n_segments))

    # Track statistics for distances
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
                # Extract observations for DTW comparison
                query = segments[i]["obs"].numpy()
                reference = segments[j]["obs"].numpy()

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

    if non_finite_count > 0:
        print(
            f"WARNING: {non_finite_count} distances were computed using fallback method due to non-finite DTW values"
        )

    return distance_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Compute DTW distance matrix between trajectory segments"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/scr2/shared/pref/datasets/robomimic/can/mg_image_dense.pt",
        help="Path to trajectory data and save path",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=64,
        help="Length of each trajectory segment",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DTW matrix file if it exists",
    )
    args = parser.parse_args()

    data_path = args.data_path
    segment_length = args.segment_length
    overwrite = args.overwrite

    dtw_matrix_file = Path(data_path).parent / f"dtw_matrix_{segment_length}.pkl"

    print(
        f"Computing DTW matrix for {data_path} with segment length {segment_length} and saving to {dtw_matrix_file}"
    )

    if os.path.exists(dtw_matrix_file) and not overwrite:
        print(f"DTW matrix file already exists: {dtw_matrix_file}")
        with open(dtw_matrix_file, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                dtw_matrix = data
            else:
                dtw_matrix = data
                print("Warning: Loaded DTW matrix does not contain segment IDs")
    else:
        if os.path.exists(dtw_matrix_file) and overwrite:
            print(f"Overwriting existing DTW matrix file: {dtw_matrix_file}")

        # Load data into trajectories
        trajectories = process_data_trajectories(data_path)

        # Dynamic segment length calculation
        segments_per_trajectory = len(trajectories[0]["obs"]) // segment_length + 1
        print(f"Using segments per trajectory: {segments_per_trajectory}")

        # Segment trajectories
        segments = []
        for trajectory in trajectories:
            segments.extend(
                segment_trajectory(trajectory, segment_length, segments_per_trajectory)
            )

        assert len(segments) == len(trajectories) * segments_per_trajectory, (
            f"Total segments: {len(segments)}, should equal len(trajectories) {len(trajectories)} * segments_per_trajectory {segments_per_trajectory} = {len(trajectories) * segments_per_trajectory}"
        )

        # Compute DTW distance matrix
        dtw_matrix = compute_dtw_distance_matrix(segments)

        # Save to cache
        print(f"Saving DTW matrix and segment IDs to cache: {dtw_matrix_file}")
        with open(dtw_matrix_file, "wb") as f:
            pickle.dump((dtw_matrix), f)

    print(f"DTW matrix shape: {dtw_matrix.shape}")

if __name__ == "__main__":
    main()
