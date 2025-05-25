"""Compute the full DTW distance matrix for trajectory segments."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import utils.dtw as dtw
from utils.data import load_tensordict, segment_episodes, segment_episodes_dynamic


def compute_dtw_distance_matrix(segments, use_relative_eef: bool):
    """Compute DTW distance matrix between segments.

    Args:
        segments: List of segments to compute distances between

    Returns:
        distance_matrix: Matrix of DTW distances between segments
    """
    n_segments = len(segments)
    print(f"Computing DTW distance matrix for {n_segments} segments")

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
                # EE positions
                query = segments[i]["obs"].numpy()[:, :3]
                reference = segments[j]["obs"].numpy()[:, :3]

                # Relative
                if use_relative_eef:
                    query = query[1:] - query[:-1]
                    reference = reference[1:] - reference[:-1]

                cost, _ = dtw.get_single_match(query, reference)

                distance_matrix[i, j] = cost
                distance_matrix[j, i] = cost

                pbar.update(1)

    if count > 0:
        avg_dist = sum_dist / count
        print(
            f"Distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}"
        )
    if non_finite_count > 0:
        print(
            f"WARNING: {non_finite_count} distances used fallback due to non-finite DTW values"
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
        "--use_relative_eef",
        action="store_true",
        help="Use relative EEF positions instead of absolute",
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
    use_relative_eef = args.use_relative_eef

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

        data = load_tensordict(data_path)
        data = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        segments, segment_indices = segment_episodes_dynamic(data, segment_length)

        dtw_matrix = compute_dtw_distance_matrix(segments, use_relative_eef)

        print(f"Saving DTW matrix and segment IDs to cache: {dtw_matrix_file}")
        with open(dtw_matrix_file, "wb") as f:
            pickle.dump((dtw_matrix, segment_indices), f)

    print(f"len(segments): {len(segments)}")
    print(f"len(segment_indices): {len(segment_indices)}")
    print(f"DTW matrix shape: {dtw_matrix.shape}")


if __name__ == "__main__":
    main()
