import torch
import numpy as np
import os
import random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import utility functions
from utils.trajectory import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    save_preprocessed_segments,
    load_preprocessed_segments,
)

# Import function to extract trajectories
from eef_clustering import extract_eef_trajectories


def main():
    """Extract and save preprocessed segments from robot trajectory data."""
    parser = argparse.ArgumentParser(
        description="Preprocess robot trajectory data and save segments"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATHS[0],
        help="Path to the PT file containing trajectory data",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="preprocessed/segments.pt",
        help="Path to save the preprocessed data",
    )
    parser.add_argument(
        "--segment_length", type=int, default=64, help="Length of trajectory segments"
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=1000,
        help="Maximum number of segments to extract (default: all)",
    )
    parser.add_argument(
        "--use_relative_differences",
        action="store_true",
        help="Use relative differences instead of absolute positions",
    )
    parser.add_argument(
        "--minimal_data",
        action="store_true",
        help="Save only minimal data (segments and indices) without observations to save space",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Preprocessing robot trajectory data")
    print(f"Data path: {args.data_path}")
    print(f"Output file: {args.output_file}")
    print(f"Segment length: {args.segment_length}")
    print(f"Max segments: {args.max_segments}")
    print(f"Use relative differences: {args.use_relative_differences}")
    print(f"Minimal data only: {args.minimal_data}")
    print("=" * 50 + "\n")

    # Check if output file already exists
    if os.path.exists(args.output_file):
        print(f"WARNING: Output file {args.output_file} already exists!")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Aborting.")
            return

    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_tensordict(args.data_path)

    # Extract trajectories
    print(f"\nExtracting trajectory segments...")
    segments, segment_indices, original_segments = extract_eef_trajectories(
        data,
        segment_length=args.segment_length,
        max_segments=args.max_segments,
        use_relative_differences=args.use_relative_differences,
    )

    print(f"Extracted {len(segments)} segments")

    # Get dataset name from file path
    dataset_name = Path(args.data_path).stem

    # Prepare data for saving
    preprocessed_data = {
        "dataset_name": dataset_name,
        "segments": segments,
        "segment_indices": segment_indices,
        "original_segments": original_segments,
        "args": vars(args),
        "use_relative_differences": args.use_relative_differences,
        "segment_length": args.segment_length,
    }

    # Include full observation data by default unless minimal_data is specified
    if not args.minimal_data:
        print("Including full observation data and images in preprocessed file")
        # Add important observation fields
        essential_fields = ["obs", "state", "action", "reward", "episode", "image"]
        for field in essential_fields:
            if field in data:
                preprocessed_data[field] = dat a[field]
                print(f"Added field '{field}' with shape {data[field].shape}")

    # Save preprocessed data
    save_preprocessed_segments(preprocessed_data, args.output_file)

    print("\nPreprocessing complete!")
    print(f"Preprocessed data saved to {args.output_file}")
    print("=" * 50)

    # Test loading the saved data
    print("\nTesting data loading...")
    loaded_data = load_preprocessed_segments(args.output_file)
    print("Test successful!")


if __name__ == "__main__":
    main()
