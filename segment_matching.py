import torch
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from IPython.display import HTML, display
import matplotlib.gridspec as gridspec
import argparse
from matplotlib import animation

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    extract_images_from_tensordict,
    compute_dinov2_embeddings,
    create_segments,
    compute_dtw_distance,
    find_top_matches,
    create_segment_animation,
    sample_segments,
)


def visualize_query_and_matches(
    images,
    query_indices,
    match_indices_list,
    distances,
    dataset_indicators=None,
    output_file=None,
):
    """Visualize a query segment and its top matches."""
    # Ensure images are on CPU for visualization
    images_cpu = images.cpu()

    start_q, end_q = query_indices

    # Create animations for query and matches
    query_anim = create_segment_animation(
        images_cpu, start_q, end_q, title="Query Segment"
    )

    # Display query animation
    print("Query Segment:")
    if dataset_indicators is not None:
        dataset_idx = dataset_indicators[start_q]
        print(f"  Dataset: {dataset_idx}")

    display(HTML(query_anim.to_jshtml()))

    # Display top matches
    print("\nTop Matches:")
    match_anims = []

    for i, (match_indices, distance) in enumerate(zip(match_indices_list, distances)):
        start_m, end_m = match_indices

        # Dataset info if available
        dataset_info = ""
        if dataset_indicators is not None:
            dataset_idx = dataset_indicators[start_m]
            dataset_info = f", Dataset: {dataset_idx}"

        print(f"  Match {i + 1}: Distance = {distance:.4f}{dataset_info}")

        # Create animation
        match_anim = create_segment_animation(
            images_cpu, start_m, end_m, title=f"Match {i + 1} (Dist: {distance:.4f})"
        )
        match_anims.append(match_anim)
        display(HTML(match_anim.to_jshtml()))

    # Create video of query and matches if requested
    if output_file:
        create_comparison_video(
            images_cpu,
            query_indices,
            match_indices_list,
            distances,
            dataset_indicators,
            output_file,
        )
        print(f"Saved comparison video to {output_file}")


def create_comparison_video(
    images,
    query_indices,
    match_indices_list,
    distances,
    dataset_indicators=None,
    output_file="segment_comparison.mp4",
):
    """Create a video comparing a query segment with its top matches."""
    # Extract trajectory data
    start_q, end_q = query_indices
    query_images = images[start_q : end_q + 1]

    match_images_list = []
    for start_m, end_m in match_indices_list:
        # Get same length as query for visualization
        length = min(end_q - start_q + 1, end_m - start_m + 1)
        match_images = images[start_m : start_m + length]

        # If match is shorter than query, pad with the last frame
        if len(match_images) < len(query_images):
            padding = len(query_images) - len(match_images)
            last_frame = match_images[-1].unsqueeze(0).repeat(padding, 1, 1, 1)
            match_images = torch.cat([match_images, last_frame])

        match_images_list.append(match_images)

    num_matches = len(match_images_list)

    # Set up figure for animation
    fig = plt.figure(figsize=(12, 4 + 4 * num_matches))
    gs = gridspec.GridSpec(1 + num_matches, 2, height_ratios=[1] + [1] * num_matches)

    # Query subplot
    ax_query = fig.add_subplot(gs[0, :])
    ax_query.set_title("Query Segment")
    ax_query.axis("off")
    img_query = ax_query.imshow(query_images[0].numpy())

    # Match subplots
    ax_matches = []
    img_matches = []

    for i in range(num_matches):
        ax = fig.add_subplot(gs[i + 1, :])
        dataset_info = ""
        if dataset_indicators is not None:
            start_m, _ = match_indices_list[i]
            dataset_idx = dataset_indicators[start_m]
            dataset_info = f", Dataset: {dataset_idx}"

        ax.set_title(f"Match {i + 1} (Distance: {distances[i]:.4f}{dataset_info})")
        ax.axis("off")
        img = ax.imshow(match_images_list[i][0].numpy())
        ax_matches.append(ax)
        img_matches.append(img)

    # Step counter
    step_text = fig.text(0.5, 0.98, "Step: 0", fontsize=14, ha="center")

    # Animation function
    def animate(i):
        # Update step counter
        step_text.set_text(f"Step: {i}")

        # Update query image
        frame_idx = min(i, len(query_images) - 1)
        img_query.set_array(query_images[frame_idx].numpy())

        # Update match images
        for j in range(num_matches):
            match_frame_idx = min(i, len(match_images_list[j]) - 1)
            img_matches[j].set_array(match_images_list[j][match_frame_idx].numpy())

        return [step_text, img_query] + img_matches

    # Create animation
    frames = len(query_images)
    print(f"Creating animation with {frames} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=True)

    # Save animation
    print(f"Saving video to {output_file}...")
    writer = animation.FFMpegWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close(fig)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Find similar segments using DTW")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=DEFAULT_DATA_PATHS,
        help="Paths to the PT files containing trajectory data",
    )
    parser.add_argument(
        "--segment_length", type=int, default=20, help="Length of segments (H)"
    )
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        default=500,
        help="Number of segments to sample from each dataset",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top matches to find"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="segment_matches",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--query_index",
        type=int,
        default=None,
        help="Index of the segment to use as query (random if not specified)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cached_data",
        help="Directory to cache embeddings",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Parameters
    H = args.segment_length
    batch_size = args.batch_size
    top_k = args.top_k

    print("\n" + "=" * 50)
    print(f"Finding similar segments with parameters:")
    print(f"  Data paths: {args.data_paths}")
    print(f"  Segment size (H): {H}")
    print(f"  Samples per dataset: {args.samples_per_dataset}")
    print(f"  Top-k matches: {top_k}")
    print(f"  Random seed: {RANDOM_SEED}")
    print("=" * 50 + "\n")

    # Lists to store data from all datasets
    all_images = []
    all_episode_ids = []
    dataset_indicators = []  # To keep track of which dataset each sample comes from

    # Load and process each dataset
    for dataset_idx, data_path in enumerate(args.data_paths):
        dataset_name = Path(data_path).stem
        print(
            f"\nProcessing dataset {dataset_idx + 1}/{len(args.data_paths)}: {dataset_name}"
        )

        # Create cache file path
        embeddings_cache = f"{args.cache_dir}/dinov2_embeddings_{dataset_name}.pt"

        # Load data
        print("Loading tensordict data...")
        data = load_tensordict(data_path)

        # Extract images and episode IDs
        print("Extracting images and episode IDs...")
        images = extract_images_from_tensordict(data)
        episode_ids = data["episode"]

        # Compute DINOv2 embeddings with caching
        print("\nProcessing embeddings...")
        embeddings = compute_dinov2_embeddings(
            images, batch_size=batch_size, cache_file=embeddings_cache
        )

        # Create segments
        print("\nSegmenting data...")
        segments, segment_indices = create_segments(embeddings, episode_ids, H)

        # Sample segments if needed
        if args.samples_per_dataset > 0 and args.samples_per_dataset < len(segments):
            print(
                f"\nSampling {args.samples_per_dataset} segments from {len(segments)} total segments..."
            )
            segments, segment_indices = sample_segments(
                segments, segment_indices, args.samples_per_dataset
            )

        # Append to combined data
        all_images.append(images)
        all_episode_ids.append(episode_ids)

        # Add dataset indicator (which dataset each sample came from)
        dataset_indicators.extend([dataset_idx] * len(images))

        # Save dataset segments for later use
        print(f"\nSaving processed segments for {dataset_name}...")
        os.makedirs(f"{args.output_dir}/segments", exist_ok=True)
        with open(f"{args.output_dir}/segments/segments_{dataset_name}.pkl", "wb") as f:
            pickle.dump(
                {
                    "segments": segments,
                    "segment_indices": segment_indices,
                    "dataset_name": dataset_name,
                    "dataset_idx": dataset_idx,
                },
                f,
            )
        print(
            f"Saved {len(segments)} segments to {args.output_dir}/segments/segments_{dataset_name}.pkl"
        )

    # Combine data from all datasets
    images = torch.cat(all_images, dim=0)
    episode_ids = torch.cat(all_episode_ids, dim=0)

    print(f"\nCombined data from {len(args.data_paths)} datasets:")
    print(f"  Total images: {len(images)}")
    print(f"  Total unique episodes: {len(torch.unique(episode_ids))}")

    # Load all segments
    all_segments = []
    all_segment_indices = []

    segment_files = list(Path(f"{args.output_dir}/segments").glob("segments_*.pkl"))
    print(f"\nLoading {len(segment_files)} segment files...")

    for segment_file in segment_files:
        with open(segment_file, "rb") as f:
            segment_data = pickle.load(f)

        all_segments.extend(segment_data["segments"])
        all_segment_indices.extend(segment_data["segment_indices"])

        print(
            f"  Loaded {len(segment_data['segments'])} segments from {segment_data['dataset_name']}"
        )

    print(f"\nTotal number of segments: {len(all_segments)}")

    # Select query segment
    if args.query_index is None:
        query_idx = random.randint(0, len(all_segments) - 1)
    else:
        query_idx = args.query_index

    print(f"\nUsing segment {query_idx} as query")
    query_segment = all_segments[query_idx]
    query_indices = all_segment_indices[query_idx]

    # Find top matches
    print(f"\nFinding top {top_k} matches for query segment...")
    top_indices, top_distances = find_top_matches(
        query_segment, all_segments, top_k=top_k
    )

    # Get segment indices for top matches
    top_match_indices = [all_segment_indices[i] for i in top_indices]

    # Visualize query and matches
    print("\nVisualizing query segment and top matches...")
    output_video = f"{args.output_dir}/segment_matches_query{query_idx}.mp4"
    visualize_query_and_matches(
        images,
        query_indices,
        top_match_indices,
        top_distances,
        dataset_indicators,
        output_file=output_video,
    )

    # Save results
    print("\nSaving match results...")
    match_results = {
        "query_idx": query_idx,
        "query_segment": query_segment.cpu(),
        "query_indices": query_indices,
        "top_indices": top_indices,
        "top_distances": top_distances,
        "top_match_indices": top_match_indices,
        "dataset_indicators": dataset_indicators,
    }
    with open(f"{args.output_dir}/match_results_query{query_idx}.pkl", "wb") as f:
        pickle.dump(match_results, f)

    print(f"\nResults saved to {args.output_dir}/match_results_query{query_idx}.pkl")
    print(f"Video saved to {output_video}")
    print("\nProcessing complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
