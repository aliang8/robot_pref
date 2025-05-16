import torch
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from IPython.display import HTML, display

# Import utility functions
from utils.trajectory import (
    DEFAULT_DATA_PATHS,
    load_tensordict,
    compute_eef_position_ranges,
)

# Import EEF clustering functions
from eef_clustering import (
    extract_eef_trajectories,
    compute_dtw_distance,
)

# Note: We'll set random seeds inside the main function using the config value


def find_top_matches(query_segment, all_segments, top_k=5, exclude_self=True):
    """Find top k segments with lowest DTW distance to the query segment."""
    distances = []

    for i, segment in enumerate(tqdm(all_segments, desc="Computing DTW distances")):
        # Skip self-comparison if needed
        if exclude_self and torch.all(query_segment == segment):
            distances.append(float("inf"))
            continue

        distance = compute_dtw_distance(query_segment, segment)
        distances.append(distance)

    # Get indices of top-k lowest distances
    distances = np.array(distances)
    top_indices = np.argsort(distances)[:top_k]
    top_distances = distances[top_indices]

    return top_indices, top_distances


def create_eef_trajectory_animation(
    eef_positions, start_idx, end_idx, title=None, global_ranges=None
):
    """Create a 3D animation of EEF trajectory for visualization.

    Args:
        eef_positions: Tensor of all EEF positions
        start_idx: Start index for the segment
        end_idx: End index for the segment
        title: Optional title for the animation
        global_ranges: Optional tuple (x_min, x_max, y_min, y_max, z_min, z_max) for consistent visualization
    """
    # Extract trajectory data
    segment_eef = eef_positions[start_idx : end_idx + 1].cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    if title:
        ax.set_title(title)

    # Plot trajectory
    (line,) = ax.plot([], [], [], marker="o", markersize=3)

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set axis limits - either from global ranges or based on this segment
    if global_ranges:
        x_min, x_max, y_min, y_max, z_min, z_max = global_ranges
    else:
        # Find min and max for this segment
        x_min, y_min, z_min = np.min(segment_eef, axis=0)
        x_max, y_max, z_max = np.max(segment_eef, axis=0)

        # Add some padding
        padding = 0.1
        x_range = max(x_max - x_min, 0.01)
        y_range = max(y_max - y_min, 0.01)
        z_range = max(z_max - z_min, 0.01)

        x_min -= padding * x_range
        y_min -= padding * y_range
        z_min -= padding * z_range
        x_max += padding * x_range
        y_max += padding * y_range
        z_max += padding * z_range

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Initialize scatter plot for current position
    point = ax.scatter([], [], [], color="red", s=50)

    # Initialize text annotation for frame number
    frame_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def animate(i):
        # Update the trajectory line
        line.set_data(segment_eef[: i + 1, 0], segment_eef[: i + 1, 1])
        line.set_3d_properties(segment_eef[: i + 1, 2])

        # Update the current position point
        point._offsets3d = (
            [segment_eef[i, 0]],
            [segment_eef[i, 1]],
            [segment_eef[i, 2]],
        )

        # Update frame text
        frame_text.set_text(f"Frame: {i}")

        return line, point, frame_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(segment_eef), interval=200, blit=True
    )

    plt.close()
    return anim


def create_comparison_video(
    eef_positions,
    query_indices,
    match_indices_list,
    distances,
    dataset_indicators=None,
    output_file="eef_segment_comparison.mp4",
    data=None,
    global_ranges=None,
):
    """Create a 3D animation comparing a query segment with its top matches, including image visualization.

    Args:
        eef_positions: Tensor of EEF positions
        query_indices: (start, end) indices for query segment
        match_indices_list: List of (start, end) indices for matches
        distances: DTW distances for each match
        dataset_indicators: Optional indicators of which dataset each frame is from
        output_file: Path to save the output video
        data: Optional TensorDict with image data for visualization
        global_ranges: Optional tuple (x_min, x_max, y_min, y_max, z_min, z_max) for consistent visualization
    """
    # Extract trajectory data
    start_q, end_q = query_indices
    query_eef = eef_positions[start_q : end_q + 1].cpu().numpy()

    match_eef_list = []
    for start_m, end_m in match_indices_list:
        match_eef = eef_positions[start_m : end_m + 1].cpu().numpy()
        match_eef_list.append(match_eef)

    num_matches = len(match_eef_list)

    # Check if we have image data
    has_images = data is not None and "image" in data

    # Set up figure layout based on whether we have images
    if has_images:
        # Create a figure with both trajectory plots and images
        fig = plt.figure(figsize=(15, 4 + 4 * num_matches))
        gs = gridspec.GridSpec(num_matches + 1, 2, width_ratios=[1, 1])
    else:
        # Create a figure with only trajectory plots
        fig = plt.figure(figsize=(10, 4 + 4 * num_matches))

    # First subplot for query trajectory
    if has_images:
        ax_query = fig.add_subplot(gs[0, 0], projection="3d")
        ax_query_img = fig.add_subplot(gs[0, 1])
        ax_query_img.set_title("Query Image")
        ax_query_img.axis("off")
    else:
        ax_query = fig.add_subplot(num_matches + 1, 1, 1, projection="3d")

    ax_query.set_title("Query Trajectory")

    # Set axis limits - either from global ranges or based on all segments
    if global_ranges:
        x_min, x_max, y_min, y_max, z_min, z_max = global_ranges
    else:
        # Find global min and max for consistent axis limits
        all_eef = np.vstack([query_eef] + match_eef_list)
        x_min, y_min, z_min = np.min(all_eef, axis=0)
        x_max, y_max, z_max = np.max(all_eef, axis=0)

        # Add padding
        padding = 0.1
        x_range = max(x_max - x_min, 0.01)
        y_range = max(y_max - y_min, 0.01)
        z_range = max(z_max - z_min, 0.01)

        x_min -= padding * x_range
        y_min -= padding * y_range
        z_min -= padding * z_range
        x_max += padding * x_range
        y_max += padding * y_range
        z_max += padding * z_range

    # Set same limits for all subplots
    ax_query.set_xlim(x_min, x_max)
    ax_query.set_ylim(y_min, y_max)
    ax_query.set_zlim(z_min, z_max)

    # Add axis labels
    ax_query.set_xlabel("X")
    ax_query.set_ylabel("Y")
    ax_query.set_zlabel("Z")

    # Initialize plot elements
    (query_line,) = ax_query.plot([], [], [], marker="o", markersize=2)
    query_point = ax_query.scatter([], [], [], color="red", s=50)

    # Initialize query image if we have images
    if has_images:
        query_img = ax_query_img.imshow(np.zeros((84, 84, 3), dtype=np.uint8))

    # Initialize match plots
    match_lines = []
    match_points = []
    match_axes = []
    match_img_axes = []
    match_imgs = []

    for i in range(num_matches):
        if has_images:
            ax = fig.add_subplot(gs[i + 1, 0], projection="3d")
            ax_img = fig.add_subplot(gs[i + 1, 1])
            ax_img.set_title(f"Match {i + 1} Image")
            ax_img.axis("off")
            match_img_axes.append(ax_img)
        else:
            ax = fig.add_subplot(num_matches + 1, 1, i + 2, projection="3d")

        # Add dataset info if available
        dataset_info = ""
        if dataset_indicators is not None:
            start_m, _ = match_indices_list[i]
            dataset_idx = dataset_indicators[start_m]
            dataset_info = f", Dataset: {dataset_idx}"

        ax.set_title(f"Match {i + 1} (Distance: {distances[i]:.4f}{dataset_info})")

        # Set same limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Add axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Initialize line and point
        (line,) = ax.plot([], [], [], marker="o", markersize=2)
        point = ax.scatter([], [], [], color="red", s=50)

        match_lines.append(line)
        match_points.append(point)
        match_axes.append(ax)

        # Initialize image if we have images
        if has_images:
            img = ax_img.imshow(np.zeros((84, 84, 3), dtype=np.uint8))
            match_imgs.append(img)

    # Add frame counter text
    frame_text = fig.text(0.5, 0.98, "Frame: 0", ha="center", fontsize=14)

    # Get image data if available
    if has_images:
        # Extract image data for query and matches
        query_images = data["image"][start_q : end_q + 1].cpu().numpy()
        match_images_list = []
        for start_m, end_m in match_indices_list:
            match_images = data["image"][start_m : end_m + 1].cpu().numpy()
            match_images_list.append(match_images)

    # Function to update animation
    def update(frame):
        frame_text.set_text(f"Frame: {frame}")

        # Calculate normalized frame position for query trajectory
        query_length = len(query_eef)
        query_frame = min(frame, query_length - 1)

        # Update query trajectory
        query_line.set_data(
            query_eef[: query_frame + 1, 0], query_eef[: query_frame + 1, 1]
        )
        query_line.set_3d_properties(query_eef[: query_frame + 1, 2])
        query_point._offsets3d = (
            [query_eef[query_frame, 0]],
            [query_eef[query_frame, 1]],
            [query_eef[query_frame, 2]],
        )

        # Update query image if we have images
        if has_images:
            query_img.set_array(query_images[query_frame])

        # Update match trajectories and images
        for i in range(num_matches):
            match_length = len(match_eef_list[i])
            match_frame = min(frame, match_length - 1)

            match_lines[i].set_data(
                match_eef_list[i][: match_frame + 1, 0],
                match_eef_list[i][: match_frame + 1, 1],
            )
            match_lines[i].set_3d_properties(match_eef_list[i][: match_frame + 1, 2])
            match_points[i]._offsets3d = (
                [match_eef_list[i][match_frame, 0]],
                [match_eef_list[i][match_frame, 1]],
                [match_eef_list[i][match_frame, 2]],
            )

            # Update match image if we have images
            if has_images:
                match_imgs[i].set_array(match_images_list[i][match_frame])

            # Rotate view for better visualization
            ax_query.view_init(elev=30, azim=(frame % 360))
            match_axes[i].view_init(elev=30, azim=(frame % 360))

        # Return all updated artists
        artists = [frame_text, query_line, query_point] + match_lines + match_points
        if has_images:
            artists += [query_img] + match_imgs
        return artists

    # Create animation
    max_length = max(len(query_eef), max(len(match) for match in match_eef_list))
    anim = animation.FuncAnimation(
        fig, update, frames=max_length, interval=50, blit=True
    )

    # Save animation
    plt.tight_layout(
        rect=[0, 0, 1, 0.98]
    )  # Adjust layout to leave room for the frame counter
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist="Me"), bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close(fig)

    return output_file


def visualize_query_and_matches(
    eef_positions,
    query_indices,
    match_indices_list,
    distances,
    dataset_indicators=None,
    output_file=None,
    data=None,
    global_ranges=None,
):
    """Visualize a query segment and its top matches.

    Args:
        eef_positions: Tensor of EEF positions
        query_indices: (start, end) indices for query segment
        match_indices_list: List of (start, end) indices for matches
        distances: DTW distances for each match
        dataset_indicators: Optional indicators of which dataset each frame is from
        output_file: Path to save the output video (None to skip video creation)
        data: Optional TensorDict with image data for visualization
        global_ranges: Optional tuple (x_min, x_max, y_min, y_max, z_min, z_max) for consistent visualization
    """
    start_q, end_q = query_indices

    # Create animations for query and matches
    query_anim = create_eef_trajectory_animation(
        eef_positions,
        start_q,
        end_q,
        title="Query Segment",
        global_ranges=global_ranges,
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
        match_anim = create_eef_trajectory_animation(
            eef_positions,
            start_m,
            end_m,
            title=f"Match {i + 1} (Dist: {distance:.4f})",
            global_ranges=global_ranges,
        )
        match_anims.append(match_anim)
        display(HTML(match_anim.to_jshtml()))

    # Create video of query and matches if requested
    if output_file:
        create_comparison_video(
            eef_positions,
            query_indices,
            match_indices_list,
            distances,
            dataset_indicators,
            output_file,
            data=data,
            global_ranges=global_ranges,
        )
        print(f"Saved comparison video to {output_file}")


@hydra.main(config_path="config", config_name="eef_segment_matching", version_base=None)
def main(cfg: DictConfig):
    """Run end-effector segment matching with Hydra configuration."""
    print("\n" + "=" * 50)
    print("Finding similar EEF trajectory segments")
    print("=" * 50)

    # Print config for visibility
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    random_seed = cfg.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print(f"Using random seed: {random_seed}")

    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)

    # Extract parameters from config
    data_paths = cfg.data.data_paths
    segment_length = cfg.data.segment_length
    samples_per_dataset = cfg.data.samples_per_dataset

    top_k = cfg.matching.top_k
    query_index = cfg.matching.query_index
    query_indices = (
        cfg.matching.query_indices if hasattr(cfg.matching, "query_indices") else []
    )
    num_random_queries = (
        cfg.matching.num_random_queries
        if hasattr(cfg.matching, "num_random_queries")
        else 1
    )

    use_shared_ranges = cfg.visualization.use_shared_ranges
    create_videos = cfg.visualization.create_videos

    output_dir = cfg.output.output_dir

    print(f"\nFinding similar EEF trajectory segments with parameters:")
    print(f"  Data paths: {data_paths}")
    print(f"  Segment size (H): {segment_length}")
    print(f"  Samples per dataset: {samples_per_dataset}")
    print(f"  Top-k matches: {top_k}")
    print(f"  Random seed: {random_seed}")
    print(f"  Use shared ranges: {use_shared_ranges}")
    if query_indices:
        print(f"  Processing {len(query_indices)} specified query indices")
    elif query_index is not None:
        print(f"  Processing single query index: {query_index}")
    else:
        print(f"  Generating {num_random_queries} random query indices")
    print("=" * 50 + "\n")

    # Compute global EEF position ranges if needed
    global_ranges = None
    if use_shared_ranges:
        global_ranges = compute_eef_position_ranges(data_paths)

    # Lists to store data from all datasets
    all_eef_positions = []
    all_episode_ids = []
    dataset_indicators = []  # To keep track of which dataset each sample comes from
    all_data = []  # Store loaded data for image visualization

    # Load and process each dataset
    for dataset_idx, data_path in enumerate(data_paths):
        dataset_name = Path(data_path).stem
        print(
            f"\nProcessing dataset {dataset_idx + 1}/{len(data_paths)}: {dataset_name}"
        )

        # Load data
        print("Loading tensordict data...")
        data = load_tensordict(data_path)
        all_data.append(data)  # Store loaded data

        # Extract EEF trajectories
        print("\nExtracting EEF trajectories...")
        eef_positions = data["obs"][:, :3].cpu()
        episode_ids = data["episode"].cpu()

        # Create segments
        print("\nCreating segments...")
        segments, segment_indices, _ = extract_eef_trajectories(
            data,
            segment_length=segment_length,
            max_segments=samples_per_dataset,
            use_relative_differences=False,
        )

        # Append to combined data
        all_eef_positions.append(eef_positions)
        all_episode_ids.append(episode_ids)

        # Add dataset indicator (which dataset each sample came from)
        dataset_indicators.extend([dataset_idx] * len(eef_positions))

        # Save dataset segments for later use
        print(f"\nSaving processed segments for {dataset_name}...")
        os.makedirs(f"{output_dir}/segments", exist_ok=True)
        with open(f"{output_dir}/segments/segments_{dataset_name}.pkl", "wb") as f:
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
            f"Saved {len(segments)} segments to {output_dir}/segments/segments_{dataset_name}.pkl"
        )

    # Combine data from all datasets
    eef_positions = torch.cat(all_eef_positions, dim=0)
    episode_ids = torch.cat(all_episode_ids, dim=0)

    # Combine tensordict data for images
    # Note: This assumes all datasets have the same structure
    combined_data = {}
    if all_data and "image" in all_data[0]:
        print("\nCombining image data for visualization...")
        # Combine images from all datasets
        all_images = [data["image"] for data in all_data]
        combined_data["image"] = torch.cat(all_images, dim=0)

    print(f"\nCombined data from {len(data_paths)} datasets:")
    print(f"  Total positions: {len(eef_positions)}")
    print(f"  Total unique episodes: {len(torch.unique(episode_ids))}")
    if "image" in combined_data:
        print(f"  Total images: {len(combined_data['image'])}")

    # Load all segments
    all_segments = []
    all_segment_indices = []

    segment_files = list(Path(f"{output_dir}/segments").glob("segments_*.pkl"))
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

    # Determine query indices to process
    query_idxs_to_process = []
    if query_indices:
        # Use the specified list of query indices
        query_idxs_to_process = query_indices
        print(
            f"\nUsing {len(query_idxs_to_process)} specified query indices: {query_idxs_to_process}"
        )
    elif query_index is not None:
        # Use the single specified query index
        query_idxs_to_process = [query_index]
        print(f"\nUsing specified query index: {query_index}")
    else:
        # Generate random query indices
        max_idx = len(all_segments) - 1
        query_idxs_to_process = random.sample(
            range(max_idx + 1), min(num_random_queries, max_idx + 1)
        )
        print(
            f"\nGenerated {len(query_idxs_to_process)} random query indices: {query_idxs_to_process}"
        )

    # Process each query index
    all_match_results = []

    for query_idx in query_idxs_to_process:
        print(f"\n{'=' * 30}")
        print(f"Processing query segment {query_idx}")
        print(f"{'=' * 30}")

        # Ensure query index is valid
        if query_idx < 0 or query_idx >= len(all_segments):
            print(f"Warning: Query index {query_idx} is out of range. Skipping.")
            continue

        # Get query segment and indices
        query_segment = all_segments[query_idx]
        query_indices = all_segment_indices[query_idx]

        # Find top matches
        print(f"\nFinding top {top_k} matches for query segment {query_idx}...")
        top_indices, top_distances = find_top_matches(
            query_segment, all_segments, top_k=top_k
        )

        # Get segment indices for top matches
        top_match_indices = [all_segment_indices[i] for i in top_indices]

        # Visualize query and matches
        print("\nVisualizing query segment and top matches...")
        output_video = None
        if create_videos:
            output_video = f"{output_dir}/eef_segment_matches_query{query_idx}.mp4"

        visualize_query_and_matches(
            eef_positions,
            query_indices,
            top_match_indices,
            top_distances,
            dataset_indicators,
            output_file=output_video,
            data=combined_data,
            global_ranges=global_ranges,
        )

        # Save results
        print("\nSaving match results...")
        match_results = {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "query_idx": query_idx,
            "query_segment": query_segment.cpu(),
            "query_indices": query_indices,
            "top_indices": top_indices,
            "top_distances": top_distances,
            "top_match_indices": top_match_indices,
            "dataset_indicators": dataset_indicators,
            "global_ranges": global_ranges,
        }
        with open(f"{output_dir}/eef_match_results_query{query_idx}.pkl", "wb") as f:
            pickle.dump(match_results, f)

        print(f"\nResults saved to {output_dir}/eef_match_results_query{query_idx}.pkl")
        if output_video:
            print(f"Video saved to {output_video}")

        # Add to list of all results
        all_match_results.append(match_results)

    print("\nProcessing complete!")
    print(f"Processed {len(all_match_results)} query segments")

    return all_match_results


if __name__ == "__main__":
    main()
