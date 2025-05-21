import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def create_video_grid(video_files, output_path, max_videos=6, fps=30, title=None):
    """Create a grid of videos from individual mp4 files.

    Args:
        video_files: List of paths to mp4 video files
        output_path: Path to save the output video
        max_videos: Maximum number of videos to include in the grid
        fps: Frames per second of the output video
        title: Optional title to display above the grid (not used if None)

    Returns:
        Path to the saved grid video
    """
    if not video_files:
        return None

    # Limit the number of videos
    video_files = video_files[:max_videos]
    n_videos = len(video_files)

    # Determine grid dimensions
    if n_videos <= 2:
        grid_dims = (1, n_videos)
    elif n_videos <= 4:
        grid_dims = (2, 2)
    elif n_videos <= 6:
        grid_dims = (2, 3)
    else:
        grid_dims = (3, 3)

    # Open all video captures
    video_captures = [cv2.VideoCapture(vf) for vf in video_files]

    # Check if videos were opened successfully
    if not all(vc.isOpened() for vc in video_captures):
        print("Warning: Could not open one or more video files")
        return None

    # Get video properties
    widths = []
    heights = []
    frame_counts = []

    for vc in video_captures:
        widths.append(int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)))
        heights.append(int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_counts.append(int(vc.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Find the size of each cell in the grid
    cell_width = max(widths)
    cell_height = max(heights)

    # Use the maximum frame count (with padding) instead of minimum
    max_frames = max(frame_counts)

    # Set up the matplotlib figure and grid with minimal spacing
    fig = plt.figure(figsize=(grid_dims[1] * 3, grid_dims[0] * 3))

    # Create a more compact grid with very minimal spacing
    grid = gridspec.GridSpec(
        grid_dims[0],
        grid_dims[1],
        figure=fig,
        wspace=0.01,  # Minimal horizontal space between cells
        hspace=0.01,  # Minimal vertical space between cells
        left=0.01,  # Minimal left margin
        right=0.99,  # Minimal right margin
        top=0.99,  # Minimal top margin
        bottom=0.01,  # Minimal bottom margin
    )

    # Create axes for each video, but only fill as many as there are videos
    axes = []
    for i in range(n_videos):
        row = i // grid_dims[1]
        col = i % grid_dims[1]
        ax = fig.add_subplot(grid[row, col])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        axes.append(ax)

    # Fill any remaining grid cells with empty axes (so we don't get "grids in grids")
    total_cells = grid_dims[0] * grid_dims[1]
    for i in range(n_videos, total_cells):
        row = i // grid_dims[1]
        col = i % grid_dims[1]
        ax = fig.add_subplot(grid[row, col])
        ax.axis("off")

    # Load first frames and create image objects
    images = []
    last_frames = []  # Store last frame of each video for padding

    for i, vc in enumerate(video_captures):
        # Read first frame
        ret, frame = vc.read()
        if ret:
            # Convert BGR to RGB (cv2 uses BGR, matplotlib uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = axes[i].imshow(frame_rgb)
            images.append(img)

            # Read all frames and store the last frame for padding
            last_frame = frame_rgb.copy()
            while True:
                ret, frame = vc.read()
                if not ret:
                    break
                last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frames.append(last_frame)
            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            black_frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            img = axes[i].imshow(black_frame)
            images.append(img)
            last_frames.append(black_frame)

    # Set up animation update function with padding
    def update(frame_idx):
        for i, vc in enumerate(video_captures):
            if frame_idx < frame_counts[i]:
                ret, frame = vc.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images[i].set_array(frame_rgb)
            else:
                images[i].set_array(last_frames[i])
        return images

    # Reset all video captures to the first frame before animation
    for vc in video_captures:
        vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, blit=True, interval=1000 / fps
    )

    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="d3rlpy"))
    anim.save(output_path, writer=writer)

    plt.close(fig)
    for vc in video_captures:
        vc.release()

    print(f"Created compact video grid with {n_videos} videos at {output_path}")
    return output_path

def plot_active_learning_metrics(model_dir, metrics, augmented_accuracy):
    """
    Plot active learning metrics including test accuracy, loss, and augmented accuracy.

    Args:
        model_dir (str): Path to the directory where the plot will be saved.
        metrics (dict): Dictionary containing 'num_labeled', 'test_accuracy', and 'test_loss'.
        augmented_accuracy (list): List of augmented accuracy values.
    Returns:
        str: Path to the saved plot.
    """
    # Create a 1x3 grid (3 plots side by side)
    fig, axs = plt.subplots(
        1, 3, figsize=(18, 6), constrained_layout=False, sharex=True
    )

    dot_size = 10
    line_color = "blue"

    # Plot test accuracy (left)
    ax1 = axs[0]
    ax1.plot(
        metrics["num_labeled"],
        metrics["test_accuracy"],
        marker="o",
        markersize=dot_size,
        color=line_color,
    )
    ax1.set_ylabel("Test Accuracy", fontsize=16)
    ax1.set_title("Test Accuracy vs Labeled Pairs")
    ax1.grid(True, alpha=0.3)
    # Remove top and right spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Plot test loss (Bradley-Terry) (right)
    ax2 = axs[1]
    ax2.plot(
        metrics["num_labeled"],
        metrics["test_loss"],
        marker="o",
        markersize=dot_size,
        color=line_color,
    )
    ax2.set_ylabel("Bradley-Terry Loss (BCE)", fontsize=16)
    ax2.set_title("Preference Learning Loss vs Labeled Pairs")
    ax2.grid(True, alpha=0.3)
    # Remove top and right spines
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Plot augmented accuracy
    if len(augmented_accuracy) > 0:
        ax3 = axs[2]
        ax3.plot(
            metrics["num_labeled"],
            augmented_accuracy,
            marker="o",
            markersize=dot_size,
            color=line_color,
        )
        ax3.set_ylabel("Augmented Accuracy", fontsize=16)
        ax3.set_title("Augmented Accuracy vs Labeled Pairs")
        ax3.grid(True, alpha=0.3)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

    # Ensure x-axis has only integer ticks
    from matplotlib.ticker import MaxNLocator

    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Make sure there's space for the common x-label
    plt.subplots_adjust(bottom=0.2)

    # Save plot in the model directory
    learning_curve_path = model_dir / "active_learning_metrics.png"
    plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")

    return learning_curve_path