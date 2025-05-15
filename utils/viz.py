import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import cv2
import os
import numpy as np


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

    # Create axes for each video
    axes = []
    for i in range(n_videos):
        row = i // grid_dims[1]
        col = i % grid_dims[1]
        ax = fig.add_subplot(grid[row, col])
        # Remove titles for a cleaner grid
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove border around each cell
        ax.set_frame_on(False)
        axes.append(ax)

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
                # Keep updating the last_frame with the latest valid frame
                last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store the last frame for padding shorter videos
            last_frames.append(last_frame)

            # Reset video capture to start
            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            # If we couldn't read a frame, create black frame
            black_frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            img = axes[i].imshow(black_frame)
            images.append(img)
            last_frames.append(black_frame)

    # Set up animation update function with padding
    def update(frame_idx):
        for i, vc in enumerate(video_captures):
            if frame_idx < frame_counts[i]:
                # If current frame exists in this video, show it
                ret, frame = vc.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images[i].set_array(frame_rgb)
            else:
                # If we've gone past this video's frame count, pad with last frame
                images[i].set_array(last_frames[i])

        return images

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, blit=True, interval=1000 / fps
    )

    # Save the animation
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="d3rlpy"))
    anim.save(output_path, writer=writer)

    # Close figure and video captures
    plt.close(fig)
    for vc in video_captures:
        vc.release()

    print(f"Created compact video grid with {n_videos} videos at {output_path}")
    return output_path
