import uuid
from typing import Any, Optional

import numpy as np
from d3rlpy.logging.logger import (
    AlgProtocol,
    LoggerAdapter,
    LoggerAdapterFactory,
    SaveProtocol,
)
from omegaconf import OmegaConf

import wandb


def wandb_init(config: dict) -> None:
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        project=config.wandb.project,
        name=config.wandb.name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def log_video_to_wandb(images, name, fps=8):
    """Log a sequence of images as a video to wandb."""
    if images is None or len(images) == 0:
        return
    
    # Convert to (N, C, H, W) format if needed
    if len(images.shape) == 4:
        if images.shape[-1] == 3:
            images = images.transpose(0, 3, 1, 2)
        
        # Log the video
        wandb.log({
            name: wandb.Video(images, fps=fps, format="mp4")
        })

def log_query_videos_to_wandb(dataset, idx_st_1, idx_st_2, labels, config, prefix="queries"):
    """Log video sequences for query pairs to wandb."""

    def combine_segments_side_by_side(images1, images2, preference=None):
        """Combine two image sequences side by side with a border around the preferred trajectory."""
        if len(images1.shape) == 4:  # (N, H, W, C)
            h1, w1 = images1.shape[1:3]
            h2, w2 = images2.shape[1:3]
            combined = np.zeros((len(images1), max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            
            # Copy the images
            combined[:, :h1, :w1] = images1
            combined[:, :h2, w1:w1+w2] = images2
            
            # Add border around preferred trajectory if preference is provided
            if preference is not None:
                border_width = h1 // 10
                border_color = np.array([0, 255, 0], dtype=np.uint8)  # Green border
                
                if preference == 0:  # First segment is preferred
                    # Add border to first segment
                    combined[:, :border_width, :w1] = border_color  # Top border
                    combined[:, -border_width:, :w1] = border_color  # Bottom border
                    combined[:, :, :border_width] = border_color  # Left border
                    combined[:, :, w1-border_width:w1] = border_color  # Right border
                elif preference == 1:  # Second segment is preferred
                    # Add border to second segment
                    combined[:, :border_width, w1:] = border_color  # Top border
                    combined[:, -border_width:, w1:] = border_color  # Bottom border
                    combined[:, :, w1:w1+border_width] = border_color  # Left border
                    combined[:, :, -border_width:] = border_color  # Right border
            
            return combined
        return None

    if "images" not in dataset:
        return
    print("Logging video sequences to wandb...")
    for i, (idx1, idx2) in enumerate(zip(idx_st_1, idx_st_2)):
        # Get image sequences for both segments
        images1 = dataset["images"][idx1:idx1 + config.segment_size]
        images2 = dataset["images"][idx2:idx2 + config.segment_size]
        # Get preference for this pair
        preference = 2  # Default to equal preference
        if i < len(labels):
            if labels[i][0] == 0 and labels[i][1] == 1:  # Second segment preferred
                preference = 1
            elif labels[i][0] == 1 and labels[i][1] == 0:  # First segment preferred
                preference = 0
        # Combine segments side by side with preference border
        combined_segments = combine_segments_side_by_side(images1, images2, preference)
        if combined_segments is not None and wandb.run:
            log_video_to_wandb(combined_segments, f"{prefix}/query_{i}")
        if i >= 9:
            break

def log_to_wandb(metrics, prefix="", epoch=None):
    """Simplified function to log metrics to wandb with prefix support.

    Args:
        metrics: Dict of metrics to log
        prefix: Prefix to add to metric names (e.g., "train", "eval")
        epoch: Current epoch (optional)
    """

    # Add prefix to keys
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"

    # Create dict with prefixed keys
    log_dict = {}

    # Add epoch if provided
    if epoch is not None:
        log_dict["epoch"] = epoch

    # Process scalar metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            log_dict[f"{prefix}{key}"] = value
        elif isinstance(value, wandb.Video):
            # Log videos separately
            wandb.run.log({f"{prefix}{key}": value})
        elif isinstance(value, wandb.Image):
            # Log images separately
            wandb.run.log({f"{prefix}{key}": value})

    # Log scalar metrics
    if log_dict:
        wandb.run.log(log_dict)

__all__ = ["WanDBAdapter", "WanDBAdapterFactory"]


class WanDBAdapter(LoggerAdapter):
    r"""WandB Logger Adapter class.

    This class logs data to Weights & Biases (WandB) for experiment tracking.

    Args:
        algo: Algorithm.
        n_steps_per_epoch: Number of steps per epoch.
        project: Project name.
    """

    def __init__(
        self,
        algo: AlgProtocol,
        experiment_name: str,
        n_steps_per_epoch: int,
        wandb_cfg: dict
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Please install wandb") from e
        assert algo.impl

        cfg = wandb_cfg.copy()
        cfg["name"] = experiment_name
        if "use_wandb" in cfg:
            del cfg["use_wandb"]

        self.run = wandb.init(**cfg)
        self.run.watch(
            tuple(algo.impl.modules.get_torch_modules().values()),
            log="gradients",
            log_freq=n_steps_per_epoch,
        )
        self._is_model_watched = False

    def write_params(self, params: dict[str, Any]) -> None:
        """Writes hyperparameters to WandB config."""
        self.run.config.update(params)

    def before_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed before writing metric."""

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        """Writes metric to WandB."""
        self.run.log({name: value, "epoch": epoch}, step=step)

    def after_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed after writing metric."""

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        """Saves models to Weights & Biases.

        Not implemented for WandB.
        """
        # Implement saving model to wandb if needed

    def close(self) -> None:
        """Closes the logger and finishes the WandB run."""
        self.run.finish()

    def watch_model(
        self,
        epoch: int,
        step: int,
    ) -> None:
        pass


class WanDBAdapterFactory(LoggerAdapterFactory):
    r"""WandB Logger Adapter Factory class.

    This class creates instances of the WandB Logger Adapter for experiment
    tracking.

    Args:
        project (Optional[str], optional): The name of the WandB project.
            Defaults to None.
    """

    _project: Optional[str]

    def __init__(self, wandb_cfg: dict) -> None:
        self._wandb_cfg = wandb_cfg

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> LoggerAdapter:
        return WanDBAdapter(
            algo=algo,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            wandb_cfg=self._wandb_cfg,
        )