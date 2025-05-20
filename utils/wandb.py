import numpy as np

import wandb
from typing import Any, Optional

from d3rlpy.logging.logger import (
    AlgProtocol,
    LoggerAdapter,
    LoggerAdapterFactory,
    SaveProtocol
)

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
        n_steps_per_epoch: int,
        wandb_cfg: dict
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Please install wandb") from e
        assert algo.impl

        cfg = wandb_cfg.copy()
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
            n_steps_per_epoch=n_steps_per_epoch,
            wandb_cfg=self._wandb_cfg,
        )