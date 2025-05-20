import numpy as np

import wandb


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
