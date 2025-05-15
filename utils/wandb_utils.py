import wandb
import numpy as np
import os
from pathlib import Path


def log_to_wandb(metrics, prefix="", epoch=None):
    """Simplified function to log metrics to wandb with prefix support.

    Args:
        metrics: Dict of metrics to log
        prefix: Prefix to add to metric names (e.g., "train", "eval")
        epoch: Current epoch (optional)
    """
    if not wandb.run:
        return

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
            wandb.log({f"{prefix}{key}": value})
        elif isinstance(value, wandb.Image):
            # Log images separately
            wandb.log({f"{prefix}{key}": value})

    # Log scalar metrics
    if log_dict:
        wandb.log(log_dict)


def log_artifact(artifact_path, artifact_type="model", metadata=None):
    """Log an artifact file to wandb.

    Args:
        artifact_path: Path to the file to log
        artifact_type: Type of artifact (model, dataset, etc.)
        metadata: Optional dict of metadata to associate with the artifact

    Returns:
        wandb.Artifact: The logged artifact object or None if logging failed
    """
    if not wandb.run:
        return None

    # Ensure path is valid
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        print(f"Warning: Artifact file {artifact_path} does not exist")
        return None

    # Create artifact name
    name = f"{artifact_type}_{artifact_path.stem}_{wandb.run.id}"

    # Create and log artifact
    artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
    artifact.add_file(str(artifact_path))
    wandb.log_artifact(artifact)

    return artifact
