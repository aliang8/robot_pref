import wandb
import numpy as np

# Track global step for consistent wandb logging
_global_step = 0

def log_to_wandb(metrics, epoch=None, prefix="", step=None):
    """Log any metrics to wandb with proper prefixing.
    
    Args:
        metrics: Dict of metrics or list of (epoch, metrics_dict) tuples from d3rlpy
        epoch: Current epoch (optional)
        prefix: Prefix to add to metric names (e.g., "train", "eval")
        step: Step to use for wandb logging (defaults to epoch if provided)
    
    Returns:
        bool: True if metrics were logged, False otherwise
    """
    global _global_step
    
    if not wandb.run:
        return False
    
    # Use epoch as step if step not specified
    if step is None and epoch is not None:
        step = epoch
    
    # If step is still None, use and increment global step
    if step is None:
        step = _global_step
        _global_step += 1
    else:
        # Ensure global step is at least as large as the provided step
        _global_step = max(_global_step, step + 1)
        
    # Ensure prefix ends with / if it's not empty
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    
    # Handle d3rlpy training_metrics format (list of tuples)
    if isinstance(metrics, list) and len(metrics) > 0 and isinstance(metrics[0], tuple) and len(metrics[0]) == 2:
        # Sort metrics by epoch to ensure increasing steps
        sorted_metrics = sorted(metrics, key=lambda x: x[0])
        
        # Log each epoch's metrics separately
        for epoch_val, epoch_metrics in sorted_metrics:
            # Create metrics dict with prefix
            log_dict = {f"{prefix}{k}": v for k, v in epoch_metrics.items() 
                      if isinstance(v, (int, float, np.int64, np.float32, np.float64, np.number))}
            
            # Add epoch
            log_dict["epoch"] = epoch_val
            
            # Use current global step for consistent ordering
            curr_step = _global_step
            _global_step += 1
            
            # Log to wandb
            if log_dict:
                wandb.log(log_dict, step=curr_step)
        
        print(f"Logged {len(metrics)} epochs of {prefix.rstrip('/')} metrics to wandb")
        return True
    
    # Handle single metrics dict
    elif isinstance(metrics, dict):
        log_dict = {}
        
        # Add epoch if provided
        if epoch is not None:
            log_dict[f"{prefix}epoch"] = epoch
        
        # Add all numerical metrics with prefix
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.int64, np.float32, np.float64, np.number)):
                log_dict[f"{prefix}{key}"] = value
        
        # Use same step for all related logs
        curr_step = step
        
        # Log histogram for returns if available
        if "returns" in metrics and isinstance(metrics["returns"], (list, np.ndarray)):
            wandb.log({f"{prefix}returns_histogram": wandb.Histogram(metrics["returns"])}, step=curr_step)
        
        # Log to wandb
        if log_dict:
            wandb.log(log_dict, step=curr_step)
            return True
    
    return False
