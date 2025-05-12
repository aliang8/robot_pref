import numpy as np
import wandb
from utils.wandb_utils import log_to_wandb, log_artifact
import os

class WandbCallback:
    """Simplified callback for d3rlpy to log metrics to wandb."""
    
    def __init__(self, use_wandb=True, prefix="train"):
        self.use_wandb = use_wandb
        self.prefix = prefix
        self.epoch = 0
        self.best_eval_metrics = None
        self.best_eval_epoch = -1
        # Track metrics across epochs
        self.training_losses = {}
    
    def __call__(self, algo, epoch, total_step):
        """Called by d3rlpy at the end of each epoch."""
        self.epoch = epoch
        
        # Basic metrics to track
        metrics = {
            "epoch": epoch,
            "total_step": total_step
        }
        
        # Get metrics from logger if available
        logger = algo._active_logger
        if hasattr(logger, '_metrics_buffer'):
            for name, buffer in logger._metrics_buffer.items():
                if buffer:  # Check if there are values
                    # Calculate the mean of accumulated values
                    mean_value = np.mean(buffer)
                    metrics[name] = mean_value
                    
                    # Track loss values for plotting
                    if name.endswith('_loss') or name.startswith('loss'):
                        self.training_losses[name] = self.training_losses.get(name, []) + [mean_value]
        
        # Log to wandb if enabled
        if self.use_wandb and wandb.run:
            log_to_wandb(metrics, prefix=self.prefix, epoch=epoch)
        
        return metrics
    
    def update_eval_metrics(self, eval_metrics, epoch):
        """Track the best evaluation metrics so far."""
        # Check if these are the best metrics so far
        is_best = False
        if self.best_eval_metrics is None:
            is_best = True
        elif 'mean_return' in eval_metrics and 'mean_return' in self.best_eval_metrics:
            if eval_metrics['mean_return'] > self.best_eval_metrics['mean_return']:
                is_best = True
        
        # Update best metrics if applicable
        if is_best:
            self.best_eval_metrics = eval_metrics.copy()
            self.best_eval_epoch = epoch
            
        # Add a flag for best metrics
        eval_metrics_with_best = eval_metrics.copy()
        eval_metrics_with_best['is_best'] = is_best
        
        return eval_metrics_with_best
    
    def get_training_summary(self):
        """Get a summary of training metrics."""
        summary = {
            "epoch": self.epoch,
            "best_eval_epoch": self.best_eval_epoch,
        }
        
        # Add latest values of each loss
        for loss_name, values in self.training_losses.items():
            if values:
                summary[f"final_{loss_name}"] = values[-1]
                summary[f"mean_{loss_name}"] = np.mean(values)
        
        return summary
    
    def log_model_artifact(self, model_path, metadata=None):
        """Log model as a wandb artifact."""
        if not self.use_wandb or not wandb.run:
            return None
            
        return log_artifact(model_path, artifact_type="model", metadata=metadata)


class CompositeCallback:
    """A callback that combines multiple callbacks into one."""
    
    def __init__(self, callbacks):
        """Initialize with a list of callbacks."""
        self.callbacks = callbacks
    
    def __call__(self, algo, epoch, total_step):
        """Call all callbacks in order."""
        results = []
        for callback in self.callbacks:
            try:
                result = callback(algo, epoch, total_step)
                results.append(result)
            except Exception as e:
                print(f"Error in callback {callback}: {e}")
        return results 