import numpy as np
import wandb

class WandbCallback:
    """Callback for d3rlpy to log metrics to wandb.
    
    This callback is designed to capture training metrics that d3rlpy logs
    during training, including loss values and evaluation scores.
    """
    
    def __init__(self, use_wandb=True, prefix="train"):
        self.use_wandb = use_wandb
        self.prefix = prefix
        self.epoch = 0
        self.best_eval_metrics = None
        self.best_eval_epoch = -1
        # Track metrics across epochs
        self.training_losses = {}
        self.current_epoch_metrics = {}
        self.evaluation_scores = []
        
    def __call__(self, algo, epoch, total_step):
        """Called by d3rlpy at the end of each epoch or update step."""
        self.epoch = epoch
        
        # Basic metrics to track
        metrics = {
            "epoch": epoch,
            "total_step": total_step
        }
        
        # Handle both old and new d3rlpy versions
        logger = getattr(algo, '_active_logger', None)
        if logger is None and hasattr(algo, '_impl'):
            logger = getattr(algo._impl, '_active_logger', None)
        
        # Get metrics from the metrics_buffer
        if hasattr(logger, '_metrics_buffer'):
            for name, buffer in logger._metrics_buffer.items():
                if buffer:  # Check if there are values
                    # Calculate the mean of accumulated values
                    mean_value = np.mean(buffer)
                    metrics[name] = mean_value
                    
                    # Store loss metrics separately for tracking over time
                    if name.endswith('_loss') or name.startswith('loss'):
                        self.training_losses[name] = self.training_losses.get(name, []) + [mean_value]
            
            # Store evaluation scores if present
            if 'evaluation' in logger._metrics_buffer and logger._metrics_buffer['evaluation']:
                eval_score = np.mean(logger._metrics_buffer['evaluation'])
                self.evaluation_scores.append((epoch, eval_score))
                metrics['evaluation_score'] = eval_score
                    
        # Store metrics for this epoch
        self.current_epoch_metrics = metrics.copy()
        
        # Log to wandb if enabled
        if self.use_wandb and wandb.run:
            self._log_to_wandb(metrics, epoch=epoch, prefix=self.prefix)
        
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
        """Get a summary of training losses and metrics."""
        summary = {
            "epoch": self.epoch,
            "best_eval_epoch": self.best_eval_epoch,
        }
        
        # Add latest values of each loss
        for loss_name, values in self.training_losses.items():
            if values:
                summary[f"final_{loss_name}"] = values[-1]
                summary[f"mean_{loss_name}"] = np.mean(values)
        
        # Add latest evaluation score if available
        if self.evaluation_scores:
            latest_eval = self.evaluation_scores[-1]
            summary["final_evaluation_score"] = latest_eval[1]
        
        return summary
        
    def _log_to_wandb(self, metrics, epoch=None, prefix="", step=None):
        """Log any metrics to wandb with proper prefixing.
        
        Args:
            metrics: Dict of metrics or list of (epoch, metrics_dict) tuples from d3rlpy
            epoch: Current epoch (optional)
            prefix: Prefix to add to metric names (e.g., "train", "eval")
            step: Step to use for wandb logging (defaults to epoch if provided)
        
        Returns:
            bool: True if metrics were logged, False otherwise
        """
        if not wandb.run:
            return False
        
        # Use epoch as step if step not specified
        if step is None and epoch is not None:
            step = epoch
            
        # Ensure prefix ends with / if it's not empty
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        
        # Handle d3rlpy training_metrics format (list of tuples)
        if isinstance(metrics, list) and len(metrics) > 0 and isinstance(metrics[0], tuple) and len(metrics[0]) == 2:
            # Log each epoch's metrics separately
            for epoch, epoch_metrics in metrics:
                # Create metrics dict with prefix
                log_dict = {f"{prefix}{k}": v for k, v in epoch_metrics.items() 
                        if isinstance(v, (int, float, np.int64, np.float32, np.float64, np.number))}
                
                # Add epoch
                log_dict["epoch"] = epoch
                
                # Log to wandb
                if log_dict:
                    wandb.log(log_dict, step=epoch)
            
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
            
            # Log histogram for returns if available
            if "returns" in metrics and isinstance(metrics["returns"], (list, np.ndarray)):
                wandb.log({f"{prefix}returns_histogram": wandb.Histogram(metrics["returns"])}, step=step)
            
            # Log to wandb
            if log_dict:
                wandb.log(log_dict, step=step)
                return True
        
        return False


class CompositeCallback:
    """A callback that combines multiple callbacks into one."""
    
    def __init__(self, callbacks):
        """Initialize with a list of callbacks.
        
        Args:
            callbacks: List of callback functions/objects to call
        """
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