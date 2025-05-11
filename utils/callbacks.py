import numpy as np
import wandb
from utils.wandb_utils import log_to_wandb, log_artifact, debug_media_file
import os

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
        
        logger = algo._active_logger
        
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
            try:
                # Use our improved log_to_wandb function
                log_to_wandb(metrics, epoch=epoch, prefix=self.prefix)
            except Exception as e:
                print(f"Warning: Failed to log metrics to wandb: {e}")
        
        return metrics
    
    def update_eval_metrics(self, eval_metrics, epoch):
        """Track the best evaluation metrics so far.
        
        Args:
            eval_metrics: Dictionary of evaluation metrics
            epoch: Current epoch
            
        Returns:
            Dictionary with evaluation metrics and is_best flag
        """
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
    
    def log_model_artifact(self, model_path, metadata=None):
        """Log model as a wandb artifact.
        
        Args:
            model_path: Path to the saved model
            metadata: Optional metadata to add to the artifact
            
        Returns:
            wandb.Artifact or None if logging failed
        """
        if not self.use_wandb or not wandb.run:
            return None
            
        try:
            return log_artifact(model_path, artifact_type="model", metadata=metadata)
        except Exception as e:
            print(f"Error logging model artifact: {e}")
            return None
    
    def log_video(self, video_path, name=None, fps=30, prefix=None):
        """Log a video file to wandb.
        
        Args:
            video_path: Path to the video file
            name: Name to use for the video (default: derived from path)
            fps: Frames per second for the video
            prefix: Prefix to use (defaults to self.prefix if None)
        """
        if not self.use_wandb or not wandb.run:
            return
            
        # Verify the video file is valid
        print(f"Checking video file: {video_path}")
        if not debug_media_file(video_path):
            print(f"Skipping invalid video file: {video_path}")
            return
            
        try:
            # Default name from path if not specified
            if name is None:
                name = f"video_{self.epoch}"
                
            # Use provided prefix or default to self.prefix
            actual_prefix = prefix if prefix is not None else self.prefix
                
            print(f"Logging video to wandb: {video_path} with name '{name}' under prefix '{actual_prefix}'")
            # Log using our improved function
            log_to_wandb({name: wandb.Video(video_path, fps=fps, format="mp4")}, 
                        prefix=actual_prefix)
            print(f"Successfully logged video: {video_path}")
        except Exception as e:
            print(f"Error logging video: {e}")
            import traceback
            traceback.print_exc()


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