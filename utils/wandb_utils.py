import wandb
import numpy as np
import os
from pathlib import Path
import torch

# Track global step for consistent wandb logging
_global_step = 0

def log_to_wandb(metrics, epoch=None, prefix="", step=None):
    """Log any metrics to wandb with proper prefixing.
    
    Args:
        metrics: Dict of metrics or list of (epoch, metrics_dict) tuples from d3rlpy
        epoch: Current epoch (optional, will be added to metrics but not used for step)
        prefix: Prefix to add to metric names (e.g., "train", "eval")
        step: Optional step info (will be stored in metrics but not used for wandb step)
    
    Returns:
        bool: True if metrics were logged, False otherwise
    """
    global _global_step
    
    if not wandb.run:
        return False
    
    try:
        # Ensure prefix ends with / if it's not empty
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        
        # Handle d3rlpy training_metrics format (list of tuples)
        if isinstance(metrics, list) and len(metrics) > 0 and isinstance(metrics[0], tuple) and len(metrics[0]) == 2:
            # Sort metrics by epoch to ensure consistent order
            sorted_metrics = sorted(metrics, key=lambda x: x[0])
            
            # Log each epoch's metrics separately
            for epoch_val, epoch_metrics in sorted_metrics:
                # Create metrics dict with prefix
                log_dict = {f"{prefix}{k}": v for k, v in epoch_metrics.items() 
                          if isinstance(v, (int, float, np.int64, np.float32, np.float64, np.number))}
                
                # Add epoch info but don't use for step
                log_dict["epoch"] = epoch_val
                if step is not None:
                    log_dict["total_step"] = step
                
                # Log to wandb using global step (no step parameter)
                if log_dict:
                    wandb.log(log_dict)
                    
                # Check for non-scalar metrics that need special handling
                for k, v in epoch_metrics.items():
                    if not isinstance(v, (int, float, np.int64, np.float32, np.float64, np.number)):
                        try:
                            # Handle special case like images, videos, etc.
                            _log_special_metric(f"{prefix}{k}", v)
                        except Exception as e:
                            print(f"Error logging special metric {k}: {e}")
                
                # Increment global step so wandb uses a new step next time
                _global_step += 1
            
            print(f"Logged {len(metrics)} epochs of {prefix.rstrip('/')} metrics to wandb")
            return True
        
        # Handle single metrics dict
        elif isinstance(metrics, dict):
            scalar_dict = {}
            special_dict = {}
            
            # Add epoch/step if provided (as regular metrics, not for wandb step)
            if epoch is not None:
                scalar_dict[f"{prefix}epoch"] = epoch
            if step is not None:
                scalar_dict[f"{prefix}step"] = step
            
            # Separate metrics based on their type
            for key, value in metrics.items():
                # Add numerical metrics to scalar_dict
                if isinstance(value, (int, float, np.int64, np.float32, np.float64, np.number)):
                    scalar_dict[f"{prefix}{key}"] = value
                # Handle returns histogram specially
                elif isinstance(value, (list, np.ndarray)) and key == "returns":
                    special_dict[f"{prefix}returns_histogram"] = wandb.Histogram(value)
                # Handle everything else as potential media
                else:
                    special_dict[f"{prefix}{key}"] = value
            
            # Log scalar metrics (no step parameter)
            if scalar_dict:
                try:
                    wandb.log(scalar_dict)
                except Exception as e:
                    print(f"Error logging scalar metrics: {e}")
            
            # Log special metrics (videos, images, etc.)
            for k, v in special_dict.items():
                try:
                    _log_special_metric(k, v)
                except Exception as e:
                    print(f"Error logging special metric {k}: {e}")
            
            # Increment global step for next call
            _global_step += 1
                    
            return (len(scalar_dict) > 0 or len(special_dict) > 0)
    
    except Exception as e:
        print(f"Error in log_to_wandb: {e}")
        return False
    
    return False

def _log_special_metric(key, value, step=None):
    """Handle special metrics like videos, images, histograms, etc.
    
    Args:
        key: Metric key/name
        value: Metric value (non-scalar)
        step: Not used, kept for backward compatibility 
    """
    try:
        # Check if it's already a wandb-specific object
        if isinstance(value, (wandb.Image, wandb.Video, wandb.Histogram, wandb.Object3D)):
            print(f"Logging pre-created wandb object: {key}, type: {type(value)}")
            wandb.log({key: value})
            return
            
        # Check for video files
        if isinstance(value, (str, Path)) and str(value).lower().endswith(('.mp4', '.avi', '.gif')):
            # It's a path to a video file
            if os.path.exists(value):
                # Verify the video file is valid
                if not debug_media_file(value):
                    print(f"Skipping invalid video file: {value}")
                    return
                
                # Get FPS either from metadata or use default
                fps = getattr(value, 'fps', 30)
                print(f"Creating wandb.Video object for {value} with fps={fps}")
                video_obj = wandb.Video(value, fps=fps, format="mp4")
                wandb.log({key: video_obj})
                print(f"Successfully logged video {key}")
            else:
                print(f"Warning: Video file {value} does not exist")
                
        # Check for image files
        elif isinstance(value, (str, Path)) and str(value).lower().endswith(('.png', '.jpg', '.jpeg')):
            # It's a path to an image file
            if os.path.exists(value):
                wandb.log({key: wandb.Image(value)})
            else:
                print(f"Warning: Image file {value} does not exist")
                
        # Handle lists of video files
        elif isinstance(value, list) and all(isinstance(x, str) for x in value) and any(x.lower().endswith(('.mp4', '.avi', '.gif')) for x in value):
            # Get only existing video files
            existing_videos = [v for v in value if os.path.exists(v)]
            if existing_videos:
                # Verify each video file
                valid_videos = []
                for v in existing_videos:
                    if debug_media_file(v):
                        valid_videos.append(v)
                    else:
                        print(f"Skipping invalid video in list: {v}")
                
                if valid_videos:
                    # Get FPS either from metadata or use default
                    fps = getattr(value[0], 'fps', 30) if value else 30
                    print(f"Creating list of {len(valid_videos)} wandb.Video objects with fps={fps}")
                    video_list = [wandb.Video(v, fps=fps, format="mp4") for v in valid_videos]
                    wandb.log({key: video_list})
                    print(f"Successfully logged video list {key}")
                else:
                    print(f"No valid videos found in list for {key}")
            else:
                print(f"Warning: None of the video files in {key} exist")
                
        # Handle lists of video objects
        elif isinstance(value, list) and all(isinstance(x, wandb.Video) for x in value):
            print(f"Logging list of {len(value)} pre-created wandb.Video objects")
            wandb.log({key: value})
            
        # Handle lists of image objects
        elif isinstance(value, list) and all(isinstance(x, wandb.Image) for x in value):
            wandb.log({key: value})
            
        # Handle numpy arrays as potential images
        elif isinstance(value, np.ndarray) and len(value.shape) >= 2:
            # Check if it's an image (H,W,C) or (H,W)
            if len(value.shape) in [2, 3]:
                wandb.log({key: wandb.Image(value)})
            else:
                print(f"Warning: Cannot log numpy array with shape {value.shape} as an image")
                
        # Handle PyTorch tensors as potential images
        elif isinstance(value, torch.Tensor) and len(value.shape) >= 2:
            # Move to CPU and convert to numpy
            value_np = value.detach().cpu().numpy()
            # Check if it's an image (H,W,C) or (H,W)
            if len(value_np.shape) in [2, 3]:
                wandb.log({key: wandb.Image(value_np)})
            else:
                print(f"Warning: Cannot log tensor with shape {value.shape} as an image")
                
        # For other types, try to log directly and let wandb handle it
        else:
            wandb.log({key: value})
            
    except Exception as e:
        print(f"Error in _log_special_metric for {key}: {e}")
        # Try a more general approach
        try:
            wandb.log({key: value})
        except:
            print(f"Could not log {key} to wandb at all")

def log_artifact(artifact_path, artifact_type="model", name=None, metadata=None):
    """Log an artifact file to wandb.
    
    Args:
        artifact_path: Path to the file to log
        artifact_type: Type of artifact (model, dataset, etc.)
        name: Name for the artifact (defaults to filename)
        metadata: Optional dict of metadata to associate with the artifact
        
    Returns:
        wandb.Artifact: The logged artifact object or None if logging failed
    """
    if not wandb.run:
        return None
        
    try:
        # Ensure path is valid
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            print(f"Warning: Artifact file {artifact_path} does not exist")
            return None
            
        # Use filename as name if not specified
        if name is None:
            name = f"{artifact_type}_{artifact_path.stem}"
            
        # Add run ID for uniqueness
        name = f"{name}_{wandb.run.id}"
        
        # Create and configure artifact
        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
        artifact.add_file(str(artifact_path))
        
        # Log artifact
        wandb.log_artifact(artifact)
        
        return artifact
        
    except Exception as e:
        print(f"Error logging artifact: {e}")
        return None

def reset_global_step(value=0):
    """Reset or set the global step counter.
    
    Args:
        value: The value to set the global step counter to (default: 0)
    """
    global _global_step
    _global_step = value
    print(f"Global wandb step counter reset to {value}")

def debug_media_file(file_path):
    """Check if a media file exists and has valid content.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        bool: True if file exists and appears valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # in MB
        print(f"File size: {file_size:.2f} MB")
        
        if file_size < 0.01:  # Less than 10KB
            print(f"Warning: File is likely empty or corrupt: {file_path}")
            return False
            
        # Check if it's a video file
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            try:
                import cv2
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    print(f"Video file cannot be opened: {file_path}")
                    return False
                    
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"Video info: {width}x{height}, {frame_count} frames")
                
                if frame_count < 2 or width == 0 or height == 0:
                    print(f"Video appears to be invalid: {file_path}")
                    return False
                    
                cap.release()
            except Exception as e:
                print(f"Error checking video file: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error checking file: {e}")
        return False
