import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Import utility functions
from trajectory_utils import (
    load_tensordict,
    create_segments,
    sample_segment_pairs
)
from utils.wandb_utils import log_to_wandb, log_artifact

# Import shared models and utilities
from models import SegmentRewardModel
from utils import (
    PreferenceDataset,
    bradley_terry_loss,
    create_data_loaders,
    evaluate_model_on_test_set,
    load_preferences_data,
    train_reward_model
)

@hydra.main(config_path="config", config_name="reward_model", version_base=None)
def main(cfg: DictConfig):
    """Train a state-action reward model using BT loss with Hydra config."""
    print("\n" + "=" * 50)
    print("Training reward model with Bradley-Terry preference learning")
    print("=" * 50)
    
    # Print config for visibility
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    random_seed = cfg.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Also set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Reduce memory fragmentation
        torch.cuda.empty_cache()
        # Use more aggressive memory caching if available (PyTorch 1.11+)
        if hasattr(torch.cuda, 'memory_stats'):
            print("Enabling memory_stats for better CUDA memory management")
            torch.cuda.memory_stats(device=None)
        # Set memory allocation strategy to avoid fragmentation
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # Use 80% of available memory to leave room for system
            torch.cuda.set_per_process_memory_fraction(0.8, 0)
            print("Set CUDA memory fraction to 80%")
            
    print(f"Using random seed: {random_seed}")
    
    # Initialize wandb
    if cfg.wandb.use_wandb:
        # Generate experiment name based on data path
        dataset_name = Path(cfg.data.data_path).stem
        
        # Set up a run name if not specified
        run_name = cfg.wandb.name
        if run_name is None:
            run_name = f"reward_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes
        )
        
        print(f"Wandb initialized: {wandb.run.name}")
    
    output_dir = cfg.output.output_dir

    # Get dataset name for the subdirectory
    dataset_name = Path(cfg.data.data_path).stem
    
    # If using preference data, include that in the output directory path
    pref_dataset_info = ""
    if hasattr(cfg.data, 'preferences_data_path') and cfg.data.preferences_data_path:
        # Extract key parameters from the preference dataset path
        pref_path = Path(cfg.data.preferences_data_path)
        pref_dir = pref_path.parent
        
        # Try to extract parameters from directory name (e.g., n50_k10_seed42_dtw500)
        dir_parts = pref_dir.name.split('_')
        
        # Extract n_queries and k_augment if present in the directory name
        n_queries = next((part[1:] for part in dir_parts if part.startswith('n') and part[1:].isdigit()), "")
        k_augment = next((part[1:] for part in dir_parts if part.startswith('k') and part[1:].isdigit()), "")
        
        if n_queries and k_augment:
            pref_dataset_info = f"_n{n_queries}_k{k_augment}"
        else:
            # If we can't extract parameters, just use the parent directory name
            pref_dataset_info = f"_{pref_dir.name}"
    
    os.makedirs(output_dir, exist_ok=True)
    model_dir = output_dir
    
    # Setup CUDA device
    if cfg.hardware.use_cpu:
        device = torch.device("cpu")
    else:
        cuda_device = f"cuda:{cfg.hardware.gpu}" if torch.cuda.is_available() else "cpu"
        device = torch.device(cuda_device)
    
    print(f"Using device: {device}")
    
    # Determine effective GPU and CPU settings
    effective_num_workers = cfg.training.num_workers
    effective_pin_memory = cfg.training.pin_memory
    
    # Adjust for CPU-only mode
    if device.type == "cpu":
        print("Running in CPU mode")
        effective_pin_memory = False
    
    # Initialize variables
    data_cpu = None
    segments = None
    segment_pairs = None
    segment_indices = None
    preferences = None
    state_dim = None
    action_dim = None
    
    # First, try to load from preference data if specified
    if hasattr(cfg.data, 'preferences_data_path') and cfg.data.preferences_data_path:
        print(f"Loading preference data from {cfg.data.preferences_data_path}")
        
        # Load the preference data
        segment_pairs, segment_indices, preferences, embedded_data = load_preferences_data(cfg.data.preferences_data_path)
        
        # Use embedded data if available
        if embedded_data is not None:
            print("Using embedded data from preference file")
            data_cpu = embedded_data
            
            # Get observation and action dimensions from embedded data
            if 'obs' in data_cpu:
                observations = data_cpu['obs']
                state_dim = observations.shape[1]
            elif 'state' in data_cpu:
                observations = data_cpu['state']
                state_dim = observations.shape[1]
            
            if 'action' in data_cpu:
                actions = data_cpu['action']
                action_dim = actions.shape[1]
            
            print(f"Embedded data contains fields: {list(data_cpu.keys())}")
            print(f"Observation shape: {observations.shape}, Action shape: {actions.shape if 'action' in data_cpu else 'N/A'}")
        else:
            print("No embedded data found in preference file. Will load from original data file.")
    
    # If we don't have data yet, load from the original file
    if data_cpu is None:
        print(f"Loading data from original file: {cfg.data.data_path}")
        data = load_tensordict(cfg.data.data_path)
        
        # Get observation and action dimensions
        observations = data["obs"] if "obs" in data else data["state"]
        actions = data["action"]
        state_dim = observations.shape[1]
        action_dim = actions.shape[1]
        
        # Ensure data is on CPU for processing
        data_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        print(f"Loaded data with {len(observations)} observations")
    
    # If we still don't have necessary preference data, generate it
    if segment_pairs is None or preferences is None:
        if segment_indices is not None and segments is None:
            # Extract segments from data using provided indices
            print("Extracting segments from data using provided segment indices...")
            segments = []
            for start_idx, end_idx in tqdm(segment_indices, desc="Extracting segments"):
                segment_obs = data_cpu["obs"][start_idx:end_idx+1]
                segments.append(segment_obs)
            print(f"Extracted {len(segments)} segments")
        
        # If we still don't have segments, generate them
        if segments is None or segment_indices is None:
            print(f"Creating segments of length {cfg.data.segment_length}...")
            num_segments = cfg.data.num_segments if cfg.data.num_segments > 0 else None
            segments, segment_indices = create_segments(data_cpu, segment_length=cfg.data.segment_length, max_segments=num_segments)
        
        # If we still don't have pairs or preferences, generate them
        if segment_pairs is None or preferences is None:
            print(f"Generating {cfg.data.num_pairs} preference pairs...")
            segment_pairs, preferences = sample_segment_pairs(
                segments, 
                segment_indices, 
                data_cpu["reward"], 
                n_pairs=cfg.data.num_pairs
            )
    else:
        print(f"Using {len(segment_pairs)} preference pairs loaded from preference data file")
    
    print(f"Final data stats - Observation dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Working with {len(segment_pairs) if segment_pairs is not None else 0} preference pairs across {len(segment_indices) if segment_indices is not None else 0} segments")
    
    # Create dataset
    preference_dataset = PreferenceDataset(
        data_cpu, 
        segment_pairs,
        segment_indices,
        preferences
    )
    
    # Create data loaders
    dataloaders = create_data_loaders(
        preference_dataset,
        train_ratio=0.8,  # 80% for training
        val_ratio=0.1,    # 10% for validation
        batch_size=cfg.training.batch_size,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        seed=random_seed  # Use the same random seed
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Log dataset information to wandb
    if cfg.wandb.use_wandb:
        wandb.config.update({
            "dataset": {
                "name": Path(cfg.data.data_path).stem,
                "total_pairs": len(preference_dataset),
                "train_size": dataloaders['train_size'],
                "val_size": dataloaders['val_size'],
                "test_size": dataloaders['test_size'],
                "observation_dim": state_dim,
                "action_dim": action_dim
            }
        })
    
    # Initialize reward model
    model = SegmentRewardModel(state_dim, action_dim, hidden_dims=cfg.model.hidden_dims)
    
    # Log model info to wandb
    if cfg.wandb.use_wandb:
        # Use a different key name to avoid conflicts with existing 'model' key
        wandb.config.update({
            "model_info": {
                "hidden_dims": cfg.model.hidden_dims,
                "total_parameters": sum(p.numel() for p in model.parameters())
            }
        })
        
        # Log model graph if possible
        if hasattr(wandb, 'watch'):
            wandb.watch(model, log="all")
    
    print(f"Reward model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Start timing the training
    start_time = time.time()
    
    # Train the model
    print("\nTraining reward model...")
    model, train_losses, val_losses = train_reward_model(
        model, train_loader, val_loader, device, 
        num_epochs=cfg.training.num_epochs, lr=cfg.model.lr,
        wandb=wandb if cfg.wandb.use_wandb else None
    )
    
    # Calculate and print training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Evaluate on test set
    model = model.to(device)
    test_metrics = evaluate_model_on_test_set(model, test_loader, device)
    
    # Log final test results to wandb
    if cfg.wandb.use_wandb:
        log_to_wandb(test_metrics, prefix="test")
    
    # Create a descriptive model filename
    dataset_name = Path(cfg.data.data_path).stem
    hidden_dims_str = "_".join(map(str, cfg.model.hidden_dims))
    
    # Also save a version with more detailed filename for versioning
    sub_dir = f"{dataset_name}{pref_dataset_info}_model_seg{cfg.data.segment_length}_hidden{hidden_dims_str}_epochs{cfg.training.num_epochs}_pairs{cfg.data.num_pairs}"
    
    os.makedirs(os.path.join(model_dir, sub_dir), exist_ok=True)
    model_path = os.path.join(model_dir, sub_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved with detailed name: {model_path}")
    
    # Log as wandb artifact
    if cfg.wandb.use_wandb:
        try:
            # Create metadata about the model
            metadata = {
                "observation_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dims": cfg.model.hidden_dims,
                "test_accuracy": test_metrics["test_accuracy"],
                "test_loss": test_metrics["test_loss"],
                "num_segments": len(segments) if segments is not None else 0,
                "num_pairs": len(segment_pairs) if segment_pairs is not None else 0,
                "segment_length": cfg.data.segment_length,
                "preference_data": cfg.data.preferences_data_path if hasattr(cfg.data, 'preferences_data_path') else None
            }
            
            # Create and log artifact
            artifact = log_artifact(
                model_path, 
                artifact_type="reward_model", 
                name=f"{dataset_name}{pref_dataset_info}_seg{cfg.data.segment_length}_pairs{cfg.data.num_pairs}_epochs{cfg.training.num_epochs}", 
                metadata=metadata
            )
            if artifact:
                print(f"Model logged to wandb as artifact: {artifact.name}")
        except Exception as e:
            print(f"Warning: Could not log model as wandb artifact: {e}")
    
    # Save segment info
    segment_info = {
        "segment_length": cfg.data.segment_length,
        "num_segments": len(segments) if segments is not None else 0,
        "num_pairs": len(segment_pairs) if segment_pairs is not None else 0,
        "observation_dim": state_dim,
        "action_dim": action_dim,
        "training_losses": train_losses,
        "validation_losses": val_losses,
        "test_metrics": test_metrics,
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    info_filename = f"info.pkl"
    info_path = os.path.join(model_dir, sub_dir, info_filename)
    with open(info_path, "wb") as f:
        pickle.dump(segment_info, f)
    
    print(f"Model information saved to {info_path}")
    
    # Finish wandb run
    if cfg.wandb.use_wandb and wandb.run:
        wandb.finish()
        
    print("\nReward model training complete!")

if __name__ == "__main__":
    main() 