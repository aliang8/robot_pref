#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path
import glob
import re
import itertools

# ============================================================================
# EDIT THESE TEMPLATE COMMANDS TO CUSTOMIZE THE PIPELINE
# ============================================================================

# Dataset configuration
DATASET = "/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt"  # Dataset path

# Reward model training configuration
REWARD_MODEL_TEMPLATE = [
    "python", "train_reward_model.py",
    "num_seeds=1",                        # Number of different random seeds to use
    f"data.data_path={DATASET}",
    "data.num_pairs=100"
]

# Grid search parameters for regular reward model
REWARD_MODEL_GRID = {
    "data.num_pairs": ["100", "500", "1000"],  # Different numbers of preference pairs
}

# Reward model training configuration for active learning
REWARD_MODEL_TEMPLATE_ACTIVE = [
    "python", "train_reward_model_active.py",
    "num_seeds=1",                        
    f"data.data_path={DATASET}"
]

# Grid search parameters for active reward model
ACTIVE_REWARD_MODEL_GRID = {
    "active_learning.uncertainty_method": ["entropy"],  
    "active_learning.max_queries": ["50"],   
    "dtw_augmentation.enabled": [True, False]
}

# Policy training configuration
POLICY_ALGORITHM = "iql_mw"  # One of: "iql", "bc"
POLICY_TEMPLATE = [
    "python", "train_policy.py",
    f"--config-name={POLICY_ALGORITHM}",
    f"data.data_path={DATASET}"
]

# Multirun configuration
USE_MULTIRUN = False  # Set to True to use multirun
RANDOM_SEEDS = "521,522,523"  # Comma-separated list of seeds to use
LAUNCHER = "slurm"  # Launcher for multirun (usually "slurm" on clusters)

# Current pipeline mode
USE_ACTIVE_LEARNING = False 

# ============================================================================
# PIPELINE CODE (you shouldn't need to edit below this line)
# ============================================================================

def generate_grid_combinations(grid_params):
    """Generate all combinations of grid parameters."""
    if not grid_params:
        return [{}]  # Return empty dict if no grid params
    
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    # Convert to list of dicts
    param_dicts = []
    for combo in combinations:
        param_dict = {}
        for i, name in enumerate(param_names):
            param_dict[name] = combo[i]
        param_dicts.append(param_dict)
    
    return param_dicts


def train_reward_model(template, grid_params=None):
    """Train a reward model with specified parameters."""
    # Start with the base template
    cmd = template.copy()
    
    # Add grid parameters if specified
    grid_desc = ""
    if grid_params:
        for param_name, param_value in grid_params.items():
            # Remove any existing parameter with the same name
            cmd = [arg for arg in cmd if not arg.startswith(f"{param_name}=")]
            # Add the new parameter
            cmd.append(f"{param_name}={param_value}")
            grid_desc += f"_{param_name.split('.')[-1]}_{param_value}"
    
    # Add multirun configuration if enabled
    if USE_MULTIRUN:
        cmd.append(f"random_seed={RANDOM_SEEDS}")
        cmd.append(f"hydra/launcher={LAUNCHER}")
        cmd.append("--multirun")
    
    # Run reward model training with output capture
    print("\n" + "=" * 80)
    print(f"TRAINING REWARD MODEL: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run the command and capture its output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    model_saved_pattern = re.compile(r"Model saved to: (.+/model_\d+\.pt)")
    model_dir = None
    model_paths = []
    
    # Process and print output in real-time
    for line in process.stdout:
        print(line, end='')
        
        # Check if this line contains the model path
        match = model_saved_pattern.search(line)
        if match:
            # Extract the path and ensure it's absolute
            model_path = match.group(1)
            model_path = os.path.abspath(model_path)
            model_paths.append(model_path)
            model_dir = os.path.dirname(model_path)
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode != 0:
        print(f"Error: Reward model training failed with exit code {process.returncode}")
        sys.exit(1)
    
    if not model_dir:
        print("Error: Could not determine model directory from command output")
        sys.exit(1)
    
    print(f"Found model directory: {model_dir}")
    print(f"Found {len(model_paths)} trained reward models")
    
    return model_dir, model_paths, grid_desc


def train_policy(reward_model_path, grid_desc=""):
    """Train a policy using the trained reward model."""
    print("\n" + "=" * 80)
    print(f"TRAINING POLICY WITH {POLICY_ALGORITHM.upper()}: Using {reward_model_path}")
    print("=" * 80)
    
    print(f"\nTraining policy with reward model: {reward_model_path}")
        
    # Build the command using the template
    cmd = POLICY_TEMPLATE.copy()
    
    # Add reward model path - ensure it's absolute
    reward_model_path = os.path.abspath(reward_model_path)
    cmd.append(f"data.reward_model_path={reward_model_path}")
    
    # Add a descriptive name based on grid parameters
    timestamp = int(time.time())
    dataset_name = Path(DATASET).stem
    cmd.append(f"wandb.name={POLICY_ALGORITHM}_{dataset_name}{grid_desc}_{timestamp}")
    
    # Add multirun configuration if enabled
    if USE_MULTIRUN:
        cmd.append(f"wandb.use_wandb=true")
        cmd.append(f"random_seed={RANDOM_SEEDS}")
        cmd.append(f"hydra/launcher={LAUNCHER}")
        cmd.append("--multirun")
    
    # Run policy training
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    # Choose which reward model template and grid to use
    if USE_ACTIVE_LEARNING:
        template = REWARD_MODEL_TEMPLATE_ACTIVE
        grid = ACTIVE_REWARD_MODEL_GRID
    else:
        template = REWARD_MODEL_TEMPLATE
        grid = REWARD_MODEL_GRID
    
    # Generate all parameter combinations
    param_combinations = generate_grid_combinations(grid)

    print("=" * 80)
    print("PARAM COMBINATIONS")
    print(param_combinations)
    print("=" * 80)
    
    # Train reward models for each parameter combination
    for i, params in enumerate(param_combinations):
        # print params
        print("=" * 80)
        print(f"ITERATION {i+1}/{len(param_combinations)}:")
        print(params)
        print("=" * 80)

        # Train reward model with these parameters
        model_dir, model_paths, grid_desc = train_reward_model(template, params)
        
        # Train policy using each reward model
        if USE_MULTIRUN:
            # Use the model directory instead of individual model paths
            train_policy(model_dir, grid_desc)
        else:
            # Otherwise, train one policy per reward model
            for model_path in model_paths:
                train_policy(model_path, grid_desc)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main() 