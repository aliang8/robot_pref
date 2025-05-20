#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path
import glob
import re
import itertools
import pty
import fcntl
import select
import errno

# ============================================================================
# EDIT THESE TEMPLATE COMMANDS TO CUSTOMIZE THE PIPELINE
# ============================================================================

# Dataset configuration
DATASET = "/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt"  # Dataset path

# Reward model training configuration
REWARD_MODEL_TEMPLATE = [
    "python", "train_reward_model.py",
    "num_seeds=1",                        
    f"data.data_path={DATASET}",
    "data.num_pairs=100"
]

# Grid search parameters for regular reward model
REWARD_MODEL_GRID = {
    "data.num_pairs": [100, 500, 1000],  
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
    "active_learning.total_queries": [10],   
    "dtw_augmentation.enabled": [True, False]
}

# Policy training configuration
POLICY_ALGORITHM = "iql_mw"  # One of: "iql", "bc"
POLICY_TEMPLATE = [
    "python", "train_policy.py",
    f"--config-name={POLICY_ALGORITHM}",
    f"data.data_path={DATASET}",
    "training.n_epochs=100",              # Number of training epochs
    "data.use_ground_truth=false",        # Don't use ground truth rewards
    "data.use_zero_rewards=false",        # Don't use zero rewards
]

# Multirun configuration
USE_MULTIRUN = False  # Set to True to use multirun
RANDOM_SEEDS = "521,522,523"  # Comma-separated list of seeds to use
LAUNCHER = "slurm"  # Launcher for multirun (usually "slurm" on clusters)

# Current pipeline mode
USE_ACTIVE_LEARNING = True 

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


def run_with_tqdm_support(cmd):
    """Run a command with proper tqdm progress bar support."""
    print(f"Running command: {' '.join(cmd)}")
    
    # Create a pseudo-terminal to make the subprocess think it's running in an interactive terminal
    master, slave = pty.openpty()
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=slave,
        stderr=subprocess.STDOUT,
        text=True,
        close_fds=True
    )
    
    # Close the slave file descriptor as we're not using it directly
    os.close(slave)
    
    # Set the master to non-blocking mode
    flags = fcntl.fcntl(master, fcntl.F_GETFL)
    fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    
    output = ""
    
    # Read from the master until the process completes
    while True:
        try:
            # Check if there's data to read
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    # Read data
                    data = os.read(master, 4096).decode('utf-8', errors='replace')
                    if data:
                        # Print the data directly to show progress bars
                        sys.stdout.write(data)
                        sys.stdout.flush()
                        output += data
                except OSError as e:
                    if e.errno != errno.EIO:  # EIO is expected when the child process exits
                        raise
                    break
            
            # Check if the process has exited
            if process.poll() is not None:
                break
                
        except KeyboardInterrupt:
            process.terminate()
            break
    
    # Close the master file descriptor
    os.close(master)
    
    # Wait for the process to finish and get the return code
    returncode = process.wait()
    
    return returncode, output


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
    print("\n" + "#" * 100)
    print(f"## REWARD MODEL TRAINING ##")
    print(f"## Command: {' '.join(cmd)}")
    print("#" * 100)
    
    # Run the command with tqdm support
    returncode, output = run_with_tqdm_support(cmd)
    
    if returncode != 0:
        print(f"Error: Reward model training failed with exit code {returncode}")
        sys.exit(1)
    
    # Search for model paths in the output
    model_saved_pattern = re.compile(r"Model saved to: (.+/*\d+\.pt)")
    model_paths = []
    model_dir = None
    
    for match in model_saved_pattern.finditer(output):
        model_path = match.group(1)
        model_path = os.path.abspath(model_path)
        model_paths.append(model_path)
        model_dir = os.path.dirname(model_path)
    
    if not model_dir:
        print("Error: Could not determine model directory from command output")
        sys.exit(1)
    
    print("\n" + "-" * 80)
    print(f">> Found model directory: {model_dir}")
    print(f">> Found {len(model_paths)} trained reward models")
    print("-" * 80)
    
    return model_dir, model_paths, grid_desc


def train_policy(reward_model_path, grid_desc=""):
    """Train a policy using the trained reward model."""
    print("\n" + "*" * 100)
    print(f"** POLICY TRAINING **")
    print(f"** Policy Algorithm: {POLICY_ALGORITHM.upper()}")
    print(f"** Reward Model: {reward_model_path}")
    print("*" * 100)
        
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
    
    # Run policy training with tqdm support
    run_with_tqdm_support(cmd)


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

    print("\n" + "+" * 100)
    print("+" + " " * 98 + "+")
    print(f"+{' PIPELINE STARTING ':^98}+")
    print(f"+{f' Total Combinations: {len(param_combinations)} ':^98}+")
    print("+" + " " * 98 + "+")
    print("+" * 100)
    
    print("\n" + "~" * 100)
    print("~ PARAMETER COMBINATIONS:")
    for i, params in enumerate(param_combinations):
        print(f"~ Combo {i+1}: {params}")
    print("~" * 100)
    
    # Train reward models for each parameter combination
    for i, params in enumerate(param_combinations):
        print("\n" + "=" * 100)
        print(f"= GRID SEARCH ITERATION {i+1}/{len(param_combinations)}")
        print(f"= Parameters: {params}")
        print("=" * 100)

        # Train reward model with these parameters
        _, model_paths, grid_desc = train_reward_model(template, params)
        
        for model_path in model_paths:
            train_policy(model_path, grid_desc)
    
    print("\n" + "+" * 100)
    print("+" + " " * 98 + "+")
    print(f"+{' PIPELINE COMPLETED SUCCESSFULLY ':^98}+")
    print("+" + " " * 98 + "+")
    print("+" * 100)


if __name__ == "__main__":
    main() 