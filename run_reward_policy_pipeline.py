#!/usr/bin/env python3
import errno
import fcntl
import itertools
import os
import pty
import re
import select
import subprocess
import sys
from pathlib import Path

# ============================================================================
# EDIT THESE TEMPLATE COMMANDS TO CUSTOMIZE THE PIPELINE
# ============================================================================

# Dataset configuration
DATASET = "/project2/biyik_1165/hongmm/pref/datasets/robomimic/can/can_mg_image_dense_balanced.pt"
ENV_NAME = "can"  # Environment name

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
    "active_learning.uncertainty_method": ["entropy", "disagreement"],  
    "active_learning.total_queries": [25,50,100],
    "dtw_augmentation.enabled": [True, False],
    "dtw_augmentation.use_heuristic_beta": [True, False],
    "random_seed": [521]
}

# Policy training configuration
POLICY_ALGORITHM = "iql_robomimic"  # One of: "iql", "bc"
POLICY_TEMPLATE = [
    "python", "train_policy.py",
    f"--config-name={POLICY_ALGORITHM}",
    f"data.data_path={DATASET}",
    f"data.env_name={ENV_NAME}",
    "training.n_epochs=125",              # Number of training epochs
    "data.use_ground_truth=false",        # Don't use ground truth rewards
    "data.use_zero_rewards=false",        # Don't use zero rewards
]

USE_MULTIRUN = True  # Set to True to use multirun
RANDOM_SEEDS = "521"  # Comma-separated list of seeds to use
LAUNCHER = "slurm_carc"  # Launcher for multirun (usually "slurm" on clusters)

# Current pipeline mode
USE_ACTIVE_LEARNING = True

def generate_grid_combinations(grid_params):
    """Generate all combinations of grid parameters, but only use the max value for active_learning.total_queries."""
    if not grid_params:
        return [{}]  # Return empty dict if no grid params

    # If "active_learning.total_queries" is in the grid, only use its max value
    param_names = list(grid_params.keys())
    param_values = []
    for name in param_names:
        if name == "active_learning.total_queries":
            # Only use the max value
            max_val = max(grid_params[name])
            param_values.append([max_val])
        else:
            param_values.append(grid_params[name])
    
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

    # Run reward model training with output capture
    print("\n" + "#" * 100)
    print("## REWARD MODEL TRAINING ##")
    print(f"## Command: {' '.join(cmd)}")
    print("#" * 100)
    
    # Run the command with tqdm support
    returncode, output = run_with_tqdm_support(cmd)
    
    if returncode != 0:
        print(f"Error: Reward model training failed with exit code {returncode}")
        sys.exit(1)
    
    # Search for model paths in the output
    model_saved_pattern = re.compile(r"Model saved to: (.+/model_\d+\.pt)")
    model_saved_pattern_active = re.compile(r"Model saved to: (.+/checkpoint_iter_\d+\.pt)")
    model_paths = []
    model_dir = None
    
    for match in model_saved_pattern.finditer(output):
        model_path = match.group(1)
        model_path = os.path.abspath(model_path)
        model_paths.append(model_path)
        model_dir = os.path.dirname(model_path)

    # If nothing matched the first pattern, check the second
    if not model_paths:
        for match in model_saved_pattern_active.finditer(output):
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
    """Launch policy training in the background without tracking."""
    print("\n" + "*" * 80)
    print("** LAUNCHING POLICY TRAINING IN BACKGROUND **")
    print(f"** Policy Algorithm: {POLICY_ALGORITHM.upper()}")
    print(f"** Reward Model: {reward_model_path}")
    print("*" * 80)
        
    # Build the command using the template
    cmd = POLICY_TEMPLATE.copy()
    
    # Add reward model path - ensure it's absolute
    reward_model_path = os.path.abspath(reward_model_path)
    cmd.append(f"data.reward_model_path={reward_model_path}")
    
    # Add a descriptive name based on grid parameters
    dataset_name = Path(DATASET).stem
    job_name = f"{POLICY_ALGORITHM}_{dataset_name}{grid_desc}"
    
    # Add multirun configuration if enabled
    if USE_MULTIRUN:
        cmd.append("wandb.use_wandb=true")
        cmd.append(f"random_seed={RANDOM_SEEDS}")
        cmd.append(f"hydra/launcher={LAUNCHER}")
        cmd.append(f"hydra.job.name={job_name}")
        cmd.append("--multirun")
    
    # Create log file for the background process
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"policy_{job_name}.log")
    
    # Start policy training in background
    print(f"Running command in background: {' '.join(cmd)}")
    print(f"Job name: {job_name}")
    print(f"Log file: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True 
        )
    
    print(f"Started process with PID: {process.pid}")


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

        # Run train_policy for model_paths that match any value in ACTIVE_REWARD_MODEL_GRID["active_learning.total_queries"]
        total_queries_list = ACTIVE_REWARD_MODEL_GRID["active_learning.total_queries"]

        for model_path in model_paths:
            # Extract the iteration number from the checkpoint filename
            # e.g., .../checkpoint_iter_10.pt -> 10
            match = re.search(r"checkpoint_iter_(\d+)\.pt", model_path)

            if match:
                iter_num = int(match.group(1))
                if iter_num in total_queries_list:
                    train_policy(model_path, grid_desc)
            else:
                # If no match, fallback to running all (shouldn't happen)
                train_policy(model_path, grid_desc)
    
    print("\n" + "+" * 100)
    print("+" + " " * 98 + "+")
    print(f"+{' PIPELINE COMPLETED SUCCESSFULLY ':^98}+")
    print(f"+{' Policy training processes are running in the background ':^98}+")
    print("+" + " " * 98 + "+")
    print("+" * 100)


if __name__ == "__main__":
    main() 