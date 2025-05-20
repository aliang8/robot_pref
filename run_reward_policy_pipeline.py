#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path
import glob
import re

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

# ============================================================================
# PIPELINE CODE (you shouldn't need to edit below this line)
# ============================================================================

def train_reward_models():
    """Train reward models with multiple seeds."""
    print("\n" + "=" * 80)
    print(f"TRAINING REWARD MODELS: dataset={DATASET}")
    print("=" * 80)
    
    # Build the command using the template
    cmd = REWARD_MODEL_TEMPLATE.copy()
    
    # Add multirun configuration if enabled
    if USE_MULTIRUN:
        cmd.append(f"random_seed={RANDOM_SEEDS}")
        cmd.append(f"hydra/launcher={LAUNCHER}")
        cmd.append("--multirun")
    
    # Run reward model training with output capture
    print(f"Running command: {' '.join(cmd)}")
    
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
    
    return model_dir, model_paths


def train_policies(reward_model_path):
    """Train policies using the trained reward models."""
    print("\n" + "=" * 80)
    print(f"TRAINING POLICIES WITH {POLICY_ALGORITHM.upper()}: Using {reward_model_path}")
    print("=" * 80)
    
    print(f"\nTraining policy with reward model: {reward_model_path}")
        
    # Build the command using the template
    cmd = POLICY_TEMPLATE.copy()
    
    # Add reward model path - ensure it's absolute
    reward_model_path = os.path.abspath(reward_model_path)
    cmd.append(f"data.reward_model_path={reward_model_path}")
    
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
    # Step 1: Train reward models
    model_dir, model_paths = train_reward_models()
    
    # Step 2: Train policies using each reward model
    # If we're using multirun, we only need to train one policy with the reward model directory
    if USE_MULTIRUN:
        # Use the model directory instead of individual model paths
        train_policies(model_dir)
    else:
        # Otherwise, train one policy per reward model
        for model_path in model_paths:
            train_policies(model_path)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main() 