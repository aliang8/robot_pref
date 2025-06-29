#!/bin/bash

data_path="/scr/shared/datasets/robot_pref/stack_panda/stack_panda.hdf5"
use_wandb=True
seeds=(7 8 9 10 11 12)

for seed in "${seeds[@]}"; do
    echo "Launching experiment with seed $seed..."
    python3 bc.py data_path=$data_path seed=$seed use_wandb=$use_wandb&
done

wait  # Wait for all background jobs to finish
echo "All experiments completed."