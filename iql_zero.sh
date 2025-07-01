#!/bin/bash

data_path="/scr/shared/datasets/robot_pref/stack_panda/stack_panda.hdf5"
use_wandb=True
seeds=(4 5 6)

# Run zero rewards IQL for three seeds
for seed in "${seeds[@]}"; do
    echo "Running seed $seed"
    python3 iql.py seed=$seed \
    data_path=$data_path trivial_reward=1 use_wandb=$use_wandb seq_len=$seq_len&
done 

wait  # Wait for all background jobs to finish
echo "All experiments completed."