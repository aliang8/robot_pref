#!/bin/bash
# data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"
data_path="/scr/shared/datasets/robot_pref/stack_mixed_sawyer/stack_mixed_sawyer.hdf5"

use_wandb=True
seeds=(1 2 3)
record_video=False
trivial_reward=(0)

for seed in "${seeds[@]}"; do
    for trivial in "${trivial_reward[@]}"; do
        echo "Running seed $seed with trivial_reward=$trivial"
        python3 iql.py seed=$seed use_wandb=$use_wandb data_path=$data_path trivial_reward=$trivial record_video=$record_video &
    done
done

wait
