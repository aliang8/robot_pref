#!/bin/bash

# data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"
# data_path="/tmp/mimicgen_stack_100/stack/demo_src_stack_task_D0_r_Panda/demo.hdf5"
data_path="/tmp/square_500/square/demo_src_square_task_D0_r_Panda/demo_success.hdf5"
# data_path="/tmp/square_500/square/demo_src_square_task_D0_r_Panda/demo.hdf5"


use_wandb=True
seeds=(1 2 3)

for seed in "${seeds[@]}"; do
    echo "Launching experiment with seed $seed..."
    python3 bc.py data_path=$data_path seed=$seed use_wandb=$use_wandb &
done

wait  # Wait for all background jobs to finish
echo "All experiments completed."