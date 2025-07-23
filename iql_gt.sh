#!/bin/bash
data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"
# data_path="/tmp/mimicgen_stack_1000/stack/demo_src_stack_task_D0_r_Sawyer/demo_exp-250_sub-250_noise-0.8.hdf5"

use_wandb=True
seeds=(7 8 9)
record_video=True
trivial_reward=(0)

for seed in "${seeds[@]}"; do
    for trivial in "${trivial_reward[@]}"; do
        echo "Running seed $seed with trivial_reward=$trivial"
        python3 iql.py seed=$seed use_wandb=$use_wandb data_path=$data_path trivial_reward=$trivial record_video=$record_video &
    done
done

wait
