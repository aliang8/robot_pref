#!/bin/bash
# run iql zero and iql gt rewards
data_path="/tmp/mimicgen_stack_100/stack/demo_src_stack_task_D0_r_Panda/demo_exp-100_sub-150.hdf5"

use_wandb=True
seeds=(521 522 523)
record_video=False
trivial_reward=(0 1)

for seed in "${seeds[@]}"; do
    for trivial in "${trivial_reward[@]}"; do
        echo "Running seed $seed with trivial_reward=$trivial"
        # Run IQL with the specified seed and trivial_reward
        python3 iql.py seed=$seed use_wandb=$use_wandb data_path=$data_path trivial_reward=$trivial record_video=$record_video &
    done
done

# Wait for all background jobs to finish
wait
