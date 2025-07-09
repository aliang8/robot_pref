#!/bin/bash
# run iql zero and iql gt rewards
data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"

use_wandb=True
seeds=(40 41 42 43 44 45)
record_video=False
trivial_reward=(1)

# iql_tau=0.3
# beta=0.5

for seed in "${seeds[@]}"; do
    for trivial in "${trivial_reward[@]}"; do
        echo "Running seed $seed with trivial_reward=$trivial"
        # Run IQL with the specified seed and trivial_reward
        python3 iql.py seed=$seed use_wandb=$use_wandb data_path=$data_path trivial_reward=$trivial record_video=$record_video &
    done
done

# Wait for all background jobs to finish
wait
