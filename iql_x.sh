#!/bin/bash
feedback_num=500

# Data
data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"
cross_data_path="/scr/shared/datasets/robot_pref/stack_mixed_sawyer/stack_mixed_sawyer.hdf5"

# RM Methods
use_cross=true
use_goal_pos=true

# Human or gt prefs
human=true

use_wandb=True
seeds=(4 5 6)

for seed in "${seeds[@]}"; do
    (
    echo "Running reward model learning for seed $seed"
    python3 learn_reward.py seed=$seed feedback_num=$feedback_num use_cross=$use_cross \
    use_wandb=$use_wandb data_path=$data_path cross_data_path=$cross_data_path human=$human use_goal_pos=$use_goal_pos
    
    echo "Running IQL on unseen embodiment with reward model for seed $seed"
    python3 iql.py use_reward_model=True seed=$seed feedback_num=$feedback_num use_cross=$use_cross \
    use_wandb=$use_wandb data_path=$cross_data_path human=$human use_goal_pos=$use_goal_pos
    ) &
done

# Wait for all background jobs to finish
wait
