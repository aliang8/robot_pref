#!/bin/bash
feedback_num=500

# Data
data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"
cross_data_path="/scr/shared/datasets/robot_pref/stack_mixed_sawyer/stack_mixed_sawyer.hdf5"

# RM Methods
eef_rm=true
single_emb=true

eef_rm_2d=true
use_cross=false
use_goal_pos=false

# Human or gt prefs
human=true

use_wandb=true
seeds=(1 2 3)

for seed in "${seeds[@]}"; do
    (
    echo "Running reward model learning for seed $seed"
    python3 learn_reward.py seed=$seed feedback_num=$feedback_num use_cross=$use_cross \
    use_wandb=$use_wandb data_path=$data_path human=$human use_goal_pos=$use_goal_pos \
    eef_rm=$eef_rm eef_rm_2d=$eef_rm_2d single_emb=$single_emb

    echo "Running IQL on unseen embodiment with reward model for seed $seed"
    python3 iql.py use_reward_model=True seed=$seed feedback_num=$feedback_num use_cross=$use_cross \
    use_wandb=$use_wandb data_path=$cross_data_path human=$human use_goal_pos=$use_goal_pos \
    eef_rm=$eef_rm eef_rm_2d=$eef_rm_2d single_emb=$single_emb
    ) &
done

# Wait for all background jobs to finish
wait
