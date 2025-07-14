#!/bin/bash
feedback_num=500

# data paths
data_path="/scr/shared/datasets/robot_pref/stack_mixed_sawyer/stack_mixed_sawyer.hdf5"
cross_data_path="" # x-emb-transfer

# RM Methods
single_emb=false
eef_rm=false
use_cross=false

# Human or gt prefs
human=true

use_wandb=True
seeds=(46 47 48)

for seed in "${seeds[@]}"; do
    (
    # echo "Running reward model learning for seed $seed"
    # python3 learn_reward.py seed=$seed feedback_num=$feedback_num \
    # eef_rm=$eef_rm use_cross=$use_cross dtw_k_augment=$dtw_k_augment single_emb=$single_emb \
    # use_wandb=$use_wandb data_path=$data_path cross_data_path=$cross_data_path human=$human

    echo "Running IQL with reward model for seed $seed"
    python3 iql.py use_reward_model=True seed=$seed feedback_num=$feedback_num \
    eef_rm=$eef_rm use_cross=$use_cross dtw_k_augment=$dtw_k_augment single_emb=$single_emb \
    use_wandb=$use_wandb data_path=$data_path human=$human
    ) &
done

# Wait for all background jobs to finish
wait
