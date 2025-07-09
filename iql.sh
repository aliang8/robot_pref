#!/bin/bash

feedback_num=1000

# Data
data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"
target_data_path="" # x-emb

# RM Methods
use_gt_prefs=true
eef_rm=false
use_dtw=false

# Human or gt prefs
human=true

use_wandb=True
seeds=(40 41 42)

for seed in "${seeds[@]}"; do
    (
    echo "Running reward model learning for seed $seed"
    python3 learn_reward.py seed=$seed feedback_num=$feedback_num \
    eef_rm=$eef_rm use_dtw=$use_dtw dtw_k_augment=$dtw_k_augment use_gt_prefs=$use_gt_prefs \
    use_wandb=$use_wandb data_path=$data_path target_data_path=$target_data_path human=$human

    echo "Running IQL with reward model for seed $seed"
    python3 iql.py use_reward_model=True seed=$seed feedback_num=$feedback_num \
    eef_rm=$eef_rm use_dtw=$use_dtw dtw_k_augment=$dtw_k_augment use_gt_prefs=$use_gt_prefs \
    use_wandb=$use_wandb data_path=$data_path human=$human
    ) &
done

# Wait for all background jobs to finish
wait
