#!/bin/bash

feedback_num=1000

# Data
data_path="/scr/shared/datasets/robot_pref/stack_mixed/stack_mixed.hdf5"
target_data_path=""

# RM Methods
eef_rm=false # 3D EEF RM
use_dtw=false
dtw_k_augment=1
use_gt_prefs=true
human=false # Use human feedback or gt rewards

use_wandb=True
seeds=(521 523 524 525 526)

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
