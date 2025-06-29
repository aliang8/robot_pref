#!/bin/bash

feedback_num=100

# Data
data_path="/scr/shared/datasets/robot_pref/stack_panda/stack_panda.hdf5"
target_data_path="/scr/shared/datasets/robot_pref/lift_sawyer/lift_sawyer.hdf5"

# RM Methods
use_distributional_model=false # Use distributional reward model
eef_rm=false # 3D EEF RM
use_dtw_augmentations=false
dtw_k_augment=1
use_gt_prefs=true

use_wandb=True
seeds=(4 5 6)

for seed in "${seeds[@]}"; do
    (
    echo "Running reward model learning for seed $seed"
    python3 learn_reward.py seed=$seed feedback_num=$feedback_num use_distributional_model=$use_distributional_model \
    eef_rm=$eef_rm use_dtw_augmentations=$use_dtw_augmentations dtw_k_augment=$dtw_k_augment use_gt_prefs=$use_gt_prefs \
    use_wandb=$use_wandb data_path=$data_path target_data_path=$target_data_path seed=$seed

    echo "Running IQL with reward model for seed $seed"
    python3 iql.py use_reward_model=True seed=$seed feedback_num=$feedback_num use_distributional_model=$use_distributional_model \
    eef_rm=$eef_rm use_dtw_augmentations=$use_dtw_augmentations dtw_k_augment=$dtw_k_augment use_gt_prefs=$use_gt_prefs \
    use_wandb=$use_wandb data_path=$data_path seed=$seed
    ) &
done

# Wait for all background jobs to finish
wait
