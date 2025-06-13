#!/bin/bash

env=robomimic_lift
data_quality=1.0
feedback_num=500
q_budget=1
feedback_type=RLT
model_type=linear_BT
epochs=3000
activation=tanh
seed=521
threshold=0.1
segment_size=32
data_aug=none
batch_size=512
ensemble_num=3
ensemble_method=mean
noise=0.0
human=False

use_dtw_augmentations=True
dtw_k_augment=1

use_gt_prefs=False
eef_rm=False

data_path="/scr/shared/datasets/robot_pref/lift_panda/lift_panda.hdf5"
target_data_path="/scr/shared/datasets/robot_pref/lift_sawyer/lift_sawyer.hdf5"

seeds=(527 528 529)


for seed in "${seeds[@]}"; do
    echo "Running reward model learning for seed $seed"
    python3 learn_reward.py --config=configs/reward.yaml --env=$env --human=$human \
    --data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
    --threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
    --segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method --batch_size=$batch_size \
    --use_dtw_augmentations=$use_dtw_augmentations --dtw_k_augment=$dtw_k_augment --eef_rm=$eef_rm \
    --data_path=$data_path --target_data_path=$target_data_path --use_gt_prefs=$use_gt_prefs

    echo "Running IQL with reward model for seed $seed"
    python3 iql.py --use_reward_model=True --config=configs/iql.yaml --env=$env \
    --data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
    --threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
    --segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
    --use_dtw_augmentations=$use_dtw_augmentations --dtw_k_augment=$dtw_k_augment --data_path=$data_path --eef_rm=$eef_rm --use_gt_prefs=$use_gt_prefs

done