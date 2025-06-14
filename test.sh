#!/bin/bash

env=robomimic_lift   # ["metaworld_button-press-topdown-v2", "dmc_cheetah-run"]: env name
data_quality=1.0    # data quality. 
feedback_num=500    # total feedback number (we use 500, 1000 feedback in the paper)
q_budget=1        # query budget (we use 100 in the paper)
feedback_type=RLT   # ["RLT", "SeqRank"]: RLT means ranked list
model_type=linear_BT       # ["BT", "linear_BT"]: BT means exponential bradley-terry model, and linear_BT use linear score function
epochs=5000          # we use 300 epochs in the paper, but more epochs (e.g., 5000) can be used for better performance
activation=tanh     # final activation function of the reward model (use tanh for bounded reward)
threshold=0.1       # Thresholds for determining tie labels (eqaully preferred pairs)
segment_size=32     # segment size
data_aug=none       # ["none", "temporal"]: if you want to use data augmentation (TDA), set data_aug=temporal
batch_size=512    
ensemble_num=3      # number of reward models to ensemble
ensemble_method=mean    # we average the reward values of the ensemble models
noise=0.0           # probability of preference labels (0.0 is noiseless label and 0.1 is 10% noise label)
human=False         # [True, False]: use human feedback or not

# augmentation settings
use_dtw_augmentations=True
dtw_subsample_size=20000
dtw_augmentation_size=2000
dtw_k_augment=1 # How many augmentations to generate for each source human
acquisition_threshold_low=0.25
acquisition_threshold_high=0.75
dtw_augment_before_training=False

use_goal_pos=False
use_relative_eef=False

data_path="/scr/shared/datasets/robot_pref/lift_panda/lift_panda.hdf5"
target_data_path="/scr/shared/datasets/robot_pref/lift_sawyer/lift_sawyer.hdf5"
use_gt_prefs=True

seeds=(42 43 44)

for seed in "${seeds[@]}"; do
    echo "Running reward model learning for seed $seed"
    CUDA_VISIBLE_DEVICES=0 python3 learn_reward.py --config=configs/reward.yaml --env=$env --human=$human \
    --data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
    --threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
    --segment_size=$segment_size --data_aug=$data_aug  --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method --batch_size=$batch_size \
    --use_dtw_augmentations=$use_dtw_augmentations --dtw_subsample_size=$dtw_subsample_size --dtw_augmentation_size=$dtw_augmentation_size \
    --dtw_k_augment=$dtw_k_augment --acquisition_threshold_low=$acquisition_threshold_low --acquisition_threshold_high=$acquisition_threshold_high \
    --dtw_augment_before_training=$dtw_augment_before_training --use_goal_pos=$use_goal_pos --use_relative_eef=$use_relative_eef --use_gt_prefs=$use_gt_prefs \
    --data_path=$data_path --target_data_path=$data_path

    echo "Running IQL with reward model for seed $seed"
    python3 iql.py --use_reward_model=True --config=configs/iql.yaml --env=$env \
    --data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
    --threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
    --segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
    --use_dtw_augmentations=$use_dtw_augmentations --dtw_subsample_size=$dtw_subsample_size --dtw_augmentation_size=$dtw_augmentation_size \
    --dtw_k_augment=$dtw_k_augment --acquisition_threshold_low=$acquisition_threshold_low --acquisition_threshold_high=$acquisition_threshold_high \
    --dtw_augment_before_training=$dtw_augment_before_training --data_path=$data_path --use_gt_prefs=$use_gt_prefs
done