# bc
python train_policy.py --config-name=bc_robomimic \
    data.env_name=can \
    data.data_path=/scr/shared/datasets/robot_pref/can_mh/can_mh.pt

# precompute dtw matrix
python preprocess_dtw_matrix.py \
    --data_path="/scr2/shared/pref/datasets/robomimic/lift/mg_image_dense.pt" \
    --segment_length=32

# train reward model with active offline learning
python train_reward_model_active.py \
    data.data_path="/scr2/shared/pref/datasets/robomimic/lift/mg_image_dense.pt" \
    active_learning.uncertainty_method=random \
    data.segment_length=32 \
    active_learning.total_queries=100

# train reward model with active offline learning + augmentations
python train_reward_model_active.py \
    data.data_path="/scr2/shared/pref/datasets/robomimic/lift/mg_image_dense.pt" \
    active_learning.uncertainty_method=random \
    data.segment_length=32 \
    active_learning.total_queries=100 \
    dtw_augmentation.enabled=True

# train policy using learned reward model
python3 train_policy.py --config-name=iql_robomimic \
    data.env_name=can \
    data.data_path="/scr2/shared/pref/datasets/robomimic/can/mg_image_dense.pt" \
    data.reward_model_path="/scr/matthewh6/robot_pref/results/active_reward_model/can_mg_image_dense_balanced_active_disagreement_max100_1_augFalse_k5/checkpoints/checkpoint_iter_100.pt" \
    random_seed=42


