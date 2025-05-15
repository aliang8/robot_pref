# bc
python train_policy.py --config-name=bc_robomimic \
    data.env_name=can \
    data.data_path=/scr2/shared/pref/datasets/robomimic/can/mg_image_dense.pt

# train reward model with active offline learning
python train_reward_model_active.py \
    data.data_path="/scr2/shared/pref/datasets/robomimic/lift/mg_image_dense.pt" \
    active_learning.uncertainty_method=random \
    data.segment_length=64 \
    active_learning.max_queries=50
    
# train reward model with active offline learning + augmentations
python train_reward_model_active.py \
    data.data_path="/scr2/shared/pref/datasets/robomimic/lift/mg_image_dense.pt" \
    active_learning.uncertainty_method=random \
    data.segment_length=64 \
    active_learning.max_queries=50 \
    dtw_augmentation.enabled=true

# train policy using learned reward model
python3 train_policy.py --config-name=iql_robomimic \
    data.env_name=can \
    data.data_path="/scr2/shared/pref/datasets/robomimic/can/mg_image_dense.pt" \
    data.reward_model_path=/scr/matthewh6/robot_pref/results/active_reward_model/ \
    random_seed=42

