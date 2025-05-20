# OLD
python preprocess_segments.py --max_segments 2000 --use_relative_differences --output_file preprocessed/assembly_v2_segments.pt

python eef_clustering.py --preprocessed_data preprocessed/assembly_v2_segments.pt --n_clusters 3 --max_dtw_segments 500 --output_dir results/eef_clustering --use_relative_differences --skip_videos --linkage_method ward

python collect_cluster_preferences.py --preprocessed_data preprocessed/assembly_v2_segments.pt --clustering_results results/eef_clustering/clustering_results.pkl --output_dir results/preferences

python preprocess_dtw_matrix.py \
    --data_path=/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2_clean.pt \
    --segment_length=32 \
    --overwrite

python train_reward_model.py \
    data.data_path=/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2_clean.pt \
    data.num_pairs=100 

python train_reward_model_active.py \
    data.data_path=/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2_clean.pt \
    active_learning.uncertainty_method=entropy \
    dtw_augmentation.enabled=true \
    --multirun

# BC
python train_policy.py \
    --config-name=bc \
    data.data_path=/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2_clean.pt \
    wandb.use_wandb=true \
    random_seed=521,522,523 \
    hydra/launcher=slurm \
    --multirun


python train_policy.py \
    --config-name=iql \
    data.data_path=/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2_clean.pt \
    wandb.use_wandb=true \
    data.use_zero_rewards=true \
    model.weight_temp=1.0 \
    random_seed=521,522,523 \
    hydra/launcher=slurm \
    --multirun

    data.reward_model_path=/scr/shared/clam/robot_pref/results/reward_model/state_action_reward_model.pt \




# using balanced dataset
# make mixed expertise dataset
python create_mixed_expertise_dataset.py \
    --data_path=/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2_clean.pt \
    --output_dir=dataset_mw

# train reward model without active learning
python train_reward_model.py \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    data.num_pairs=100 

# preprocess DTW matrix
python preprocess_dtw_matrix.py \
    --data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    --segment_length=32 \
    --overwrite

# train reward model with active offline learning
python train_reward_model_active.py \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    active_learning.uncertainty_method=disagreement \
    --multirun

# with augmentations
python train_reward_model_active.py \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    active_learning.uncertainty_method=disagreement \
    dtw_augmentation.enabled=true \
    training.num_epochs=100
    --multirun

# baseline - use ground truth rewards
python train_policy.py \
    --config-name=iql \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    wandb.use_wandb=true \
    data.use_ground_truth=true \
    data.scale_rewards=true \
    random_seed=521,522,523 \
    hydra/launcher=slurm \
    --multirun

# baseline - zero rewards
python train_policy.py \
    --config-name=iql \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    wandb.use_wandb=true \
    data.use_zero_rewards=true \
    random_seed=521,522,523 \
    hydra/launcher=slurm \
    --multirun

# baseline - BC
# BC
python train_policy.py \
    --config-name=bc \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    wandb.use_wandb=true \
    random_seed=521,522,523 \
    hydra/launcher=slurm \
    --multirun


# train policy using learned reward model
python train_policy.py \
    --config-name=iql \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    wandb.use_wandb=true \
    random_seed=521,522,523 \
    data.reward_model_path=/scr/aliang80/robot_pref/results/active_reward_model/

# train reward model with active
python train_reward_model_active.py \
    data.data_path=/scr/aliang80/robot_pref/dataset_mw/buffer_assembly-v2_balanced.pt \
    active_learning.uncertainty_method=entropy,disagreement \
    active_learning.max_queries=50 \
    dtw_augmentation.enabled=true \
    hydra/launcher=slurm \
    --multirun