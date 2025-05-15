python preprocess_segments.py --max_segments 2000 --use_relative_differences --output_file preprocessed/assembly_v2_segments.pt

python eef_clustering.py --preprocessed_data preprocessed/assembly_v2_segments.pt --n_clusters 3 --max_dtw_segments 500 --output_dir results/eef_clustering --use_relative_differences --skip_videos --linkage_method ward

python collect_cluster_preferences.py --preprocessed_data preprocessed/assembly_v2_segments.pt --clustering_results results/eef_clustering/clustering_results.pkl --output_dir results/preferences

# train reward model without active learning
python train_reward_model.py \
    data.data_path=/scr/aliang80/robot_pref/labeled_datasets/buffer_assembly-v2_balanced.pt \
    data.num_pairs=100 

# train reward model with active offline learning
python train_reward_model_active.py \
    data.data_path=/scr/aliang80/robot_pref/labeled_datasets/buffer_assembly-v2_balanced.pt \
    active_learning.uncertainty_method=entropy,disagreement \
    --multirun

# train policy using learned reward model
python train_policy.py \
    --config-name=iql \
    data.data_path=/scr/aliang80/robot_pref/labeled_datasets/buffer_assembly-v2_balanced.pt \
    wandb.use_wandb=true \
    random_seed=521,522,523 \
    data.reward_model_path=/scr/aliang80/robot_pref/results/active_reward_model/

# train reward model with active
python train_reward_model_active.py \
    data.data_path=/scr/aliang80/robot_pref/labeled_datasets/buffer_assembly-v2_balanced.pt \
    active_learning.uncertainty_method=entropy,disagreement \
    active_learning.max_queries=300 \
    dtw_augmentation.enabled=true \
    hydra/launcher=slurm \
    --multirun