# Robot Preference Learning

## Setup

```bash
pip install wandb hydra-core d3rlpy
```

## Dataset Filtering

```bash
# List available datasets and use the first one
python create_mixed_expertise_dataset.py

# Specify a particular dataset by index
python create_mixed_expertise_dataset.py --dataset_idx 1

# Specify a custom dataset path and output directory
python create_mixed_expertise_dataset.py --data_path "/path/to/dataset.pt" --output_dir "my_balanced_datasets"
```

## End-Effector Clustering

```bash
python eef_clustering.py clustering.n_clusters=7 clustering.linkage_method=ward
```

## End-Effector Segment Matching

```bash
# Basic segment matching using default parameters
python eef_segment_matching.py

# Customize data parameters
python eef_segment_matching.py data.data_paths='["/path/to/dataset1.pt", "/path/to/dataset2.pt"]'
python eef_segment_matching.py data.segment_length=32 data.samples_per_dataset=300

# Customize matching parameters
python eef_segment_matching.py matching.top_k=10
python eef_segment_matching.py matching.query_indices=[5,10,15]  # Process multiple specific query segments

# Visualization options
python eef_segment_matching.py visualization.create_videos=false
python eef_segment_matching.py visualization.use_shared_ranges=true

# Change random seed and output directory
python eef_segment_matching.py random_seed=123 output.output_dir="./segment_matching_results"
```

## Collect Preferences

```bash
# Basic preference collection using clustering results
python collect_cluster_preferences.py data.clustering_results="/path/to/clustering_results.pkl"

# Specify data path and output directory
python collect_cluster_preferences.py data.data_path="/path/to/dataset.pt" output.output_dir="./preference_results"

# Customize preference collection parameters
python collect_cluster_preferences.py preferences.n_representatives=5 preferences.max_comparisons=20

# Skip video generation for faster collection
python collect_cluster_preferences.py preferences.skip_videos=true

# Use automatic preferences based on ground truth rewards (no user input required)
python collect_cluster_preferences.py preferences.use_automatic_preferences=true
```

## Collect Sequential Preferences with Similarity-Based Augmentation

```bash
# Basic sequential preference collection
python collect_sequential_pref.py

# Customize preference collection parameters
python collect_sequential_pref.py preferences.n_queries=50 preferences.k_augment=10

# Specify data path and output directory
python collect_sequential_pref.py data.data_path="/path/to/dataset.pt" output.output_dir="./custom_results"

# Adjust segment parameters
python collect_sequential_pref.py data.segment_length=20 data.max_segments=1000

# Configure DTW distance options
python collect_sequential_pref.py preferences.use_dtw_distance=true preferences.max_dtw_segments=500

# Control visualization generation
python collect_sequential_pref.py visualize=true max_visualizations=5 max_augmentations=10
```

Results will be saved in a directory structure like:
```
output_dir/
├── n50_k10_seed42_dtw500/
│   ├── augmentation_visualizations/
│   ├── preference_dataset.pkl
│   └── raw_preferences.pkl
└── n100_k5_seed42_dtw1000/
    ├── augmentation_visualizations/
    ├── preference_dataset.pkl
    └── raw_preferences.pkl
```

## Train Reward Model

```bash
# Basic reward model training
python train_reward_model.py data.data_path="/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt"

# Custom parameters
python train_reward_model.py data.data_path="/path/to/dataset.pt" model.hidden_dims=[256,256] training.num_epochs=50

# Run with SLURM
python train_reward_model.py --multirun
```

## Train Reward Model with Active Learning

```bash
# Train reward model with active learning using disagreement-based uncertainty sampling
python train_reward_model_sampling.py active_learning.uncertainty_method="disagreement" active_learning.num_models=5

# Enable fine-tuning between active learning iterations
python train_reward_model_sampling.py active_learning.fine_tune=true active_learning.fine_tune_lr=5e-5
```

## Train Policy

### IQL (Implicit Q-Learning)

```bash
# Basic IQL policy training
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" data.reward_model_path="reward_model/state_action_reward_model.pt"

# With video recording
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" evaluation.record_video=true

# With custom parameters
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" training.n_epochs=200 model.actor_learning_rate=3e-4
```

### BC (Behavior Cloning)

```bash
# Basic BC policy training
python train_policy.py --config-name=bc data.data_path="/path/to/dataset.pt"

# With custom parameters
python train_policy.py --config-name=bc data.data_path="/path/to/dataset.pt" model.learning_rate=1e-4 training.n_epochs=200 evaluation.record_video=true

/scr/aliang80/robot_pref/labeled_datasets/buffer_assembly-v2_balanced.pt