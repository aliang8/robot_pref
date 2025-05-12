# Robot Preference Learning

## Setup

```bash
pip install wandb hydra-core d3rlpy
```

## Dataset Filtering

Create a balanced dataset with equal numbers of random/medium/expert trajectories:

```bash
# List available datasets and use the first one
python create_mixed_expertise_dataset.py

# Specify a particular dataset by index
python create_mixed_expertise_dataset.py --dataset_idx 1

# Specify a custom dataset path and output directory
python create_mixed_expertise_dataset.py --data_path "/path/to/dataset.pt" --output_dir "my_balanced_datasets"
```

## End-Effector Clustering

Cluster robot trajectories based on end-effector movements using Hydra for configuration:

```bash
# Basic clustering using default parameters
python eef_clustering.py

# Customize data parameters
python eef_clustering.py data.data_path="/path/to/dataset.pt" data.segment_length=128

# Customize clustering parameters
python eef_clustering.py clustering.n_clusters=7 clustering.linkage_method=complete

# Change random seed for reproducibility
python eef_clustering.py random_seed=100

# Turn on/off visualization features
python eef_clustering.py visualization.skip_videos=true visualization.use_shared_ranges=false

# Specify output directory
python eef_clustering.py output.output_dir="./my_clustering_results"

# Use preprocessed data (skips extraction step)
python eef_clustering.py data.preprocessed_data="/path/to/preprocessed.pkl"
```

The script uses a configuration file at `config/eef_clustering.yaml` that defines all available parameters, organized into these sections:

- `random_seed`: Controls reproducibility (default: 42)
- `data`: Dataset path, segment length, etc.
- `clustering`: Number of clusters, linkage method
- `visualization`: Video generation and display options
- `output`: Output directory for results
- `wandb`: Weights & Biases logging configuration

## End-Effector Segment Matching

Find similar end-effector trajectory segments across multiple datasets using DTW distance. This script uses Hydra configuration and shares code with the clustering script for trajectory extraction and DTW calculations.

```bash
# Basic segment matching using default parameters
python eef_segment_matching.py

# Customize data parameters
python eef_segment_matching.py data.data_paths='["/path/to/dataset1.pt", "/path/to/dataset2.pt"]'
python eef_segment_matching.py data.segment_length=32 data.samples_per_dataset=300

# Customize matching parameters
python eef_segment_matching.py matching.top_k=10
python eef_segment_matching.py matching.query_index=5  # Use a specific segment as query

# Visualization options
python eef_segment_matching.py visualization.create_videos=false
python eef_segment_matching.py visualization.use_shared_ranges=true

# Change random seed and output directory
python eef_segment_matching.py random_seed=123 output.output_dir="./segment_matching_results"
```

The script uses a configuration file at `config/eef_segment_matching.yaml` with the following sections:

- `random_seed`: Controls reproducibility (default: 42)
- `data`: Dataset paths, segment length, samples per dataset
- `matching`: Top-k segments to find, query segment index
- `visualization`: Video generation and display options
- `output`: Output directory for results
- `wandb`: Weights & Biases logging configuration

## Collect Preferences

Generate preference pairs from trajectory segments:

```bash
# Basic preference collection using returns as the criterion
python collect_preferences.py --data_path "/path/to/dataset.pt" --num_pairs 1000

# Save to a specific output file
python collect_preferences.py --data_path "/path/to/dataset.pt" --num_pairs 1000 --output_file "my_preferences.pkl"
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

## Train Policy

Train a policy using different algorithms:

### IQL (Implicit Q-Learning)

IQL is the default algorithm. You can run it directly with:

```bash
# Basic IQL policy training
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" data.reward_model_path="reward_model/state_action_reward_model.pt"

# With video recording
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" evaluation.record_video=true

# With custom parameters
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" training.n_epochs=200 model.actor_learning_rate=3e-4
```

### BC (Behavior Cloning)

To train with BC instead of IQL:

```bash
# Basic BC policy training
python train_policy.py --config-name=bc data.data_path="/path/to/dataset.pt"

# With custom parameters
python train_policy.py --config-name=bc data.data_path="/path/to/dataset.pt" model.learning_rate=1e-4 training.n_epochs=200 evaluation.record_video=true
```

# With custom parameters
python train_policy.py --config-name=bc data.data_path="/path/to/dataset.pt" model.learning_rate=1e-4 training.n_epochs=200 evaluation.record_video=true