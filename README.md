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

Cluster robot trajectories based on end-effector movements:

```bash
# Basic clustering using default parameters
python eef_clustering.py --data_path "/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt"

# More advanced options
python eef_clustering.py --data_path "path/to/dataset.pt" --n_clusters 5 --segment_length 64 --max_segments 1000 --linkage_method average
```

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

## Parallel Evaluation

The codebase supports parallel evaluation of policies for faster performance:

```bash
# Enable parallel evaluation with 4 processes
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" evaluation.parallel_eval=true evaluation.eval_workers=4

# Adjust frequency of evaluations
python train_policy.py --config-name=iql data.data_path="/path/to/dataset.pt" evaluation.parallel_eval=true training.eval_interval=10
```

Note: The parallel evaluation uses a pickle-safe environment creation mechanism to avoid serialization issues when using multiprocessing. 


python train_policy.py --config-name=iql data.use_ground_truth=true data.scale_rewards=True wandb.use_wandb=True data.reward_model_path="/scr/aliang80/robot_pref/reward_model/state_action_reward_model.pt"


python train_policy.py --config-name=bc  \
    data.data_path="/scr/aliang80/robot_pref/labeled_datasets/buffer_assembly-v2_balanced.pt" \

python train_policy.py --config-name=iql  \
    data.data_path="/scr/aliang80/robot_pref/labeled_datasets/buffer_assembly-v2_balanced.pt" \
    data.reward_model_path="
```

## SLURM Job Submission

The codebase supports running jobs on SLURM clusters while maintaining the same output structure as local runs:

```bash
# Run with SLURM using the default configuration
python train_reward_model.py --multirun

# Customize SLURM settings at runtime
python train_policy.py --config-name=iql hydra.launcher.partition=gpu hydra.launcher.timeout_min=1440

# Run multiple hyperparameter combinations
python train_policy.py --multirun training.n_epochs=50,100,200 model.actor_learning_rate=1e-4,3e-4
```

The SLURM configuration can be customized in `config/hydra/launcher.yaml`. Job outputs are stored in the same `outputs/YYYY-MM-DD/HH-MM-SS` directory structure as local runs, with SLURM logs in `.submitit` subdirectories.