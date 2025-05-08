# Robot Trajectory Preference Learning

This repository contains scripts for training reward models from robot trajectory preferences and using them to train reinforcement learning policies. The codebase leverages DINOv2 visual embeddings for processing trajectory segments.

## Setup

### Requirements

Install the necessary packages:

```bash
pip install torch numpy matplotlib tqdm pillow metaworld d3rlpy tensordict
```

You also need to install the custom DTW implementation used for trajectory comparison.

## Workflow

The overall workflow consists of these main steps:

1. **Process and analyze trajectory data** from MetaWorld environments
2. **Train a reward model** using Bradley-Terry preference learning
3. **Train an IQL policy** using the learned reward model
4. **Evaluate the policy** on MetaWorld environments

## Scripts

### Shared Utilities

- `trajectory_utils.py`: Common utilities for trajectory processing, including:
  - Loading trajectory data from TensorDict files
  - Computing DINOv2 embeddings for images
  - Creating fixed-length segments
  - Computing DTW distances between segments

### Data Processing and Analysis

- `hierarchical_clustering.py`: Process trajectory data and cluster segments
- `segment_matching.py`: Find segments similar to a query segment using DTW
- `analyze_preferences.py`: Visualize and analyze preference data

### Reward Learning and Policy Training

- `train_reward_model.py`: Train a reward model using Bradley-Terry loss on segment preferences
- `train_iql_policy.py`: Train an IQL policy using the learned reward model
- `evaluate_policy.py`: Evaluate the trained policy on MetaWorld environments

## Usage Examples

### Process and analyze trajectory data

```bash
# Analyze trajectories with hierarchical clustering
python hierarchical_clustering.py

# Find similar segments across datasets
python segment_matching.py --samples_per_dataset 500 --top_k 5
```

### Train reward model

```bash
# Train reward model on a MetaWorld task
python train_reward_model.py \
  --data_path "/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt" \
  --num_segments 5000 \
  --num_pairs 10000 \
  --num_epochs 50
```

### Train and evaluate IQL policy

```bash
# Train IQL policy using the trained reward model
python train_iql_policy.py \
  --data_path "/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt" \
  --reward_model_path "reward_model/reward_model.pt" \
  --iql_epochs 100

# Evaluate trained policy
python evaluate_policy.py \
  --model_path "iql_model/iql_assembly-v2.pt" \
  --task_name "assembly-v2" \
  --num_episodes 20 \
  --record
```

## Data Paths

The default data paths for MetaWorld tasks are:

```
/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt
/scr/shared/clam/datasets/metaworld/bin-picking-v2/buffer_bin-picking-v2.pt
/scr/shared/clam/datasets/metaworld/peg-insert-side-v2/buffer_peg-insert-side-v2.pt
```

## Workflow Details

1. **Segment Generation**:
   - Load trajectory data from `.pt` files
   - Compute DINOv2 embeddings for visual features
   - Create fixed-length segments respecting episode boundaries

2. **Reward Model Training**:
   - Generate synthetic preference pairs based on original rewards
   - Train an MLP reward model using Bradley-Terry loss
   - Evaluate model on held-out preference pairs

3. **IQL Policy Training**:
   - Create an MDP dataset with learned rewards
   - Train an IQL policy using d3rlpy
   - Evaluate policy on the original MetaWorld environment 