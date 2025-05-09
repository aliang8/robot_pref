# Robot Preference Learning

This repository contains code for training robot manipulation policies using preference learning techniques. The codebase is designed to work with [MetaWorld](https://meta-world.github.io/) environments and [D3RLPy](https://github.com/takuseno/d3rlpy) for offline reinforcement learning.

## Setup

Install the required packages:

```bash
pip install wandb hydra-core d3rlpy
```

## Key Components

1. **Reward Model Training**: Learns a reward function from preferences between trajectory segments using Bradley-Terry preference learning
2. **IQL Policy Training**: Trains a policy using the learned reward model with Implicit Q-Learning (IQL)

## Configuration with Hydra

This project uses [Hydra](https://hydra.cc/) for configuration management, allowing for flexible experiment configuration without modifying code.

### Configuration Files

- `config/train_reward_model/config.yaml`: Configuration for reward model training
- `config/train_iql/config.yaml`: Configuration for IQL policy training

### Overriding Config Values

You can override config values from the command line:

```bash
# Train reward model with custom parameters
python train_reward_model.py data.data_path="/path/to/dataset.pt" model.hidden_dims=[512,512]

# Train IQL policy with custom parameters
python train_iql_policy.py data.data_path="/path/to/dataset.pt" training.iql_epochs=200
```

## Weights & Biases Integration

This codebase integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking.

### Enabling/Disabling Wandb

In the configuration files, set `wandb.use_wandb` to `true` or `false` to enable/disable wandb.

```yaml
# Enable wandb logging
wandb:
  use_wandb: true
  project: "robot_preference_learning"
  entity: null  # Your wandb username or team
```

### Tracked Metrics

The following metrics are tracked in wandb:

#### Reward Model Training:
- Training and validation loss
- Test accuracy
- Model architecture details
- Training curve visualization

#### IQL Policy Training:
- Evaluation returns and success rates
- Training metrics from D3RLPy
- Model artifacts

## Example Usage

### Train a Reward Model

```bash
python train_reward_model.py data.data_path="/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt" model.hidden_dims=[256,256] training.num_epochs=50
```

### Train an IQL Policy

```bash
python train_iql_policy.py data.data_path="/scr/shared/clam/datasets/metaworld/assembly-v2/buffer_assembly-v2.pt" data.reward_model_path="reward_model/state_action_reward_model.pt"
```

## Video Recording

The IQL training script supports recording evaluation videos. To enable:

```bash
python train_iql_policy.py evaluation.record_video=true
```

Videos will be saved in the output directory. 