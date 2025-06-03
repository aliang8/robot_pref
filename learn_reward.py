import numpy as np
import torch

import gym

import pyrallis
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import random, os, tqdm, copy, rich

import wandb
import uuid
from dataclasses import asdict, dataclass

import reward_utils
from reward_utils import collect_feedback, collect_human_feedback, consist_test_dataset
from models.reward_model import RewardModel
from utils.analyze_rewards_legacy import analyze_rewards_legacy, create_episodes_from_dataset, plot_preference_return_analysis_legacy, plot_segment_return_scatter_analysis_legacy

import sys

sys.path.append("../LiRE/algorithms")
import utils_env


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "metaworld_box-close-v2"  # environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    checkpoints_path: Optional[str] = None  # checkpoints path
    load_model: str = ""  # Model load file name, "" doesn't load
    # preference learning
    feedback_num: int = 1000
    data_quality: float = 5.0  # Replay buffer size (data_quality * 100000)
    segment_size: int = 25
    normalize: bool = True
    threshold: float = 0.0
    data_aug: str = "none"
    q_budget: int = 10000
    feedback_type: str = "RLT"
    model_type: str = "BT"
    noise: float = 0.0
    human: bool = False
    # MLP
    epochs: int = int(1e3)
    batch_size: int = 256
    activation: str = "tanh"  # Final Activation function
    lr: float = 1e-3
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"
    # Wandb logging
    project: str = "Reward Learning"
    group: str = "Reward learning"
    name: str = "Reward"

    def __post_init__(self):
        self.group = f"{self.env}_data_{self.data_quality}_fn_{self.feedback_num}_qb_{self.q_budget}_ft_{self.feedback_type}_m_{self.model_type}_e_{self.epochs}_n_{self.noise}"
        checkpoints_name = f"{self.name}/{self.env}/data_{self.data_quality}/fn_{self.feedback_num}/qb_{self.q_budget}/ft_{self.feedback_type}/m_{self.model_type}/n_{self.noise}/e_{self.epochs}/s_{self.seed}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                self.checkpoints_path, checkpoints_name
            )
            if not os.path.exists(self.checkpoints_path):
                os.makedirs(self.checkpoints_path)
        self.name = f"seed_{self.seed}"


def wandb_init(config: dict) -> None:
    wandb.init(
        mode="offline",
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@pyrallis.wrap()
def train(config: TrainConfig):
    rich.print(config)
    reward_utils.set_seed(config.seed)

    if "metaworld" in config.env:
        env_name = config.env.replace("metaworld-", "")
        env = utils_env.make_metaworld_env(env_name, config.seed)
        dataset = utils_env.MetaWorld_dataset(config)
    elif "dmc" in config.env:
        env_name = config.env.replace("dmc-", "")
        print("env_name ", env_name)
        env = utils_env.make_dmc_env(env_name, config.seed)
        dataset = utils_env.DMC_dataset(config)
        config.threshold *= 0.1  # because reward scaling is different from metaworld

    N = dataset["observations"].shape[0]
    traj_total = N // 500  # each trajectory has 500 steps

    if config.normalize:
        state_mean, state_std = reward_utils.compute_mean_std(
            dataset["observations"], eps=1e-3
        )
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = reward_utils.normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = reward_utils.normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    assert config.q_budget >= 1
    if config.human == False:
        multiple_ranked_list = collect_feedback(dataset, traj_total, config)
    elif config.human == True:
        multiple_ranked_list = collect_human_feedback(dataset, config)

    idx_st_1 = []
    idx_st_2 = []
    labels = []
    # construct the preference pairs
    for single_ranked_list in multiple_ranked_list:
        sub_index_set = []
        for i, group in enumerate(single_ranked_list):
            for tup in group:
                sub_index_set.append((tup[0], i, tup[1]))
        for i in range(len(sub_index_set)):
            for j in range(i + 1, len(sub_index_set)):
                idx_st_1.append(sub_index_set[i][0])
                idx_st_2.append(sub_index_set[j][0])
                if sub_index_set[i][1] < sub_index_set[j][1]:
                    labels.append([0, 1])
                else:
                    labels.append([0.5, 0.5])
    labels = np.array(labels)
    idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
    idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
    obs_act_1 = np.concatenate(
        (dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1
    )
    obs_act_2 = np.concatenate(
        (dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1
    )
    return_1 = dataset["rewards"][idx_1].sum(axis=1)
    return_2 = dataset["rewards"][idx_2].sum(axis=1)
    
    # test query set (for debug the training, not used for training)
    test_feedback_num = 5000
    test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels = (
        consist_test_dataset(
            dataset,
            test_feedback_num,
            traj_total,
            segment_size=config.segment_size,
            threshold=config.threshold,
        )
    )

    wandb_init(asdict(config))

    dimension = obs_act_1.shape[-1]
    reward_model = RewardModel(config, obs_act_1, obs_act_2, labels, dimension)

    reward_model.save_test_dataset(
        test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
    )

    reward_model.train_model()
    reward_model.save_model(config.checkpoints_path)
    
    # Run reward analysis using the legacy-compatible functions
    print("Running reward analysis...")
    
    # Create episodes from the dataset for analysis
    episodes = create_episodes_from_dataset(dataset, device=config.device, episode_length=500)
    
    # Determine reward range for normalization if rewards are available
    reward_min = None
    reward_max = None
    if "rewards" in dataset:
        reward_min = dataset["rewards"].min()
        reward_max = dataset["rewards"].max()
    
    # Generate reward analysis plot
    if config.checkpoints_path:
        reward_grid_path = os.path.join(config.checkpoints_path, f"reward_analysis_seed_{config.seed}.png")
        analyze_rewards_legacy(
            reward_model=reward_model,
            episodes=episodes,
            output_file=reward_grid_path,
            num_episodes=min(9, len(episodes)),
            reward_min=reward_min,
            reward_max=reward_max,
            wandb_run=wandb.run if wandb.run else None,
            random_seed=config.seed
        )
        print(f"Reward analysis saved to: {reward_grid_path}")
        
        # Generate preference return analysis plot
        preference_return_path = os.path.join(config.checkpoints_path, f"preference_return_analysis_seed_{config.seed}.png")
        plot_preference_return_analysis_legacy(
            reward_model=reward_model,
            obs_act_1=obs_act_1,
            obs_act_2=obs_act_2,
            labels=labels,
            gt_return_1=return_1,
            gt_return_2=return_2,
            segment_size=config.segment_size,
            output_file=preference_return_path,
            wandb_run=wandb.run if wandb.run else None
        )
        print(f"Preference return analysis saved to: {preference_return_path}")
        
        # Generate reward scatter analysis plot
        scatter_analysis_path = os.path.join(config.checkpoints_path, f"segment_return_scatter_analysis_seed_{config.seed}.png")
        plot_segment_return_scatter_analysis_legacy(
            reward_model=reward_model,
            obs_act_1=obs_act_1,
            obs_act_2=obs_act_2,
            gt_return_1=return_1,
            gt_return_2=return_2,
            segment_size=config.segment_size,
            output_file=scatter_analysis_path,
            max_samples=5000,
            wandb_run=wandb.run if wandb.run else None,
            random_seed=config.seed
        )
        print(f"Segment return scatter analysis saved to: {scatter_analysis_path}")


if __name__ == "__main__":
    train()
