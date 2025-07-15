import inspect
import random
import time
from pathlib import Path

from env.robomimic_lowdim import RobomimicLowdimWrapper
import os
import pickle as pkl
import copy
import os
import random
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import hydra
import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

import models.reward_model as reward_model
import utils.env as utils_env
import wandb
from learn_reward import build_rm_checkpoint_path
from models.flow_policy import FlowNoisePredictionNet, FlowPolicy
from utils.eval import eval_actor
from utils.wandb import wandb_init

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from gym.wrappers.time_limit import TimeLimit


try:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
except:
    pass
import os

os.environ["MUJOCO_GL"] = "egl"


def get_metaworld_env(env_name, seed=42):
    """Create a MetaWorld environment with the exact specified name.

    Args:
        env_name: Exact name of the MetaWorld environment to create
        seed: Random seed for the environment

    Returns:
        MetaWorld environment instance
    """
    from metaworld.envs import (
        ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    )

    # Try to find the environment in the goal observable environments
    if env_name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        env_constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    # Otherwise, try to find it in the goal hidden environments
    elif env_name in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN:
        env_constructor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name]
    else:
        # If not found, raise a clear error with available options
        print("Available goal observable environments:")
        for name in sorted(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()):
            print(f"  - {name}")
        print("\nAvailable goal hidden environments:")
        for name in sorted(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys()):
            print(f"  - {name}")
        raise ValueError(
            f"Environment '{env_name}' not found in MetaWorld environments. Please use one of the listed environments."
        )

    # Create the environment with the specified seed
    env = env_constructor(seed=seed)
    return env


class MetaWorldEnvCreator:
    """A picklable environment creator for MetaWorld environments."""

    def __init__(self, env_name="assembly-v2-goal-observable"):
        """Initialize the creator with a specific environment name.

        Args:
            env_name: The exact name of the MetaWorld environment to create.
                      Must match one of the keys in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
                      or ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.
        """
        self.env_name = env_name

    def __call__(self):
        """Create a new environment with a random seed."""
        # Generate a unique seed each time this function is called
        unique_seed = int(time.time() * 1000) % 100000 + random.randint(0, 10000)
        return get_metaworld_env(self.env_name, seed=unique_seed)


def make_metaworld_env(env_name, seed):
    env_name = env_name.replace("metaworld_", "")
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    # print("partially observe", env._partially_observable) Ture
    # print("env._freeze_rand_vec", env._freeze_rand_vec) True
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def make_dmc_env(env_name, seed):
    env_name = env_name.replace("dmc_", "")
    domain_name, task_name = env_name.split("-")
    domain_name = domain_name.lower()
    task_name = task_name.lower()
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
    )
    return env


def MetaWorld_dataset(config):
    """
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of dummy rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if config.human == False:
        base_path = os.path.join(os.getcwd(), "dataset/MetaWorld/")
        env_name = config.env
        base_path += str(env_name.replace("metaworld_", ""))
        dataset = dict()

        # TODO: hardcode this for now
        paths = [
            "/project2/biyik_1165/hongmm/dataset/MetaWorld/button-press-topdown-v2/replay_buffer_step_150000_seed12345.pkl",
            "/project2/biyik_1165/hongmm/dataset/MetaWorld/button-press-topdown-v2/replay_buffer_step_150000_seed23451.pkl",
            "/project2/biyik_1165/hongmm/dataset/MetaWorld/button-press-topdown-v2/replay_buffer_step_150000_seed34512.pkl",
        ]
        # for seed in range(3):
        for path in paths:
            # path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
            print(f"loading data from: {path}")

            with open(path, "rb") as f:
                load_dataset = pkl.load(f)

            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    : int(config.data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                : int(config.data_quality * 100_000)
            ]
            load_dataset.pop("dones", None)

            for key in load_dataset.keys():
                if key not in dataset:
                    dataset[key] = load_dataset[key]
                else:
                    dataset[key] = np.concatenate(
                        (dataset[key], load_dataset[key]), axis=0
                    )
    elif config.human == True:
        base_path = os.path.join(os.getcwd(), "human_feedback/")
        base_path += f"{config.env}/dataset.pkl"
        with open(base_path, "rb") as f:
            dataset = pkl.load(f)
            dataset["observations"] = np.array(dataset["observations"])
            dataset["actions"] = np.array(dataset["actions"])
            dataset["next_observations"] = np.array(dataset["next_observations"])
            dataset["rewards"] = np.zeros(len(dataset["observations"]))  # dummy rewards
            dataset["terminals"] = np.array(dataset["dones"])
            dataset["images"] = np.array(dataset["images"])

    N = dataset["observations"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    images_ = []

    dataset["terminals"] = dataset["terminals"].reshape(-1)
    dataset["rewards"] = dataset["rewards"].reshape(-1)

    for i in range(N):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        images = dataset["images"][i].astype(np.uint8)
        done_bool = bool(dataset["terminals"][i])
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        images_.append(images)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "images": np.array(images_),
    }


def DMC_dataset(config):
    base_path = os.path.join(os.getcwd(), "dataset/DMControl/")
    env_name = config.env.replace("dmc_", "")
    base_path += str(env_name)
    dataset = dict()
    for seed in range(3):
        path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pkl.load(f)

        if "humanoid" in env_name:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    200000 : int(config.data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                0 : int(config.data_quality * 100_000) - 200000
            ]
            load_dataset.pop("dones", None)
        else:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    0 : int(config.data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                0 : int(config.data_quality * 100_000) - 0
            ]
            load_dataset.pop("dones", None)

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)
        # print("shape", load_dataset["rewards"].shape, "from seed ", seed, end=",  ")
    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def Robomimic_dataset(data_path, return_images=False, clip_last=False, goal_state=True):
    """
    Load Robomimic dataset and build:
    If clip_last, we don't use the last transition for IQL
    """

    print(f"Loading data from: {data_path}")

    with h5py.File(data_path, "r") as f:
        data = f["data"]

        all_observations = []
        all_next_observations = []
        all_actions = []
        all_rewards = []
        all_images = []
        all_terminals = []

        all_goal_points = []

        print(f"Found {len(data.keys())} trajectories in dataset")

        for demo in tqdm(
            sorted(data.keys(), key=lambda x: int(x.split("_")[1])),
            desc="Processing demos",
        ):
            demo_data = data[demo]
            import ipdb

            ipdb.set_trace()
            if goal_state:
                # Concatenate observation components
                obs = np.concatenate(
                    [
                        demo_data["obs"]["robot0_eef_pos"][:],
                        demo_data["obs"]["robot0_eef_quat"][:],
                        demo_data["obs"]["robot0_gripper_qpos"][:],
                        demo_data["obs"]["object"][:],
                    ],
                    axis=1,
                )
            else:
                obs = np.concatenate(
                    [
                        demo_data["obs"]["robot0_eef_pos"][:],
                        demo_data["obs"]["robot0_eef_quat"][:],
                        demo_data["obs"]["robot0_gripper_qpos"][:],
                    ],
                    axis=1,
                )

            if clip_last:
                next_obs = obs[1:]
                obs = obs[:-1]

                acts = demo_data["actions"][:-1]
                rewards = demo_data["rewards"][:-1]
                images = demo_data["obs"]["agentview_image"][:-1]

                goal_points = demo_data["goal_points"][:-1]
            else:
                next_obs = obs

                acts = demo_data["actions"]
                rewards = demo_data["rewards"]
                images = demo_data["obs"]["agentview_image"]

                goal_points = demo_data["goal_points"]

            all_observations.append(obs)
            all_next_observations.append(next_obs)
            all_actions.append(acts)
            all_rewards.append(rewards)

            all_goal_points.append(goal_points)

            if images.shape[1] != 84:
                # Assuming images are in format (batch, height, width, channels)
                resized_images = np.array([cv2.resize(img, (84, 84)) for img in images])
                all_images.append(resized_images)
            else:
                all_images.append(images)
            # Create terminals array - True only for the last step of each episode
            episode_length = len(acts)
            terminals = np.zeros(episode_length, dtype=bool)
            terminals[-1] = True  # Mark the last step as terminal
            all_terminals.append(terminals)

        # Convert to numpy arrays
        observations = np.concatenate(all_observations, axis=0)
        next_observations = np.concatenate(all_next_observations, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        images = np.concatenate(all_images, axis=0)
        terminals = np.concatenate(all_terminals, axis=0)
        goal_points = np.concatenate(all_goal_points, axis=0)

    print(f"Total number of transitions: {len(observations)}")

    dataset = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "goal_points": goal_points,
    }
    if return_images:
        dataset["images"] = images

    return dataset
    

def get_robomimic_env(
    data_path,
    render=False,
    render_offscreen=True,
    use_image_obs=False,
    seed=42,
):
    """Create a Robomimic environment.

    Args:
        env_name: Name of the environment (e.g. "lift", "can")
        render: Whether to enable rendering (default: False)
        render_offscreen: Whether to use offscreen rendering (default: True)
        use_image_obs: Whether to use image observations (default: False)
        base_path: Base path to look for datasets
        seed: Random seed for the environment

    Returns:
        Robomimic environment wrapped in RobomimicLowdimWrapper
    """
    try:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
    except ImportError:
        raise ImportError("Please install robomimic to use Robomimic environments")

    env_meta = FileUtils.get_env_metadata_from_dataset(data_path)

    obs_modality_dict = {
        "low_dim": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
            "object",
        ],
        "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
    }

    # Force EGL for offscreen rendering
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["DISPLAY"] = ""  # Clear any display settings

    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
    )

    env.env.hard_reset = False

    from env.robomimic_lowdim import RobomimicLowdimWrapper

    env = RobomimicLowdimWrapper(env)
    env.seed(seed)

    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        # Handle different state formats
        if isinstance(state, (list, tuple)):
            # If state is wrapped in a container, extract the observation
            if len(state) == 2:
                state = state[0]
            else:
                # For other cases, convert to numpy array
                state = np.array(state)

        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Apply normalization
        normalized = (state - state_mean) / state_std

        # Ensure output format matches input format
        return normalized.astype(np.float32)

    def scale_reward(reward):
        return reward_scale * reward

    # Apply transformations
    env = gym.wrappers.TransformObservation(env, normalize_state)

    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)

    return env
