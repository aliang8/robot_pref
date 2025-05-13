import time
import random
import torch
import numpy as np
from env.robomimic_lowdim import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import os
from pathlib import Path

def get_metaworld_env(env_name, seed=42):
    """Create a MetaWorld environment with the exact specified name.
    
    Args:
        env_name: Exact name of the MetaWorld environment to create
        seed: Random seed for the environment
    
    Returns:
        MetaWorld environment instance
    """
    from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                               ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
    
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
        raise ValueError(f"Environment '{env_name}' not found in MetaWorld environments. Please use one of the listed environments.")
    
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
    
class RobomimicEnvCreator:
    """A picklable environment creator for Robomimic environments."""
    
    def __init__(self, env_name="lift"):
        """Initialize the creator with the dataset name."""
        self.env_name = env_name

    def __call__(self):
        """Create a new environment with a random seed."""
        # Generate a unique seed each time this function is called
        unique_seed = int(time.time() * 1000) % 100000 + random.randint(0, 10000)
        return get_robomimic_env(self.env_name, seed=unique_seed)

def get_robomimic_env(
    env_name,
    render=True,
    render_offscreen=True,
    use_image_obs=True,
    base_path="/scr/matthewh6/robomimic/robomimic/datasets",
    seed=42,
):
    dataset_path = f"{base_path}/{env_name}/mg/demo_v15.hdf5"
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    obs_modality_dict = {
        "low_dim": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
            "object",
        ],
        "rgb": ["agentview_image"],
    }

    if render_offscreen or use_image_obs:
        os.environ["MUJOCO_GL"] = "egl"

    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=render,
        # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
    )

    env.env.hard_reset = False

    env = RobomimicLowdimWrapper(env)
    env.seed(seed)

    return env