import time
import random
import torch
import numpy as np

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