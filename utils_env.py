import os
import pickle as pkl

# import dmc2gym
# import metaworld.envs.mujoco.env_dict as _env_dict
import numpy as np
from gym.wrappers.time_limit import TimeLimit

from rlkit.envs.wrappers import NormalizedBoxEnv


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

def Robomimic_dataset(data_path, seq_len=3):
    """
    Load Robomimic dataset and build:
        - action sequences of length seq_len (staying inside episodes)
        - corresponding single starting observations
    """
    import h5py
    import numpy as np

    print(f"Loading data from: {data_path}")

    with h5py.File(data_path, 'r') as f:
        data = f["data"]

        all_observations = []
        all_actions = []
        all_rewards = []
        episode_boundaries = [0]  # Track episode start indices
        
        print(f"Found {len(data.keys())} trajectories in dataset")

        for demo in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
            demo_data = data[demo]
            
            # Concatenate observation components
            obs = np.concatenate([
                demo_data["obs"]["robot0_eef_pos"][:],
                demo_data["obs"]["robot0_eef_quat"][:],
                demo_data["obs"]["robot0_gripper_qpos"][:],
                demo_data["obs"]["object"][:]
            ], axis=1)

            all_observations.append(obs)
            all_actions.append(demo_data["actions"][:])
            all_rewards.append(demo_data["rewards"][:])
            episode_boundaries.append(episode_boundaries[-1] + len(demo_data["actions"]))

        # Convert to numpy arrays
        observations = np.concatenate(all_observations, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)

    print(f"Total number of transitions: {len(observations)}")

    # === Build single starting obs + action sequences ===
    obs_starts = []
    next_obs_starts = []
    action_seqs = []
    reward_starts = []
    terminal_starts = []

    # Process each episode
    for i in range(len(episode_boundaries) - 1):
        start_idx = episode_boundaries[i]
        end_idx = episode_boundaries[i + 1]
        
        ep_len = end_idx - start_idx
        
        # Skip episodes that are too short (need at least seq_len + 1 for next_obs)
        if ep_len < seq_len + 1:
            continue
            
        # Create sequences within this episode
        # For next_obs at end of action sequence, we need seq_len more observations
        max_seq_start = ep_len - seq_len
        
        for seq_start in range(max_seq_start):
            abs_start = start_idx + seq_start
            
            # Single observation at start of sequence
            obs_start = observations[abs_start]
            # Next observation at the end of the action sequence
            next_obs_start = observations[abs_start + seq_len]
            
            # Action sequence
            action_seq = actions[abs_start:abs_start + seq_len]
            
            # Reward and terminal at start
            rew_start = rewards[abs_start]
            terminal_start = (seq_start == max_seq_start - 1)  # True if this is the last possible sequence in episode
            
            obs_starts.append(obs_start)
            next_obs_starts.append(next_obs_start)
            action_seqs.append(action_seq)
            reward_starts.append(rew_start)
            terminal_starts.append(terminal_start)

    # Convert to numpy arrays
    obs_starts = np.array(obs_starts)
    next_obs_starts = np.array(next_obs_starts)
    action_seqs = np.array(action_seqs)
    reward_starts = np.array(reward_starts)
    terminal_starts = np.array(terminal_starts)

    print(f"Built {len(action_seqs)} action sequences of length {seq_len}")
    print(f"Observation shape: {obs_starts.shape}")
    print(f"Action sequence shape: {action_seqs.shape}")
    
    return {
        'observations': obs_starts,
        'next_observations': next_obs_starts,
        'actions': action_seqs,
        'rewards': reward_starts,
        'terminals': terminal_starts
    }

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
        paths = ["/project2/biyik_1165/hongmm/dataset/MetaWorld/button-press-topdown-v2/replay_buffer_step_150000_seed12345.pkl", "/project2/biyik_1165/hongmm/dataset/MetaWorld/button-press-topdown-v2/replay_buffer_step_150000_seed23451.pkl", "/project2/biyik_1165/hongmm/dataset/MetaWorld/button-press-topdown-v2/replay_buffer_step_150000_seed34512.pkl"]
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
