import os
from pathlib import Path

import hydra
import numpy as np
import rich
from omegaconf import DictConfig, OmegaConf

import utils_env
from models.reward_model import RewardModel
from reward_utils import (
    compute_dtw_matrix_cross,
    create_augmented_preferences_from_dtw,
    get_feedbacks,
    set_seed,
)
from utils.wandb import log_query_videos_to_wandb, wandb_init


@hydra.main(config_path="configs", config_name="reward", version_base=None)
def train(config: DictConfig):
    # Initialize  if enabled
    wandb_init(config) if getattr(config, 'use_wandb', False) else None

    rich.print(config)
    set_seed(config.seed)

    # Set up checkpoint paths
    if getattr(config, 'checkpoints_path', None) is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        
        checkpoint_name = build_rm_checkpoint_path(config)
        config.checkpoints_path = os.path.join(config.checkpoints_path, checkpoint_name)
        
        os.makedirs(config.checkpoints_path, exist_ok=True)
        OmegaConf.save(config=config, f=os.path.join(config.checkpoints_path, "config.yaml"))

    # Load datasets
    if "metaworld" in config.env:
        dataset = utils_env.MetaWorld_dataset(config)
    elif "dmc" in config.env:
        dataset = utils_env.DMC_dataset(config)
        config.threshold *= 0.1  # because reward scaling is different from metaworld
    elif "robomimic" in config.env:
        dataset = utils_env.Robomimic_dataset(config.data_path, return_images=True)
        cross_dataset = utils_env.Robomimic_dataset(config.cross_data_path, return_images=True) if config.cross_data_path else None
    else:
        raise ValueError(f"Unsupported environment type: {config.env}")
    
    # Normalize observations if required (+ goal points for DTW matrix)
    if config.normalize:
        state_mean = dataset["observations"].mean(axis=0)
        state_std = dataset["observations"].std(axis=0) + 1e-8

        # Bound the std to prevent large values
        min_std = 1e-2
        state_std = np.maximum(state_std, min_std)

        dataset["observations"] = (dataset["observations"] - state_mean) / state_std
        dataset["next_observations"] = (dataset["next_observations"] - state_mean) / state_std

        goal_points_mean = dataset["goal_points"].mean(axis=0)
        goal_points_std = dataset["goal_points"].std(axis=0) + 1e-8
        dataset["goal_points"] = (dataset["goal_points"] - goal_points_mean) / goal_points_std

        if cross_dataset is not None:
            state_mean = cross_dataset["observations"].mean(axis=0)
            state_std = cross_dataset["observations"].std(axis=0) + 1e-8

            # Bound the std to prevent large values
            min_std = 1e-2
            state_std = np.maximum(state_std, min_std)

            cross_dataset["observations"] = (cross_dataset["observations"] - state_mean) / state_std
            cross_dataset["next_observations"] = (cross_dataset["next_observations"] - state_mean) / state_std

            goal_points_mean = cross_dataset["goal_points"].mean(axis=0)
            goal_points_std = cross_dataset["goal_points"].std(axis=0) + 1e-8
            cross_dataset["goal_points"] = (cross_dataset["goal_points"] - goal_points_mean) / goal_points_std
    
    # Get source and target dataset preferences
    labels, idx_st_1, idx_st_2, val_labels, val_idx_st_1, val_idx_st_2 = get_feedbacks(config.data_path, config.feedback_num, human=config.human)

    val_episodes_path = Path(config.data_path).parent / "val_episodes.npy"
    val_episodes = np.load(val_episodes_path, allow_pickle=True)

    # val_episodes_path = Path(config.cross_data_path).parent / "val_episodes.npy"
    # val_episodes = np.load(val_episodes_path, allow_pickle=True)
    # if cross_dataset is not None:
    #     target_labels, target_idx_st_1, target_idx_st_2 = get_feedbacks(config.cross_data_path, config.test_feedback_num, human=config.human) if config.cross_data_path else (None, None, None)
    
    obs_act_dim = dataset["observations"].shape[-1] + dataset["actions"].shape[-1]
    print(f"Observation-action dimension: {obs_act_dim}")

    # Convert labels to [1,0], [0,1], or [0.5,0.5] format for both labels and target_labels
    def convert_labels_to_array(label_list):
        new_labels = []
        for label in label_list:
            if label == 1:  # First segment preferred
                new_labels.append([1, 0])
            elif label == 0:  # Second segment preferred
                new_labels.append([0, 1])
            else:  # Equal preference
                new_labels.append([0.5, 0.5])
        return np.array(new_labels)

    labels = convert_labels_to_array(labels)
    val_labels = convert_labels_to_array(val_labels)
    
    # if cross_dataset is not None:
    #     target_labels = convert_labels_to_array(target_labels)
    
    # After collecting feedback log the source preferences
    log_query_videos_to_wandb(dataset, idx_st_1, idx_st_2, labels, config, prefix="prefs")

    if config.single_emb:
        print(f"Using ground truth preferences for {config.data_path} for reward learning")
        # Split the data into train and validation sets
        train_labels = labels
        train_idx_st_1 = idx_st_1
        train_idx_st_2 = idx_st_2

        val_labels = val_labels
        val_idx_st_1 = val_idx_st_1
        val_idx_st_2 = val_idx_st_2
        
        # Create indices for segments
        train_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in train_idx_st_1]
        train_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in train_idx_st_2]

        val_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in val_idx_st_1]
        val_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in val_idx_st_2]

        # Get observations and actions for segments
        train_obs_act_1 = np.concatenate(
            (dataset["observations"][train_idx_1], dataset["actions"][train_idx_1]), axis=-1
        )
        train_obs_act_2 = np.concatenate(
            (dataset["observations"][train_idx_2], dataset["actions"][train_idx_2]), axis=-1
        )

        train_images1 = dataset["images"][train_idx_1] if "images" in dataset else None
        train_images2 = dataset["images"][train_idx_2] if "images" in dataset else None
        
        val_obs_act_1 = np.concatenate(
            (dataset["observations"][val_idx_1], dataset["actions"][val_idx_1]), axis=-1
        )
        val_obs_act_2 = np.concatenate(
            (dataset["observations"][val_idx_2], dataset["actions"][val_idx_2]), axis=-1
        )

        # Get images for segments if available
        val_images1 = dataset["images"][val_idx_1] if "images" in dataset else None
        val_images2 = dataset["images"][val_idx_2] if "images" in dataset else None
        
        obs_act_dim = train_obs_act_1.shape[-1]

        reward_model = RewardModel(config, dataset, train_obs_act_1, train_obs_act_2, train_labels, obs_act_dim, train_images1, train_images2)
        
        reward_model.save_test_dataset(val_obs_act_1, val_obs_act_2, val_labels, val_labels, val_images1, val_images2, val_episodes, test_dataset=dataset)
        reward_model.train_model()
        
    # elif config.eef_rm:
    #     print("Using 3d EEF positions as reward model input")
        
    #     # Create segment indices
    #     train_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
    #     train_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
    #     val_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in target_idx_st_1]
    #     val_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in target_idx_st_2]

    #     # Get observations and actions for segments
    #     train_obs_act_1 = np.concatenate(
    #         (dataset["observations"][train_idx_1][:, :, :3], dataset["actions"][train_idx_1]), axis=-1
    #     )
    #     train_obs_act_2 = np.concatenate(
    #         (dataset["observations"][train_idx_2][:, :, :3], dataset["actions"][train_idx_2]), axis=-1
    #     )

    #     train_images1 = dataset["images"][train_idx_1] if "images" in dataset else None
    #     train_images2 = dataset["images"][train_idx_2] if "images" in dataset else None
        
    #     val_obs_act_1 = np.concatenate(
    #         (dataset["observations"][val_idx_1][:, :, :3], dataset["actions"][val_idx_1]), axis=-1
    #     )
    #     val_obs_act_2 = np.concatenate(
    #         (dataset["observations"][val_idx_2][:, :, :3], dataset["actions"][val_idx_2]), axis=-1
    #     )

    #     # Get images for segments if available
    #     val_images1 = dataset["images"][val_idx_1] if "images" in dataset else None
    #     val_images2 = dataset["images"][val_idx_2] if "images" in dataset else None

    #     eef_act_dim = train_obs_act_1.shape[-1]
    #     reward_model = RewardModel(config, dataset, train_obs_act_1, train_obs_act_2, labels, eef_act_dim, train_images1, train_images2)

    #     print("Reward model architecture:")
    #     print(reward_model.net)

    #     reward_model.save_test_dataset(val_obs_act_1, val_obs_act_2, target_labels, target_labels, val_images1, val_images2)
    #     reward_model.train_model()
        
    elif config.cross_data_path:
        # Load source and target segment indices
        data_path = Path(config.data_path)
        seg_indices_path = Path(data_path).parent / "segment_start_end_indices.npy"
        seg_indices = np.load(seg_indices_path, allow_pickle=True)

        cross_data_path = Path(config.cross_data_path)
        seg_indices_path = cross_data_path.parent / "segment_start_end_indices.npy"
        cross_seg_indices = np.load(seg_indices_path, allow_pickle=True)
        val_episodes_path = cross_data_path.parent / "val_episodes.npy"
        val_episodes = np.load(val_episodes_path, allow_pickle=True)

        # labels, idx_st_1, idx_st_2, val_labels, val_idx_st_1, val_idx_st_2

        # Use pre-computeed cross-embodiment DTW matrix if exists, else compute it
        new_data_path = data_path.parent.parent / f"{data_path.parent.name}_X_{cross_data_path.parent.name}"
        new_data_path.mkdir(parents=True, exist_ok=True)

        cross_dtw_file = "cross_dtw_matrix"
        # Different dtw metrics
        if config.use_relative_eef:
            cross_dtw_file = f"{cross_dtw_file}_relative_eef"
        if config.use_goal_pos:
            cross_dtw_file = f"{cross_dtw_file}_goal_pos"

        dtw_matrix_path = new_data_path / f"{cross_dtw_file}.npy"
        print(f"DTW matrix path: {dtw_matrix_path}")

        if dtw_matrix_path.exists():
            print(f"Loading existing DTW matrix from {dtw_matrix_path}")
            cross_dtw_matrix = np.load(dtw_matrix_path)
        else:
            print("Computing cross-embodiment DTW matrix...")
            cross_dtw_matrix = compute_dtw_matrix_cross(dataset, seg_indices, cross_dataset, cross_seg_indices, config)
            np.save(dtw_matrix_path, cross_dtw_matrix)
            print(f"Saved DTW matrix to {dtw_matrix_path}")

        # Create augmented source preferences from DTW matrix
        cross_labels, cross_idx_st_1, cross_idx_st_2 = create_augmented_preferences_from_dtw(
            cross_dtw_matrix,
            labels,
            idx_st_1,
            idx_st_2,
            seg_indices,
            cross_seg_indices,
            config,
        )

        log_query_videos_to_wandb(cross_dataset, cross_idx_st_1, cross_idx_st_2, cross_labels, config, prefix="cross_prefs")

        # Create indices for augmented segments
        aug_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in cross_idx_st_1]
        aug_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in cross_idx_st_2]

        # Get observations and actions for augmented segments
        aug_obs_act_1 = np.concatenate(
            (cross_dataset["observations"][aug_idx_1], cross_dataset["actions"][aug_idx_1]), axis=-1
        )
        aug_obs_act_2 = np.concatenate(
            (cross_dataset["observations"][aug_idx_2], cross_dataset["actions"][aug_idx_2]), axis=-1
        )

        train_images1 = cross_dataset["images"][aug_idx_1] if "images" in cross_dataset else None
        train_images2 = cross_dataset["images"][aug_idx_2] if "images" in cross_dataset else None

        # Create reward model with augmented preferences
        print("Training reward model with augmented preferences...")
        reward_model = RewardModel(config, cross_dataset, aug_obs_act_1, aug_obs_act_2, cross_labels, obs_act_dim, train_images1, train_images2)

        print(reward_model)

        # TODO: fix this, for now just run with whatever
        # # Create indices for target segments
        # target_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in cross_idx_st_1]
        # target_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in cross_idx_st_2]

        # Get observations and actions for target segments
        test_obs_act_1 = np.concatenate(
            (cross_dataset["observations"][aug_idx_1], cross_dataset["actions"][aug_idx_1]), axis=-1
        )
        test_obs_act_2 = np.concatenate(
            (cross_dataset["observations"][aug_idx_2], cross_dataset["actions"][aug_idx_2]), axis=-1
        )

        # Get images for target segments if available
        test_images1 = cross_dataset["images"][aug_idx_1] if "images" in cross_dataset else None
        test_images2 = cross_dataset["images"][aug_idx_2] if "images" in cross_dataset else None
                        
        # Save test dataset
        reward_model.save_test_dataset(
            test_obs_act_1, test_obs_act_2, cross_labels, cross_labels, test_images1, test_images2, val_episodes, test_dataset=cross_dataset
        )
        
        reward_model.train_model()
    else:
        raise ValueError("Invalid reward learning method.")
    
    # save the trained model
    reward_model.save_model(config.checkpoints_path)


def build_rm_checkpoint_path(config: DictConfig) -> str:
    """Build reward learning checkpoint path based on config parameters."""
    
    # Build checkpoint path components
    checkpoint_components = [
        f"{config.env}",
        f"fn_{config.feedback_num}",
        f"gt_{int(config.single_emb)}",
        f"eef_{int(config.eef_rm)}",
        f"dtw_{int(config.use_cross)}",
        f"s_{config.seed}",
        # f"dist_{int(config.use_distributional_model)}"
    ]
    
    # checkpoint_components.append(f"s_{getattr(config, 'seed', 0)}")

    # if getattr(config, 'use_cross', False):
    #     checkpoint_components.append(f"dtw_k_{getattr(config, 'dtw_k_augment', None)}")
    
    checkpoints_name = "/".join(checkpoint_components)
    return checkpoints_name




if __name__ == "__main__":
    train()
