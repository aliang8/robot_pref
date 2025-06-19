import os

import hydra
import numpy as np
import rich
from omegaconf import DictConfig, OmegaConf

import utils_env
from models.reward_model import DistributionalRewardModel, RewardModel
from reward_utils import *
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
        dataset = utils_env.Robomimic_dataset(config.data_path)
        target_dataset = utils_env.Robomimic_dataset(config.target_data_path) if config.target_data_path else None
    else:
        raise ValueError(f"Unsupported environment type: {config.env}")

    # Normalize observations if required
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-4)
        target_state_mean, target_state_std = compute_mean_std(
            dataset["next_observations"], eps=1e-4
        )
    else:
        state_mean, state_std = 0, 1
        target_state_mean, target_state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    if target_dataset is not None:
        target_dataset["observations"] = normalize_states(
            target_dataset["observations"], state_mean, state_std
        )
        target_dataset["next_observations"] = normalize_states(
            target_dataset["next_observations"], target_state_mean, target_state_std
        )

    # Get source and target dataset preferences
    labels, idx_st_1, idx_st_2 = get_human_feedbacks(config.data_path, getattr(config, 'feedback_num', 100))
    target_labels, target_idx_st_1, target_idx_st_2 = get_human_feedbacks(config.target_data_path, getattr(config, 'test_feedback_num', 100))

    obs_act = np.concatenate(
        (dataset["observations"][0], dataset["actions"][0]), axis=-1
    )
    obs_act_dim = obs_act.shape[-1]
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
    target_labels = convert_labels_to_array(target_labels)

    # After collecting feedback log the source preferences
    log_query_videos_to_wandb(dataset, idx_st_1, idx_st_2, labels, config)

    if getattr(config, 'use_gt_prefs', False):
        print(f"Using ground truth preferences for {config.data_path} for reward learning")

        # Split into train/val sets (80/20 split)
        n_feedbacks = len(labels)
        n_train = int(0.9 * n_feedbacks)
        
        # Shuffle indices
        indices = np.random.permutation(n_feedbacks)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        print(f"Using {len(train_indices)} train indices and {len(val_indices)} val indices")
        
        # Split the data into train and validation sets
        train_labels = labels[train_indices]
        train_idx_st_1 = idx_st_1[train_indices]
        train_idx_st_2 = idx_st_2[train_indices]
        
        val_labels = labels[val_indices]
        val_idx_st_1 = idx_st_1[val_indices]
        val_idx_st_2 = idx_st_2[val_indices] 
        
        # Create indices for segments
        train_idx_1 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in train_idx_st_1]
        train_idx_2 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in train_idx_st_2]
        
        val_idx_1 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in val_idx_st_1]
        val_idx_2 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in val_idx_st_2]
        
        # Get observations and actions for segments
        train_obs_act_1 = np.concatenate(
            (dataset["observations"][train_idx_1], dataset["actions"][train_idx_1]), axis=-1
        )
        train_obs_act_2 = np.concatenate(
            (dataset["observations"][train_idx_2], dataset["actions"][train_idx_2]), axis=-1
        )
        
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

        if getattr(config, 'use_distributional_model', False):
            print("Using distributional reward model")
            reward_model = DistributionalRewardModel(config, train_obs_act_1, train_obs_act_2, train_labels, obs_act_dim)
        else:
            print("Using regular reward model")
            reward_model = RewardModel(config, train_obs_act_1, train_obs_act_2, train_labels, obs_act_dim)

        print(reward_model)
        
        reward_model.save_test_dataset(val_obs_act_1, val_obs_act_2, val_labels, val_labels, val_images1, val_images2)
        reward_model.train_model()
        
    elif getattr(config, 'eef_rm', False):
        print("Using 3d EEF positions as reward model input")
        
        # Create segment indices
        train_idx_1 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in idx_st_1]
        train_idx_2 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in idx_st_2]
        val_idx_1 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in target_idx_st_1]
        val_idx_2 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in target_idx_st_2]

        # Get observations and actions for segments
        train_obs_act_1 = np.concatenate(
            (dataset["observations"][train_idx_1][:, :, :3], dataset["actions"][train_idx_1]), axis=-1
        )
        train_obs_act_2 = np.concatenate(
            (dataset["observations"][train_idx_2][:, :, :3], dataset["actions"][train_idx_2]), axis=-1
        )
        
        val_obs_act_1 = np.concatenate(
            (dataset["observations"][val_idx_1][:, :, :3], dataset["actions"][val_idx_1]), axis=-1
        )
        val_obs_act_2 = np.concatenate(
            (dataset["observations"][val_idx_2][:, :, :3], dataset["actions"][val_idx_2]), axis=-1
        )

        # Get images for segments if available
        val_images1 = dataset["images"][val_idx_1] if "images" in dataset else None
        val_images2 = dataset["images"][val_idx_2] if "images" in dataset else None

        eef_act_dim = train_obs_act_1.shape[-1]
        reward_model = RewardModel(config, train_obs_act_1, train_obs_act_2, labels, eef_act_dim)

        print("Reward model architecture:")
        print(reward_model.net)

        reward_model.save_test_dataset(val_obs_act_1, val_obs_act_2, target_labels, target_labels, val_images1, val_images2)
        reward_model.train_model()
        
    elif getattr(config, 'use_dtw_augmentations', False):
        # Load source and target segment indices
        data_path = Path(config.data_path)
        seg_indices_path = data_path.parent / "segment_start_end_indices.npy"
        seg_indices = np.load(seg_indices_path, allow_pickle=True)
        target_data_path = Path(config.target_data_path)
        seg_indices_path = target_data_path.parent / "segment_start_end_indices.npy"
        target_seg_indices = np.load(seg_indices_path, allow_pickle=True)

        # Compute cross-embodiment DTW matrix
        cross_dtw_matrix = compute_dtw_matrix_cross(dataset, seg_indices, target_dataset, target_seg_indices, config)
        
        # Create augmented source preferences from DTW matrix
        aug_labels, aug_idx_st_1, aug_idx_st_2 = create_augmented_preferences_from_dtw(
            cross_dtw_matrix,
            labels,
            idx_st_1,
            idx_st_2,
            seg_indices,
            target_seg_indices,
            config,
        )

        log_query_videos_to_wandb(target_dataset, aug_idx_st_1, aug_idx_st_2, aug_labels, config, prefix="aug_prefs")
        
        # Create indices for augmented segments
        aug_idx_1 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in aug_idx_st_1]
        aug_idx_2 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in aug_idx_st_2]
        
        # Get observations and actions for augmented segments
        aug_obs_act_1 = np.concatenate(
            (dataset["observations"][aug_idx_1], dataset["actions"][aug_idx_1]), axis=-1
        )
        aug_obs_act_2 = np.concatenate(
            (target_dataset["observations"][aug_idx_2], target_dataset["actions"][aug_idx_2]), axis=-1
        )
        
        # Create reward model with augmented preferences
        print("Training reward model with augmented preferences...")
        reward_model = RewardModel(config, aug_obs_act_1, aug_obs_act_2, aug_labels, obs_act_dim)

        print(reward_model)
        
        # Use target embodiment human preferences for testing
        if target_labels is not None:
            # Create indices for target segments
            target_idx_1 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in target_idx_st_1]
            target_idx_2 = [[j for j in range(i, i + getattr(config, 'segment_size', 25))] for i in target_idx_st_2]
            
            # Get observations and actions for target segments
            test_obs_act_1 = np.concatenate(
                (target_dataset["observations"][target_idx_1], target_dataset["actions"][target_idx_1]), axis=-1
            )
            test_obs_act_2 = np.concatenate(
                (target_dataset["observations"][target_idx_2], target_dataset["actions"][target_idx_2]), axis=-1
            )

            # Get images for target segments if available
            test_images1 = target_dataset["images"][target_idx_1] if "images" in target_dataset else None
            test_images2 = target_dataset["images"][target_idx_2] if "images" in target_dataset else None
                            
            # Save test dataset
            reward_model.save_test_dataset(
                test_obs_act_1, test_obs_act_2, target_labels, target_labels, test_images1, test_images2
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
        f"{getattr(config, 'env', 'unknown')}",
        f"fn_{getattr(config, 'feedback_num', 100)}",
        f"gt_{int(getattr(config, 'use_gt_prefs', False))}",
        f"eef_{int(getattr(config, 'eef_rm', False))}",
        f"dist_{int(getattr(config, 'use_distributional_model', False))}"
    ]
    
    checkpoint_components.append(f"s_{getattr(config, 'seed', 0)}")

    if getattr(config, 'use_dtw_augmentations', False):
        checkpoint_components.append(f"dtw_k_{getattr(config, 'dtw_k_augment', None)}")
    
    checkpoints_name = "/".join(checkpoint_components)
    return checkpoints_name




if __name__ == "__main__":
    train()
