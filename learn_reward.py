import os
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pyrallis
import rich

import reward_utils
import wandb
from models.reward_model import RewardModel
from reward_utils import *

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
    data_path: str = ""  # Path to the dataset file
    target_data_path: Optional[str] = None  # Path to the dataset file for DTW augmentations

    feedback_num: int = 100 
    test_feedback_num: int = 100 

    data_quality: float = 5.0  # Replay buffer size (data_quality * 100000)
    segment_size: int = 25
    normalize: bool = True
    threshold: float = 0.0
    data_aug: str = "none"
    q_budget: int = 1
    feedback_type: str = "RLT"
    model_type: str = "BT"
    noise: float = 0.0
    human: bool = False
    use_relative_eef: bool = False
    use_goal_pos: bool = False
    use_gt_prefs: bool = False  # Use ground truth preferences for reward learning
    # DTW augmentations
    use_dtw_augmentations: bool = False
    dtw_augment_before_training: bool = False  # If True: augment first then train, If False: train then augment
    dtw_subsample_size: int = 10000  # Number of segments to sample for DTW matrix
    dtw_augmentation_size: int = 2000  # Number of augmentation pairs to create from DTW matrix
    dtw_k_augment: int = 5  # Number of similar segments to find for each original preference pair
    dtw_preference_ratios: List[float] = None  # Ratios for [seg1_better, seg2_better, equal_pref] sampling. None = use all available
    acquisition_threshold_low: float = 0.25  # 25th percentile for acquisition filtering
    acquisition_threshold_high: float = 0.75  # 75th percentile for acquisition filtering
    acquisition_method: str = "entropy"  # "entropy", "disagreement", "combined", "variance"
    # Cache paths for reproducibility
    segment_indices_path: Optional[str] = None  # Path to save/load segment indices
    dtw_matrix_path: Optional[str] = None  # Path to save/load DTW matrix
    precomputed_dtw_matrix_path: Optional[str] = None  # Path to pre-computed DTW matrix for a different dataset
    # MLP
    epochs: int = int(1e3)
    batch_size: int = 256
    activation: str = "tanh"  # Final Activation function
    lr: float = 1e-3
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"
    # Class weighting for preference balancing
    use_class_weights: bool = False  # Apply inverse frequency weighting to balance preference types
    # Wandb logging
    project: str = "Reward Learning"
    group: str = "Reward learning"
    name: str = "Reward"

    eef_rm: bool = False
    use_gt_prefs: bool = False
    
    # Custom checkpoint naming (optional override)
    custom_checkpoint_params: Optional[List[str]] = None  # e.g., ["env", "data_quality", "dtw_settings"]

    def __post_init__(self):
        # Set default equal ratios for DTW preference sampling if not specified
        if self.dtw_preference_ratios is None:
            self.dtw_preference_ratios = [0.0, 1.0, 0.0]  # [seg1_better, seg2_better, equal_pref]
            
        # Validate ratios sum to 1.0
        if self.dtw_preference_ratios is not None:
            ratio_sum = sum(self.dtw_preference_ratios)
            if abs(ratio_sum - 1.0) > 1e-6:
                print(f"Warning: DTW preference ratios sum to {ratio_sum:.6f}, normalizing to sum to 1.0")
                self.dtw_preference_ratios = [r / ratio_sum for r in self.dtw_preference_ratios]
        
        # Set up cache paths if checkpoints_path is available
        if self.checkpoints_path is not None:
            if self.segment_indices_path is None:
                self.segment_indices_path = os.path.join(self.checkpoints_path, "segment_indices_cache.pkl")
            if self.dtw_matrix_path is None:
                self.dtw_matrix_path = os.path.join(self.checkpoints_path, "dtw_matrix_cache.pkl")
        
        # Define all available parameter mappings for flexible naming
        param_mappings = {
            "env": self.env,
            "data_quality": self.data_quality,
            "feedback_num": self.feedback_num,
            "q_budget": self.q_budget,
            "feedback_type": self.feedback_type,
            "model_type": self.model_type,
            "epochs": self.epochs,
            "noise": self.noise,
            "seed": self.seed,
            "dtw_enabled": int(self.use_dtw_augmentations),
            "dtw_subsample": f"{self.dtw_subsample_size//1000}k",
            "dtw_augmentation": self.dtw_augmentation_size,
            "dtw_ratios": f"{self.dtw_preference_ratios[0]:.2f}-{self.dtw_preference_ratios[1]:.2f}-{self.dtw_preference_ratios[2]:.2f}",
            "acquisition_thresholds": f"{self.acquisition_threshold_low}-{self.acquisition_threshold_high}",
            "dtw_settings": f"{int(self.use_dtw_augmentations)}-{int(self.dtw_augment_before_training)}-{self.dtw_subsample_size//1000}k-{self.dtw_augmentation_size}-R{self.dtw_preference_ratios[0]:.2f}-{self.dtw_preference_ratios[1]:.2f}-{self.dtw_preference_ratios[2]:.2f}-A{self.acquisition_threshold_low}-{self.acquisition_threshold_high}",
            "gt_prefs": int(self.use_gt_prefs),  # Add ground truth preference flag
            "eef_rm": int(self.eef_rm)  # Add EEF reward model flag
        }
        
        # Use custom parameters if specified, otherwise use default structure
        if self.custom_checkpoint_params:
            # Build group string from custom parameters
            group_parts = []
            checkpoint_parts = []
            
            for param_name in self.custom_checkpoint_params:
                if param_name in param_mappings:
                    value = param_mappings[param_name]
                    group_parts.append(f"{param_name}_{value}")
                    checkpoint_parts.append(f"{param_name}_{value}")
                else:
                    print(f"Warning: Unknown parameter '{param_name}' in custom_checkpoint_params")
            
            self.group = "_".join(group_parts)
            checkpoint_components = [self.name] + checkpoint_parts
            
        else:
            # Default checkpoint parameter groups for flexible naming
            checkpoint_params = [
                # Environment and data
                ["env", self.env],
                ["data", self.data_quality], 
                ["fn", self.feedback_num],
                ["m", self.model_type],
                ["e", self.epochs],
                ["gt", int(self.use_gt_prefs)],  # Add ground truth preference flag
                ["dtw", f"{int(self.use_dtw_augmentations)}"] if self.use_dtw_augmentations else None,
                ["s", self.seed]
            ]
            
            # Filter out None entries and build group string
            active_params = [param for param in checkpoint_params if param is not None]
            self.group = "_".join([f"{param[0]}_{param[1]}" for param in active_params])
            
            # Build checkpoint path components
            checkpoint_components = [
                f"{self.name}",
                f"{self.env}",
                f"fn_{self.feedback_num}",
                f"m_{self.model_type}",
                f"e_{self.epochs}",
                f"gt_{int(self.use_gt_prefs)}",  # Add ground truth preference flag
                f"eef_{int(self.eef_rm)}"  # Add EEF reward model flag
            ]
            
            checkpoint_components.append(f"s_{self.seed}")
        
        checkpoints_name = "/".join(checkpoint_components)
        
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                self.checkpoints_path, checkpoints_name
            )
            if not os.path.exists(self.checkpoints_path):
                os.makedirs(self.checkpoints_path)
        self.name = f"seed_{self.seed}"


def wandb_init(config: dict) -> None:
    wandb.init(
        # mode="offline",
        config=config,
        project=config["project"],
        # group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def log_video_to_wandb(images, name, fps=8):
    """Log a sequence of images as a video to wandb."""
    if images is None or len(images) == 0:
        return
    
    # Convert to (N, C, H, W) format if needed
    if len(images.shape) == 4:
        if images.shape[-1] == 3:
            images = images.transpose(0, 3, 1, 2)
        
        # Log the video
        wandb.log({
            name: wandb.Video(images, fps=fps, format="mp4")
        })


def combine_segments_side_by_side(images1, images2, preference=None):
    """Combine two image sequences side by side with a border around the preferred trajectory."""
    if len(images1.shape) == 4:  # (N, H, W, C)
        h1, w1 = images1.shape[1:3]
        h2, w2 = images2.shape[1:3]
        combined = np.zeros((len(images1), max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        # Copy the images
        combined[:, :h1, :w1] = images1
        combined[:, :h2, w1:w1+w2] = images2
        
        # Add border around preferred trajectory if preference is provided
        if preference is not None:
            border_width = h1 // 10
            border_color = np.array([0, 255, 0], dtype=np.uint8)  # Green border
            
            if preference == 0:  # First segment is preferred
                # Add border to first segment
                combined[:, :border_width, :w1] = border_color  # Top border
                combined[:, -border_width:, :w1] = border_color  # Bottom border
                combined[:, :, :border_width] = border_color  # Left border
                combined[:, :, w1-border_width:w1] = border_color  # Right border
            elif preference == 1:  # Second segment is preferred
                # Add border to second segment
                combined[:, :border_width, w1:] = border_color  # Top border
                combined[:, -border_width:, w1:] = border_color  # Bottom border
                combined[:, :, w1:w1+border_width] = border_color  # Left border
                combined[:, :, -border_width:] = border_color  # Right border
        
        return combined
    return None


def get_segment_rewards(dataset, indices, default_value=0.0):
    """Safely get rewards for segments, returning default value if rewards not available."""
    if "rewards" not in dataset:
        return np.array([default_value] * len(indices))
    return dataset["rewards"][indices].sum(axis=1)


def log_query_videos_to_wandb(dataset, idx_st_1, idx_st_2, labels, config, prefix="queries"):
    """Log video sequences for query pairs to wandb."""
    if "images" not in dataset:
        return
    print("Logging video sequences to wandb...")
    for i, (idx1, idx2) in enumerate(zip(idx_st_1, idx_st_2)):
        # Get image sequences for both segments
        images1 = dataset["images"][idx1:idx1 + config.segment_size]
        images2 = dataset["images"][idx2:idx2 + config.segment_size]
        # Get preference for this pair
        preference = 2  # Default to equal preference
        if i < len(labels):
            if labels[i][0] == 0 and labels[i][1] == 1:  # Second segment preferred
                preference = 1
            elif labels[i][0] == 1 and labels[i][1] == 0:  # First segment preferred
                preference = 0
        # Combine segments side by side with preference border
        combined_segments = combine_segments_side_by_side(images1, images2, preference)
        if combined_segments is not None and wandb.run:
            log_video_to_wandb(combined_segments, f"{prefix}/query_{i}")
        if i >= 9:
            break


@pyrallis.wrap()
def train(config: TrainConfig):
    rich.print(config)
    reward_utils.set_seed(config.seed)

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
    
    # Initialize wandb
    wandb_init(asdict(config))

    # Normalize observations if required
    state_mean, state_std = (reward_utils.compute_mean_std(dataset["observations"], eps=1e-3)
                             if config.normalize else (0, 1))
    
    for key in ["observations", "next_observations"]:
        dataset[key] = reward_utils.normalize_states(dataset[key], state_mean, state_std)

    if target_dataset is not None and config.normalize:
        for key in ["observations", "next_observations"]:
            target_dataset[key] = reward_utils.normalize_states(target_dataset[key], state_mean, state_std)

    # Get source and target dataset preferences
    labels, idx_st_1, idx_st_2 = get_human_feedbacks(config.data_path, config.feedback_num)
    target_labels, target_idx_st_1, target_idx_st_2 = get_human_feedbacks(config.target_data_path, config.test_feedback_num)

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

    if config.eef_rm:
        print("Using 3d EEF positions as reward model input")
        
        # Create segment indices
        train_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
        train_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
        val_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in target_idx_st_1]
        val_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in target_idx_st_2]

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

        reward_model.save_test_dataset(val_obs_act_1, val_obs_act_2, target_labels, target_labels, val_images1, val_images2)
        reward_model.train_model()
        reward_model.save_model(config.checkpoints_path)
    elif config.use_gt_prefs:
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
        reward_model = RewardModel(config, train_obs_act_1, train_obs_act_2, train_labels, obs_act_dim)
        
        reward_model.save_test_dataset(val_obs_act_1, val_obs_act_2, val_labels, val_labels, val_images1, val_images2)
        reward_model.train_model()
        reward_model.save_model(config.checkpoints_path)
    elif config.use_dtw_augmentations:
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
        
        # Create indices for augmented segments
        aug_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in aug_idx_st_1]
        aug_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in aug_idx_st_2]
        
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
        
        # Use target embodiment human preferences for testing
        if target_labels is not None:
            # Create indices for target segments
            target_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in target_idx_st_1]
            target_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in target_idx_st_2]
            
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
        
        # Train the model
        reward_model.train_model()

        # Save final model
        reward_model.save_model(config.checkpoints_path)
        
        # Save cross-embodiment DTW matrix for analysis
        if config.checkpoints_path:
            np.save(os.path.join(config.checkpoints_path, "cross_dtw_matrix.npy"), cross_dtw_matrix)
            print(f"Saved cross-embodiment DTW matrix to {config.checkpoints_path}")
    else:
        raise ValueError("Invalid reward learning method.")

    # TODO: add reward analysis
    # # Run reward analysis using the legacy-compatible functions
    # print("Running reward analysis...")
    
    # # Create episodes from the dataset for analysis
    # episodes = create_episodes_from_dataset(dataset, device=config.device, episode_length=500)
    
    # # Generate reward analysis plot and log to wandb
    # if wandb.run:
    #     analyze_rewards_legacy(
    #         reward_model=reward_model,
    #         episodes=episodes,
    #         output_file=None,  # Don't save to file
    #         num_episodes=min(9, len(episodes)),
    #         wandb_run=wandb.run,
    #         random_seed=config.seed
    #     )


if __name__ == "__main__":
    train()
