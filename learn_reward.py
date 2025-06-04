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
from reward_utils import collect_feedback, collect_human_feedback, consist_test_dataset, collect_dtw_augmentations, compute_acquisition_scores, filter_augmentations_by_acquisition, collect_simple_pairwise_feedback, analyze_dtw_augmentation_quality, compare_baseline_vs_augmented_performance, display_preference_label_stats, plot_baseline_vs_augmented_scatter_analysis, plot_individual_test_example_deltas
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
    feedback_num: int = 100
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
    # DTW augmentations
    use_dtw_augmentations: bool = False
    dtw_augment_before_training: bool = False  # If True: augment first then train, If False: train then augment
    dtw_subsample_size: int = 10000  # Number of segments to sample for DTW matrix
    dtw_augmentation_size: int = 2000  # Number of augmentation pairs to create from DTW matrix
    dtw_k_augment: int = 5  # Number of similar segments to find for each original preference pair
    dtw_exclude_equal_pref: bool = True  # If True: only augment non-equal preference pairs (exclude [0.5, 0.5])
    acquisition_threshold_low: float = 0.25  # 25th percentile for acquisition filtering
    acquisition_threshold_high: float = 0.75  # 75th percentile for acquisition filtering
    acquisition_method: str = "entropy"  # "entropy", "disagreement", "combined", "variance"
    # Cache paths for reproducibility
    segment_indices_path: Optional[str] = None  # Path to save/load segment indices
    dtw_matrix_path: Optional[str] = None  # Path to save/load DTW matrix
    # MLP
    epochs: int = int(1e3)
    batch_size: int = 256
    activation: str = "tanh"  # Final Activation function
    lr: float = 1e-3
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"
    # Class weighting for preference balancing
    use_class_weights: bool = True  # Apply inverse frequency weighting to balance preference types
    # Wandb logging
    project: str = "Reward Learning"
    group: str = "Reward learning"
    name: str = "Reward"
    
    # Custom checkpoint naming (optional override)
    custom_checkpoint_params: Optional[List[str]] = None  # e.g., ["env", "data_quality", "dtw_settings"]

    def __post_init__(self):
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
            "dtw_mode": "before" if self.dtw_augment_before_training else "after",
            "dtw_subsample": f"{self.dtw_subsample_size//1000}k",
            "dtw_augmentation": self.dtw_augmentation_size,
            "dtw_exclude_equal": int(self.dtw_exclude_equal_pref),
            "acquisition_thresholds": f"{self.acquisition_threshold_low}-{self.acquisition_threshold_high}",
            "dtw_settings": f"{int(self.use_dtw_augmentations)}-{int(self.dtw_augment_before_training)}-{self.dtw_subsample_size//1000}k-{self.dtw_augmentation_size}-E{int(self.dtw_exclude_equal_pref)}-A{self.acquisition_threshold_low}-{self.acquisition_threshold_high}"
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
                ["qb", self.q_budget],
                ["ft", self.feedback_type],
                ["m", self.model_type],
                ["e", self.epochs],
                ["n", self.noise],
                ["th", self.threshold],
                # DTW augmentation settings (only if enabled)
                ["dtw", f"{int(self.use_dtw_augmentations)}-{int(self.dtw_augment_before_training)}-{self.dtw_subsample_size//1000}k-{self.dtw_augmentation_size}-E{int(self.dtw_exclude_equal_pref)}-A{self.acquisition_threshold_low}-{self.acquisition_threshold_high}"] if self.use_dtw_augmentations else None,
                # Seed (always last)
                ["s", self.seed]
            ]
            
            # Filter out None entries and build group string
            active_params = [param for param in checkpoint_params if param is not None]
            self.group = "_".join([f"{param[0]}_{param[1]}" for param in active_params])
            
            # Build checkpoint path components
            checkpoint_components = [
                f"{self.name}",
                f"{self.env}",
                f"data_{self.data_quality}",
                f"fn_{self.feedback_num}",
                f"qb_{self.q_budget}",
                f"ft_{self.feedback_type}",
                f"m_{self.model_type}",
                f"n_{self.noise}",
                f"e_{self.epochs}",
                f"th_{self.threshold}"
            ]
            
            # Add DTW components if enabled
            if self.use_dtw_augmentations:
                dtw_mode = "before" if self.dtw_augment_before_training else "after"
                checkpoint_components.extend([
                    f"dtw_{dtw_mode}",
                    f"sub_{self.dtw_subsample_size//1000}k",
                    f"aug_{self.dtw_augmentation_size}"
                ])
            
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
    # if config.human == False:
    #     multiple_ranked_list = collect_feedback(dataset, traj_total, config)
    # elif config.human == True:
    #     multiple_ranked_list = collect_human_feedback(dataset, config)
    multiple_ranked_list, segment_indices = collect_simple_pairwise_feedback(dataset, traj_total, config)
    
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

    # Display stats about the original preference labels
    display_preference_label_stats(labels, return_1, return_2, config, title="Original Preference Labels")
    
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
    
    # Compute test ground truth returns by recreating the test indices
    # (using same logic as consist_test_dataset)
    np.random.seed(config.seed)  # Use same seed as test dataset creation
    test_traj_idx = np.random.choice(traj_total, 2 * test_feedback_num, replace=True)
    test_idx = [
        500 * i + np.random.randint(0, 500 - config.segment_size) for i in test_traj_idx
    ]
    test_idx_st_1 = test_idx[:test_feedback_num]
    test_idx_st_2 = test_idx[test_feedback_num:]
    test_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in test_idx_st_1]
    test_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in test_idx_st_2]
    test_gt_return_1 = dataset["rewards"][test_idx_1].sum(axis=1)
    test_gt_return_2 = dataset["rewards"][test_idx_2].sum(axis=1)

    wandb_init(asdict(config))

    dimension = obs_act_1.shape[-1]
    reward_model = RewardModel(config, obs_act_1, obs_act_2, labels, dimension)

    reward_model.save_test_dataset(
        test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
    )

    # DTW Augmentation Strategy
    if config.use_dtw_augmentations and config.dtw_augment_before_training:
        print("\nStrategy: DTW augmentations BEFORE training...")
        
        # First, train a baseline model on original data only for comparison
        print("Step 1: Training baseline model on original data only...")
        baseline_reward_model = RewardModel(config, obs_act_1, obs_act_2, labels, dimension)
        baseline_reward_model.save_test_dataset(
            test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
        )
        baseline_reward_model.train_model()
        
        # Prepare original preference pairs for DTW augmentation
        original_pairs = list(zip(idx_st_1, idx_st_2))
        original_preferences = labels.tolist()
        
        # Collect DTW augmentations
        dtw_multiple_ranked_list, dtw_distances_dict, candidate_segment_indices = collect_dtw_augmentations(
            dataset, 
            traj_total, 
            config,
            original_pairs,
            original_preferences,
            use_relative_eef=config.use_relative_eef
        )
        
        # Extract augmentation data in the same format as original training data
        dtw_idx_st_1 = []
        dtw_idx_st_2 = []
        dtw_labels = []
        
        # Process DTW ranking lists to extract pairs
        for single_ranked_list in dtw_multiple_ranked_list:
            dtw_sub_index_set = []
            for i, group in enumerate(single_ranked_list):
                for tup in group:
                    dtw_sub_index_set.append((tup[0], i, tup[1]))
            for i in range(len(dtw_sub_index_set)):
                for j in range(i + 1, len(dtw_sub_index_set)):
                    dtw_idx_st_1.append(dtw_sub_index_set[i][0])
                    dtw_idx_st_2.append(dtw_sub_index_set[j][0])
                    if dtw_sub_index_set[i][1] < dtw_sub_index_set[j][1]:
                        dtw_labels.append([0, 1])
                    else:
                        dtw_labels.append([0.5, 0.5])

        if len(dtw_idx_st_1) > 0:
            dtw_labels = np.array(dtw_labels)
            dtw_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in dtw_idx_st_1]
            dtw_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in dtw_idx_st_2]
            dtw_obs_act_1 = np.concatenate(
                (dataset["observations"][dtw_idx_1], dataset["actions"][dtw_idx_1]), axis=-1
            )
            dtw_obs_act_2 = np.concatenate(
                (dataset["observations"][dtw_idx_2], dataset["actions"][dtw_idx_2]), axis=-1
            )
            dtw_return_1 = dataset["rewards"][dtw_idx_1].sum(axis=1)
            dtw_return_2 = dataset["rewards"][dtw_idx_2].sum(axis=1)
            
            # Display stats about DTW augmentation labels
            display_preference_label_stats(dtw_labels, dtw_return_1, dtw_return_2, config, title="DTW Augmentation Labels")
            
            # Combine original data with ALL DTW augmentations (no acquisition filtering)
            print("Step 2: Combining original data with DTW augmentations...")
            combined_obs_act_1 = np.concatenate([obs_act_1, dtw_obs_act_1], axis=0)
            combined_obs_act_2 = np.concatenate([obs_act_2, dtw_obs_act_2], axis=0)  
            combined_labels = np.concatenate([labels, dtw_labels], axis=0)
            combined_multiple_ranked_list = multiple_ranked_list + dtw_multiple_ranked_list
            combined_return_1 = np.concatenate([return_1, dtw_return_1], axis=0)
            combined_return_2 = np.concatenate([return_2, dtw_return_2], axis=0)
            
            # Display stats about combined dataset
            display_preference_label_stats(combined_labels, combined_return_1, combined_return_2, config, title="Combined (Original + DTW) Labels")
            
            print(f"Combined dataset sizes:")
            print(f"  Original: {len(obs_act_1)} pairs")
            print(f"  DTW augmentations: {len(dtw_obs_act_1)} pairs")
            print(f"  Combined: {len(combined_obs_act_1)} pairs")
            
            # Create reward model with combined data from the start
            print("Step 3: Training augmented reward model with combined data...")
            reward_model = RewardModel(config, combined_obs_act_1, combined_obs_act_2, combined_labels, dimension)
            reward_model.save_test_dataset(
                test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
            )
            
            # Update variables for analysis
            obs_act_1, obs_act_2, labels = combined_obs_act_1, combined_obs_act_2, combined_labels
            return_1, return_2 = combined_return_1, combined_return_2
            
            # Save DTW matrix for analysis
            if config.checkpoints_path:
                np.save(os.path.join(config.checkpoints_path, "dtw_matrix.npy"), dtw_distances_dict)
                np.save(os.path.join(config.checkpoints_path, "dtw_segment_indices.npy"), candidate_segment_indices)
                print(f"Saved DTW analysis data to {config.checkpoints_path}")
        else:
            print("No DTW augmentations generated, using baseline model as final model")
            reward_model = baseline_reward_model
        
        # Train the augmented model
        print("Training augmented reward model...")
        reward_model.train_model()
        
        # Step 4: Compare baseline vs augmented model performance
        if len(dtw_idx_st_1) > 0:
            print("Step 4: Comparing baseline vs augmented model performance...")
            comparison_path = os.path.join(config.checkpoints_path, f"baseline_vs_augmented_comparison_seed_{config.seed}.png") if config.checkpoints_path else None
            comparison_results = compare_baseline_vs_augmented_performance(
                baseline_reward_model=baseline_reward_model,
                augmented_reward_model=reward_model,
                test_obs_act_1=test_obs_act_1,
                test_obs_act_2=test_obs_act_2,
                test_labels=test_labels,
                test_binary_labels=test_binary_labels,
                test_gt_return_1=test_gt_return_1,
                test_gt_return_2=test_gt_return_2,
                config=config,
                output_path=comparison_path,
                wandb_run=wandb.run if wandb.run else None
            )
            
            # Create individual test example deltas chart
            print("Creating individual test example deltas chart...")
            deltas_chart_path = os.path.join(config.checkpoints_path, f"individual_test_deltas_seed_{config.seed}.png") if config.checkpoints_path else None
            deltas_results = plot_individual_test_example_deltas(
                baseline_reward_model=baseline_reward_model,
                augmented_reward_model=reward_model,
                test_obs_act_1=test_obs_act_1,
                test_obs_act_2=test_obs_act_2,
                test_gt_return_1=test_gt_return_1,
                test_gt_return_2=test_gt_return_2,
                output_file=deltas_chart_path,
                max_examples=200,  # Show more examples in dedicated chart
                wandb_run=wandb.run if wandb.run else None,
                random_seed=config.seed
            )
        
        # Analyze DTW augmentation quality if DTW was used
        if config.use_dtw_augmentations and len(dtw_multiple_ranked_list) > 0:
            print("Step 5: Analyzing DTW augmentation quality...")
            dtw_analysis_path = os.path.join(config.checkpoints_path, f"dtw_augmentation_analysis_seed_{config.seed}.png") if config.checkpoints_path else None
            analyze_dtw_augmentation_quality(
                reward_model=baseline_reward_model,
                dtw_obs_act_1=dtw_obs_act_1,
                dtw_obs_act_2=dtw_obs_act_2,
                dtw_labels=dtw_labels,
                dataset=dataset,
                segment_start_indices_1=dtw_idx_st_1,
                segment_start_indices_2=dtw_idx_st_2,
                config=config,
                output_path=dtw_analysis_path,
                wandb_run=wandb.run if wandb.run else None
            )
            
            # Step 6: Create baseline vs augmented scatter analysis 
            print("Step 6: Creating baseline vs augmented scatter analysis...")
            scatter_comparison_path = os.path.join(config.checkpoints_path, f"baseline_vs_augmented_scatter_seed_{config.seed}.png") if config.checkpoints_path else None
            
            # Use original data for fair comparison (both models see same data)
            # Combine both segments from pairs into individual segments
            original_obs_act_1 = np.concatenate(
                (dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1
            )
            original_obs_act_2 = np.concatenate(
                (dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1
            )
            original_return_1 = dataset["rewards"][idx_1].sum(axis=1)
            original_return_2 = dataset["rewards"][idx_2].sum(axis=1)
            
            # Combine both segments and returns into single arrays
            all_segments = np.concatenate([original_obs_act_1, original_obs_act_2], axis=0)
            all_returns = np.concatenate([original_return_1, original_return_2], axis=0)
            
            scatter_results = plot_baseline_vs_augmented_scatter_analysis(
                baseline_reward_model=baseline_reward_model,
                augmented_reward_model=reward_model,
                obs_act_segments=all_segments,
                gt_returns=all_returns,
                segment_size=config.segment_size,
                output_file=scatter_comparison_path,
                max_samples=5000,
                wandb_run=wandb.run if wandb.run else None,
                random_seed=config.seed
            )

    else:
        # Original approach: Train first, then optionally augment
        print("\nStrategy: Training FIRST, then optional DTW augmentations...")
        
        # Stage 1: Initial training (this serves as our baseline)
        print("Stage 1: Initial reward model training (baseline)...")
        reward_model.train_model()
        baseline_reward_model = reward_model  # Keep reference to baseline
        
        # Stage 2: DTW augmentations with acquisition filtering (if enabled)
        if config.use_dtw_augmentations:
            print("\nStage 2: DTW augmentations with acquisition filtering...")
            
            # Prepare original preference pairs for DTW augmentation
            original_pairs = list(zip(idx_st_1, idx_st_2))
            original_preferences = labels.tolist()
            
            # Collect DTW augmentations
            dtw_multiple_ranked_list, dtw_distances_dict, candidate_segment_indices = collect_dtw_augmentations(
                dataset, 
                traj_total, 
                config,
                original_pairs,
                original_preferences,
                use_relative_eef=config.use_relative_eef
            )
            
            # Extract augmentation data in the same format as original training data
            dtw_idx_st_1 = []
            dtw_idx_st_2 = []
            dtw_labels = []
            
            # Process DTW ranking lists to extract pairs
            for single_ranked_list in dtw_multiple_ranked_list:
                dtw_sub_index_set = []
                for i, group in enumerate(single_ranked_list):
                    for tup in group:
                        dtw_sub_index_set.append((tup[0], i, tup[1]))
                for i in range(len(dtw_sub_index_set)):
                    for j in range(i + 1, len(dtw_sub_index_set)):
                        dtw_idx_st_1.append(dtw_sub_index_set[i][0])
                        dtw_idx_st_2.append(dtw_sub_index_set[j][0])
                        if dtw_sub_index_set[i][1] < dtw_sub_index_set[j][1]:
                            dtw_labels.append([0, 1])
                        else:
                            dtw_labels.append([0.5, 0.5])
            
            if len(dtw_idx_st_1) > 0:
                dtw_labels = np.array(dtw_labels)
                dtw_idx_1 = [[j for j in range(i, i + config.segment_size)] for i in dtw_idx_st_1]
                dtw_idx_2 = [[j for j in range(i, i + config.segment_size)] for i in dtw_idx_st_2]
                dtw_obs_act_1 = np.concatenate(
                    (dataset["observations"][dtw_idx_1], dataset["actions"][dtw_idx_1]), axis=-1
                )
                dtw_obs_act_2 = np.concatenate(
                    (dataset["observations"][dtw_idx_2], dataset["actions"][dtw_idx_2]), axis=-1
                )
                dtw_return_1 = dataset["rewards"][dtw_idx_1].sum(axis=1)
                dtw_return_2 = dataset["rewards"][dtw_idx_2].sum(axis=1)
                
                # Display stats about DTW augmentation labels
                display_preference_label_stats(dtw_labels, dtw_return_1, dtw_return_2, config, title="DTW Augmentation Labels")
                
                # Compute acquisition scores for DTW augmentations
                print("Computing acquisition scores for DTW augmentations...")
                acquisition_scores = compute_acquisition_scores(
                    reward_model, dtw_obs_act_1, dtw_obs_act_2, dtw_labels, config
                )
                
                # Filter augmentations based on acquisition scores
                print("Filtering DTW augmentations based on acquisition scores...")
                filtered_dtw_ranked_list, filtered_dtw_obs_act_1, filtered_dtw_obs_act_2, filtered_dtw_labels = filter_augmentations_by_acquisition(
                    dtw_multiple_ranked_list,
                    dtw_obs_act_1,
                    dtw_obs_act_2, 
                    dtw_labels,
                    acquisition_scores,
                    threshold_low=config.acquisition_threshold_low,
                    threshold_high=config.acquisition_threshold_high
                )
                
                # Display stats about filtered DTW augmentation labels
                if len(filtered_dtw_obs_act_1) > 0:
                    # Compute returns for filtered augmentations
                    filtered_dtw_return_1 = dtw_return_1[acquisition_scores >= np.percentile(acquisition_scores, config.acquisition_threshold_low * 100)]
                    filtered_dtw_return_2 = dtw_return_2[acquisition_scores >= np.percentile(acquisition_scores, config.acquisition_threshold_low * 100)]
                    display_preference_label_stats(filtered_dtw_labels, filtered_dtw_return_1, filtered_dtw_return_2, config, title="Filtered DTW Augmentation Labels")
                
                # Combine original data with filtered augmentations
                print("Combining original data with filtered DTW augmentations...")
                combined_obs_act_1 = np.concatenate([obs_act_1, filtered_dtw_obs_act_1], axis=0)
                combined_obs_act_2 = np.concatenate([obs_act_2, filtered_dtw_obs_act_2], axis=0)  
                combined_labels = np.concatenate([labels, filtered_dtw_labels], axis=0)
                combined_multiple_ranked_list = multiple_ranked_list + filtered_dtw_ranked_list
                
                # Display stats about combined dataset
                if len(filtered_dtw_obs_act_1) > 0:
                    combined_return_1 = np.concatenate([return_1, filtered_dtw_return_1], axis=0)
                    combined_return_2 = np.concatenate([return_2, filtered_dtw_return_2], axis=0)
                    display_preference_label_stats(combined_labels, combined_return_1, combined_return_2, config, title="Combined (Original + Filtered DTW) Labels")
                
                print(f"Combined dataset sizes:")
                print(f"  Original: {len(obs_act_1)} pairs")
                print(f"  DTW augmentations: {len(filtered_dtw_obs_act_1)} pairs")
                print(f"  Combined: {len(combined_obs_act_1)} pairs")
                
                # Create new reward model with combined data
                print("Retraining reward model with augmented data...")
                combined_reward_model = RewardModel(config, combined_obs_act_1, combined_obs_act_2, combined_labels, dimension)
                combined_reward_model.save_test_dataset(
                    test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
                )
                combined_reward_model.train_model()
                
                # Compare baseline vs augmented model performance
                print("Comparing baseline vs augmented model performance...")
                comparison_path = os.path.join(config.checkpoints_path, f"baseline_vs_augmented_comparison_seed_{config.seed}.png") if config.checkpoints_path else None
                comparison_results = compare_baseline_vs_augmented_performance(
                    baseline_reward_model=baseline_reward_model,
                    augmented_reward_model=combined_reward_model,
                    test_obs_act_1=test_obs_act_1,
                    test_obs_act_2=test_obs_act_2,
                    test_labels=test_labels,
                    test_binary_labels=test_binary_labels,
                    test_gt_return_1=test_gt_return_1,
                    test_gt_return_2=test_gt_return_2,
                    config=config,
                    output_path=comparison_path,
                    wandb_run=wandb.run if wandb.run else None
                )
                
                # Create individual test example deltas chart
                print("Creating individual test example deltas chart...")
                deltas_chart_path = os.path.join(config.checkpoints_path, f"individual_test_deltas_seed_{config.seed}.png") if config.checkpoints_path else None
                deltas_results = plot_individual_test_example_deltas(
                    baseline_reward_model=baseline_reward_model,
                    augmented_reward_model=combined_reward_model,
                    test_obs_act_1=test_obs_act_1,
                    test_obs_act_2=test_obs_act_2,
                    test_gt_return_1=test_gt_return_1,
                    test_gt_return_2=test_gt_return_2,
                    output_file=deltas_chart_path,
                    max_examples=200,  # Show more examples in dedicated chart
                    wandb_run=wandb.run if wandb.run else None,
                    random_seed=config.seed
                )
                
                # Use the combined model for final results
                reward_model = combined_reward_model
                obs_act_1, obs_act_2, labels = combined_obs_act_1, combined_obs_act_2, combined_labels
                return_1 = np.concatenate([return_1, dtw_return_1[acquisition_scores >= np.percentile(acquisition_scores, config.acquisition_threshold_low * 100)]], axis=0)
                return_2 = np.concatenate([return_2, dtw_return_2[acquisition_scores >= np.percentile(acquisition_scores, config.acquisition_threshold_low * 100)]], axis=0)
                
                # Analyze DTW augmentation quality after retraining
                print("Analyzing DTW augmentation quality after retraining...")
                dtw_analysis_path = os.path.join(config.checkpoints_path, f"dtw_augmentation_analysis_seed_{config.seed}.png") if config.checkpoints_path else None
                analyze_dtw_augmentation_quality(
                    reward_model=combined_reward_model,
                    dtw_obs_act_1=dtw_obs_act_1,
                    dtw_obs_act_2=dtw_obs_act_2,
                    dtw_labels=dtw_labels,
                    dataset=dataset,
                    segment_start_indices_1=dtw_idx_st_1,
                    segment_start_indices_2=dtw_idx_st_2,
                    config=config,
                    output_path=dtw_analysis_path,
                    wandb_run=wandb.run if wandb.run else None
                )
                
                # Save DTW matrix and acquisition scores for analysis
                if config.checkpoints_path:
                    np.save(os.path.join(config.checkpoints_path, "dtw_matrix.npy"), dtw_distances_dict)
                    np.save(os.path.join(config.checkpoints_path, "acquisition_scores.npy"), acquisition_scores)
                    np.save(os.path.join(config.checkpoints_path, "dtw_segment_indices.npy"), candidate_segment_indices)
                    print(f"Saved DTW analysis data to {config.checkpoints_path}")
                
                # Create baseline vs augmented scatter analysis
                print("Creating baseline vs augmented scatter analysis...")
                scatter_comparison_path = os.path.join(config.checkpoints_path, f"baseline_vs_augmented_scatter_seed_{config.seed}.png") if config.checkpoints_path else None
                
                # Use original data for fair comparison (both models see same data)
                # Combine both segments from pairs into individual segments
                original_obs_act_1 = np.concatenate(
                    (dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1
                )
                original_obs_act_2 = np.concatenate(
                    (dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1
                )
                original_return_1 = dataset["rewards"][idx_1].sum(axis=1)
                original_return_2 = dataset["rewards"][idx_2].sum(axis=1)
                
                # Combine both segments and returns into single arrays
                all_segments = np.concatenate([original_obs_act_1, original_obs_act_2], axis=0)
                all_returns = np.concatenate([original_return_1, original_return_2], axis=0)
                
                scatter_results = plot_baseline_vs_augmented_scatter_analysis(
                    baseline_reward_model=baseline_reward_model,
                    augmented_reward_model=combined_reward_model,
                    obs_act_segments=all_segments,
                    gt_returns=all_returns,
                    segment_size=config.segment_size,
                    output_file=scatter_comparison_path,
                    max_samples=5000,
                    wandb_run=wandb.run if wandb.run else None,
                    random_seed=config.seed
                )
            else:
                print("No DTW augmentations generated, using baseline model")

    # Save final model
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
        
        # only compute returns for the original data
        obs_act_1 = np.concatenate(
            (dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1
        )
        obs_act_2 = np.concatenate(
            (dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1
        )
        return_1 = dataset["rewards"][idx_1].sum(axis=1)
        return_2 = dataset["rewards"][idx_2].sum(axis=1)
        
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


if __name__ == "__main__":
    train()
