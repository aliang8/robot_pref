from pathlib import Path
import hydra
import numpy as np
import rich
from omegaconf import DictConfig

from utils.data import (
    load_datasets,
    normalize_datasets,
    convert_labels_to_array,
    create_segment_indices,
    get_obs_act_data,
    get_images_data,
    get_eef_data,
    setup_checkpoint_paths,
)
from models.reward_model import RewardModel
from utils.reward import (
    compute_dtw_matrix_cross,
    create_augmented_preferences_from_dtw,
    get_feedbacks,
)
from utils.seed import set_seed
from utils.wandb import log_query_videos_to_wandb, wandb_init


@hydra.main(config_path="configs", config_name="reward", version_base=None)
def train(config: DictConfig):
    # Initialize wandb and seed
    if getattr(config, "use_wandb", False):
        wandb_init(config)

    rich.print(config)
    set_seed(config.seed)

    # Setup checkpoint paths
    setup_checkpoint_paths(config)

    # Load datasets
    dataset, cross_dataset = load_datasets(config)

    # Normalize observations if required
    if config.normalize:
        normalize_datasets(dataset)

    # Get preferences and validation data
    labels, idx_st_1, idx_st_2, val_labels, val_idx_st_1, val_idx_st_2 = get_feedbacks(
        config.data_path, config.feedback_num, human=config.human
    )

    val_episodes_path = Path(config.data_path).parent / "val_episodes.npy"
    val_episodes = np.load(val_episodes_path, allow_pickle=True)

    obs_act_dim = dataset["observations"].shape[-1] + dataset["actions"].shape[-1]
    print(f"Observation-action dimension: {obs_act_dim}")

    # Convert labels to preference arrays
    labels = convert_labels_to_array(labels)
    val_labels = convert_labels_to_array(val_labels)

    # Log source preferences
    log_query_videos_to_wandb(
        dataset, idx_st_1, idx_st_2, labels, config, prefix="prefs"
    )

    # Train reward model based on configuration
    if config.single_emb:
        if getattr(config, "eef_rm", False):
            train_eef_reward_model(
                config,
                dataset,
                labels,
                idx_st_1,
                idx_st_2,
                val_labels,
                val_idx_st_1,
                val_idx_st_2,
                val_episodes,
            )
        else:
            train_single_embodiment(
                config,
                dataset,
                labels,
                idx_st_1,
                idx_st_2,
                val_labels,
                val_idx_st_1,
                val_idx_st_2,
                val_episodes,
                obs_act_dim,
            )
    elif config.use_cross and config.cross_data_path:
        train_cross_embodiment(
            config,
            dataset,
            cross_dataset,
            labels,
            idx_st_1,
            idx_st_2,
            val_episodes,
            obs_act_dim,
        )


def train_single_embodiment(
    config,
    dataset,
    labels,
    idx_st_1,
    idx_st_2,
    val_labels,
    val_idx_st_1,
    val_idx_st_2,
    val_episodes,
    obs_act_dim,
):
    """Train reward model on single embodiment data."""
    print(f"Using ground truth preferences for {config.data_path} for reward learning")

    # Create segment indices
    train_idx_1 = create_segment_indices(idx_st_1, config.segment_size)
    train_idx_2 = create_segment_indices(idx_st_2, config.segment_size)
    val_idx_1 = create_segment_indices(val_idx_st_1, config.segment_size)
    val_idx_2 = create_segment_indices(val_idx_st_2, config.segment_size)

    # Get training data
    train_obs_act_1 = get_obs_act_data(dataset, train_idx_1)
    train_obs_act_2 = get_obs_act_data(dataset, train_idx_2)
    train_images1 = get_images_data(dataset, train_idx_1)
    train_images2 = get_images_data(dataset, train_idx_2)

    # Get validation data
    val_obs_act_1 = get_obs_act_data(dataset, val_idx_1)
    val_obs_act_2 = get_obs_act_data(dataset, val_idx_2)
    val_images1 = get_images_data(dataset, val_idx_1)
    val_images2 = get_images_data(dataset, val_idx_2)

    # Create and train reward model
    reward_model = RewardModel(
        config,
        dataset,
        train_obs_act_1,
        train_obs_act_2,
        labels,
        obs_act_dim,
        train_images1,
        train_images2,
    )

    reward_model.save_test_dataset(
        val_obs_act_1,
        val_obs_act_2,
        val_labels,
        val_labels,
        val_images1,
        val_images2,
        val_episodes,
        test_dataset=dataset,
    )

    reward_model.train_model()
    reward_model.save_model(config.checkpoints_path)


def train_cross_embodiment(
    config,
    dataset,
    cross_dataset,
    labels,
    idx_st_1,
    idx_st_2,
    val_episodes,
    obs_act_dim,
):
    """Train reward model with cross-embodiment data using DTW."""
    # Load segment indices
    data_path = Path(config.data_path)
    cross_data_path = Path(config.cross_data_path)

    seg_indices = np.load(
        data_path.parent / "segment_start_end_indices.npy", allow_pickle=True
    )
    cross_seg_indices = np.load(
        cross_data_path.parent / "segment_start_end_indices.npy", allow_pickle=True
    )
    val_episodes = np.load(
        cross_data_path.parent / "val_episodes.npy", allow_pickle=True
    )

    # Get or compute DTW matrix
    cross_dtw_matrix = maybe_compute_dtw_matrix(
        config,
        dataset,
        cross_dataset,
        seg_indices,
        cross_seg_indices,
        data_path,
        cross_data_path,
    )

    # Create augmented preferences from DTW matrix
    cross_labels, cross_idx_st_1, cross_idx_st_2 = (
        create_augmented_preferences_from_dtw(
            cross_dtw_matrix,
            labels,
            idx_st_1,
            idx_st_2,
            seg_indices,
            cross_seg_indices,
            config,
        )
    )

    log_query_videos_to_wandb(
        cross_dataset,
        cross_idx_st_1,
        cross_idx_st_2,
        cross_labels,
        config,
        prefix="cross_prefs",
    )

    # Create augmented segment indices and data
    aug_idx_1 = create_segment_indices(cross_idx_st_1, config.segment_size)
    aug_idx_2 = create_segment_indices(cross_idx_st_2, config.segment_size)

    aug_obs_act_1 = get_obs_act_data(cross_dataset, aug_idx_1)
    aug_obs_act_2 = get_obs_act_data(cross_dataset, aug_idx_2)
    train_images1 = get_images_data(cross_dataset, aug_idx_1)
    train_images2 = get_images_data(cross_dataset, aug_idx_2)

    # Create and train reward model
    print("Training reward model with augmented preferences...")
    reward_model = RewardModel(
        config,
        cross_dataset,
        aug_obs_act_1,
        aug_obs_act_2,
        cross_labels,
        obs_act_dim,
        train_images1,
        train_images2,
    )

    print(reward_model)

    # Save test dataset (using same augmented data for now)
    reward_model.save_test_dataset(
        aug_obs_act_1,
        aug_obs_act_2,
        cross_labels,
        cross_labels,
        train_images1,
        train_images2,
        val_episodes,
        test_dataset=cross_dataset,
    )

    reward_model.train_model()
    reward_model.save_model(config.checkpoints_path)


def train_eef_reward_model(
    config,
    dataset,
    labels,
    idx_st_1,
    idx_st_2,
    val_labels,
    val_idx_st_1,
    val_idx_st_2,
    val_episodes,
):
    """Train reward model that takes 3D EEF and maybe 2D goal position as input."""

    eef_dim = 3 if not config.eef_rm_2d else 5

    # Create segment indices
    train_idx_1 = create_segment_indices(idx_st_1, config.segment_size)
    train_idx_2 = create_segment_indices(idx_st_2, config.segment_size)
    val_idx_1 = create_segment_indices(val_idx_st_1, config.segment_size)
    val_idx_2 = create_segment_indices(val_idx_st_2, config.segment_size)

    # Get training data
    train_eef_1 = get_eef_data(dataset, train_idx_1, with_goal=config.eef_rm_2d)
    train_eef_2 = get_eef_data(dataset, train_idx_2, with_goal=config.eef_rm_2d)
    train_images1 = get_images_data(dataset, train_idx_1)
    train_images2 = get_images_data(dataset, train_idx_2)

    # Get validation data
    val_eef_1 = get_eef_data(dataset, val_idx_1, with_goal=config.eef_rm_2d)
    val_eef_2 = get_eef_data(dataset, val_idx_2, with_goal=config.eef_rm_2d)
    val_images1 = get_images_data(dataset, val_idx_1)
    val_images2 = get_images_data(dataset, val_idx_2)

    # Create and train reward model
    print("Training reward model with augmented preferences...")
    reward_model = RewardModel(
        config,
        dataset,
        train_eef_1,
        train_eef_2,
        labels,
        eef_dim,
        train_images1,
        train_images2,
    )

    print(reward_model)

    # Save test dataset
    reward_model.save_test_dataset(
        val_eef_1,
        val_eef_2,
        val_labels,
        val_labels,
        val_images1,
        val_images2,
        val_episodes,
        test_dataset=dataset,
    )

    reward_model.train_model()
    reward_model.save_model(config.checkpoints_path)


def maybe_compute_dtw_matrix(
    config,
    dataset,
    cross_dataset,
    seg_indices,
    cross_seg_indices,
    data_path,
    cross_data_path,
):
    """Get existing DTW matrix or compute new one."""
    # Create DTW matrix path
    new_data_path = (
        data_path.parent.parent
        / f"{data_path.parent.name}_X_{cross_data_path.parent.name}"
    )
    new_data_path.mkdir(parents=True, exist_ok=True)

    # Build DTW filename based on config
    dtw_filename = "cross_dtw_matrix"
    if config.use_relative_eef:
        dtw_filename += "_relative_eef"
    if config.use_goal_pos:
        dtw_filename += "_goal_pos"

    dtw_matrix_path = new_data_path / f"{dtw_filename}.npy"
    print(f"DTW matrix path: {dtw_matrix_path}")

    # Load existing or compute new DTW matrix
    if dtw_matrix_path.exists():
        print(f"Loading existing DTW matrix from {dtw_matrix_path}")
        return np.load(dtw_matrix_path)
    else:
        print("Computing cross-embodiment DTW matrix...")
        cross_dtw_matrix = compute_dtw_matrix_cross(
            dataset, seg_indices, cross_dataset, cross_seg_indices, config
        )
        np.save(dtw_matrix_path, cross_dtw_matrix)
        print(f"Saved DTW matrix to {dtw_matrix_path}")
        return cross_dtw_matrix


if __name__ == "__main__":
    train()
