import os
from pathlib import Path

import hydra
import torch
from d3rlpy.algos import BCConfig, IQLConfig
from d3rlpy.logging import WanDBAdapterFactory
from omegaconf import DictConfig, OmegaConf

import wandb
from models import RewardModel
from utils.data import AttrDict, load_dataset, load_tensordict
from utils.eval import create_env, eval_model
from utils.seed import set_seed


def print_model_architecture(algo):
    """Print the architecture details of a d3rlpy algorithm.

    Args:
        algo: A d3rlpy algorithm instance (IQL, BC, etc.)
    """
    print("\nModel Architecture Details:")
    print("=" * 50)
    print("Policy Network:")
    print(algo._impl._modules.policy)

    print("\nQ Functions:")
    for i, q_func in enumerate(algo._impl._modules.q_funcs):
        print(f"Q-Function {i}:")
        print(q_func)

    print("\nValue Function:")
    print(algo._impl._modules.value_func)
    print("=" * 50)


@hydra.main(config_path="config", config_name="iql")
def main(cfg: DictConfig):
    """Train a policy using specified algorithm with Hydra config."""
    # Register custom resolvers for path operations
    OmegaConf.register_resolver("basename", lambda path: Path(path).stem)

    # Convert OmegaConf config to AttrDict for easier access and serialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = AttrDict.from_nested_dict(cfg_dict)

    if cfg.debug:
        cfg.training.n_epochs = 5
        cfg.training.n_steps_per_epoch = 10

    algorithm_name = cfg.algorithm
    dataset_name = Path(cfg.data.data_path).stem

    # Replace dataset name placeholder in template strings
    if hasattr(cfg.output, "model_dir_name"):
        cfg.output.model_dir_name = cfg.output.model_dir_name.replace(
            "DATASET_NAME", dataset_name
        )

    print("\n" + "=" * 50)
    print(f"Training {algorithm_name.upper()} policy")
    print("=" * 50)

    # Print config for visibility (using original OmegaConf for pretty printing)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))

    # Set random seed for reproducibility
    random_seed = cfg.random_seed
    set_seed(random_seed)
    print(f"Global random seed set to {random_seed}")

    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get experiment name based on data path
    experiment_name = f"{algorithm_name.upper()}_{dataset_name}"

    # Load data
    print(f"Loading data from {cfg.data.data_path}")
    data = load_tensordict(cfg.data.data_path)

    # Get observation and action dimensions
    observations = data["obs"] if "obs" in data else data["state"]
    state_dim = observations.shape[1]
    action_dim = data["action"].shape[1]
    print(f"Observation dimension: {state_dim}, Action dimension: {action_dim}")

    # Create MDP dataset based on the algorithm
    if algorithm_name.lower() == "iql":
        # For IQL, we need a reward model
        # Load reward model
        if not cfg.data.use_zero_rewards and not cfg.data.use_ground_truth:
            reward_model = RewardModel(
                state_dim, action_dim, hidden_dims=cfg.model.hidden_dims
            )
            reward_model.load_state_dict(torch.load(cfg.data.reward_model_path))
            reward_model = reward_model.to(device)
            reward_model.eval()
            print(f"Loaded reward model from {cfg.data.reward_model_path}")

        else:
            reward_model = None

        # Check if we should use ground truth rewards
        use_ground_truth = cfg.data.get("use_ground_truth", False)
        if use_ground_truth:
            print("Using ground truth rewards instead of reward model predictions.")

        # Get reward scaling options
        scale_rewards = cfg.data.get("scale_rewards", False)
        reward_min = cfg.data.get("reward_min", -1.0)
        reward_max = cfg.data.get("reward_max", 1.0)
        if scale_rewards:
            print(f"Will scale rewards to range [{reward_min}, {reward_max}]")

        # Create MDP dataset
        print("Creating MDP dataset with rewards...")
        dataset = load_dataset(
            data,
            reward_model=reward_model,
            device=device,
            use_ground_truth=use_ground_truth,
            max_segments=cfg.data.max_segments,
            reward_batch_size=cfg.data.reward_batch_size,
            scale_rewards=scale_rewards,
            reward_min=reward_min,
            reward_max=reward_max,
            use_zero_rewards=cfg.data.get("use_zero_rewards", False),
        )
    else:  # BC or other algorithms that don't need a reward model
        # For BC, we can directly use the demonstrations
        print("Creating MDP dataset from demonstrations...")

        # Get reward scaling options (also apply to BC for consistency)
        scale_rewards = cfg.data.get("scale_rewards", False)
        reward_min = cfg.data.get("reward_min", -1.0)
        reward_max = cfg.data.get("reward_max", 1.0)
        if scale_rewards:
            print(f"Will scale rewards to range [{reward_min}, {reward_max}]")

        dataset = load_dataset(
            data,
            scale_rewards=scale_rewards,
            reward_min=reward_min,
            reward_max=reward_max,
            use_zero_rewards=cfg.data.get("use_zero_rewards", False),
        )

    # Initialize algorithm based on the algorithm_name
    print(f"Initializing {algorithm_name.upper()} algorithm...")

    if algorithm_name.lower() == "iql":
        # Initialize IQL algorithm
        iql_config = IQLConfig(**cfg.iql)
        algo = iql_config.create()
        
        # This is for wandb logging
        algo.create_impl(observation_shape=[state_dim], action_size=action_dim)

    elif algorithm_name.lower() == "bc":
        # Initialize BC algorithm
        bc_config = BCConfig(**cfg.bc)
        algo = bc_config.create()

        # This is for wandb logging
        algo.create_impl(observation_shape=[state_dim], action_size=action_dim)

        # For BC with weight decay
        if hasattr(cfg.model, "use_weight_decay") and cfg.model.use_weight_decay:
            if hasattr(algo, "create_impl"):
                impl = algo.create_impl(
                    state_dim, action_dim, algo._encoder_factory, algo._optim_factory
                )
                # Set weight decay if it's used
                if hasattr(impl.optim, "param_groups"):
                    for param_group in impl.optim.param_groups:
                        param_group["weight_decay"] = cfg.model.weight_decay

            # Fallback for older d3rlpy versions
            if hasattr(algo, "_impl") and hasattr(algo._impl, "optim"):
                for param_group in algo._impl.optim.param_groups:
                    param_group["weight_decay"] = cfg.model.weight_decay

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    # Print model architecture details
    print_model_architecture(algo)

    # Set up the environment
    env = create_env(cfg)

    # Set up the fitter for training
    print("Setting up the fitter for training...")
    fitter_kwargs = dict(
        dataset=dataset,
        n_steps=cfg.training.n_epochs * cfg.training.n_steps_per_epoch,
        n_steps_per_epoch=cfg.training.n_steps_per_epoch,
        experiment_name=experiment_name,
        with_timestamp=True,
        show_progress=True,
        save_interval=cfg.training.save_interval,
    )

    # Initialize WanDBAdapterFactory for logging
    if cfg.wandb.use_wandb:
        print("Initializing WanDBAdapterFactory for logging...")
        wandb_adapter_factory = WanDBAdapterFactory(project=cfg.wandb.project)
        fitter_kwargs["logger_adapter"] = wandb_adapter_factory

    # Training loop
    print(f"Training {algorithm_name.upper()} for {cfg.training.n_epochs} epochs...")
    for epoch, _ in algo.fitter(**fitter_kwargs):
        # For first epoch update configs
        if cfg.wandb.use_wandb and epoch == 1:
            wandb.run.config.update(cfg_dict)

        if env is not None:
            eval_model(env=env, algo=algo, cfg=cfg, epoch=epoch)

    # Get the model directory name from config if available, or fall back to default
    if hasattr(cfg.output, "model_dir_name"):
        model_dir_name = cfg.output.model_dir_name
        # Add zero reward indicator if using that mode
        if cfg.data.get("use_zero_rewards", False):
            model_dir_name += "_zero_rewards"

        # Create subdirectory based on the name template
        model_dir = os.path.join(cfg.output.output_dir, model_dir_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{algorithm_name.lower()}.pt")
    else:
        # Fall back to the original naming scheme
        zero_suffix = "_zero_rewards" if cfg.data.get("use_zero_rewards", False) else ""
        model_path = f"{cfg.output.output_dir}/{algorithm_name.lower()}_{Path(cfg.data.data_path).stem}{zero_suffix}.pt"

    print(f"Model saved to {model_path}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
