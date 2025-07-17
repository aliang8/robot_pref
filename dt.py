import os

import hydra
import rich
import torch
from tqdm import trange

import torch.nn.functional as F
from utils.env import wrap_env
import wandb
from utils.eval import eval_actor, log_evaluation_results
from utils.wandb import wandb_init
from utils.seed import set_seed
from utils.data import normalize_datasets, SequentialReplayBuffer, setup_environment_and_dataset, setup_reward_model, print_dataset_statistics
from models.dt.decision_transformer import DecisionTransformer


def print_model_info(model):
    """Print model architecture and number of parameters."""
    print("Model Architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")

@hydra.main(config_path="configs", config_name="dt", version_base=None)
def train(config):
    """Main training function."""
    # Initialize wandb
    if config.use_wandb:
        wandb_init(config)
    
    rich.print("Config:", config)
    
    # Setup environment and dataset
    env, dataset = setup_environment_and_dataset(config)
    
    # Get dimensions
    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Normalize dataset
    state_mean, state_std = normalize_datasets(dataset)
    
    # Setup reward model and label rewards if needed
    if config.use_reward_model:
        dataset = setup_reward_model(config, dataset)
    elif config.trivial_reward == 1:
        print("Using zero rewards (trivial reward)")
        dataset["rewards"] *= 0.0
    else:
        print("Using ground truth rewards")
    
    # Print dataset statistics
    print_dataset_statistics(dataset)
    
    # Wrap environment
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    
    # Setup replay buffer
    replay_buffer = SequentialReplayBuffer(state_dim, action_dim, config.buffer_size, K=config.K, device=config.device)
    replay_buffer.load_dataset(dataset)
    
    # Set seed
    set_seed(config.seed, env)
    
    # Setup networks
    dt = DecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=config.hidden_size,
        max_length=config.K).to(config.device)
    
    # batch = replay_buffer.sample(config.batch_size)
    # (observations, actions, rewards, rtgs, next_observations, dones, timesteps), attention_mask = batch

    # dt(states=observations, actions=actions, rewards=rewards, returns_to_go=rtgs, timesteps=timesteps, attention_mask=attention_mask)

    print_model_info(dt)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(dt.parameters(), lr=config.lr, weight_decay=1e-4)

    # Training loop
    for t in trange(int(config.max_timesteps)):
        batch, attention_mask = replay_buffer.sample(config.batch_size)
        action_preds = dt(batch, attention_mask=attention_mask)

        loss = F.mse_loss(action_preds, batch[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training metrics
        if wandb.run is not None:
            wandb.log({"loss": loss.item()}, step=t)

        # Evaluate periodically
        if (t + 1) % config.eval_freq == 0:
            print(f"Evaluation at step: {t + 1}")
            
            eval_results = eval_actor(
                env, dt, config.n_episodes, config.seed, 
                record_video=config.record_video, has_seq=True
            )
            
            eval_mean_rewards, eval_success, _ = eval_results
            eval_mean_reward = eval_mean_rewards.mean()
            eval_mean_success = eval_success.mean()
            
            print("---------------------------------------")
            print(f"Evaluation over {config.n_episodes} episodes: "
                  f"Reward: {eval_mean_reward:.3f}, Success: {eval_mean_success * 100:.1f}%")
            print("---------------------------------------")
            
            # Log evaluation results
            log_evaluation_results(config, t, eval_results)


if __name__ == "__main__":
    train()