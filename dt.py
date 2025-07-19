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
from utils.data import normalize_datasets, DTSequentialReplayBuffer, setup_environment_and_dataset, setup_reward_model, print_dataset_statistics
from models.decision_transformer import DecisionTransformer
from utils.log import print_model_info

@hydra.main(config_path="configs", config_name="dt", version_base=None)
def train(config):
    set_seed(config.seed)

    if config.use_wandb:
        wandb_init(config)
    
    rich.print("Config:", config)
    
    # Setup env and dataset
    env, dataset = setup_environment_and_dataset(config)

    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0]

    # Setup dataset
    state_mean, state_std = normalize_datasets(dataset)
    if config.use_reward_model:
        dataset = setup_reward_model(config, dataset)
    elif config.trivial_reward == 1:
        print("Using zero rewards (trivial reward)")
        dataset["rewards"] *= 0.0
    else:
        print("Using ground truth rewards")
    replay_buffer = DTSequentialReplayBuffer(state_dim, action_dim, config.buffer_size, K=config.K, device=config.device)
    replay_buffer.load_dataset(dataset)

    print_dataset_statistics(dataset)

    # Setup env
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    
    # Networks
    dt = DecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=config.hidden_size,
        max_length=config.K, nhead=config.nhead, nlayer=config.nlayer).to(config.device)
    print_model_info({"Decision Transformer": dt})
    optimizer = torch.optim.Adam(dt.parameters(), lr=config.lr, weight_decay=1e-4)

    # Training loop
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        action_preds = dt(batch)

        loss = F.mse_loss(action_preds, batch[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if wandb.run is not None:
            wandb.log({"loss": loss.item()}, step=t)

        if (t + 1) % config.eval_freq == 0:
            print(f"Evaluation at step: {t + 1}")
            
            eval_results = eval_actor(
                env, dt, config.n_episodes, config.seed, 
                record_video=config.record_video, has_seq=True, target_return=config.target_return,
            )
            
            eval_mean_rewards, eval_success, _ = eval_results
            eval_mean_reward = eval_mean_rewards.mean()
            eval_mean_success = eval_success.mean()
            
            print("---------------------------------------")
            print(f"Evaluation over {config.n_episodes} episodes: "
                  f"Reward: {eval_mean_reward:.3f}, Success: {eval_mean_success * 100:.1f}%")
            print("---------------------------------------")
            log_evaluation_results(config, t, eval_results)


if __name__ == "__main__":
    train()