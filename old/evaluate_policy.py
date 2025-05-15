import os
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import metaworld
import d3rlpy
from d3rlpy.algos import IQL

# Import utility functions
from utils.trajectory import RANDOM_SEED
from train_iql_policy import get_metaworld_env

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def record_video(env, algo, task_name, video_path=None, num_frames=500):
    """Record a video of the policy performance."""
    try:
        import imageio
        import numpy as np

        if video_path is None:
            video_path = f"videos/{task_name}_policy.mp4"

        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # Reset environment
        obs = env.reset()
        frames = []

        # Collect frames
        for _ in tqdm(range(num_frames), desc="Recording video"):
            # Render frame
            frame = env.render(mode="rgb_array")
            frames.append(frame)

            # Take action
            action = algo.predict([obs])[0]
            obs, _, done, _ = env.step(action)

            if done:
                obs = env.reset()

        # Save video
        print(f"Saving video to {video_path}")
        imageio.mimsave(video_path, frames, fps=30)
        return video_path
    except Exception as e:
        print(f"Error recording video: {e}")
        return None


def evaluate(model_path, task_name, num_episodes=20, render=False, record=False):
    """Evaluate a trained IQL policy on a MetaWorld environment."""
    # Create environment
    env = get_metaworld_env(task_name)

    # Load the IQL model
    algo = IQL()
    algo.build_with_env(env)
    algo.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Evaluation stats
    returns = []
    success_rates = []
    episode_lengths = []

    # Evaluate for multiple episodes
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0
        success = False
        steps = 0

        while not done:
            # Select action
            action = algo.predict([obs])[0]

            # Take action in environment
            next_obs, reward, done, info = env.step(action)

            # Render if requested
            if render:
                env.render()

            # Update stats
            ep_return += reward
            steps += 1

            # Check for success
            if "success" in info and info["success"]:
                success = True
                break

            # Update observation
            obs = next_obs

            # Break if episode is too long
            if steps >= 500:  # MetaWorld episodes are typically 500 steps
                break

        # Record episode stats
        returns.append(ep_return)
        success_rates.append(float(success))
        episode_lengths.append(steps)

        # Print episode result
        print(
            f"Episode {ep + 1}/{num_episodes}: Return = {ep_return:.2f}, Success = {success}, Steps = {steps}"
        )

    # Compute summary statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = np.mean(success_rates) * 100
    mean_length = np.mean(episode_lengths)

    print("\nEvaluation Results:")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Mean Episode Length: {mean_length:.2f}")

    # Record video if requested
    video_path = None
    if record:
        video_path = record_video(env, algo, task_name)

    # Return results
    results = {
        "returns": returns,
        "success_rates": success_rates,
        "episode_lengths": episode_lengths,
        "mean_return": mean_return,
        "std_return": std_return,
        "success_rate": success_rate,
        "mean_length": mean_length,
        "video_path": video_path,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained IQL policy")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained IQL model"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="MetaWorld task name or path to dataset",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record a video of the policy"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to save evaluation results"
    )

    args = parser.parse_args()

    # Evaluate the policy
    results = evaluate(
        args.model_path,
        args.task_name,
        num_episodes=args.num_episodes,
        render=args.render,
        record=args.record,
    )

    # Save results if requested
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {args.output_path}")

    # Plot returns and success rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot returns
    ax1.plot(range(1, args.num_episodes + 1), results["returns"], marker="o")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title(f"Episode Returns (Mean: {results['mean_return']:.2f})")
    ax1.grid(True)

    # Plot success rates (cumulative)
    cumulative_success = np.cumsum(results["success_rates"]) / np.arange(
        1, args.num_episodes + 1
    )
    ax2.plot(range(1, args.num_episodes + 1), cumulative_success * 100, marker="o")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title(f"Cumulative Success Rate (Final: {results['success_rate']:.2f}%)")
    ax2.grid(True)
    ax2.set_ylim(0, 105)

    plt.tight_layout()

    # Save plot if output path is provided
    if args.output_path:
        plot_path = args.output_path.replace(".pkl", ".png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
