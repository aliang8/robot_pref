import os
import sys

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import cv2
import h5py
import imageio
from pathlib import Path
import numpy as np

import utils_env


def main():
    # define this
    expert_data_path = "/tmp/mimicgen_stack_100/stack/demo_src_stack_task_D0_r_Panda/demo.hdf5"
    
    # expert/suboptimal split
    expert_trajs = 100
    suboptimal_trajs = 200

    output_path = Path(expert_data_path).parent / f"demo_exp-{expert_trajs}_sub-{suboptimal_trajs}.hdf5"
    video_path = Path("/scr/matthewh6/robot_pref/mixed_data_vids")

    env = utils_env.get_robomimic_env(expert_data_path)

    with h5py.File(expert_data_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Copy file-level attributes (root-level)
        data_group_out = f_out.require_group("data")
        for attr_key, attr_val in f_in["data"].attrs.items():
            print(f"Copying attribute {attr_key} with value {attr_val}")
            data_group_out.attrs[attr_key] = attr_val

        # Get all available demo keys from the input file
        all_demo_keys = list(f_in["data"].keys())
        print(f"Found {len(all_demo_keys)} demos in the input file")
        
        # Sample expert trajectories (with replacement if needed)
        if len(all_demo_keys) >= expert_trajs:
            expert_demo_keys = np.random.choice(all_demo_keys, expert_trajs, replace=False)
        else:
            expert_demo_keys = np.random.choice(all_demo_keys, expert_trajs, replace=True)
            print(f"Warning: Only {len(all_demo_keys)} demos available, sampling with replacement")

        demo_key_num = 0
        
        # Copy expert trajectories to output file and create videos
        print(f"Copying {len(expert_demo_keys)} expert trajectories...")
        for demo_key in expert_demo_keys:
            demo = f_in["data"][demo_key]
            
            # Create new demo group in output file
            new_demo_name = f"demo_{demo_key_num}"
            group = f_out.require_group(f"data/{new_demo_name}")
            
            # Copy observation data
            f_in.copy(demo["obs"], group, name="obs")
            
            # Copy other datasets
            for k in ["states", "actions", "rewards"]:
                group.create_dataset(k, data=demo[k][:], compression="gzip")

            # Copy attributes
            for attr_key, attr_val in demo.attrs.items():
                group.attrs[attr_key] = attr_val
            
            # Create video for expert trajectory
            frames = []
            actions = demo["actions"][:]
            initial_state = demo["states"][0]
            
            print(f"Creating video for expert demo {demo_key_num+1}/{len(expert_demo_keys)} from {demo_key} with {actions.shape[0]} timesteps")
            
            # Reset environment to initial state
            env.env.reset_to({"states": initial_state})
            
            # Replay the expert actions to create video
            for action in actions:
                obs = env.env.get_observation()
                obs["agentview_image"] = env.render(mode="rgb_array")
                rew = env.env.get_reward()
                
                rgb_frame = obs["agentview_image"]
                frame_with_text = rgb_frame.copy()
                cv2.putText(
                    frame_with_text,
                    f"Expert - Reward: {rew:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                
                env.step(action)
                frames.append(frame_with_text)
            
            # Save expert video
            frames = np.array(frames)            
            imageio.mimsave(video_path / f"expert_{demo_key_num}.mp4", frames, fps=30)
            print(f"Saved expert video to {video_path}")
                
            demo_key_num += 1

        print(f"Copied {len(expert_demo_keys)} expert trajectories")
        
        # Generate suboptimal trajectories
        print(f"Generating {suboptimal_trajs} suboptimal trajectories...")
        for i in range(suboptimal_trajs):
            # Sample a random expert trajectory to add noise to
            sampled_demo_key = np.random.choice(expert_demo_keys)
            demo = f_in["data"][sampled_demo_key]
            
            collected_obs = []
            collected_states = []
            collected_actions = []
            collected_rewards = []
            frames = []  # for visualization

            actions = demo["actions"][:]                     # (T, action_dim)
            initial_state = demo["states"][0]                # (state_dim,)

            print(f"Generating suboptimal demo {i+1}/{suboptimal_trajs} from {sampled_demo_key} with {actions.shape[0]} timesteps")

            # inject noise to actions
            noise_std = 0.3 # adjust noise level
            noisy_actions = actions + np.random.randn(*actions.shape) * noise_std

            # reset env to start of demo and random half the time
            if i % 2 == 0:
                env.env.reset_to({"states": initial_state})
            else:
                env.reset()

            # replay actions
            for action in noisy_actions:
                obs = env.env.get_observation()
                obs["agentview_image"] = env.render(mode="rgb_array") # add rgb image to observation
                state = env.env.get_state()["states"]
                rew = env.env.get_reward()

                collected_obs.append(obs)
                collected_states.append(state)
                collected_actions.append(action)
                collected_rewards.append(rew)

                rgb_frame = obs["agentview_image"]
                
                frame_with_text = rgb_frame.copy()
                cv2.putText(
                    frame_with_text,
                    f"Reward: {rew:.2f}",
                    (10, 30),  # position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,           # font scale (smaller)
                    (0, 255, 0),   # color (B, G, R)
                    1,             # thickness (you can also reduce this)
                    cv2.LINE_AA
                )

                env.step(action)
                frames.append(frame_with_text)

            collected_obs = {k: np.stack([obs[k] for obs in collected_obs]) for k in collected_obs[0]}  # {key: (T, ...shape...)}
            collected_states = np.stack(collected_states)  # (T, state_dim)
            collected_actions = np.stack(collected_actions)  # (T, action_dim)
            collected_rewards = np.array(collected_rewards)  # (T,)

            # add the newly collected data to the output file
            # Write the new (noisy) demo
            new_demo_name = f"demo_{demo_key_num}"
            new_demo_group = f_out.require_group(f"data/{new_demo_name}")

            # write obs sub-datasets
            obs_group = new_demo_group.require_group("obs")
            for k, v in collected_obs.items():
                obs_group.create_dataset(k, data=v, compression="gzip")

            # write other arrays
            for k, v in zip(["states", "actions", "rewards"], 
                            [collected_states, collected_actions, collected_rewards]):
                new_demo_group.create_dataset(k, data=v, compression="gzip")

            demo_key_num += 1

            # Save video
            frames = np.array(frames)  # (T, H, W, C)
            imageio.mimsave(video_path / f"sub_{demo_key_num}.mp4", frames, fps=30)
            print(f"Saved video to {video_path}")

        print(f"Data augmentation complete! Generated {expert_trajs} expert + {suboptimal_trajs} suboptimal = {expert_trajs + suboptimal_trajs} total trajectories")
        print(f"Output file saved to {output_path}")


if __name__ == "__main__":
    main()