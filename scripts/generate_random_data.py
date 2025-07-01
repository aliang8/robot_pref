import os
import sys

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import h5py
import imageio
import numpy as np

import utils_env


def main():
    data_path = "/tmp/mimicgen_stack_100/stack/demo_src_stack_task_D0_r_Panda/demo.hdf5"
    output_path = "/tmp/mimicgen_stack_100/stack/demo_src_stack_task_D0_r_Panda/demo_aug.hdf5"

    env = utils_env.get_robomimic_env(data_path)

    with h5py.File(data_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Currently using all demos in the file, so len(demo_aug) == len(demo) * 2
        demo_keys = list(f_in["data"].keys())
        demo_key_num = len(demo_keys)

        for demo_key in demo_keys:
            collected_obs = []
            collected_states = []
            collected_actions = []
            collected_rewards = []

            frames = []  # for visualization

            # Open the file and sample a random demo
            demo = f_in["data"][demo_key]
            actions = demo["actions"][:]                     # (T, action_dim)
            initial_state = demo["states"][0]                # (state_dim,)

            print(f"Sampled demo: {demo_key} with {actions.shape[0]} timesteps")

            # inject noise to actions
            noise_std = 0.2 # adjust noise level
            noisy_actions = actions + np.random.randn(*actions.shape) * noise_std

            # reset env to start of demo
            env.env.reset_to({"states": initial_state})

            # replay actions
            for action in noisy_actions:
                obs = env.env.get_observation()
                obs["agentview_image"] = env.render(mode="rgb_array") # add rgb image to observation
                action = action
                state = env.env.get_state()["states"]
                rew = env.env.get_reward()

                collected_obs.append(obs)
                collected_states.append(state)
                collected_actions.append(action)
                collected_rewards.append(rew)
                
                env.step(action)

                frames.append(obs["agentview_image"])

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

            # Also copy the original demo
            group = f_out.require_group(f"data/{demo_key}")
            f_in.copy(demo["obs"], group, name="obs")
            for k in ["states", "actions", "rewards"]:
                group.create_dataset(k, data=demo[k][:], compression="gzip")

            demo_key_num += 1

            frames = np.array(frames)  # (T, H, W, C)
            video_path = f"/scr/matthewh6/robot_pref/random_data_vids/{demo_key_num}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()