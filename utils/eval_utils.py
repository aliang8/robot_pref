import os
import gym
import numpy as np
import random
import time
import torch
from tqdm import tqdm
import imageio
from multiprocessing import get_context

# Default random seed for reproducibility
RANDOM_SEED = 42


class SimpleVideoRecorder:
    """A simple video recorder using imageio for saving environment frames as videos."""

    def __init__(self, env, path, fps=30):
        """Initialize a video recorder.

        Args:
            env: The environment to record
            path: Path to save the video file
            fps: Frames per second for the video
        """
        self.env = env
        self.path = path
        self.fps = fps
        self.frames = []
        self._closed = False

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        print(f"Video recorder initialized for: {self.path}")

    def capture_frame(self):
        """Capture a frame from the environment."""
        if self._closed:
            return

        try:
            # Try different render modes
            try:
                frame = self.env.render(mode="rgb_array")
            except:
                try:
                    # Try without mode specification
                    frame = self.env.render()
                except Exception as e:
                    print(f"Failed to render environment: {e}")
                    return

            # Skip if frame is None or not an ndarray
            if frame is None or not isinstance(frame, np.ndarray):
                print(f"Warning: Invalid frame from render, got {type(frame)}")
                return

            # Ensure frame has the right format (height, width, channels)
            if len(frame.shape) != 3:
                print(f"Warning: Frame has unexpected shape: {frame.shape}")
                return

            # Store the frame
            self.frames.append(frame)

        except Exception as e:
            print(f"Error capturing frame: {e}")

    def close(self):
        """Close the recorder and save the video."""
        if self._closed:
            return

        try:
            if self.frames:
                print(f"Saving video with {len(self.frames)} frames to {self.path}")
                # Save frames as a video using imageio
                imageio.mimsave(self.path, self.frames, fps=self.fps)
                print(f"Video saved successfully: {self.path}")
            else:
                print(f"No frames captured, video not saved: {self.path}")

        except Exception as e:
            print(f"Error saving video to {self.path}: {e}")

        # Mark as closed even if there was an error
        self._closed = True
        self.frames = []  # Clear frames to free memory


class RenderWrapper(gym.Wrapper):
    """A wrapper that ensures proper rendering, particularly useful for MuJoCo environments."""

    def __init__(self, env, seed=None):
        super().__init__(env)
        self.env = env

        # Set a unique seed if provided
        if seed is not None:
            self.seed(seed)

        # Ensure proper render mode
        if hasattr(env, "render_mode"):
            try:
                self.env.render_mode = "rgb_array"
            except (AttributeError, TypeError):
                # If render_mode is not settable, try to use it as is
                pass

        # Try to initialize rendering
        try:
            _ = self.render(mode="rgb_array")
        except:
            pass

    def reset(self, **kwargs):
        """Reset the environment and ensure rendering is initialized."""
        result = self.env.reset(**kwargs)
        # Reinitialize rendering after reset
        try:
            _ = self.render(mode="rgb_array")
        except:
            pass
        return result

    def render(self, mode="rgb_array", **kwargs):
        """Render with proper mode, ensuring rgb_array is used if available."""
        try:
            return self.env.render(mode=mode, **kwargs)
        except:
            # If rgb_array fails, try without specifying mode
            return self.env.render(**kwargs)

    def seed(self, seed=None):
        """Seed the environment with the given seed."""
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return [seed]


def create_video_recorder(env, video_path, episode_idx, fps=30):
    """Create a video recorder for the given environment and episode.

    Args:
        env: The environment to record
        video_path: Base path for video files
        episode_idx: Episode index for the filename
        fps: Frames per second for the video

    Returns:
        SimpleVideoRecorder object or None if creation failed
    """
    try:
        import os

        # Ensure directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # Initialize rendering for MuJoCo-based environments
        if hasattr(env, "render_mode"):
            # Make sure render mode is set to rgb_array
            try:
                env.render_mode = "rgb_array"
            except (AttributeError, TypeError):
                pass  # Attribute can't be set

        # For MuJoCo/MetaWorld environments, try to ensure rendering is initialized
        if hasattr(env, "model"):
            try:
                env.render(mode="rgb_array")
            except:
                try:
                    env.render()
                except:
                    pass

        # Create video path with episode index
        episode_video_path = f"{video_path}_episode_{episode_idx}.mp4"

        # Create recorder
        recorder = SimpleVideoRecorder(env, episode_video_path, fps=fps)

        return recorder
    except Exception as e:
        print(f"Failed to create video recorder for episode {episode_idx}: {e}")
        return None


def evaluate_episode_worker(worker_args):
    """Worker function for parallel episode evaluation."""
    env_creator, algo, episode_idx, record_video, video_path, video_fps = worker_args

    # The env_creator function is already configured with the proper seed
    # as we're using a lambda with default argument in the parent function
    env_instance = env_creator()

    # Wrap the environment to ensure proper rendering if video recording is requested
    if record_video:
        env = RenderWrapper(env_instance)
    else:
        env = env_instance

    # Setup video recording if requested
    video_recorder = None
    if record_video and video_path:
        try:
            import os

            # Create video directory if needed
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            episode_video_path = f"{video_path}_episode_{episode_idx}.mp4"
            try:
                # Create the recorder
                video_recorder = SimpleVideoRecorder(
                    env, episode_video_path, fps=video_fps
                )
            except Exception as e:
                print(f"Error setting up video recorder: {e}")
                video_recorder = None
        except Exception as e:
            print(f"Error setting up video recording: {e}")

    # Reset environment and ensure rendering is ready
    observation = env.reset()

    # For MetaWorld environments, we may need to explicitly initialize the rendering
    if record_video and hasattr(env, "model"):
        # Initialize rendering for MuJoCo-based environments
        if hasattr(env, "render_mode"):
            # Make sure render mode is set to rgb_array
            try:
                env.render_mode = "rgb_array"
            except (AttributeError, TypeError):
                pass  # Attribute can't be set

        # Some environments need an explicit render call to initialize
        try:
            env.render(mode="rgb_array")
        except:
            try:
                # Try without specifying mode
                env.render()
            except:
                pass

    done = False
    episode_return = 0
    steps = 0
    success = False

    # Record initial frame if recording
    if video_recorder:
        try:
            video_recorder.capture_frame()
        except Exception as e:
            print(f"Warning: Could not capture frame: {e}")

    # Run episode
    while not done and steps < 1000:  # Step limit to prevent infinite loops
        # Extract observation if it's a tuple
        if isinstance(observation, tuple):
            obs_array = observation[0]  # Use only the observation part
        else:
            obs_array = observation

        # Ensure observation is a numpy array with batch dimension
        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array, dtype=np.float32)

        if len(obs_array.shape) == 1:
            obs_array = np.expand_dims(obs_array, axis=0)

        # Run prediction
        action = algo.predict(obs_array)[0]

        # Handle different step return formats
        step_result = env.step(action)

        # MetaWorld environments return 4 values: obs, reward, done, info
        if len(step_result) == 4:
            observation, reward, done, info = step_result
        # Some environments might return 5 values including truncated flag (gym>=0.26)
        elif len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            print(f"Warning: Unexpected step result format: {len(step_result)} values")
            break

        episode_return += reward
        steps += 1

        # Record frame if recording
        if video_recorder:
            try:
                video_recorder.capture_frame()
            except Exception as e:
                print(f"Warning: Could not capture frame: {e}")

        # Check for success
        if isinstance(info, dict) and "success" in info and info["success"]:
            success = True
            break

    # Close video recorder
    if video_recorder:
        try:
            video_recorder.close()
        except Exception as e:
            print(f"Warning: Error closing video recorder: {e}")

    # Return episode results
    return {"return": episode_return, "steps": steps, "success": success}


def evaluate_policy_manual(
    env,
    algo,
    n_episodes=10,
    verbose=True,
    record_video=False,
    video_path=None,
    video_fps=30,
    parallel=True,
    num_workers=None,
):
    """Manually evaluate policy on environment and return detailed metrics.

    Args:
        env: Environment to evaluate on, or a function that creates environments
        algo: Algorithm/policy to evaluate
        n_episodes: Number of episodes to evaluate
        verbose: Whether to print evaluation results
        record_video: Whether to record videos
        video_path: Path to save videos
        video_fps: FPS for recorded videos
        parallel: Whether to run evaluation in parallel
        num_workers: Number of parallel workers (default: min(n_episodes, cpu_count))

    Returns:
        Dictionary with evaluation metrics
    """
    # Check if environment is available
    if env is None:
        print("No environment available for evaluation.")
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "success_rate": 0.0,
            "num_episodes": 0,
        }

    # Generate base seed values for each episode to ensure diversity
    # We'll use these seeds for both parallel and sequential evaluation
    base_seeds = []
    for i in range(n_episodes):
        # Create a unique seed for each episode that's different from the global random seed
        # This ensures diversity across episodes and consistency across runs
        base_seed = RANDOM_SEED + i * 10000 + int(time.time() % 1000)
        base_seeds.append(base_seed)

    # Determine if we should use parallel evaluation
    use_parallel = parallel and n_episodes > 1

    if use_parallel:
        try:
            import multiprocessing as mp
            from multiprocessing import get_context

            # Create a function that can recreate environments
            if not callable(env):
                # Store the environment configuration for recreation
                env_type = type(env)
                if hasattr(env, "spec") and hasattr(env.spec, "id"):
                    env_id = env.spec.id
                    env_creator = lambda seed=None: gym.make(env_id)
                else:
                    # For MetaWorld environments, we need to recreate using fallback methods
                    print(
                        "Warning: Using approximate environment recreation for parallel evaluation"
                    )
                    original_env = env

                    # Create a generic environment recreation function
                    # Note: The implementation may need to be adjusted based on your specific environments
                    def create_similar_env(seed=None):
                        try:
                            # Try to use a specific recreation method if available
                            # You'll need to implement this based on your environment type
                            from metaworld.envs import (
                                ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            )

                            # Try to determine the environment type/name from the original
                            if hasattr(original_env, "__class__") and hasattr(
                                original_env.__class__, "__name__"
                            ):
                                env_name = original_env.__class__.__name__
                                for (
                                    name,
                                    constructor,
                                ) in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.items():
                                    if env_name in name:
                                        return constructor(seed=seed)
                        except:
                            pass

                        # If creation with seed fails, return original env
                        if seed is not None and hasattr(original_env, "seed"):
                            # If we can't create a new env, at least try to seed the original
                            original_env.seed(seed)
                        return original_env

                    env_creator = create_similar_env
            else:
                # If environment is already a creator function, wrap it to accept a seed
                original_creator = env

                def seeded_env_creator(seed=None):
                    new_env = original_creator()
                    if seed is not None and hasattr(new_env, "seed"):
                        new_env.seed(seed)
                    return new_env

                env_creator = seeded_env_creator

            # Determine number of workers
            if num_workers is None:
                num_workers = min(n_episodes, mp.cpu_count())

            if verbose:
                print(
                    f"Running parallel evaluation with {num_workers} workers for {n_episodes} episodes"
                )
                print(
                    f"Using different seeds for each environment to ensure diverse initial states"
                )

            # Prepare arguments for workers
            worker_args = [
                (
                    lambda seed=base_seeds[i]: env_creator(seed),
                    algo,
                    i,
                    record_video and i < 3,
                    video_path,
                    video_fps,
                )
                for i in range(n_episodes)
            ]

            # Use 'spawn' context for better compatibility across platforms
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(evaluate_episode_worker, worker_args),
                        total=n_episodes,
                        desc="Evaluating policy",
                    )
                )

            # Process results
            returns = [r["return"] for r in results]
            steps_to_complete = [r["steps"] for r in results]
            success_rate = sum(r["success"] for r in results) / n_episodes

        except ImportError:
            print(
                "Warning: multiprocessing not available. Falling back to sequential evaluation."
            )
            use_parallel = False
        except Exception as e:
            print(
                f"Error in parallel evaluation: {e}. Falling back to sequential evaluation."
            )
            import traceback

            traceback.print_exc()
            use_parallel = False

    # Sequential evaluation (fallback or if parallel not requested)
    if not use_parallel:
        returns = []
        success_rate = 0
        steps_to_complete = []

        # Setup video recording if requested
        video_recorders = []
        if record_video and video_path:
            try:
                # Record at most 3 episodes to save space
                n_videos = min(n_episodes, 3)

                # For video recording, we'll create a recorder for each episode
                # but initialize it only when needed to avoid rendering issues
                for i in range(n_videos):
                    video_recorders.append(None)  # Placeholder

                print(
                    f"Will record up to {n_videos} evaluation episodes to {video_path}_episode_*.mp4"
                )
            except Exception as e:
                print(f"Error setting up video recording: {e}")
                record_video = False

        if verbose:
            print(f"Running sequential evaluation for {n_episodes} episodes")
            print(
                f"Using different seeds for each episode to ensure diverse initial states"
            )

        for episode in tqdm(range(n_episodes), desc="Evaluating policy"):
            # Get the base seed for this episode
            episode_seed = base_seeds[episode]

            # Create a fresh environment for each episode
            if callable(env):
                # If env is a creator function, create a fresh environment
                episode_env_instance = env()
            else:
                # Otherwise use the provided environment
                # (we'll still seed it differently for each episode)
                episode_env_instance = env

            # Set the seed for this episode
            if hasattr(episode_env_instance, "seed"):
                episode_env_instance.seed(episode_seed)

            # For recording episodes, create a fresh wrapped environment
            if record_video and episode < len(video_recorders):
                # Create a wrapper for rendering
                episode_env = RenderWrapper(episode_env_instance)
            else:
                # Use the unwrapped environment directly
                episode_env = episode_env_instance

            # Print environment info for the first episode
            if verbose and episode == 0:
                print(f"Evaluating on environment with:")
                print(f"  Observation space: {episode_env.observation_space}")
                print(f"  Action space: {episode_env.action_space}")

            # Reset environment
            observation = episode_env.reset()

            # For MetaWorld environments, we may need to explicitly reset the rendering
            # This ensures rendering works properly for all episodes
            if record_video and hasattr(episode_env, "model"):
                # Reset the environment's MuJoCo visualization if it exists
                # This is needed for MetaWorld environments
                if hasattr(episode_env, "render_mode"):
                    # Re-set the render mode to ensure the viewer is active
                    try:
                        episode_env.render_mode = "rgb_array"
                    except (AttributeError, TypeError):
                        pass  # Attribute can't be set
                # Some environments need an explicit render call to initialize the viewer
                try:
                    episode_env.render(mode="rgb_array")
                except:
                    try:
                        # Try without specifying mode
                        episode_env.render()
                    except:
                        pass

            done = False
            episode_return = 0
            steps = 0

            # Determine if this episode should be recorded
            record_this_episode = record_video and episode < len(video_recorders)
            if record_this_episode:
                # Create a new recorder for this episode
                video_recorders[episode] = create_video_recorder(
                    episode_env, video_path, episode, fps=video_fps
                )
                if video_recorders[episode]:
                    try:
                        video_recorders[episode].capture_frame()
                    except Exception as e:
                        print(f"Warning: Could not capture initial frame: {e}")

            while not done and steps < 1000:  # Add step limit to prevent infinite loops
                # Extract observation if it's a tuple (some environments return additional info)
                if isinstance(observation, tuple):
                    obs_array = observation[0]  # Use only the observation part
                else:
                    obs_array = observation

                # Ensure observation is a numpy array with batch dimension
                if not isinstance(obs_array, np.ndarray):
                    obs_array = np.array(obs_array, dtype=np.float32)

                if len(obs_array.shape) == 1:
                    obs_array = np.expand_dims(obs_array, axis=0)

                # Run prediction
                action = algo.predict(obs_array)[0]

                # Handle different step return formats
                step_result = episode_env.step(action)

                # MetaWorld environments return 4 values: obs, reward, done, info
                if len(step_result) == 4:
                    observation, reward, done, info = step_result
                # Some environments might return 5 values including truncated flag (gym>=0.26)
                elif len(step_result) == 5:
                    observation, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    if verbose:
                        print(
                            f"Warning: Unexpected step result format: {len(step_result)} values"
                        )
                    break

                episode_return += reward
                steps += 1

                # Record video frame if needed
                if record_this_episode and video_recorders[episode]:
                    try:
                        video_recorders[episode].capture_frame()
                    except Exception as e:
                        print(f"Warning: Could not capture frame: {e}")

                # Check for success
                if isinstance(info, dict) and "success" in info and info["success"]:
                    success_rate += 1
                    break

            returns.append(episode_return)
            steps_to_complete.append(steps)

            # Close video recorder for this episode
            if record_this_episode and video_recorders[episode]:
                try:
                    video_recorders[episode].close()
                except Exception as e:
                    print(f"Warning: Error closing video recorder: {e}")

        # Make sure all video recorders are closed
        for recorder in video_recorders:
            if recorder:  # Only proceed if recorder is not None
                try:
                    # Only close if not already closed
                    if hasattr(recorder, "_closed") and not recorder._closed:
                        recorder.close()
                except Exception as e:
                    print(f"Warning: Error closing video recorder: {e}")

        # Calculate success rate for sequential evaluation
        success_rate = success_rate / n_episodes if n_episodes > 0 else 0

    # Process results
    if len(returns) > 0:
        success_rate_pct = 100.0 * success_rate
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_steps = np.mean(steps_to_complete)

        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")
            print(f"  Success rate: {success_rate_pct:.1f}%")
            print(f"  Average steps per episode: {mean_steps:.1f}")
            print(f"  Evaluated over {n_episodes} episodes")

        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "success_rate": success_rate,
            "mean_steps": mean_steps,
            "num_episodes": n_episodes,
            "returns": returns,
            "steps": steps_to_complete,
        }
    else:
        if verbose:
            print("No complete episodes during evaluation.")
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "success_rate": 0.0,
            "num_episodes": 0,
        }


def custom_evaluate_on_environment(env):
    """Custom environment evaluation function that handles observation conversion properly."""

    def scorer(algo, *args, **kwargs):
        # Create a fresh environment for evaluation if env is a creator function
        eval_env = env() if callable(env) else env

        # Set a unique seed to ensure diverse initial states
        if hasattr(eval_env, "seed"):
            unique_seed = int(time.time() * 1000) % 100000 + random.randint(0, 10000)
            eval_env.seed(unique_seed)

        # Check environment compatibility with the model
        env_obs_dim = eval_env.observation_space.shape[0]

        # Get model's expected observation dimension through various ways
        model_obs_dim = None
        if hasattr(algo, "observation_shape"):
            model_obs_dim = algo.observation_shape[0]
        elif hasattr(algo, "_impl") and hasattr(algo._impl, "observation_shape"):
            model_obs_dim = algo._impl.observation_shape[0]

        if model_obs_dim is not None and env_obs_dim != model_obs_dim:
            print(
                f"Warning: Environment observation dimension ({env_obs_dim}) doesn't match model's expected dimension ({model_obs_dim})"
            )
            print("Skipping evaluation with incompatible environment")
            return 0.0

        total_reward = 0.0
        n_episodes = 5  # Number of episodes to evaluate on for each call

        for episode in range(n_episodes):
            # Set a different seed for each episode to ensure diversity
            if hasattr(eval_env, "seed"):
                episode_seed = unique_seed + episode * 100
                eval_env.seed(episode_seed)

            observation = eval_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # Extract observation if it's a tuple
                if isinstance(observation, tuple):
                    obs_array = observation[0]
                else:
                    obs_array = observation

                # Ensure observation is a numpy array with batch dimension
                if not isinstance(obs_array, np.ndarray):
                    obs_array = np.array(obs_array, dtype=np.float32)

                if len(obs_array.shape) == 1:
                    obs_array = np.expand_dims(obs_array, axis=0)

                action = algo.predict(obs_array)[0]

                # Handle different step return formats
                step_result = eval_env.step(action)

                # MetaWorld environments return 4 values: obs, reward, done, info
                if len(step_result) == 4:
                    observation, reward, done, _ = step_result
                # Some environments might return 5 values including truncated flag (gym>=0.26)
                elif len(step_result) == 5:
                    observation, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    print(f"Warning: Unexpected step result format: {step_result}")
                    break

                episode_reward += reward

            total_reward += episode_reward

        # Return average reward across episodes
        avg_reward = total_reward / n_episodes
        print(
            f"Evaluation during training: {avg_reward:.2f} average reward over {n_episodes} episodes"
        )
        return avg_reward

    return scorer
