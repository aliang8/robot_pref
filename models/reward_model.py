import logging

logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)

import io
import os
import tempfile

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

import wandb


class RewardModel:
    def __init__(
        self,
        config,
        dataset,
        obs_act_1,
        obs_act_2,
        labels,
        dimension,
        train_images1=None,
        train_images2=None,
    ):
        self.env = config.env
        self.config = config
        self.dimension = dimension
        self.dataset = dataset
        self.device = config.device
        self.obs_act_1 = obs_act_1
        self.obs_act_2 = obs_act_2
        self.labels = labels
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.activation = config.activation
        # self.data_aug: For TDA data augmentation
        # (SURF: Semi-supervised Reward Learning with Data Augmentation for Feedback-efficient Preference-based Reinforcement Learning)
        self.data_aug = config.data_aug
        self.segment_size = config.segment_size
        self.lr = config.lr
        self.hidden_sizes = config.hidden_sizes
        self.dropout = config.dropout
        self.n_layers = config.n_layers
        self.loss = None
        self.model_type = config.model_type
        if self.model_type == "BT":
            self.loss = self.BT_loss
        elif self.model_type == "linear_BT":
            self.loss = self.linear_BT_loss
        self.ensemble_num = config.ensemble_num
        self.ensemble_method = config.ensemble_method
        self.paramlist = []
        self.optimizer = []
        self.lr_scheduler = []
        self.net = None
        self.ensemble_model = None
        self.feedback_type = config.feedback_type
        self.train_images1 = train_images1
        self.train_images2 = train_images2
        # Calculate class weights based on preference distribution
        self.use_class_weights = getattr(config, "use_class_weights", False)
        if self.use_class_weights:
            self.class_weights = self._calculate_class_weights()
            print(f"Class weights calculated: {self.class_weights}")
        else:
            self.class_weights = None
            print("Class weighting disabled")

    def _calculate_class_weights(self):
        """Calculate inverse frequency class weights for preference types."""
        # Count preference types
        seg1_better_count = 0
        seg2_better_count = 0
        equal_pref_count = 0

        for label in self.labels:
            if np.array_equal(label, [1, 0]):  # Segment 1 better
                seg1_better_count += 1
            elif np.array_equal(label, [0, 1]):  # Segment 2 better
                seg2_better_count += 1
            elif np.array_equal(label, [0.5, 0.5]):  # Equal preference
                equal_pref_count += 1

        total_samples = len(self.labels)
        num_classes = 3

        # Calculate class weights (inverse frequency)
        seg1_better_weight = (
            total_samples / (num_classes * seg1_better_count)
            if seg1_better_count > 0
            else 0.0
        )
        seg2_better_weight = (
            total_samples / (num_classes * seg2_better_count)
            if seg2_better_count > 0
            else 0.0
        )
        equal_pref_weight = (
            total_samples / (num_classes * equal_pref_count)
            if equal_pref_count > 0
            else 0.0
        )

        print("Training data preference distribution:")
        print(
            f"  Segment 1 better: {seg1_better_count} ({seg1_better_count / total_samples * 100:.1f}%) - weight: {seg1_better_weight:.3f}"
        )
        print(
            f"  Segment 2 better: {seg2_better_count} ({seg2_better_count / total_samples * 100:.1f}%) - weight: {seg2_better_weight:.3f}"
        )
        print(
            f"  Equal preference: {equal_pref_count} ({equal_pref_count / total_samples * 100:.1f}%) - weight: {equal_pref_weight:.3f}"
        )

        return {
            "seg1_better": seg1_better_weight,
            "seg2_better": seg2_better_weight,
            "equal_pref": equal_pref_weight,
        }

    def save_test_dataset(
        self,
        test_obs_act_1,
        test_obs_act_2,
        test_labels,
        test_binary_labels,
        test_images1=None,
        test_images2=None,
        test_episodes=None,  # held out episodes from dataset for validation
        test_dataset=None,
    ):
        self.test_obs_act_1 = torch.from_numpy(test_obs_act_1).float().to(self.device)
        self.test_obs_act_2 = torch.from_numpy(test_obs_act_2).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels).float().to(self.device)
        self.test_binary_labels = (
            torch.from_numpy(test_binary_labels).float().to(self.device)
        )
        if test_images1 is not None:
            self.test_images1 = test_images1
        if test_images2 is not None:
            self.test_images2 = test_images2
        if test_episodes is not None:
            self.test_episodes = test_episodes
        if test_dataset is not None:
            self.test_dataset = test_dataset

    def model_net(self, in_dim=39, out_dim=1, H=128, n_layers=2):
        """
        Create a neural network for reward modeling.

        Args:
            in_dim: Input dimension (obs + action)
            out_dim: Output dimension (typically 1 for reward)
            H: Hidden layer size
            n_layers: Number of hidden layers
        """
        net = []

        # Input layer
        net.append(nn.Linear(in_dim, H))
        net.append(nn.LayerNorm(H))  # Add layer norm for stability
        net.append(nn.LeakyReLU())
        net.append(nn.Dropout(self.dropout)) if self.dropout > 0 else None

        # Hidden layers
        for i in range(n_layers - 1):  # -1 because we already added the first layer
            net.append(nn.Linear(H, H))
            net.append(nn.LayerNorm(H))
            net.append(nn.Mish())
            net.append(nn.Dropout(self.dropout)) if self.dropout > 0 else None

        # Output layer (no activation in the middle, we'll add it at the end)
        net.append(nn.Linear(H, out_dim))

        # Final activation based on configuration
        if self.activation == "tanh":
            net.append(nn.Tanh())
        elif self.activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif self.activation == "relu":
            net.append(nn.ReLU())
        elif self.activation == "leaky_relu":
            net.append(nn.LeakyReLU())
        elif self.activation == "gelu":
            net.append(nn.GELU())
        elif self.activation == "none":
            pass  # No final activation
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return nn.Sequential(*net)

    def construct_ensemble(self):
        ensemble_model = []

        for i in range(self.ensemble_num):
            ensemble_model.append(
                self.model_net(
                    in_dim=self.dimension,
                    out_dim=1,
                    H=self.hidden_sizes,
                    n_layers=self.n_layers,
                ).to(self.device)
            )

        print("----------------------------------------")
        print(f"Model Architecture for {self.ensemble_num} ensemble members:")
        print(ensemble_model[0])
        print("----------------------------------------")

        return ensemble_model

    def single_model_forward(self, obs_act):
        return self.net(obs_act)

    def ensemble_model_forward(self, obs_act):
        pred = []
        for i in range(self.ensemble_num):
            pred.append(self.ensemble_model[i](obs_act))
        pred = torch.stack(pred, dim=1)
        if self.ensemble_method == "mean":
            return torch.mean(pred, dim=1)
        elif self.ensemble_method == "min":
            return torch.min(pred, dim=1).values
        elif self.ensemble_method == "uwo":
            return torch.mean(pred, dim=1) - 5 * torch.std(pred, dim=1)

    def BT_loss(self, pred_hat, label, apply_class_weights=False):
        # https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax
        logprobs = F.log_softmax(pred_hat, dim=1)
        losses = -(label * logprobs).sum(dim=1)  # Per-sample losses

        if (
            apply_class_weights
            and self.use_class_weights
            and self.class_weights is not None
        ):
            # Apply class weights based on preference type using vectorized operations
            # Create boolean masks for each preference type
            seg1_better_mask = (label[:, 0] == 1.0) & (label[:, 1] == 0.0)
            seg2_better_mask = (label[:, 0] == 0.0) & (label[:, 1] == 1.0)
            equal_pref_mask = (label[:, 0] == 0.5) & (label[:, 1] == 0.5)

            # Apply weights using vectorized operations
            weights = torch.ones_like(losses)
            weights[seg1_better_mask] = self.class_weights["seg1_better"]
            weights[seg2_better_mask] = self.class_weights["seg2_better"]
            weights[equal_pref_mask] = self.class_weights["equal_pref"]

            losses = losses * weights

        return losses.sum()

    def linear_BT_loss(self, pred_hat, label, apply_class_weights=False):
        pred_hat += self.segment_size + 1e-5
        pred_prob = pred_hat / torch.sum(pred_hat, dim=1, keepdim=True)
        # label and pred_hat cross entropy loss
        losses = -torch.sum(label * torch.log(pred_prob), dim=1)  # Per-sample losses

        if (
            apply_class_weights
            and self.use_class_weights
            and self.class_weights is not None
        ):
            # Apply class weights based on preference type using vectorized operations
            # Create boolean masks for each preference type
            seg1_better_mask = (label[:, 0] == 1.0) & (label[:, 1] == 0.0)
            seg2_better_mask = (label[:, 0] == 0.0) & (label[:, 1] == 1.0)
            equal_pref_mask = (label[:, 0] == 0.5) & (label[:, 1] == 0.5)

            # Apply weights using vectorized operations
            weights = torch.ones_like(losses)
            weights[seg1_better_mask] = self.class_weights["seg1_better"]
            weights[seg2_better_mask] = self.class_weights["seg2_better"]
            weights[equal_pref_mask] = self.class_weights["equal_pref"]

            losses = losses * weights

        return torch.sum(losses)

    def save_model(self, path):
        print(f"Saving {self.ensemble_num} ensemble members to {path}")
        for member in range(self.ensemble_num):
            # join path + member number
            member_path = os.path.join(path, "reward_" + str(member) + ".pt")
            torch.save(self.ensemble_model[member].state_dict(), member_path)

    def load_model(self, path):
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            member_path = os.path.join(path, "reward_" + str(member) + ".pt")
            self.ensemble_model[member].load_state_dict(torch.load(member_path))

    def get_reward(self, dataset):
        """Get reward predictions from the ensemble."""
        obs = dataset["observations"]
        act = dataset["actions"]
        goals = dataset["goal_points"]

        if self.dimension == 3:  # eef rm
            obs_act = obs[:, :3]
        elif self.dimension == 5:  # eef rm + 2D goal
            obs_act = np.concatenate((obs[:, :3], goals), axis=-1)
        else:
            obs_act = np.concatenate((obs, act), axis=-1)

        obs_act = torch.from_numpy(obs_act).float().to(self.device)

        with torch.no_grad():
            for i in range((obs_act.shape[0] - 1) // 10000 + 1):
                obs_act_batch = obs_act[i * 10000 : (i + 1) * 10000]
                pred_batch = self.ensemble_model_forward(obs_act_batch).reshape(-1)
                dataset["rewards"][i * 10000 : (i + 1) * 10000] = (
                    pred_batch.squeeze(-1).cpu().numpy()
                )
        return dataset["rewards"]

    def get_trajectory_rewards(self, dataset, trajectory_indices):
        """
        Get reward predictions for each trajectory as a list of lists (num_selected_trajectories x trajectory_length),
        but only for trajectories whose indices are in trajectory_indices.
        Returns rewards in the same order as trajectory_indices.
        """
        obs = dataset["observations"]
        goal = dataset["goal_points"]
        act = dataset["actions"]
        terminals = dataset["terminals"]

        if self.dimension == 3: # eef rm
            obs_act = obs[:, :3]  
        elif self.dimension == 5: # eef rm + 2D goal
            obs_act = np.concatenate((obs[:, :3], goal), axis=-1)
        else:
            obs_act = np.concatenate((obs, act), axis=-1)
        obs_act = torch.from_numpy(obs_act).float().to(self.device)

        with torch.no_grad():
            # Find the start and end indices for each trajectory
            trajectory_starts = [0]
            trajectory_ends = []
            for i in range(len(terminals)):
                if terminals[i]:
                    trajectory_ends.append(i + 1)
                    if i + 1 < len(terminals):
                        trajectory_starts.append(i + 1)

            # Create a dictionary to store rewards for each trajectory
            trajectory_rewards_dict = {}

            # For each trajectory whose index is in trajectory_indices, get reward predictions
            for idx, (start, end) in enumerate(zip(trajectory_starts, trajectory_ends)):
                if idx not in trajectory_indices:
                    continue

                traj_obs_act = obs_act[start:end]
                traj_rewards = []

                # Process in batches to handle memory constraints
                for i in range((traj_obs_act.shape[0] - 1) // 10000 + 1):
                    start_idx = i * 10000
                    end_idx = min((i + 1) * 10000, traj_obs_act.shape[0])
                    obs_act_batch = traj_obs_act[start_idx:end_idx]
                    pred_batch = self.ensemble_model_forward(obs_act_batch).reshape(-1)
                    traj_rewards.extend(pred_batch.cpu().numpy())

                trajectory_rewards_dict[idx] = traj_rewards

            # Return rewards in the same order as trajectory_indices
            rewards = []
            for traj_idx in trajectory_indices:
                if traj_idx in trajectory_rewards_dict:
                    rewards.append(trajectory_rewards_dict[traj_idx])
                else:
                    # This shouldn't happen if trajectory_indices is valid
                    print(f"Warning: Trajectory {traj_idx} not found in dataset")
                    rewards.append([])

        return rewards

    def _get_trajectory_boundaries(self, terminals):
        """Helper function to get trajectory start and end indices."""
        trajectory_starts = [0]
        trajectory_ends = []

        for i in range(len(terminals)):
            if terminals[i]:
                trajectory_ends.append(i + 1)
                if i + 1 < len(terminals):
                    trajectory_starts.append(i + 1)

        return trajectory_starts, trajectory_ends

    def val_trajectory_viz(self, dataset, name, epoch=0):
        """
        Visualize predicted vs ground truth rewards for each test trajectory in dataset.
        """
        # Get predicted rewards for all test trajectories
        trajectory_rewards = self.get_trajectory_rewards(dataset, self.test_episodes)

        # Dataset info
        terminals = dataset["terminals"]
        gt_rewards_all = dataset["rewards"]

        min_rew = gt_rewards_all.min()
        max_rew = gt_rewards_all.max()

        images_all = dataset["images"]
        trajectory_starts, trajectory_ends = self._get_trajectory_boundaries(terminals)

        videos = []

        for i, traj_idx in enumerate(self.test_episodes):
            pred_rewards = np.array(trajectory_rewards[i])

            # Trajectory start/end
            traj_start = trajectory_starts[traj_idx]
            traj_end = trajectory_ends[traj_idx]

            # Ground truth rewards and images for this trajectory
            gt_rewards = gt_rewards_all[traj_start:traj_end]
            traj_images = images_all[traj_start:traj_end]

            assert len(pred_rewards) == len(gt_rewards)

            # Mean rewards
            total_pred_reward = np.mean(pred_rewards)
            total_gt_reward = np.mean(gt_rewards)

            # Create figure: left = reward curves, right = trajectory image
            fig, (ax_plot, ax_img) = plt.subplots(1, 2, figsize=(12, 4))
            fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.1)

            ax_plot.set_title(
                f"Pred: {total_pred_reward:.3f}, GT: {total_gt_reward:.3f}", fontsize=10
            )
            ax_plot.set_xlabel("timestep", fontsize=9)
            ax_plot.set_ylabel("reward", fontsize=9)
            ax_plot.grid(True, alpha=0.3)
            ax_plot.set_xlim(0, len(gt_rewards) - 1)
            ax_plot.set_ylim(-1, 1)

            # Plot lines
            (pred_line,) = ax_plot.plot([], [], label="Predicted", color="blue")
            (gt_line,) = ax_plot.plot([], [], label="Ground Truth", color="red")
            ax_plot.legend(loc="upper right", fontsize=8)

            # Image display
            ax_img.set_title("Trajectory", fontsize=10)
            ax_img.axis("off")
            img_display = ax_img.imshow(np.zeros_like(traj_images[0]), animated=True)

            # Animation functions
            def init():
                pred_line.set_data([], [])
                gt_line.set_data([], [])
                img_display.set_data(np.zeros_like(traj_images[0]))
                return pred_line, gt_line, img_display

            def animate(frame):
                x = np.arange(frame + 1)
                pred_line.set_data(x, pred_rewards[: frame + 1])
                gt_line.set_data(x, gt_rewards[: frame + 1])

                img = traj_images[frame]
                if img.max() > 1.0:
                    img = img / 255.0
                img_display.set_data(img)

                return pred_line, gt_line, img_display

            ani = animation.FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=len(gt_rewards),
                interval=100,
                blit=True,
            )

            # Save video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                ani.save(tmp.name, writer="ffmpeg", fps=10)
                with open(tmp.name, "rb") as f:
                    video_buf = io.BytesIO(f.read())
            os.unlink(tmp.name)
            plt.close(fig)

            # Add to wandb videos
            video_buf.seek(0)
            videos.append(
                wandb.Video(
                    video_buf,
                    caption=f"Traj {traj_idx} | Pred: {total_pred_reward:.3f}, GT: {total_gt_reward:.3f}",
                    format="mp4",
                )
            )

        if videos:
            wandb.log({f"{name}/trajectory_videos": videos}, step=epoch)

    def eval(
        self,
        obs_act_1,
        obs_act_2,
        labels,
        binary_labels,
        name,
        epoch,
        images1=None,
        images2=None,
    ):
        """Evaluate the ensemble of distributional reward models."""
        eval_acc = 0
        eval_loss = 0
        total_samples = 0

        with torch.no_grad():
            vis_data = []

            for batch in range((obs_act_1.shape[0] - 1) // self.batch_size + 1):
                obs_act_1_batch = obs_act_1[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                obs_act_2_batch = obs_act_2[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                labels_batch = labels[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                binary_labels_batch = binary_labels[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]

                pred_1 = self.ensemble_model_forward(obs_act_1_batch)
                pred_2 = self.ensemble_model_forward(obs_act_2_batch)

                pred_hat = torch.stack(
                    [pred_1.sum(dim=1), pred_2.sum(dim=1)], dim=1
                ).squeeze()

                # Handle single sample case properly
                if pred_hat.dim() == 1:
                    pred_labels = pred_hat.argmax(dim=0)
                else:
                    pred_labels = pred_hat.argmax(dim=1).squeeze()

                eval_acc += (
                    (pred_labels == binary_labels_batch.argmax(dim=1)).sum().item()
                )

                # Normalize losses by batch size
                batch_size = labels_batch.shape[0]
                eval_loss += self.loss(pred_hat, labels_batch).item()
                total_samples += batch_size

                if images1 is not None and images2 is not None and len(vis_data) < 4:
                    batch_indices = list(range(len(pred_1)))
                    if len(batch_indices) > (4 - len(vis_data)):
                        batch_indices = np.random.choice(
                            batch_indices, size=4 - len(vis_data), replace=False
                        )

                    for i in batch_indices:
                        vis_data.append(
                            {
                                "reward1": pred_1[i].cpu().numpy(),
                                "reward2": pred_2[i].cpu().numpy(),
                                "gt_pref": torch.argmax(binary_labels_batch[i]).item(),
                                "pred_pref": pred_labels[i].item(),
                                "images1": images1[batch * self.batch_size + i],
                                "images2": images2[batch * self.batch_size + i],
                            }
                        )

        # Normalize by total number of samples
        eval_loss /= total_samples
        eval_acc /= float(total_samples)

        wandb.log(
            {name + "/loss": eval_loss, name + "/acc": eval_acc}, step=epoch
        ) if wandb.run is not None else None

        if len(vis_data) > 0 and wandb.run is not None:
            self._create_visualization(vis_data, name, epoch)

    def _create_visualization(self, vis_data, name, epoch):
        """Create visualization of reward predictions."""
        videos = []
        for idx, ex in enumerate(vis_data):
            r1 = ex["reward1"]  # mean rewards
            r2 = ex["reward2"]  # mean rewards
            gt_pref = ex["gt_pref"]
            pred_pref = ex["pred_pref"]

            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1.2, 1, 1]}
            )
            fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.1)

            ax1.set_title(
                f"Test Pair {idx + 1}: GT Pref={gt_pref}, Pred={pred_pref}", fontsize=10
            )
            ax1.set_xlabel("Timestep", fontsize=9)
            ax1.set_ylabel("Reward", fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Plot mean rewards
            (line1,) = ax1.plot([], [], label="Traj 1", color="blue")
            (line2,) = ax1.plot([], [], label="Traj 2", color="red")

            ax1.legend(loc="upper right", fontsize=8)
            ax1.tick_params(axis="both", which="major", labelsize=8)

            ax2.set_title("Trajectory 1", fontsize=10)
            ax2.axis("off")
            img1 = ax2.imshow(np.zeros((64, 64, 3)), animated=True)

            ax3.set_title("Trajectory 2", fontsize=10)
            ax3.axis("off")
            img2 = ax3.imshow(np.zeros((64, 64, 3)), animated=True)

            max_len = max(len(r1), len(r2))
            ax1.set_xlim(0, max_len - 1)
            ax1.set_ylim(-1, 1)

            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                img1.set_data(np.zeros((64, 64, 3)))
                img2.set_data(np.zeros((64, 64, 3)))
                return line1, line2, img1, img2

            def animate(i):
                # Update mean reward lines
                x = np.arange(i + 1)
                line1.set_data(x, r1[: i + 1])
                line2.set_data(x, r2[: i + 1])

                # Update images
                img1_data = ex["images1"][i]
                img2_data = ex["images2"][i]

                if img1_data.max() > 1.0:
                    img1_data = img1_data / 255.0
                    img2_data = img2_data / 255.0

                img1.set_data(img1_data)
                img2.set_data(img2_data)

                return line1, line2, img1, img2

            ani = animation.FuncAnimation(
                fig, animate, init_func=init, frames=max_len, interval=100, blit=True
            )

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                ani.save(tmp.name, writer="ffmpeg", fps=10)
                with open(tmp.name, "rb") as f:
                    buf = io.BytesIO(f.read())
            os.unlink(tmp.name)

            plt.close(fig)
            buf.seek(0)
            videos.append(
                wandb.Video(
                    buf,
                    caption=f"Test Pair {idx + 1}: GT Pref={gt_pref}, Pred={pred_pref}",
                    format="mp4",
                )
            )

        wandb.log(
            {f"{name}/reward_videos": videos}, step=epoch
        ) if wandb.run is not None else None

    def train_model(self):
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            self.ensemble_model[member].train()
            self.optimizer.append(
                optim.Adam(
                    self.ensemble_model[member].parameters(),
                    lr=self.lr,
                    weight_decay=1e-3,
                )
            )
            self.lr_scheduler.append(
                optim.lr_scheduler.StepLR(
                    self.optimizer[member],
                    step_size=5 if self.epochs <= 500 else 1000,
                    gamma=0.7,
                )
            )

        self.obs_act_1 = torch.from_numpy(self.obs_act_1).float().to(self.device)
        self.obs_act_2 = torch.from_numpy(self.obs_act_2).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).float().to(self.device)
        for epoch in tqdm.tqdm(range(1, self.epochs + 1)):
            train_loss = 0
            for member in range(self.ensemble_num):
                self.optimizer[member].zero_grad()
                self.net = self.ensemble_model[member]
                # shuffle data
                idx = np.random.permutation(self.obs_act_1.shape[0])
                obs_act_1 = self.obs_act_1[idx]
                obs_act_2 = self.obs_act_2[idx]
                labels = self.labels[idx]

                for batch in range((obs_act_1.shape[0] - 1) // self.batch_size + 1):
                    loss = 0
                    obs_act_1_batch = obs_act_1[
                        batch * self.batch_size : (batch + 1) * self.batch_size
                    ]
                    obs_act_2_batch = obs_act_2[
                        batch * self.batch_size : (batch + 1) * self.batch_size
                    ]
                    labels_batch = labels[
                        batch * self.batch_size : (batch + 1) * self.batch_size
                    ]
                    if self.data_aug == "temporal":
                        # cut random segment from self.segment_size (20 ~ 25)
                        short_segment_size = np.random.randint(
                            self.segment_size - 5, self.segment_size + 1
                        )
                        start_idx_1 = np.random.randint(
                            0, self.segment_size - short_segment_size + 1
                        )
                        start_idx_2 = np.random.randint(
                            0, self.segment_size - short_segment_size + 1
                        )
                        obs_act_1_batch = obs_act_1_batch[
                            :, start_idx_1 : start_idx_1 + short_segment_size, :
                        ]
                        obs_act_2_batch = obs_act_2_batch[
                            :, start_idx_2 : start_idx_2 + short_segment_size, :
                        ]
                    pred_1 = self.single_model_forward(obs_act_1_batch)
                    pred_2 = self.single_model_forward(obs_act_2_batch)

                    pred_seg_sum_1 = torch.sum(pred_1, dim=1)
                    pred_seg_sum_2 = torch.sum(pred_2, dim=1)
                    pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)
                    loss = self.loss(pred_hat, labels_batch) / labels_batch.shape[0]
                    train_loss += loss.item() * labels_batch.shape[0]
                    loss.backward()
                    self.optimizer[member].step()
                self.lr_scheduler[member].step()

            train_loss /= obs_act_1.shape[0] * self.ensemble_num

            if epoch % 20 == 0 and wandb.run is not None:
                wandb.log({"train/loss": train_loss}, step=epoch)

            # Create training visualization every 2000 epochs
            if epoch % 2000 == 0 and wandb.run is not None:
                train_vis_data = []

                # Sample a few training examples for visualization
                with torch.no_grad():
                    # Use a subset of training data for visualization
                    vis_indices = np.random.choice(
                        len(self.obs_act_1), min(4, len(self.obs_act_1)), replace=False
                    )

                    for idx in vis_indices:
                        obs_act_1_sample = self.obs_act_1[idx : idx + 1]
                        obs_act_2_sample = self.obs_act_2[idx : idx + 1]
                        labels_sample = self.labels[idx : idx + 1]

                        pred_1 = self.ensemble_model_forward(obs_act_1_sample)
                        pred_2 = self.ensemble_model_forward(obs_act_2_sample)

                        pred_hat = torch.stack(
                            [pred_1.sum(dim=1), pred_2.sum(dim=1)], dim=1
                        ).squeeze()

                        # Handle single sample case properly
                        if pred_hat.dim() == 1:
                            pred_labels = pred_hat.argmax(dim=0)
                        else:
                            pred_labels = pred_hat.argmax(dim=1).squeeze()

                        vis_data_point = {
                            "reward1": pred_1[0].cpu().numpy(),
                            "reward2": pred_2[0].cpu().numpy(),
                            "gt_pref": torch.argmax(labels_sample[0]).item(),
                            "pred_pref": pred_labels.item()
                            if pred_labels.numel() == 1
                            else pred_labels[0].item(),
                        }

                        # Add images if available
                        if (
                            self.train_images1 is not None
                            and self.train_images2 is not None
                        ):
                            vis_data_point["images1"] = self.train_images1[idx]
                            vis_data_point["images2"] = self.train_images2[idx]

                        train_vis_data.append(vis_data_point)

                if len(train_vis_data) > 0:
                    self._create_visualization(train_vis_data, "train", epoch)

            # Full trajectory reward visualization
            if epoch % 2000 == 0 and wandb.run is not None:
                self.val_trajectory_viz(
                    self.test_dataset
                    if hasattr(self, "test_dataset")
                    else self.dataset,
                    name="eval",
                    epoch=epoch,
                )

            if epoch % 100 == 0:
                self.eval(
                    self.obs_act_1,
                    self.obs_act_2,
                    self.labels,
                    self.labels,
                    "train",
                    epoch,
                )
                self.eval(
                    self.test_obs_act_1,
                    self.test_obs_act_2,
                    self.test_labels,
                    self.test_binary_labels,
                    "eval",
                    epoch,
                    images1=self.test_images1
                    if hasattr(self, "test_images1") and epoch % 2000 == 0
                    else None,
                    images2=self.test_images2
                    if hasattr(self, "test_images2") and epoch % 2000 == 0
                    else None,
                )
