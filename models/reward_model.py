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
    def __init__(self, config, obs_act_1, obs_act_2, labels, dimension):
        self.env = config.env
        self.config = config
        self.dimension = dimension
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

        # Calculate class weights based on preference distribution
        self.use_class_weights = getattr(config, 'use_class_weights', False)
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
        seg1_better_weight = total_samples / (num_classes * seg1_better_count) if seg1_better_count > 0 else 0.0
        seg2_better_weight = total_samples / (num_classes * seg2_better_count) if seg2_better_count > 0 else 0.0
        equal_pref_weight = total_samples / (num_classes * equal_pref_count) if equal_pref_count > 0 else 0.0
        
        print("Training data preference distribution:")
        print(f"  Segment 1 better: {seg1_better_count} ({seg1_better_count/total_samples*100:.1f}%) - weight: {seg1_better_weight:.3f}")
        print(f"  Segment 2 better: {seg2_better_count} ({seg2_better_count/total_samples*100:.1f}%) - weight: {seg2_better_weight:.3f}")
        print(f"  Equal preference: {equal_pref_count} ({equal_pref_count/total_samples*100:.1f}%) - weight: {equal_pref_weight:.3f}")
        
        return {
            'seg1_better': seg1_better_weight,
            'seg2_better': seg2_better_weight,
            'equal_pref': equal_pref_weight
        }

    def save_test_dataset(
        self,
        test_obs_act_1,
        test_obs_act_2,
        test_labels,
        test_binary_labels,
        test_images1=None,
        test_images2=None,
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

    def model_net(self, in_dim=39, out_dim=1, H=128, n_layers=1):
        net = []
        for i in range(n_layers):
            net.append(nn.Linear(in_dim, H))
            net.append(nn.LeakyReLU())
            net.append(nn.Dropout(0.2))
            in_dim = H
        net.append(nn.Linear(H, out_dim))
        if self.activation == "tanh":
            net.append(nn.Tanh())
        elif self.activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif self.activation == "relu":
            net.append(nn.ReLU())
        elif self.activation == "leaky_relu":
            net.append(nn.LeakyReLU())
        elif self.activation == "none":
            pass
        elif self.activation == "gelu":
            net.append(nn.GELU())

        return nn.Sequential(*net)

    def construct_ensemble(self):
        ensemble_model = []
        for i in range(self.ensemble_num):
            ensemble_model.append(
                self.model_net(
                    in_dim=self.dimension, out_dim=1, H=self.hidden_sizes
                ).to(self.device)
            )
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
        
        if apply_class_weights and self.use_class_weights and self.class_weights is not None:
            # Apply class weights based on preference type using vectorized operations
            # Create boolean masks for each preference type
            seg1_better_mask = (label[:, 0] == 1.0) & (label[:, 1] == 0.0)
            seg2_better_mask = (label[:, 0] == 0.0) & (label[:, 1] == 1.0)
            equal_pref_mask = (label[:, 0] == 0.5) & (label[:, 1] == 0.5)
            
            # Apply weights using vectorized operations
            weights = torch.ones_like(losses)
            weights[seg1_better_mask] = self.class_weights['seg1_better']
            weights[seg2_better_mask] = self.class_weights['seg2_better']
            weights[equal_pref_mask] = self.class_weights['equal_pref']
            
            losses = losses * weights
        
        return losses.sum()

    def linear_BT_loss(self, pred_hat, label, apply_class_weights=False):
        pred_hat += self.segment_size + 1e-5
        pred_prob = pred_hat / torch.sum(pred_hat, dim=1, keepdim=True)
        # label and pred_hat cross entropy loss
        losses = -torch.sum(label * torch.log(pred_prob), dim=1)  # Per-sample losses
        
        if apply_class_weights and self.use_class_weights and self.class_weights is not None:
            # Apply class weights based on preference type using vectorized operations
            # Create boolean masks for each preference type
            seg1_better_mask = (label[:, 0] == 1.0) & (label[:, 1] == 0.0)
            seg2_better_mask = (label[:, 0] == 0.0) & (label[:, 1] == 1.0)
            equal_pref_mask = (label[:, 0] == 0.5) & (label[:, 1] == 0.5)
            
            # Apply weights using vectorized operations
            weights = torch.ones_like(losses)
            weights[seg1_better_mask] = self.class_weights['seg1_better']
            weights[seg2_better_mask] = self.class_weights['seg2_better']
            weights[equal_pref_mask] = self.class_weights['equal_pref']
            
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

        if self.dimension == 10:
            obs = obs[:, :3]  # eef pos

        obs_act = np.concatenate((obs, act), axis=-1)
        obs_act = torch.from_numpy(obs_act).float().to(self.device)
        
        with torch.no_grad():
            for i in range((obs_act.shape[0] - 1) // 10000 + 1):
                obs_act_batch = obs_act[i * 10000 : (i + 1) * 10000]
                pred_batch = self.ensemble_model_forward(obs_act_batch).reshape(-1)
                dataset["rewards"][
                    i * 10000 : (i + 1) * 10000
                ] = pred_batch.squeeze(-1).cpu().numpy()
        return dataset["rewards"]

    def eval(self, obs_act_1, obs_act_2, labels, binary_labels, name, epoch, images1=None, images2=None):
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

                pred_hat = torch.stack([
                    pred_1.sum(dim=1),
                    pred_2.sum(dim=1)
                ], dim=1).squeeze()

                pred_labels = pred_hat.argmax(dim=1).squeeze()
                eval_acc += (pred_labels == binary_labels_batch.argmax(dim=1)).sum().item()
                                
                # Normalize losses by batch size
                batch_size = labels_batch.shape[0]
                eval_loss += self.loss(pred_hat, labels_batch).item()
                total_samples += batch_size
                
                if images1 is not None and images2 is not None and len(vis_data) < 4:
                    batch_indices = list(range(len(pred_1)))
                    if len(batch_indices) > (4 - len(vis_data)):
                        batch_indices = np.random.choice(batch_indices, size=4 - len(vis_data), replace=False)
                    
                    for i in batch_indices:
                        vis_data.append({
                            'reward1': pred_1[i].cpu().numpy(),
                            'reward2': pred_2[i].cpu().numpy(),
                            'gt_pref': torch.argmax(binary_labels_batch[i]).item(),
                            'pred_pref': pred_labels[i].item(),
                            'images1': images1[batch * self.batch_size + i],
                            'images2': images2[batch * self.batch_size + i],
                        })
        
        # Normalize by total number of samples
        eval_loss /= total_samples
        eval_acc /= float(total_samples)
        
        wandb.log({
            name + "/loss": eval_loss,
            name + "/acc": eval_acc
        }, step=epoch) if wandb.run is not None else None
        
        if len(vis_data) > 0 and wandb.run is not None:
            self._create_visualization(vis_data, name, epoch)

    def _create_visualization(self, vis_data, name, epoch):
        """Create visualization of reward predictions."""
        videos = []
        for idx, ex in enumerate(vis_data):
            r1 = ex['reward1']  # mean rewards
            r2 = ex['reward2']  # mean rewards
            gt_pref = ex['gt_pref']
            pred_pref = ex['pred_pref']
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [1.2, 1, 1]})
            fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.1)
            
            ax1.set_title(f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}", fontsize=10)
            ax1.set_xlabel("Timestep", fontsize=9)
            ax1.set_ylabel("Reward", fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Plot mean rewards
            line1, = ax1.plot([], [], label="Traj 1", color="blue")
            line2, = ax1.plot([], [], label="Traj 2", color="red")
            
            ax1.legend(loc="upper right", fontsize=8)
            ax1.tick_params(axis='both', which='major', labelsize=8)
            
            ax2.set_title("Trajectory 1", fontsize=10)
            ax2.axis('off')
            img1 = ax2.imshow(np.zeros((64, 64, 3)), animated=True)
            
            ax3.set_title("Trajectory 2", fontsize=10)
            ax3.axis('off')
            img2 = ax3.imshow(np.zeros((64, 64, 3)), animated=True)
            
            max_len = max(len(r1), len(r2))
            ax1.set_xlim(0, max_len-1)
            min_y = min(np.min(r1), np.min(r2))
            max_y = max(np.max(r1), np.max(r2))
            ax1.set_ylim(min_y - 0.1 * abs(min_y), max_y + 0.1 * abs(max_y))
            
            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                img1.set_data(np.zeros((64, 64, 3)))
                img2.set_data(np.zeros((64, 64, 3)))
                return line1, line2, img1, img2
            
            def animate(i):
                # Update mean reward lines
                x = np.arange(i+1)
                line1.set_data(x, r1[:i+1])
                line2.set_data(x, r2[:i+1])
                
                # Update images
                img1_data = ex['images1'][i]
                img2_data = ex['images2'][i]
                
                if img1_data.max() > 1.0:
                    img1_data = img1_data / 255.0
                    img2_data = img2_data / 255.0
                
                img1.set_data(img1_data)
                img2.set_data(img2_data)
                
                return line1, line2, img1, img2
            
            ani = animation.FuncAnimation(
                fig, animate, init_func=init, frames=max_len,
                interval=100, blit=True
            )
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                ani.save(tmp.name, writer="ffmpeg", fps=10)
                with open(tmp.name, 'rb') as f:
                    buf = io.BytesIO(f.read())
            os.unlink(tmp.name)
            
            plt.close(fig)
            buf.seek(0)
            videos.append(wandb.Video(buf, caption=f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}", format="mp4"))
        
        wandb.log({f"{name}/reward_videos": videos}, step=epoch) if wandb.run is not None else None

    def train_model(self):
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            self.ensemble_model[member].train()
            self.optimizer.append(
                optim.Adam(self.ensemble_model[member].parameters(), lr=self.lr, weight_decay=1e-4)
            )
            self.lr_scheduler.append(
                optim.lr_scheduler.StepLR(
                    self.optimizer[member],
                    step_size=10 if self.epochs <= 500 else 1000,
                    gamma=0.9,
                )
            )

        self.obs_act_1 = torch.from_numpy(self.obs_act_1).float().to(self.device)
        self.obs_act_2 = torch.from_numpy(self.obs_act_2).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).float().to(self.device)
        for epoch in tqdm.tqdm(range(self.epochs)):
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
                    images1=self.test_images1 if hasattr(self, 'test_images1') and epoch % 1000 == 0 else None,
                    images2=self.test_images2 if hasattr(self, 'test_images2') and epoch % 1000 == 0 else None,
                )


class DistributionalRewardModel:
    """Distributional reward model with separate mean and variance branches."""

    def __init__(self, config, obs_act_1, obs_act_2, labels, dimension):
        self.env = config.env
        self.config = config
        self.dimension = dimension
        self.device = config.device
        self.obs_act_1 = obs_act_1
        self.obs_act_2 = obs_act_2
        self.labels = labels
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.activation = config.activation
        self.data_aug = config.data_aug
        self.segment_size = config.segment_size
        self.lr = config.lr
        self.hidden_sizes = config.hidden_sizes
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

        # Calculate class weights based on preference distribution
        self.use_class_weights = getattr(config, 'use_class_weights', True)
        if self.use_class_weights:
            self.class_weights = self._calculate_class_weights()
            print(f"Class weights calculated: {self.class_weights}")
        else:
            self.class_weights = None
            print("Class weighting disabled")

    def _calculate_class_weights(self):
        # TODO: this is unused for now
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
        seg1_better_weight = total_samples / (num_classes * seg1_better_count) if seg1_better_count > 0 else 0.0
        seg2_better_weight = total_samples / (num_classes * seg2_better_count) if seg2_better_count > 0 else 0.0
        equal_pref_weight = total_samples / (num_classes * equal_pref_count) if equal_pref_count > 0 else 0.0
        
        print("Training data preference distribution:")
        print(f"  Segment 1 better: {seg1_better_count} ({seg1_better_count/total_samples*100:.1f}%) - weight: {seg1_better_weight:.3f}")
        print(f"  Segment 2 better: {seg2_better_count} ({seg2_better_count/total_samples*100:.1f}%) - weight: {seg2_better_weight:.3f}")
        print(f"  Equal preference: {equal_pref_count} ({equal_pref_count/total_samples*100:.1f}%) - weight: {equal_pref_weight:.3f}")
        
        return {
            'seg1_better': seg1_better_weight,
            'seg2_better': seg2_better_weight,
            'equal_pref': equal_pref_weight
        }
    
    def save_test_dataset(
        self,
        test_obs_act_1,
        test_obs_act_2,
        test_labels,
        test_binary_labels,
        test_images1=None,
        test_images2=None,
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

    def model_net(self, in_dim=39, out_dim=1, H=128, n_layers=2):
        """Create a distributional reward model network with mean and variance branches."""
        # Shared feature extractor
        shared_layers = []
        prev_dim = in_dim
        
        for i in range(n_layers):
            shared_layers.append(nn.Linear(prev_dim, H))
            shared_layers.append(nn.LayerNorm(H))
            shared_layers.append(nn.LeakyReLU(0.1))
            prev_dim = H
            
        shared_features = nn.Sequential(*shared_layers)

        split_dim = H // 2
        
        mean_branch = nn.Sequential(
            nn.Linear(split_dim, split_dim),
            nn.LayerNorm(split_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(split_dim, out_dim),
            nn.Tanh() # [-1, 1]
        )
        
        # Variance branch (output log variance for numerical stability)
        variance_branch = nn.Sequential(
            nn.Linear(split_dim, split_dim),
            nn.LayerNorm(split_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(split_dim, out_dim)
        )

        return shared_features, mean_branch, variance_branch

    def construct_ensemble(self):
        """Construct an ensemble of distributional reward models."""
        ensemble_model = []
        for i in range(self.ensemble_num):
            shared_features, mean_branch, variance_branch = self.model_net(
                in_dim=self.dimension, out_dim=1, H=self.hidden_sizes
            )
            model = {
                'shared_features': shared_features.to(self.device),
                'mean_branch': mean_branch.to(self.device),
                'variance_branch': variance_branch.to(self.device)
            }
            ensemble_model.append(model)
        return ensemble_model

    def single_model_forward(self, obs_act):
        """Forward pass through a single distributional reward model."""
        shared_features = self.net['shared_features'](obs_act)
        
        # Split features along the embedding dimension for mean and variance branches
        split_dim = shared_features.shape[-1] // 2
        mean_features = shared_features[..., :split_dim]
        var_features = shared_features[..., split_dim:]
        
        # Compute mean and variance
        reward_mean = self.net['mean_branch'](mean_features)
        log_variance = self.net['variance_branch'](var_features)
        
        # Convert log variance to variance (ensure positive)
        reward_variance = torch.exp(log_variance) + 1e-6
        
        return reward_mean, reward_variance

    def ensemble_model_forward(self, obs_act):
        """Forward pass through the ensemble of distributional reward models."""
        means = []
        variances = []
        
        for model in self.ensemble_model:
            shared_features = model['shared_features'](obs_act)
            split_dim = shared_features.shape[-1] // 2
            mean_features = shared_features[..., :split_dim]
            var_features = shared_features[..., split_dim:]
            
            mean = model['mean_branch'](mean_features)
            log_var = model['variance_branch'](var_features)
            var = torch.exp(log_var) + 1e-6
            
            means.append(mean)
            variances.append(var)
        
        means = torch.stack(means, dim=1)
        variances = torch.stack(variances, dim=1)
        
        if self.ensemble_method == "mean":
            return torch.mean(means, dim=1), torch.mean(variances, dim=1)
        elif self.ensemble_method == "min":
            return torch.min(means, dim=1).values, torch.mean(variances, dim=1)
        elif self.ensemble_method == "uwo":
            return torch.mean(means, dim=1) - 5 * torch.std(means, dim=1), torch.mean(variances, dim=1)

    def BT_loss(self, mean1, mean2, var1, var2, label, apply_class_weights=True):
        """Bradley-Terry loss for distributional reward model."""

        # Split the concatenated predictions back into segments
        # mean and variance are [batch_size, 2] tensors
        # mean1 = mean[:, 0]  # First column is first segment
        # mean2 = mean[:, 1]  # Second column is second segment
        # var1 = variance[:, 0]
        # var2 = variance[:, 1]
        
        # Convert labels to preference format (1 = first segment preferred, 0 = second segment preferred)
        prefs = (label[:, 0] == 1.0).float()

        # Apply class weights if enabled
        # if apply_class_weights and self.use_class_weights and self.class_weights is not None:
        #     weights = torch.ones_like(prefs)
        #     weights[label[:, 0] == 1.0] = self.class_weights['seg1_better']
        #     weights[label[:, 1] == 1.0] = self.class_weights['seg2_better']
        #     weights[(label[:, 0] == 0.5) & (label[:, 1] == 0.5)] = self.class_weights['equal_pref']
        # else:
        #     weights = None

        # Compute distributional Bradley-Terry loss
        from utils.loss import distributional_reward_loss
        total_loss, bt_loss_mean, bt_loss_samples, reg_loss = distributional_reward_loss(
            mean1, mean2,
            var1, var2,
            prefs,
        )

        # Apply class weights if enabled
        # if weights is not None:
        #     loss = loss * weights.mean()

        return total_loss, bt_loss_mean, bt_loss_samples, reg_loss

    def linear_BT_loss(self, mean1, mean2, var1, var2, label, apply_class_weights=True):
        """Linear Bradley-Terry loss for distributional reward model."""

        # Add segment size TODO
        mean1 = mean1
        mean2 = mean2
        
        # Convert labels to preference format
        prefs = (label[:, 0] == 1.0).float()
        
        # Apply class weights if enabled
        # if apply_class_weights and self.use_class_weights and self.class_weights is not None:
        #     weights = torch.ones_like(prefs)
        #     weights[label[:, 0] == 1.0] = self.class_weights['seg1_better']
        #     weights[label[:, 1] == 1.0] = self.class_weights['seg2_better']
        #     weights[(label[:, 0] == 0.5) & (label[:, 1] == 0.5)] = self.class_weights['equal_pref']
        # else:
        #     weights = None
        
        # Compute distributional Bradley-Terry loss
        from utils.loss import distributional_reward_loss
        total_loss, bt_loss_mean, bt_loss_samples, reg_loss = distributional_reward_loss(
            mean1, mean2,
            var1, var2,
            prefs,
        )

        # Apply class weights if enabled
        # if weights is not None:
        #     loss = loss * weights.mean()
        
        return total_loss, bt_loss_mean, bt_loss_samples, reg_loss

    def save_model(self, path):
        """Save the ensemble of distributional reward models."""
        for member in range(self.ensemble_num):
            member_path = os.path.join(path, f"reward_{member}")
            os.makedirs(member_path, exist_ok=True)
            
            torch.save(self.ensemble_model[member]['shared_features'].state_dict(), 
                      os.path.join(member_path, "shared_features.pt"))
            torch.save(self.ensemble_model[member]['mean_branch'].state_dict(), 
                      os.path.join(member_path, "mean_branch.pt"))
            torch.save(self.ensemble_model[member]['variance_branch'].state_dict(), 
                      os.path.join(member_path, "variance_branch.pt"))
            
            print(f"Model for member {member} saved at {member_path}")

    def load_model(self, path):
        """Load the ensemble of distributional reward models."""
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            member_path = os.path.join(path, f"reward_{member}")
            
            self.ensemble_model[member]['shared_features'].load_state_dict(
                torch.load(os.path.join(member_path, "shared_features.pt")))
            self.ensemble_model[member]['mean_branch'].load_state_dict(
                torch.load(os.path.join(member_path, "mean_branch.pt")))
            self.ensemble_model[member]['variance_branch'].load_state_dict(
                torch.load(os.path.join(member_path, "variance_branch.pt")))

    def get_reward(self, dataset):
        """Get reward predictions from the ensemble."""
        obs = dataset["observations"]
        act = dataset["actions"]

        if self.dimension == 10:
            obs = obs[:, :3]  # eef pos

        obs_act = np.concatenate((obs, act), axis=-1)
        obs_act = torch.from_numpy(obs_act).float().to(self.device)
        
        with torch.no_grad():
            for i in range((obs_act.shape[0] - 1) // 10000 + 1):
                obs_act_batch = obs_act[i * 10000 : (i + 1) * 10000]
                mean, _ = self.ensemble_model_forward(obs_act_batch)
                dataset["rewards"][i * 10000 : (i + 1) * 10000] = mean.squeeze(-1).cpu().numpy()
        
        return dataset["rewards"]

    def train_model(self):
        """Train the ensemble of distributional reward models."""
        self.ensemble_model = self.construct_ensemble()
        
        for member in range(self.ensemble_num):
            # Set up optimizer for all components
            params = list(self.ensemble_model[member]['shared_features'].parameters()) + \
                    list(self.ensemble_model[member]['mean_branch'].parameters()) + \
                    list(self.ensemble_model[member]['variance_branch'].parameters())
            
            self.optimizer.append(optim.Adam(params, lr=self.lr, weight_decay=1e-4))  # Added weight decay
            self.lr_scheduler.append(
                optim.lr_scheduler.StepLR(
                    self.optimizer[member],
                    step_size=10 if self.epochs <= 500 else 1000,
                    gamma=0.9,
                )
            )

        self.obs_act_1 = torch.from_numpy(self.obs_act_1).float().to(self.device)
        self.obs_act_2 = torch.from_numpy(self.obs_act_2).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).float().to(self.device)
        
        for epoch in tqdm.tqdm(range(self.epochs)):
            train_loss = 0
            train_bt_mean_loss = 0
            train_bt_samples_loss = 0
            train_reg_loss = 0
            total_samples = 0
            
            for member in range(self.ensemble_num):
                self.optimizer[member].zero_grad()
                self.net = self.ensemble_model[member]
                
                # Shuffle data
                idx = np.random.permutation(self.obs_act_1.shape[0])
                obs_act_1 = self.obs_act_1[idx]
                obs_act_2 = self.obs_act_2[idx]
                labels = self.labels[idx]

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
                    
                    if self.data_aug == "temporal":
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
                    
                    # Compute predictions for both segments in a single call to avoid redundancy
                    pred_means_1, pred_vars_1 = self.single_model_forward(obs_act_1_batch)
                    pred_means_2, pred_vars_2 = self.single_model_forward(obs_act_2_batch)

                    # Compute all losses in one call
                    total_loss, bt_loss_mean, bt_loss_samples, reg_loss = self.loss(
                        pred_means_1, pred_means_2, pred_vars_1, pred_vars_2, labels_batch
                    )

                    batch_size = labels_batch.shape[0]
                    # Accumulate losses (weighted by batch size for averaging later)
                    train_loss += total_loss.item() * batch_size
                    train_bt_mean_loss += bt_loss_mean.item() * batch_size
                    train_bt_samples_loss += bt_loss_samples.item() * batch_size
                    train_reg_loss += reg_loss.item() * batch_size
                    total_samples += batch_size

                    # Normalize loss for backward pass
                    total_loss.backward()
                    
                    # # Add gradient clipping
                    # torch.nn.utils.clip_grad_norm_(self.net['shared_features'].parameters(), max_norm=1.0)
                    # torch.nn.utils.clip_grad_norm_(self.net['mean_branch'].parameters(), max_norm=1.0)
                    # torch.nn.utils.clip_grad_norm_(self.net['variance_branch'].parameters(), max_norm=1.0)
                    
                    self.optimizer[member].step()
                
                self.lr_scheduler[member].step()

            # Normalize by total samples
            train_loss /= total_samples
            train_bt_mean_loss /= total_samples
            train_bt_samples_loss /= total_samples
            train_reg_loss /= total_samples

            if epoch % 20 == 0 and wandb.run is not None:
                wandb.log({
                    "train/loss": train_loss, 
                    "train/bt_mean_loss": train_bt_mean_loss, 
                    "train/bt_samples_loss": train_bt_samples_loss,
                    "train/reg_loss": train_reg_loss
                }, step=epoch)

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
                    images1=self.test_images1 if hasattr(self, 'test_images1') and epoch % 1000 == 0 else None,
                    images2=self.test_images2 if hasattr(self, 'test_images2') and epoch % 1000 == 0 else None,
                )

    def eval(self, obs_act_1, obs_act_2, labels, binary_labels, name, epoch, images1=None, images2=None):
        """Evaluate the ensemble of distributional reward models."""
        eval_acc = 0
        eval_loss = 0
        eval_bt_mean_loss = 0
        eval_bt_samples_loss = 0
        eval_reg_loss = 0
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
                
                pred_1_mean, pred_1_var = self.ensemble_model_forward(obs_act_1_batch)
                pred_2_mean, pred_2_var = self.ensemble_model_forward(obs_act_2_batch)

                pred_hat = torch.stack([
                    pred_1_mean.sum(dim=1),
                    pred_2_mean.sum(dim=1)
                ], dim=1)

                pred_labels = pred_hat.argmax(dim=1).squeeze()
                eval_acc += (pred_labels == binary_labels_batch.argmax(dim=1)).sum().item()
                
                total_loss, bt_loss_mean, bt_loss_samples, reg_loss = self.loss(pred_1_mean, pred_2_mean, pred_1_var, pred_2_var, labels_batch)

                # Normalize losses by batch size
                batch_size = labels_batch.shape[0]
                eval_loss += total_loss.item() * batch_size
                eval_bt_mean_loss += bt_loss_mean.item() * batch_size
                eval_bt_samples_loss += bt_loss_samples.item() * batch_size
                eval_reg_loss += reg_loss.item() * batch_size
                total_samples += batch_size
                
                if images1 is not None and images2 is not None and len(vis_data) < 4:
                    batch_indices = list(range(len(pred_1_mean)))
                    if len(batch_indices) > (4 - len(vis_data)):
                        batch_indices = np.random.choice(batch_indices, size=4 - len(vis_data), replace=False)
                    
                    for i in batch_indices:
                        vis_data.append({
                            'reward1': pred_1_mean[i].cpu().numpy(),
                            'reward2': pred_2_mean[i].cpu().numpy(),
                            'variance1': pred_1_var[i].cpu().numpy(),
                            'variance2': pred_2_var[i].cpu().numpy(),
                            'gt_pref': torch.argmax(binary_labels_batch[i]).item(),
                            'pred_pref': pred_labels[i].item(),
                            'images1': images1[batch * self.batch_size + i],
                            'images2': images2[batch * self.batch_size + i],
                        })
        
        # Normalize by total number of samples
        eval_loss /= total_samples
        eval_bt_mean_loss /= total_samples
        eval_bt_samples_loss /= total_samples
        eval_reg_loss /= total_samples
        eval_acc /= float(total_samples)
        
        wandb.log({
            name + "/loss": eval_loss,
            name + "/bt_mean_loss": eval_bt_mean_loss,
            name + "/bt_samples_loss": eval_bt_samples_loss,
            name + "/reg_loss": eval_reg_loss,
            name + "/acc": eval_acc
        }, step=epoch) if wandb.run is not None else None
        
        if len(vis_data) > 0 and wandb.run is not None:
            self._create_visualization(vis_data, name, epoch)

    def _create_visualization(self, vis_data, name, epoch):
        """Create visualization of reward predictions with variance bands."""
        import io
        import os
        import tempfile

        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        
        videos = []
        for idx, ex in enumerate(vis_data):
            r1 = ex['reward1']  # mean rewards
            r2 = ex['reward2']  # mean rewards
            v1 = ex['variance1']  # variances
            v2 = ex['variance2']  # variances
            gt_pref = ex['gt_pref']
            pred_pref = ex['pred_pref']
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [1.2, 1, 1]})
            fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.1)
            
            ax1.set_title(f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}", fontsize=10)
            ax1.set_xlabel("Timestep", fontsize=9)
            ax1.set_ylabel("Reward", fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Plot mean rewards
            line1, = ax1.plot([], [], label="Traj 1", color="blue")
            line2, = ax1.plot([], [], label="Traj 2", color="red")
            
            # Create empty variance bands
            band1 = ax1.fill_between([], [], [], [], alpha=0.2, color="blue")
            band2 = ax1.fill_between([], [], [], [], alpha=0.2, color="red")
            
            ax1.legend(loc="upper right", fontsize=8)
            ax1.tick_params(axis='both', which='major', labelsize=8)
            
            ax2.set_title("Trajectory 1", fontsize=10)
            ax2.axis('off')
            img1 = ax2.imshow(np.zeros((64, 64, 3)), animated=True)
            
            ax3.set_title("Trajectory 2", fontsize=10)
            ax3.axis('off')
            img2 = ax3.imshow(np.zeros((64, 64, 3)), animated=True)
            
            max_len = max(len(r1), len(r2))
            ax1.set_xlim(0, max_len-1)
            min_y = min(np.min(r1 - 2*np.sqrt(v1)), np.min(r2 - 2*np.sqrt(v2)))
            max_y = max(np.max(r1 + 2*np.sqrt(v1)), np.max(r2 + 2*np.sqrt(v2)))
            ax1.set_ylim(min_y - 0.1 * abs(min_y), max_y + 0.1 * abs(max_y))
            
            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                band1.set_verts([])
                band2.set_verts([])
                img1.set_data(np.zeros((64, 64, 3)))
                img2.set_data(np.zeros((64, 64, 3)))
                return line1, line2, band1, band2, img1, img2
            
            def animate(i):
                # Update mean reward lines
                x = np.arange(i+1)
                line1.set_data(x, r1[:i+1])
                line2.set_data(x, r2[:i+1])
                
                # Update variance bands
                band1.set_verts([np.column_stack([
                    np.concatenate([x, x[::-1]]),
                    np.concatenate([r1[:i+1] + 2*np.sqrt(v1[:i+1]), 
                                  (r1[:i+1] - 2*np.sqrt(v1[:i+1]))[::-1]])
                ])])
                
                band2.set_verts([np.column_stack([
                    np.concatenate([x, x[::-1]]),
                    np.concatenate([r2[:i+1] + 2*np.sqrt(v2[:i+1]), 
                                  (r2[:i+1] - 2*np.sqrt(v2[:i+1]))[::-1]])
                ])])
                
                # Update images
                img1_data = ex['images1'][i]
                img2_data = ex['images2'][i]
                
                if img1_data.max() > 1.0:
                    img1_data = img1_data / 255.0
                    img2_data = img2_data / 255.0
                
                img1.set_data(img1_data)
                img2.set_data(img2_data)
                
                return line1, line2, band1, band2, img1, img2
            
            ani = animation.FuncAnimation(
                fig, animate, init_func=init, frames=max_len,
                interval=100, blit=True
            )
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                ani.save(tmp.name, writer="ffmpeg", fps=10)
                with open(tmp.name, 'rb') as f:
                    buf = io.BytesIO(f.read())
            os.unlink(tmp.name)
            
            plt.close(fig)
            buf.seek(0)
            videos.append(wandb.Video(buf, caption=f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}", format="mp4"))
        
        wandb.log({f"{name}/reward_videos": videos}, step=epoch) if wandb.run is not None else None



