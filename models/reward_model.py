import os

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
        self.use_class_weights = getattr(config, 'use_class_weights', True)
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

    def model_net(self, in_dim=39, out_dim=1, H=128, n_layers=2):
        net = []
        for i in range(n_layers):
            net.append(nn.Linear(in_dim, H))
            net.append(nn.LeakyReLU())
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

    def BT_loss(self, pred_hat, label, apply_class_weights=True):
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

    def linear_BT_loss(self, pred_hat, label, apply_class_weights=True):
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
        for member in range(self.ensemble_num):
            # join path + member number
            member_path = os.path.join(path, "reward_" + str(member) + ".pt")
            torch.save(self.ensemble_model[member].state_dict(), member_path)

            print(f"Model for member {member} saved at {member_path}")

    def load_model(self, path):
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            member_path = os.path.join(path, "reward_" + str(member) + ".pt")
            self.ensemble_model[member].load_state_dict(torch.load(member_path))

    def get_reward(self, dataset):
        obs = dataset["observations"]
        act = dataset["actions"]

        if self.dimension == 10:
            obs = obs[:, :3] # eef pos

        obs_act = np.concatenate((obs, act), axis=-1)
        obs_act = torch.from_numpy(obs_act).float().to(self.device)
        with torch.no_grad():
            for i in range((obs_act.shape[0] - 1) // 10000 + 1):
                obs_act_batch = obs_act[i * 10000 : (i + 1) * 10000]
                pred_batch = self.ensemble_model_forward(obs_act_batch).reshape(-1)
                dataset["rewards"][
                    i * 10000 : (i + 1) * 10000
                ] = pred_batch.cpu().numpy()
        return dataset["rewards"]

    def eval(self, obs_act_1, obs_act_2, labels, binary_labels, name, epoch, images1=None, images2=None):
        eval_acc = 0
        eval_loss = 0
        for member in range(self.ensemble_num):
            self.ensemble_model[member].eval()
        with torch.no_grad():
            # Store visualization data for top predictions
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
                pred_seg_sum_1 = torch.sum(pred_1, dim=1)
                pred_seg_sum_2 = torch.sum(pred_2, dim=1)
                pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)
                pred_labels = torch.argmax(pred_hat, dim=-1)
                eval_acc += torch.sum(
                    pred_labels == torch.argmax(binary_labels_batch, dim=-1)
                ).item()
                eval_loss += self.loss(pred_hat, labels_batch, apply_class_weights=False).item()
                
                # Store data for visualization (random 4 examples)
                if images1 is not None and images2 is not None and len(vis_data) < 4:
                    # Get all indices in this batch
                    batch_indices = list(range(len(pred_1)))
                    # Randomly sample indices if we have more than needed
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
        
        eval_loss /= obs_act_1.shape[0]
        eval_acc /= float(obs_act_1.shape[0])
        wandb.log({name + "/loss": eval_loss, name + "/acc": eval_acc}, step=epoch)
        
        # Create visualization if we have data
        if len(vis_data) > 0 and wandb.run is not None:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import io
            import tempfile
            import os
            
            videos = []
            for idx, ex in enumerate(vis_data):
                r1 = ex['reward1']
                r2 = ex['reward2']
                gt_pref = ex['gt_pref']
                pred_pref = ex['pred_pref']
                
                # Create a matplotlib animation of the reward curves
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [1.2, 1, 1]})
                fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.1)
                
                ax1.set_title(f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}", fontsize=10)
                ax1.set_xlabel("Timestep", fontsize=9)
                ax1.set_ylabel("Reward", fontsize=9)
                ax1.grid(True, alpha=0.3)
                line1, = ax1.plot([], [], label="Traj 1", color="blue")
                line2, = ax1.plot([], [], label="Traj 2", color="red")
                ax1.legend(loc="upper right", fontsize=8)
                ax1.tick_params(axis='both', which='major', labelsize=8)
                
                # Set up image displays side by side
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
                    line1.set_data(range(i+1), r1[:i+1])
                    line2.set_data(range(i+1), r2[:i+1])
                    
                    # Get the current frame from the image data
                    img1_data = ex['images1'][i]  # Shape: (H, W, C)
                    img2_data = ex['images2'][i]  # Shape: (H, W, C)                
                    
                    # Normalize to [0, 1] range if needed
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
                
                # Save animation to buffer
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    ani.save(tmp.name, writer="ffmpeg", fps=10)
                    # Read the temporary file into memory
                    with open(tmp.name, 'rb') as f:
                        buf = io.BytesIO(f.read())
                # Clean up the temporary file
                os.unlink(tmp.name)
                
                plt.close(fig)
                buf.seek(0)
                videos.append(wandb.Video(buf, caption=f"Test Pair {idx+1}: GT Pref={gt_pref}, Pred={pred_pref}", format="mp4"))
            
            wandb.log({f"{name}/reward_videos": videos}, step=epoch)

    def train_model(self):
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            self.ensemble_model[member].train()
            self.optimizer.append(
                optim.Adam(self.ensemble_model[member].parameters(), lr=self.lr)
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

            if epoch % 20 == 0:
                wandb.log({"train_eval/loss": train_loss}, step=epoch)

            if epoch % 100 == 0:
                self.eval(
                    self.obs_act_1,
                    self.obs_act_2,
                    self.labels,
                    self.labels,
                    "train_eval",
                    epoch,
                )
                self.eval(
                    self.test_obs_act_1,
                    self.test_obs_act_2,
                    self.test_labels,
                    self.test_binary_labels,
                    "test_eval",
                    epoch,
                    images1=self.test_images1 if hasattr(self, 'test_images1') and epoch % 1000 == 0 else None,
                    images2=self.test_images2 if hasattr(self, 'test_images2') and epoch % 1000 == 0 else None,
                )
