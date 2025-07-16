from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisePredictionNet(nn.Module, ABC):
    @abstractmethod
    def forward(self, sample, timestep, global_cond):
        raise NotImplementedError


class FlowNoisePredictionNet(NoisePredictionNet):
    def __init__(self, action_dim, global_cond_dim, hidden_dim=256):
        """
        action_dim: dimension of action
        global_cond_dim: dimension of observation to condition on
        """
        super().__init__()
        input_dim = action_dim + 1 + global_cond_dim  # sample + timestep + global_cond

        self.noise_pred_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),  # predict noise for action
        )

    def forward(self, sample, timestep, global_cond):
        """
        sample: (B, action_dim)
        timestep: (B,)
        global_cond: (B, global_cond_dim)
        """
        timestep = timestep.view(-1, 1)  # (B,1)

        # Concatenate sample, timestep, and global_cond
        x = torch.cat([sample, timestep, global_cond], dim=-1)  # (B, input_dim)
        out = self.noise_pred_net(x)  # (B, action_dim)
        return out


class FlowPolicy(nn.Module):
    def __init__(
        self,
        action_dim,
        noise_pred_net,
        max_action=1.0,
        num_train_steps=100,
        num_inference_steps=10,
        timeshift=1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        assert isinstance(noise_pred_net, NoisePredictionNet)
        self.noise_pred_net = noise_pred_net

        self.max_action = max_action

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps

        # Create rescaled inference timesteps
        timesteps = torch.linspace(1, 0, self.num_inference_steps + 1)
        self.timesteps = (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps)

    @torch.no_grad()
    def sample(self, obs):
        """
        obs: (B, global_cond_dim)
        """
        B = obs.shape[0]
        action = torch.randn(B, self.action_dim, device=obs.device)

        for tcont, tcont_next in zip(self.timesteps[:-1], self.timesteps[1:]):
            # Scale to training timestep range
            t = (tcont * self.num_train_steps).long().expand(B).to(obs.device)
            noise_pred = self.noise_pred_net(action, t, global_cond=obs)

            # Flow step
            action = action + (tcont_next - tcont) * noise_pred

        return action

    def forward(self, obs, action):
        """
        obs: (B, global_cond_dim)
        action: (B, action_dim)
        """
        noise = torch.randn_like(action)  # target

        # Uniform random timestep in [0,1]
        tcont = torch.rand((action.shape[0],), device=action.device)
        direction = noise - action

        # Forward noisy action: move along direction by tcont
        noisy_action = action + tcont.view(-1, 1) * direction

        t = (tcont * self.num_train_steps).long()
        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs)

        # Flow matching loss
        loss = F.mse_loss(noise_pred, direction, reduction="none").mean(dim=1)
        return loss
