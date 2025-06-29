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
        action_dim: dimension of action at each timestep (D)
        global_cond_dim: dimension of observation to condition on
        """
        super().__init__()
        # input = action + timestep + global_cond
        input_dim = action_dim + 1 + global_cond_dim

        self.noise_pred_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)  # predict noise per action
        )

    def forward(self, sample, timestep, global_cond):
        """
        sample: (B, T, action_dim)
        timestep: (B,)
        global_cond: (B, state_dim)
        """
        B, T, action_dim = sample.shape
        
        # Expand timestep and global_cond over time dimension
        timestep = timestep.view(B, 1, 1).expand(-1, T, -1)  # (B,T,1)
        global_cond = global_cond.unsqueeze(1).expand(-1, T, -1)  # (B,T,state_dim)
        
        # Concatenate
        x = torch.cat([sample, timestep, global_cond], dim=-1)  # (B,T,action_dim+1+state_dim)
        
        # Flatten to apply MLP over each timestep
        x_flat = x.reshape(B*T, -1)
        
        out = self.noise_pred_net(x_flat)  # (B*T, action_dim)
        
        return out.view(B, T, action_dim)


class FlowPolicy(nn.Module):
    def __init__(
        self,
        action_len,
        action_dim,
        noise_pred_net,
        num_train_steps=100,
        num_inference_steps=10,
        timeshift=1.0,
    ):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim

        # Noise prediction net
        assert isinstance(noise_pred_net, NoisePredictionNet)
        self.noise_pred_net = noise_pred_net

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(1, 0, self.num_inference_steps + 1)
        self.timesteps = (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps)

    @torch.no_grad()
    def sample(self, obs):
        # Initialize sample
        action = torch.randn(
            (obs.shape[0], self.action_len, self.action_dim), device=obs.device
        )

        for tcont, tcont_next in zip(self.timesteps[:-1], self.timesteps[1:]):
            # Predict noise
            t = (tcont * self.num_train_steps).long()
            noise_pred = self.noise_pred_net(action, t, global_cond=obs)

            # Flow step
            action = action + (tcont_next - tcont) * noise_pred

        return action

    def forward(self, obs, action):
        # Sample random noise
        noise = torch.randn_like(action)

        # Sample random timestep
        tcont = torch.rand((action.shape[0],), device=action.device)

        # Forward flow step
        direction = noise - action
        noisy_action = (
            action + tcont.view(-1, *[1 for _ in range(action.dim() - 1)]) * direction
        )

        # Flow matching loss
        t = (tcont * self.num_train_steps).long()
        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs)
        loss = F.mse_loss(noise_pred, direction)
        return loss