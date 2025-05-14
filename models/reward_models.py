import torch
import torch.nn as nn
import numpy as np
from utils.dataset_utils import bradley_terry_loss

class RewardModel(nn.Module):
    """MLP-based reward model that takes state and action as input."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], output_dim=1):
        super(RewardModel, self).__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = state_dim + action_dim  # Concatenated state and action
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Add layer normalization for stability
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with a better scheme for stability."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, obs, action):
        """Forward pass using observation and action.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Reward tensor [batch_size]
        """
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=-1)
        
        # Apply model to get logits
        logits = self.model(x).squeeze(-1)  # Squeeze to remove last dimension if batch size = 1
        
        result = torch.tanh(logits)
        return result
        

class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], num_models=5):
        super(EnsembleRewardModel, self).__init__()
        self.models = nn.ModuleList([
            RewardModel(state_dim, action_dim, hidden_dims)
            for _ in range(num_models)
        ])
        self.num_models = num_models
    
    def forward(self, observations, actions):
        """Return rewards from all models in the ensemble.
        
        Args:
            observations: Observations tensor [batch_size, seq_len, obs_dim]
            actions: Actions tensor [batch_size, seq_len, action_dim]
            
        Returns:
            Tensor of rewards with shape [num_models, batch_size]
        """

        rewards = []
        for model in self.models:
            rewards.append(model(observations, actions))
            
        return torch.stack(rewards, dim=0)
    
    def mean_reward(self, observations, actions):
        """Return mean reward across all models."""
        rewards = self(observations, actions)
        return rewards.mean(dim=0)
    
    def std_reward(self, observations, actions):
        """Return standard deviation of rewards across all models."""
        rewards = self(observations, actions)
        return rewards.std(dim=0)
    
    def disagreement(self, observations, actions):
        """Return disagreement (variance) across models."""
        rewards = self(observations, actions)
        return rewards.var(dim=0)