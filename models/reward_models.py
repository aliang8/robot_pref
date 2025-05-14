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
    

class StateActionRewardModel(nn.Module):
    """MLP-based reward model that takes state and action as input."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], output_dim=1):
        super(StateActionRewardModel, self).__init__()
        
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
        """Forward pass using observation and action."""
        # Check for NaN inputs
        if torch.isnan(obs).any() or torch.isnan(action).any():
            print("WARNING: NaN detected in model inputs")
            
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=-1)
        
        # Apply model
        result = self.model(x).squeeze(-1)  # Squeeze to remove last dimension if batch size = 1
        
        # Check for NaN outputs
        if torch.isnan(result).any():
            print("WARNING: NaN detected in model outputs")
            
        return result
    
    def logpdf(self, obs, action, reward):
        """Compute log probability of reward given observation and action."""
        # Assuming Gaussian distribution with unit variance
        pred_reward = self(obs, action)
        log_prob = -0.5 * ((pred_reward - reward) ** 2) - 0.5 * np.log(2 * np.pi)
        return log_prob

class SegmentRewardModel(nn.Module):
    """Model that computes reward for a segment of observation-action pairs."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(SegmentRewardModel, self).__init__()
        self.reward_model = StateActionRewardModel(state_dim, action_dim, hidden_dims)
    
    def forward(self, observations, actions):
        """Compute reward for a sequence of observation-action pairs."""
        # Handle both single segments and batches
        if observations.dim() == 2:  # Single segment (seq_len, obs_dim)
            # Observations and actions should be the same length now
            if observations.size(0) != actions.size(0):
                # If they're not the same, adjust them
                min_len = min(observations.size(0), actions.size(0))
                observations = observations[:min_len]
                actions = actions[:min_len]
                
            # Process all steps in the segment at once (vectorized)
            batch_obs = observations.unsqueeze(0)  # Add batch dim
            batch_actions = actions.unsqueeze(0)
            
            # Concatenate observations and actions along the feature dimension
            combined_inputs = torch.cat([
                batch_obs.reshape(-1, batch_obs.size(-1)), 
                batch_actions.reshape(-1, batch_actions.size(-1))
            ], dim=1)
            
            # Process through the reward model
            rewards = self.reward_model.model(combined_inputs)
            rewards = rewards.reshape(batch_obs.size(0), -1)
            return rewards.sum(1)[0]  # Sum over sequence length and remove batch dim
        
        elif observations.dim() == 3:  # Batch of segments (batch_size, seq_len, obs_dim)
            batch_size = observations.size(0)
            
            # Observations and actions should be the same length now
            if observations.size(1) != actions.size(1):
                # If they're not the same, adjust them
                min_len = min(observations.size(1), actions.size(1))
                observations = observations[:, :min_len]
                actions = actions[:, :min_len]
            
            # Flatten the batch and sequence dimensions
            flat_obs = observations.reshape(-1, observations.size(-1))
            flat_actions = actions.reshape(-1, actions.size(-1))
            
            # Concatenate observations and actions along the feature dimension
            combined_inputs = torch.cat([flat_obs, flat_actions], dim=1)
            
            # Process through the reward model
            flat_rewards = self.reward_model.model(combined_inputs)
            rewards = flat_rewards.reshape(batch_size, -1)
            return rewards.sum(1)  # Sum over sequence length
        
        else:
            raise ValueError(f"Unexpected input shape: observations {observations.shape}, actions {actions.shape}")
    
    def logpdf(self, observations, actions, rewards):
        """Compute log probability of rewards given observations and actions."""
        # Vectorized implementation
        if observations.dim() == 2:  # Single segment
            segment_reward = self(observations, actions)
            return self.reward_model.logpdf(observations.mean(0, keepdim=True), 
                                          actions.mean(0, keepdim=True) if len(actions) > 0 else actions[0:1], 
                                          rewards)
        elif observations.dim() == 3:  # Batch of segments
            # Compute mean observation and action for each segment
            mean_obs = observations.mean(1)  # Average over sequence length
            mean_actions = actions.mean(1) if actions.size(1) > 0 else actions[:, 0]
            return self.reward_model.logpdf(mean_obs, mean_actions, rewards)
        else:
            raise ValueError(f"Unexpected input shape for logpdf")