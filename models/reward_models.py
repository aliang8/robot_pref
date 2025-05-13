import torch
import torch.nn as nn
import numpy as np

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
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=-1)
        
        # Apply model
        result = self.model(x).squeeze(-1)  # Squeeze to remove last dimension if batch size = 1
        
        return result
    
    def logpdf(self, obs, action, reward):
        """Compute log probability of reward given observation and action.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            reward: Reward tensor
            
        Returns:
            Log probability of the reward under the model
        """
        # Get predicted reward from the model
        pred_reward = self(obs, action)
        
        # Compute log probability using Gaussian distribution with unit variance
        # Using more stable implementation to avoid numerical issues
        log_prob = -0.5 * ((pred_reward - reward) ** 2) - 0.5 * np.log(2 * np.pi)
        
        return log_prob

class SegmentRewardModel(nn.Module):
    """Model that computes reward for a segment of observation-action pairs."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(SegmentRewardModel, self).__init__()
        self.reward_model = StateActionRewardModel(state_dim, action_dim, hidden_dims)
    
    def forward(self, observations, actions, return_ensemble_format=False):
        """Compute reward for a sequence of observation-action pairs.
        
        Args:
            observations: Observations tensor
            actions: Actions tensor
            return_ensemble_format: If True, add an extra dimension to mimic ensemble output
            
        Returns:
            Reward tensor, with shape [batch_size] or [1, batch_size] if return_ensemble_format=True
        """
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
            result = rewards.sum(1)[0]  # Sum over sequence length and remove batch dim
            
            # Add ensemble dimension if requested
            if return_ensemble_format:
                return result.unsqueeze(0)  # Shape: [1]
            return result
        
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
            result = rewards.sum(1)  # Sum over sequence length, shape: [batch_size]
            
            # Add ensemble dimension if requested
            if return_ensemble_format:
                return result.unsqueeze(0)  # Shape: [1, batch_size]
            return result
        
        else:
            raise ValueError(f"Unexpected input shape: observations {observations.shape}, actions {actions.shape}")
    
    def compute_paired_rewards(self, obs1, actions1, obs2, actions2):
        """Compute rewards for two segments.
        
        This is a convenience method for training that computes rewards for both
        segments in a preference pair.
        
        Args:
            obs1: First segment observations
            actions1: First segment actions
            obs2: Second segment observations
            actions2: Second segment actions
            
        Returns:
            Tuple of (reward1, reward2) with shape [1, batch_size]
        """
        reward1 = self(obs1, actions1, return_ensemble_format=True)
        reward2 = self(obs2, actions2, return_ensemble_format=True)
        return reward1, reward2
    
    def logpdf(self, observations, actions, rewards):
        """Compute log probability of rewards given observations and actions.
        
        Args:
            observations: Batch of observation sequences or single observation sequence
            actions: Batch of action sequences or single action sequence
            rewards: Target reward values
            
        Returns:
            Log probability of rewards under the model
        """
        # Vectorized implementation
        if observations.dim() == 2:  # Single segment
            # Compute segment reward
            segment_reward = self(observations, actions)
            
            # Use mean observation and action for the whole segment to compute log probability
            mean_obs = observations.mean(0, keepdim=True)
            mean_action = actions.mean(0, keepdim=True) if len(actions) > 0 else actions[0:1]
            
            # Get log probability from base reward model
            return self.reward_model.logpdf(mean_obs, mean_action, rewards)
            
        elif observations.dim() == 3:  # Batch of segments
            # Compute mean observation and action for each segment
            mean_obs = observations.mean(1)  # Average over sequence length
            mean_actions = actions.mean(1) if actions.size(1) > 0 else actions[:, 0]
            
            # Get log probability from base reward model
            return self.reward_model.logpdf(mean_obs, mean_actions, rewards)
        else:
            raise ValueError(f"Unexpected input shape for logpdf: {observations.shape}")

class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation."""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], num_models=5):
        super(EnsembleRewardModel, self).__init__()
        self.models = nn.ModuleList([
            SegmentRewardModel(state_dim, action_dim, hidden_dims)
            for _ in range(num_models)
        ])
        self.num_models = num_models
    
    def forward(self, observations, actions):
        """Return rewards from all models in the ensemble.
        
        Returns:
            Tensor of rewards with shape [num_models, batch_size]
        """
        rewards = []
        for model in self.models:
            rewards.append(model(observations, actions))
        
        # Stack rewards from all models along first dimension
        # Shape: [num_models, batch_size] or [num_models] for single sample
        stacked_rewards = torch.stack(rewards, dim=0)
        
        # Ensure we always return [num_models, batch_size] even for single samples
        if stacked_rewards.dim() == 1:  # [num_models]
            stacked_rewards = stacked_rewards.unsqueeze(-1)  # [num_models, 1]
            
        return stacked_rewards
    
    def compute_paired_rewards(self, obs1, actions1, obs2, actions2):
        """Compute rewards for two segments across all models.
        
        This is a convenience method for training that computes rewards for both
        segments in a preference pair.
        
        Args:
            obs1: First segment observations
            actions1: First segment actions
            obs2: Second segment observations
            actions2: Second segment actions
            
        Returns:
            Tuple of (reward1, reward2) with shape [num_models, batch_size]
        """
        reward1 = self(obs1, actions1)
        reward2 = self(obs2, actions2)
        return reward1, reward2
    
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