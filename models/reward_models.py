import torch
import torch.nn as nn
import numpy as np
from .image_embedder import MultiInputEmbedder
from omegaconf import OmegaConf


class RewardModel(nn.Module):
    """MLP-based reward model that takes state, action, and optionally images as input."""

    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dims=[256, 256], 
        output_dim=1,
        use_images=False,
        image_model="resnet50",
        embedding_dim=256,
        use_image_embeddings=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(RewardModel, self).__init__()
        
        self.use_images = use_images
        self.use_image_embeddings = use_image_embeddings
        
        # Configure input modalities
        input_modalities = ["states"]
        if use_images:
            if use_image_embeddings:
                input_modalities.append("image_embed")
            else:
                input_modalities.append("image")

        # Create embedder config
        embedder_cfg = {
            "embedding_dim": embedding_dim,
            "embedding_model": image_model,
            "input_embed_dim": embedding_dim,
        }
        
        # Initialize multi-input embedder
        self.embedder = MultiInputEmbedder(
            OmegaConf.create(embedder_cfg),
            input_modalities=input_modalities,
            state_dim=state_dim,
            image_shape=(224, 224) if use_images and not use_image_embeddings else None,
            seq_len=1
        )

        self.action_embedder = nn.Linear(action_dim, embedding_dim)
        
        # Build MLP layers for reward prediction
        layers = []
        # Use embeddings if images are used, otherwise use raw state+action
        if use_images:
            prev_dim = embedding_dim * 2  # state embedding + action embedding
        else:
            prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Add layer normalization for stability
            layers.append(nn.LeakyReLU(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

        # Initialize weights properly
        self.apply(self._init_weights)
        
        # Ensure model is in float32
        self.type(torch.float32)

    def _init_weights(self, module):
        """Initialize weights with a better scheme for stability."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, obs, action, images=None):
        """Forward pass using observation, action, and optionally images.

        Args:
            obs: Observation tensor [batch_size, T, obs_dim]
            action: Action tensor [batch_size, T, action_dim]
            images: Optional image tensor [batch_size, T, C, H, W] or image embeddings [batch_size, T, embed_dim]

        Returns:
            Reward tensor [batch_size, T]
        """
        # Ensure inputs are float32
        obs = obs.float()
        action = action.float()
        if images is not None:
            images = images.float()
        
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        if self.use_images and images is not None:
            # Prepare inputs for embedder
            inputs = {"states": obs}
            if self.use_image_embeddings:
                inputs["image_embed"] = images
            else:
                inputs["image"] = images

            # Get embeddings
            embeddings = self.embedder(inputs)
            action_embeddings = self.action_embedder(action)

            # Concatenate with actions
            x = torch.cat([embeddings, action_embeddings], dim=-1)
        else:
            # Use raw state and action concatenation
            x = torch.cat([obs, action], dim=-1)

        # Apply model to get logits
        logits = self.model(x).squeeze(-1)  # Squeeze to remove last dimension if batch size = 1
        result = torch.tanh(logits)
        return result


class DistributionalRewardModel(nn.Module):
    """Distributional reward model with separate mean and variance branches."""

    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dims=[256, 256], 
        output_dim=1,
        use_images=False,
        image_model="resnet50",
        embedding_dim=256,
        use_image_embeddings=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(DistributionalRewardModel, self).__init__()
        
        self.use_images = use_images
        self.use_image_embeddings = use_image_embeddings
        self.embedding_dim = embedding_dim
        
        # Configure input modalities
        input_modalities = ["states"]
        if use_images:
            if use_image_embeddings:
                input_modalities.append("image_embed")
            else:
                input_modalities.append("image")

        # Create embedder config
        embedder_cfg = {
            "embedding_dim": embedding_dim,
            "embedding_model": image_model,
            "input_embed_dim": embedding_dim,
        }
        
        # Initialize multi-input embedder
        self.embedder = MultiInputEmbedder(
            OmegaConf.create(embedder_cfg),
            input_modalities=input_modalities,
            state_dim=state_dim,
            image_shape=(224, 224) if use_images and not use_image_embeddings else None,
            seq_len=1
        )

        self.action_embedder = nn.Linear(action_dim, embedding_dim)
        
        # Shared feature extractor
        shared_layers = []
        # Use embeddings if images are used, otherwise use raw state+action
        if use_images:
            prev_dim = embedding_dim * 2  # state embedding + action embedding
        else:
            prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.LeakyReLU(0.1))
            prev_dim = hidden_dim
            
        self.shared_features = nn.Sequential(*shared_layers)
        
        # Split the final hidden dimension for mean and variance branches
        split_dim = prev_dim // 2
        
        # Mean branch
        self.mean_branch = nn.Sequential(
            nn.Linear(split_dim, split_dim),
            nn.LayerNorm(split_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(split_dim, output_dim)
        )
        
        # Variance branch (output log variance for numerical stability)
        self.variance_branch = nn.Sequential(
            nn.Linear(split_dim, split_dim),
            nn.LayerNorm(split_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(split_dim, output_dim)
        )

        # Initialize weights properly
        self.apply(self._init_weights)
        
        # Ensure model is in float32
        self.type(torch.float32)

    def _init_weights(self, module):
        """Initialize weights with a better scheme for stability."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, obs, action, images=None, return_distribution=True):
        """Forward pass using observation, action, and optionally images.

        Args:
            obs: Observation tensor [batch_size, T, obs_dim]
            action: Action tensor [batch_size, T, action_dim]
            images: Optional image tensor [batch_size, T, C, H, W] or image embeddings [batch_size, T, embed_dim]
            return_distribution: If True, return both mean and variance. If False, return only mean.

        Returns:
            If return_distribution=True: tuple of (reward_mean, reward_variance)
            If return_distribution=False: reward_mean only
            Each tensor has shape [batch_size, T]
        """
        # Ensure inputs are float32
        obs = obs.float()
        action = action.float()
        if images is not None:
            images = images.float()
        
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        if self.use_images and images is not None:
            # Prepare inputs for embedder
            inputs = {"states": obs}
            if self.use_image_embeddings:
                inputs["image_embed"] = images
            else:
                inputs["image"] = images

            # Get embeddings
            embeddings = self.embedder(inputs)
            action_embeddings = self.action_embedder(action)

            # Concatenate with actions
            x = torch.cat([embeddings, action_embeddings], dim=-1)
        else:
            # Use raw state and action concatenation
            x = torch.cat([obs, action], dim=-1)
        
        # Extract shared features
        shared_features = self.shared_features(x)  # [batch_size, T, hidden_dim]
        
        # Split features along the embedding dimension for mean and variance branches
        split_dim = shared_features.shape[-1] // 2
        mean_features = shared_features[..., :split_dim]
        var_features = shared_features[..., split_dim:]
        
        # Compute mean and variance
        reward_mean = self.mean_branch(mean_features).squeeze(-1)  # [batch_size, T]
        reward_mean = torch.tanh(reward_mean)
        log_variance = self.variance_branch(var_features).squeeze(-1)  # [batch_size, T]
        
        # Convert log variance to variance (ensure positive)
        reward_variance = torch.exp(log_variance) + 1e-6  # Add small epsilon for numerical stability
        
        if return_distribution:
            return reward_mean, reward_variance
        else:
            return reward_mean

    def sample_rewards(self, obs, action, images=None, num_samples=1):
        """Sample rewards from the learned distribution.
        
        Args:
            obs: Observation tensor [batch_size, T, obs_dim]
            action: Action tensor [batch_size, T, action_dim]
            images: Optional image tensor
            num_samples: Number of samples to draw
            
        Returns:
            Sampled rewards [num_samples, batch_size, T]
        """
        reward_mean, reward_variance = self.forward(obs, action, images, return_distribution=True)
        
        # Sample from Gaussian distribution
        std = torch.sqrt(reward_variance)
        eps = torch.randn(num_samples, *reward_mean.shape, device=reward_mean.device)
        samples = reward_mean.unsqueeze(0) + std.unsqueeze(0) * eps
        
        return samples


class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation.
    
    Can create ensembles of either regular RewardModel or DistributionalRewardModel.
    """

    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dims=[256, 256], 
        num_models=5,
        model_type="regular",  # "regular" or "distributional"
        use_images=False,
        image_model="resnet50",
        embedding_dim=256,
        use_image_embeddings=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(EnsembleRewardModel, self).__init__()
        
        self.num_models = num_models
        self.model_type = model_type
        
        # Create ensemble of the specified model type
        if model_type == "distributional":
            self.models = nn.ModuleList([
                DistributionalRewardModel(
                    state_dim, 
                    action_dim, 
                    hidden_dims,
                    use_images=use_images,
                    image_model=image_model,
                    embedding_dim=embedding_dim,
                    use_image_embeddings=use_image_embeddings,
                    device=device
                ) for _ in range(num_models)
            ])
        else:  # regular
            self.models = nn.ModuleList([
                RewardModel(
                    state_dim, 
                    action_dim, 
                    hidden_dims,
                    use_images=use_images,
                    image_model=image_model,
                    embedding_dim=embedding_dim,
                    use_image_embeddings=use_image_embeddings,
                    device=device
                ) for _ in range(num_models)
            ])

    def forward(self, observations, actions, images=None, return_distribution=None):
        """Return rewards from all models in the ensemble.

        Args:
            observations: Observations tensor [batch_size, seq_len, obs_dim]
            actions: Actions tensor [batch_size, seq_len, action_dim]
            images: Optional images tensor (ignored)
            return_distribution: For distributional models, whether to return distribution parameters

        Returns:
            If model_type == "regular": 
                Tensor of rewards with shape [num_models, batch_size, seq_len]
            If model_type == "distributional":
                If return_distribution=True: tuple of (means, variances) each with shape [num_models, batch_size, seq_len]
                If return_distribution=False: means with shape [num_models, batch_size, seq_len]
        """
        if self.model_type == "distributional":
            if return_distribution is None:
                return_distribution = True
                
            if return_distribution:
                means, variances = [], []
                for model in self.models:
                    mean, var = model(observations, actions, images, return_distribution=True)
                    means.append(mean)
                    variances.append(var)
                return torch.stack(means, dim=0), torch.stack(variances, dim=0)
            else:
                means = []
                for model in self.models:
                    mean = model(observations, actions, images, return_distribution=False)
                    means.append(mean)
                return torch.stack(means, dim=0)
        else:  # regular models
            rewards = []
            for model in self.models:
                rewards.append(model(observations, actions, images))
            return torch.stack(rewards, dim=0)

    def mean_reward(self, observations, actions, images=None):
        """Return mean reward across all models."""
        if self.model_type == "distributional":
            means = self.forward(observations, actions, images, return_distribution=False)
        else:
            means = self.forward(observations, actions, images)
        return means.mean(dim=0)

    def sample_rewards(self, observations, actions, images=None, num_samples=1):
        """Sample rewards from the ensemble.
        
        Only available for distributional models.
        
        Returns:
            Sampled rewards [num_models, num_samples, batch_size, seq_len]
        """
        if self.model_type != "distributional":
            raise ValueError("Sampling is only available for distributional models")
            
        samples = []
        for model in self.models:
            model_samples = model.sample_rewards(observations, actions, images, num_samples)
            samples.append(model_samples)
        return torch.stack(samples, dim=0)

    def get_uncertainty(self, observations, actions, images=None):
        """Get both aleatoric and epistemic uncertainty.
        
        Only available for distributional models.
        
        Returns:
            dict with keys:
                - 'aleatoric': Average variance across models [batch_size, seq_len]
                - 'epistemic': Variance of means across models [batch_size, seq_len]
                - 'total': Sum of aleatoric and epistemic [batch_size, seq_len]
        """
        if self.model_type != "distributional":
            raise ValueError("Uncertainty estimation is only available for distributional models")
            
        means, variances = self.forward(observations, actions, images, return_distribution=True)
        
        # Aleatoric uncertainty (average variance across models)
        aleatoric_uncertainty = variances.mean(dim=0)
        
        # Epistemic uncertainty (variance of means across models)
        epistemic_uncertainty = means.var(dim=0)
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return {
            'aleatoric': aleatoric_uncertainty,
            'epistemic': epistemic_uncertainty,
            'total': total_uncertainty
        } 