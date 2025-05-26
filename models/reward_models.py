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
        
        # Build MLP layers for reward prediction
        layers = []
        prev_dim = embedding_dim + action_dim  # Embedder output + action

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
            Reward tensor [batch_size]
        """
        # Ensure inputs are float32
        obs = obs.float()
        action = action.float()
        if images is not None:
            images = images.float()
        
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        # Prepare inputs for embedder
        inputs = {"states": obs}
        if self.use_images:
            if self.use_image_embeddings:
                inputs["image_embed"] = images
            else:
                inputs["image"] = images

        # Get embeddings
        embeddings = self.embedder(inputs)
        
        # Concatenate with actions
        x = torch.cat([embeddings, action], dim=-1)

        # Apply model to get logits
        logits = self.model(x).squeeze(-1)  # Squeeze to remove last dimension if batch size = 1
        result = torch.tanh(logits)
        return result


class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation."""

    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dims=[256, 256], 
        num_models=5,
        use_images=False,
        image_model="resnet50",
        device="cuda"
    ):
        super(EnsembleRewardModel, self).__init__()
        self.models = nn.ModuleList([
            RewardModel(
                state_dim, 
                action_dim, 
                hidden_dims,
                use_images=use_images,
                image_model=image_model,
                device=device
            ) for _ in range(num_models)
        ])
        self.num_models = num_models
        self.use_images = use_images

    def forward(self, observations, actions, images=None):
        """Return rewards from all models in the ensemble.

        Args:
            observations: Observations tensor [batch_size, seq_len, obs_dim]
            actions: Actions tensor [batch_size, seq_len, action_dim]
            images: Optional images tensor [batch_size, seq_len, C, H, W]

        Returns:
            Tensor of rewards with shape [num_models, batch_size]
        """
        rewards = []
        for model in self.models:
            if self.use_images:
                rewards.append(model(observations, actions, images))
            else:
                rewards.append(model(observations, actions))

        return torch.stack(rewards, dim=0)

    def mean_reward(self, observations, actions, images=None):
        """Return mean reward across all models."""
        rewards = self(observations, actions, images)
        return rewards.mean(dim=0)
