import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkillEncoder(nn.Module):
    """LSTM encoder that encodes sub-trajectories into Gaussian latent skill distributions."""
    
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim=256, num_layers=2):
        super(SkillEncoder, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input dimension is obs + action concatenated
        input_dim = obs_dim + action_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projections for mean and log variance
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.zero_()
    
    def forward(self, obs, actions):
        """
        Encode sub-trajectory into latent distribution.
        
        Args:
            obs: Observations [batch_size, seq_len, obs_dim]
            actions: Actions [batch_size, seq_len, action_dim]
            
        Returns:
            mean: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
        """
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        # Concatenate observations and actions
        x = torch.cat([obs, actions], dim=-1)  # [batch_size, seq_len, obs_dim + action_dim]
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]
        
        # Use final hidden state
        final_hidden = h_n[-1]  # [batch_size, hidden_dim]
        
        # Project to latent parameters
        mean = self.mean_proj(final_hidden)
        logvar = self.logvar_proj(final_hidden)
        
        return mean, logvar


class SkillDecoder(nn.Module):
    """LSTM decoder that reconstructs actions given latent skill and observations."""
    
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim=256, num_layers=2):
        super(SkillDecoder, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input dimension is obs + latent skill
        input_dim = obs_dim + latent_dim
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection to actions
        self.action_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.zero_()
    
    def forward(self, obs, z):
        """
        Decode actions given observations and latent skill.
        
        Args:
            obs: Observations [batch_size, seq_len, obs_dim]
            z: Latent skill [batch_size, latent_dim]
            
        Returns:
            actions: Reconstructed actions [batch_size, seq_len, action_dim]
        """
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        # Expand latent skill to match sequence length
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, latent_dim]
        
        # Concatenate observations and latent skill
        x = torch.cat([obs, z_expanded], dim=-1)  # [batch_size, seq_len, obs_dim + latent_dim]
        
        # LSTM decoding
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        
        # Project to actions
        actions = self.action_proj(lstm_out)  # [batch_size, seq_len, action_dim]
        
        return actions


class SkillPrior(nn.Module):
    """Learned prior that predicts latent skill given start and end observations."""
    
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super(SkillPrior, self).__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Input is start and end observations concatenated
        input_dim = obs_dim * 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output projections for mean and log variance
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, o0, oH):
        """
        Predict latent skill given start and end observations.
        
        Args:
            o0: Start observations [batch_size, obs_dim]
            oH: End observations [batch_size, obs_dim]
            
        Returns:
            mean: Prior mean [batch_size, latent_dim]
            logvar: Prior log variance [batch_size, latent_dim]
        """
        # Concatenate start and end observations
        x = torch.cat([o0, oH], dim=-1)  # [batch_size, obs_dim * 2]
        
        # Forward pass
        features = self.network(x)
        
        # Project to latent parameters
        mean = self.mean_proj(features)
        logvar = self.logvar_proj(features)
        
        return mean, logvar


class TemporalPredictor(nn.Module):
    """Model that predicts temporal difference between two sub-trajectories."""
    
    def __init__(self, latent_dim, hidden_dim=256):
        super(TemporalPredictor, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Input is two latent skill means concatenated
        input_dim = latent_dim * 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output single time difference value
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, z1_mean, z2_mean):
        """
        Predict temporal difference between two skill embeddings.
        
        Args:
            z1_mean: First skill mean [batch_size, latent_dim]
            z2_mean: Second skill mean [batch_size, latent_dim]
            
        Returns:
            time_diff: Predicted time difference [batch_size, 1]
        """
        # Concatenate skill means
        x = torch.cat([z1_mean, z2_mean], dim=-1)  # [batch_size, latent_dim * 2]
        
        # Predict time difference
        time_diff = self.network(x)  # [batch_size, 1]
        
        return time_diff


class SkillVAE(nn.Module):
    """Complete Skill VAE with encoder, decoder, prior, and temporal predictor."""
    
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=256, num_layers=2):
        super(SkillVAE, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Components
        self.encoder = SkillEncoder(obs_dim, action_dim, latent_dim, hidden_dim, num_layers)
        self.decoder = SkillDecoder(obs_dim, action_dim, latent_dim, hidden_dim, num_layers)
        self.prior = SkillPrior(obs_dim, latent_dim, hidden_dim)
        self.temporal_predictor = TemporalPredictor(latent_dim, hidden_dim)
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for sampling from Gaussian distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, obs, actions):
        """Encode sub-trajectory into latent distribution."""
        return self.encoder(obs, actions)
    
    def decode(self, obs, z):
        """Decode actions given observations and latent skill."""
        return self.decoder(obs, z)
    
    def get_prior(self, o0, oH):
        """Get prior distribution given start and end observations."""
        return self.prior(o0, oH)
    
    def predict_temporal_diff(self, z1_mean, z2_mean):
        """Predict temporal difference between two skill embeddings."""
        return self.temporal_predictor(z1_mean, z2_mean)
    
    def forward(self, obs, actions):
        """
        Forward pass through the complete model.
        
        Args:
            obs: Observations [batch_size, seq_len, obs_dim]
            actions: Actions [batch_size, seq_len, action_dim]
            
        Returns:
            Dictionary containing all outputs
        """
        # Encode
        z_mean, z_logvar = self.encode(obs, actions)
        
        # Sample latent skill
        z = self.reparameterize(z_mean, z_logvar)
        
        # Decode
        reconstructed_actions = self.decode(obs, z)

        reconstructed_actions = torch.tanh(reconstructed_actions)
        
        # Get prior
        o0 = obs[:, 0]  # Start observation
        oH = obs[:, -1]  # End observation
        prior_mean, prior_logvar = self.get_prior(o0, oH)
        
        return {
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'z': z,
            'reconstructed_actions': reconstructed_actions,
            'prior_mean': prior_mean,
            'prior_logvar': prior_logvar
        }
    
    def compute_vae_loss(self, obs, actions, beta=1.0):
        """
        Compute VAE loss (Equation 1 in the paper).
        
        Args:
            obs: Observations [batch_size, seq_len, obs_dim]
            actions: Actions [batch_size, seq_len, action_dim]
            beta: Weight for KL divergence term
            
        Returns:
            Dictionary containing loss components
        """
        outputs = self.forward(obs, actions)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstructed_actions'], actions, reduction='mean')
        
        # KL divergence between posterior and prior
        posterior = torch.distributions.Normal(outputs['z_mean'], torch.exp(0.5 * outputs['z_logvar']))
        prior = torch.distributions.Normal(outputs['prior_mean'], torch.exp(0.5 * outputs['prior_logvar']))
        kl_loss = torch.distributions.kl_divergence(posterior, prior).mean()
        
        # Total VAE loss
        vae_loss = recon_loss + beta * kl_loss
        
        return {
            'vae_loss': vae_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def compute_temporal_loss(self, obs1, actions1, obs2, actions2, time_diff):
        """
        Compute temporal predictability loss (Equation 2 in the paper).
        
        Args:
            obs1: First sub-trajectory observations [batch_size, seq_len, obs_dim]
            actions1: First sub-trajectory actions [batch_size, seq_len, action_dim]
            obs2: Second sub-trajectory observations [batch_size, seq_len, obs_dim]
            actions2: Second sub-trajectory actions [batch_size, seq_len, action_dim]
            time_diff: True temporal difference [batch_size]
            
        Returns:
            Temporal predictability loss
        """
        # Encode both sub-trajectories
        z1_mean, _ = self.encode(obs1, actions1)
        z2_mean, _ = self.encode(obs2, actions2)
        
        # Predict temporal difference
        predicted_time_diff = self.predict_temporal_diff(z1_mean, z2_mean).squeeze(-1)
        
        # MSE loss
        temporal_loss = F.mse_loss(predicted_time_diff, time_diff.float())
        
        return temporal_loss
    
    def compute_total_loss(self, obs1, actions1, obs2, actions2, time_diff, beta=1.0, alpha=1.0):
        """
        Compute total skill learning loss combining VAE and temporal predictability.
        
        Args:
            obs1: First sub-trajectory observations [batch_size, seq_len, obs_dim]
            actions1: First sub-trajectory actions [batch_size, seq_len, action_dim]
            obs2: Second sub-trajectory observations [batch_size, seq_len, obs_dim]
            actions2: Second sub-trajectory actions [batch_size, seq_len, action_dim]
            time_diff: True temporal difference [batch_size]
            beta: Weight for KL divergence in VAE loss
            alpha: Weight for temporal predictability loss
            
        Returns:
            Dictionary containing all loss components
        """
        # VAE loss on first sub-trajectory
        vae_losses1 = self.compute_vae_loss(obs1, actions1, beta)
        
        # VAE loss on second sub-trajectory
        vae_losses2 = self.compute_vae_loss(obs2, actions2, beta)
        
        # Average VAE losses
        vae_loss = (vae_losses1['vae_loss'] + vae_losses2['vae_loss']) / 2
        recon_loss = (vae_losses1['recon_loss'] + vae_losses2['recon_loss']) / 2
        kl_loss = (vae_losses1['kl_loss'] + vae_losses2['kl_loss']) / 2
        
        # Temporal predictability loss
        temporal_loss = self.compute_temporal_loss(obs1, actions1, obs2, actions2, time_diff)
        
        # Total loss
        total_loss = vae_loss + alpha * temporal_loss
        
        return {
            'total_loss': total_loss,
            'vae_loss': vae_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'temporal_loss': temporal_loss
        } 