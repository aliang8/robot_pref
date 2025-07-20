import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import MLP
import math
from typing import Any, Dict, List, Tuple
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TwinTransformerQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, seq_len: int, hidden_dim: int = 256
    ):
        super(TwinTransformerQ, self).__init__()
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.action_emb = nn.Linear(action_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.transformer_action_processor = nn.TransformerEncoder(
            encoder_layer, num_layers=1, norm=nn.LayerNorm(hidden_dim)
        )
        self.position_embedding = PositionalEncoding(hidden_dim, max_len=100)

        dims = [hidden_dim * 2, hidden_dim, hidden_dim // 2, 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = actions.shape

        state_embed = self.state_emb(state)

        # (B, S, D) -> (B * S, D)
        actions = actions.reshape(B * actions.shape[1], D)
        action_embed = self.action_emb(actions)
        action_embed = action_embed.reshape(B, S, -1)

        action_embed = self.position_embedding(action_embed)
        action_embed = self.transformer_action_processor(action_embed)
        action_embed = action_embed.mean(dim=1)
        q_input = torch.cat([state_embed, action_embed], dim=1)

        return self.q1(q_input), self.q2(q_input)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, actions))


class ActionChunkingTransformer(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, seq_len: int, hidden_dim: int = 256
    ):
        super(ActionChunkingTransformer, self).__init__()
        self.state_emb = nn.Linear(state_dim, hidden_dim)

        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 2]
        self.latent_pi = MLP(dims)

        last_layer_dim = dims[-1]

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=last_layer_dim, nhead=8, batch_first=True
        )
        self.action_transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=1, norm=nn.LayerNorm(last_layer_dim)
        )
        self.triangular_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"),
            diagonal=1,
        )
        self.position_embedding = PositionalEncoding(last_layer_dim, max_len=seq_len)

        dims = [last_layer_dim, last_layer_dim // 2, last_layer_dim // 2, action_dim]
        self.mu = MLP(dims)
        self.log_std = MLP(dims)

        self.seq_len = seq_len

    def get_action_dist_params(self, state):
        state_embed = self.state_emb(state)
        latent_pi = self.latent_pi(state_embed)

        mean_actions_intermediate = self.position_embedding(
            latent_pi.unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        mean_actions_intermediate = self.action_transformer(
            mean_actions_intermediate,
            memory=mean_actions_intermediate,
        )

        mu = self.mu(mean_actions_intermediate)
        log_std_intermediate = mean_actions_intermediate
        log_std = self.log_std(log_std_intermediate)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mu, log_std

    def forward(self, state, deterministic=True):
        mu, log_std = self.get_action_dist_params(state)

        pred_actions = self.action_dist.actions_from_params(
            mu, log_std, deterministic=deterministic
        )

        return pred_actions
