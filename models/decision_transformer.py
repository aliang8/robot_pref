import torch
import torch.nn as nn
import transformers
from models.trajectory_gpt2 import GPT2Model

class DecisionTransformer(nn.Module):
    """
    Decision Transformer using GPT2Model as the backbone.
    Models (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=256,
        action_tanh=True,
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,  # not used
            n_embd=hidden_size,
            n_layer=kwargs.get('nlayer', 3),
            n_head=kwargs.get('nhead', 4),
            **{k: v for k, v in kwargs.items() if k not in ['nlayer', 'nhead']}
        )
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        """
        Decision Transformer forward pass.
        Args:
            states: (B, T, state_dim)
            actions: (B, T, act_dim)
            returns_to_go: (B, T, 1)
            timesteps: (B, T)
            attention_mask: (B, T) or None
        Returns:
            state_preds: (B, T, state_dim)
            action_preds: (B, T, act_dim)
            return_preds: (B, T, 1)
        """
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # add time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # stack as (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # stack attention mask
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape to (batch, seq, 3, hidden)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        # return state_preds, action_preds, return_preds
        return action_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        device = states.device

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat([
                torch.zeros(self.max_length - states.shape[1]),
                torch.ones(states.shape[1])
            ])
            attention_mask = attention_mask.to(dtype=torch.long, device=device).reshape(1, -1)
            states = torch.cat([
                torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim), device=device),
                states
            ], dim=1).to(dtype=torch.float32)
            actions = torch.cat([
                torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=device),
                actions
            ], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([
                torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1), device=device),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([
                torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=device),
                timesteps
            ], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds = self.forward(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask
        )

        return action_preds[0, -1]
