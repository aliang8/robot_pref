import torch
import torch.nn as nn

import transformers

from models.dt.trajectory_gpt2 import GPT2Model


class DecisionTransformer(nn.Module):

    """
    Model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length,
            nhead=4,
            nlayer=3,
            max_ep_len=512,
            action_tanh=False,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.max_length = max_length

        self.hidden_size = hidden_size

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=nlayer)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.action_head = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        # self.predict_return = torch.nn.Linear(hidden_size, 1)

    # def _generate_causal_mask(self, seq_len):
    #     mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=1).bool()
    #     return mask
    
    def _generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, batch):
        """
        Decision Transformer forward pass.
        Args:
            batch: tuple of tensors
                states: (B, T, state_dim)
                actions: (B, T, act_dim)
                rewards: (B, T, 1)
                rtgs: (B, T, 1)
                dones: (B, T, 1)
                timesteps: (B, T)
                attention_mask: (B, T) or None
        Returns:
            state_preds: (B, T, state_dim)
            action_preds: (B, T, act_dim)
            return_preds: (B, T, 1)
        """
        states, actions, _, rtgs, _, timesteps, attention_mask = batch

        batch_size, seq_length, device = states.shape[0], states.shape[1], states.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(rtgs)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # memory = self.embed_ln(stacked_inputs)
        memory = stacked_inputs

        # masks
        memory_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        tgt_mask = self._generate_causal_mask(seq_length).to(device)
        
        # padding masks
        # create dummy sequence embeddings to use as query for the transformer decoder
        tgt = torch.zeros(
            batch_size, seq_length, action_embeddings.shape[-1], device=device # [B, T, D]
        )
        tgt = tgt + time_embeddings

        x = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            memory_key_padding_mask=(memory_attention_mask == 0),
        )

        action_preds = self.action_head(x)

        return action_preds

    def sample(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        batch = (
            states, actions, rewards, returns_to_go, None, timesteps, attention_mask
        )

        action_preds = self.forward(
            batch, **kwargs)

        return action_preds[0,-1]