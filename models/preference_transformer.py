from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="mean")


def cross_ent_loss(logits, target_onehot):
    """Cross‑entropy supporting soft labels (target_onehot shape [..., 2])."""
    return (-target_onehot * F.log_softmax(logits, dim=-1)).sum(-1).mean()


# -----------------------------------------------------------------------------
# Backbone: TransRewardModel
# -----------------------------------------------------------------------------

class TransRewardModel(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        state_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.cfg = config
        self.embed_dim = config["n_embd"]
        self.max_episode_steps = config.get("max_episode_steps", 1000)
        self.use_weighted_sum = bool(config.get("use_weighted_sum", False))

        # Embeddings ----------------------------------------------------------------
        self.state_proj = nn.Linear(state_dim, self.embed_dim)
        self.action_proj = nn.Linear(action_dim, self.embed_dim)
        self.timestep_emb = nn.Embedding(self.max_episode_steps + 1, self.embed_dim)

        # Transformer ----------------------------------------------------------------
        ff_dim = config.get("n_inner", 4 * self.embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["n_head"],
            dim_feedforward=ff_dim,
            dropout=config["resid_pdrop"],
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre‑LayerNorm like GPT‑2
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config["n_layer"])

        # Heads ----------------------------------------------------------------------
        if self.use_weighted_sum:
            d_attn = config["pref_attn_embd_dim"]
            self.qkv_proj = nn.Linear(self.embed_dim, 2 * d_attn + 1)
        else:
            hidden = self.embed_dim // 2
            self.value_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

    # -------------------------------------------------------------------------
    def forward(
        self,
        states: torch.Tensor,  # (B, T, obs_dim)
        actions: torch.Tensor,  # (B, T, act_dim)
        timesteps: torch.Tensor,  # (B, T)
        attn_mask: Optional[torch.Tensor] = None,  # (B, T) bool, 1 = keep
        reverse: bool = False,
        target_idx: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], list]:
        """Forward pass mimicking the Flax implementation."""
        B, T, _ = states.shape
        device = states.device

        if attn_mask is None:
            attn_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Base embeddings ------------------------------------------------------
        s = self.state_proj(states) + self.timestep_emb(timesteps)
        a = self.action_proj(actions) + self.timestep_emb(timesteps)

        # Interleave action & state streams ------------------------------------
        if reverse:
            stacked = torch.stack((s, a), dim=2)  # B, T, 2, E
        else:
            stacked = torch.stack((a, s), dim=2)
        x = stacked.permute(0, 2, 1, 3).reshape(B, 2 * T, self.embed_dim)  # B, 2T, E

        # Transformer encoding --------------------------------------------------
        key_padding = ~torch.stack((attn_mask, attn_mask), dim=2).reshape(B, 2 * T)
        x = self.transformer(x, src_key_padding_mask=key_padding)  # B, 2T, E

        # De‑interleave ----------------------------------------------------------
        x = x.reshape(B, 2, T, self.embed_dim)  # B, 2, T, E
        hidden = x[:, target_idx]  # B, T, E (select state or action stream)

        if self.use_weighted_sum:
            d_attn = self.cfg["pref_attn_embd_dim"]
            qkv = self.qkv_proj(hidden)  # B, T, 2*d_attn + 1
            q, k, v = torch.split(qkv, [d_attn, d_attn, 1], dim=-1)  # v is scalar

            # Scaled‑dot product attention (single head) -----------------------
            attn = (q @ k.transpose(-2, -1)) / (d_attn ** 0.5)
            attn.masked_fill_(~attn_mask.unsqueeze(1), float("-inf"))
            w = torch.softmax(attn, dim=-1)  # B, T, T
            weighted = (w @ v).squeeze(-1)  # B, T
            return {"weighted_sum": weighted.unsqueeze(-1), "value": v.squeeze(-1)}, [w]

        else:
            value = self.value_mlp(hidden)  # B, T, 1
            return {"value": value}, []


# -----------------------------------------------------------------------------
# Trainer: PrefTransformer
# -----------------------------------------------------------------------------

class PrefTransformer:
    @staticmethod
    def default_config(updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = dict(
            trans_lr=1e-4,
            optimizer_type="adamw",
            scheduler_type="CosineDecay",
            vocab_size=1,
            n_layer=3,
            embd_dim=256,
            n_embd=256,
            n_head=1,
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            pref_attn_embd_dim=256,
            train_type="mean",  # mean | sum | last
            use_weighted_sum=False,
            warmup_steps=1000,
            total_steps=100_000,
        )
        if updates:
            cfg.update(updates)
        return cfg

    # ---------------------------------------------------------------------
    def __init__(self, config: Dict[str, Any], trans: TransRewardModel, device: str = None):
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trans = trans.to(self.device)

        # Optimiser ---------------------------------------------------------
        opt_cls = {"adam": Adam, "adamw": AdamW, "sgd": SGD}[config["optimizer_type"]]
        self.optimizer = opt_cls(self.trans.parameters(), lr=config["trans_lr"])

        if config["scheduler_type"] == "CosineDecay":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["total_steps"])
        else:
            self.scheduler = None

        self._ce_loss = nn.CrossEntropyLoss()
        self.total_steps = 0

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=self.device)) for k, v in batch.items()}

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.cfg["train_type"]
        if mode == "mean":
            return x.mean(1, keepdim=True)  # B,1
        elif mode == "sum":
            return x.sum(1, keepdim=True)
        elif mode == "last":
            return x[:, -1:]
        else:
            raise ValueError(f"Unknown train_type '{mode}'")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def get_reward(self, batch: Dict[str, Any]):
        batch = self._to_device(batch)
        outputs, attn = self.trans(
            batch["observations"],
            batch["actions"],
            batch["timestep"],
            attn_mask=batch.get("attn_mask"),
        )
        return outputs["value"], attn[-1] if attn else None

    @torch.no_grad()
    def evaluation(self, batch: Dict[str, Any]):
        batch = self._to_device(batch)
        logits = self._compute_logits(batch, training=False)
        targets = torch.argmax(batch["labels"], dim=-1)
        loss = self._ce_loss(logits, targets)
        return {"eval_cse_loss": loss.item(), "eval_trans_loss": loss.item()}

    def train(self, batch: Dict[str, Any]):
        batch = self._to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self._compute_logits(batch, training=True)
        targets = torch.argmax(batch["labels"], dim=-1)
        loss = self._ce_loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.total_steps += 1
        return {"cse_loss": loss.item(), "trans_loss": loss.item()}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _compute_logits(self, batch: Dict[str, Any], training: bool):
        kwargs = dict(attn_mask=None, reverse=False)

        o1 = self.trans(
            batch["observations"],
            batch["actions"],
            batch["timestep_1"],
            **kwargs,
        )[0]["value"]  # B, T, 1
        o2 = self.trans(
            batch["observations_2"],
            batch["actions_2"],
            batch["timestep_2"],
            **kwargs,
        )[0]["value"]

        p1 = self._reduce(o1).squeeze(-1)  # B
        p2 = self._reduce(o2).squeeze(-1)
        logits = torch.stack((p1, p2), dim=1)  # B, 2
        return logits

    # Expose params for compatibility ---------------------------------------
    @property
    def train_params(self):
        return self.trans.state_dict()

    @property
    def model_keys(self):
        return ("trans",)