"""256-dimensional H0/H1/H2 towers with fixed-width auxiliary features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.models import (
    CategoricalNumericEncoder,
    HybridTower,
    MaskedAttention,
    _alpha_parameter,
)


MODEL_NAMES = ("h0_256", "h1_256", "h2_256")


class HybridTower256(HybridTower):
    """H2 architecture where only the retrieval space grows to 256d."""

    def __init__(
        self,
        variant: str,
        user_vocab: int,
        item_vocab: int,
        user_category_sizes: list[int],
        item_category_sizes: list[int],
        content_dim: int = 768,
        retrieval_dim: int = 256,
        id_dim: int = 128,
        alpha_init: float = 0.05,
    ):
        nn.Module.__init__(self)
        if variant not in MODEL_NAMES:
            raise ValueError(variant)
        self.variant = variant
        self.use_side = variant != "h0_256"
        self.use_id = variant == "h2_256"
        self.retrieval_dim = retrieval_dim
        self.id_dim = id_dim
        self.content = nn.Linear(content_dim, retrieval_dim, bias=False)
        self.item_meta = CategoricalNumericEncoder(
            item_category_sizes, 4, retrieval_dim, category_dim=16
        )
        self.user_profile = CategoricalNumericEncoder(
            user_category_sizes, 2, retrieval_dim, category_dim=16
        )
        self.item_id = nn.Embedding(item_vocab + 1, id_dim, padding_idx=0)
        self.user_id = nn.Embedding(user_vocab + 1, id_dim, padding_idx=0)
        self.item_id_adapter = nn.Linear(id_dim, retrieval_dim, bias=False)
        self.user_id_adapter = nn.Linear(id_dim, retrieval_dim, bias=False)
        nn.init.xavier_uniform_(self.item_id_adapter.weight)
        nn.init.xavier_uniform_(self.user_id_adapter.weight)
        self.attention = MaskedAttention(retrieval_dim)
        self.history_projection = nn.Sequential(
            nn.Linear(retrieval_dim, retrieval_dim),
            nn.GELU(),
            nn.Linear(retrieval_dim, retrieval_dim),
        )
        self.alpha_meta_raw = _alpha_parameter(alpha_init)
        self.alpha_profile_raw = _alpha_parameter(alpha_init)
        self.alpha_item_id_raw = _alpha_parameter(alpha_init)
        self.alpha_user_id_raw = _alpha_parameter(alpha_init)
        nn.init.normal_(self.item_id.weight[1:], std=0.02)
        nn.init.normal_(self.user_id.weight[1:], std=0.02)
        with torch.no_grad():
            self.item_id.weight[0].zero_()
            self.user_id.weight[0].zero_()
        if not self.use_side:
            self.item_meta.requires_grad_(False)
            self.user_profile.requires_grad_(False)
            self.alpha_meta_raw.requires_grad_(False)
            self.alpha_profile_raw.requires_grad_(False)
        if not self.use_id:
            self.item_id.requires_grad_(False)
            self.user_id.requires_grad_(False)
            self.item_id_adapter.requires_grad_(False)
            self.user_id_adapter.requires_grad_(False)
            self.alpha_item_id_raw.requires_grad_(False)
            self.alpha_user_id_raw.requires_grad_(False)

    @staticmethod
    def _alpha(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)

    def encode_item_values(self, content, item_id_row, categorical, numeric):
        value = self.content(content)
        if self.use_side:
            value = value + self._alpha(self.alpha_meta_raw) * self.item_meta(
                categorical, numeric
            )
        if self.use_id:
            identifier = self.item_id_adapter(self.item_id(item_id_row))
            value = value + self._alpha(self.alpha_item_id_raw) * identifier
        return F.normalize(value, dim=-1)

    def encode_item(self, batch: dict, prefix: str) -> torch.Tensor:
        return self.encode_item_values(
            batch[f"{prefix}_content"],
            batch[f"{prefix}_item_id_row"],
            batch[f"{prefix}_categorical"],
            batch[f"{prefix}_numeric"],
        )

    def query(self, batch: dict) -> torch.Tensor:
        history = self.encode_item(batch, "history")
        pooled = self.attention(history, batch["history_mask"])
        value = pooled + self.history_projection(pooled)
        if self.use_side:
            value = value + self._alpha(self.alpha_profile_raw) * self.user_profile(
                batch["user_categorical"], batch["user_numeric"]
            )
        if self.use_id:
            identifier = self.user_id_adapter(self.user_id(batch["user_id_row"]))
            value = value + self._alpha(self.alpha_user_id_raw) * identifier
        return F.normalize(value, dim=-1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(batch), self.encode_item(batch, "target")

    def alpha_values(self) -> dict[str, float]:
        return {
            name: float(self._alpha(getattr(self, f"alpha_{name}_raw")).detach())
            for name in ("meta", "profile", "item_id", "user_id")
        }

