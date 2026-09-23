"""ID64 direct-concat and residual-concat hybrid two-tower models."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.models import (
    CategoricalNumericEncoder,
    MaskedAttention,
)


def _positive_parameter(initial: float) -> nn.Parameter:
    return nn.Parameter(torch.tensor(math.log(math.expm1(initial)), dtype=torch.float32))


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, zero_last: bool = False):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if zero_last:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input(value)
        value = F.gelu(value)
        value = self.norm(value)
        value = self.dropout(value)
        return self.output(value)


class ConcatFusionTwoTower(nn.Module):
    """Shared item tower used for candidates, targets, negatives and history."""

    def __init__(
        self,
        variant: str,
        user_vocab: int,
        item_vocab: int,
        user_category_sizes: list[int],
        item_category_sizes: list[int],
        content_dim: int = 768,
        output_dim: int = 128,
        id_dim: int = 64,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        beta_init: float = 0.05,
    ):
        super().__init__()
        if variant not in {"m1_direct_concat_id64", "m2_concat_residual_id64"}:
            raise ValueError(variant)
        self.variant = variant
        self.residual = variant == "m2_concat_residual_id64"
        self.output_dim = output_dim
        self.id_dim = id_dim

        self.content = nn.Linear(content_dim, output_dim, bias=False)
        self.item_meta = CategoricalNumericEncoder(item_category_sizes, 4, output_dim)
        self.user_profile = CategoricalNumericEncoder(user_category_sizes, 2, output_dim)
        self.item_id = nn.Embedding(item_vocab + 1, id_dim, padding_idx=0)
        self.user_id = nn.Embedding(user_vocab + 1, id_dim, padding_idx=0)
        self.attention = MaskedAttention(output_dim)
        self.history_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim)
        )
        fusion_input = output_dim * 2 + id_dim + 1
        self.item_fusion = FusionMLP(fusion_input, fusion_hidden_dim, output_dim, dropout, zero_last=self.residual)
        self.user_fusion = FusionMLP(fusion_input, fusion_hidden_dim, output_dim, dropout, zero_last=self.residual)
        self.beta_item_raw = _positive_parameter(beta_init)
        self.beta_user_raw = _positive_parameter(beta_init)

        nn.init.normal_(self.item_id.weight[1:], std=0.02)
        nn.init.normal_(self.user_id.weight[1:], std=0.02)
        with torch.no_grad():
            self.item_id.weight[0].zero_()
            self.user_id.weight[0].zero_()
        if not self.residual:
            self.beta_item_raw.requires_grad_(False)
            self.beta_user_raw.requires_grad_(False)

    @staticmethod
    def _beta(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)

    @staticmethod
    def _seen(row: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return row.ne(0).to(dtype).unsqueeze(-1)

    def item_components(
        self,
        content: torch.Tensor,
        item_id_row: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        content_value = self.content(content)
        metadata = self.item_meta(categorical, numeric)
        seen = self._seen(item_id_row, content_value.dtype)
        identifier = self.item_id(item_id_row) * seen
        delta = self.item_fusion(torch.cat((content_value, metadata, identifier, seen), -1))
        value = content_value + self._beta(self.beta_item_raw) * delta if self.residual else delta
        return {
            "final": F.normalize(value, dim=-1),
            "base": content_value,
            "delta": delta,
            "seen": seen,
        }

    def encode_item_values(self, content, item_id_row, categorical, numeric):
        return self.item_components(content, item_id_row, categorical, numeric)["final"]

    def encode_item(self, batch: dict, prefix: str) -> torch.Tensor:
        return self.encode_item_values(
            batch[f"{prefix}_content"],
            batch[f"{prefix}_item_id_row"],
            batch[f"{prefix}_categorical"],
            batch[f"{prefix}_numeric"],
        )

    def user_components(self, batch: dict) -> dict[str, torch.Tensor]:
        history_items = self.encode_item(batch, "history")
        pooled = self.attention(history_items, batch["history_mask"])
        history = pooled + self.history_mlp(pooled)
        profile = self.user_profile(batch["user_categorical"], batch["user_numeric"])
        seen = self._seen(batch["user_id_row"], history.dtype)
        identifier = self.user_id(batch["user_id_row"]) * seen
        delta = self.user_fusion(torch.cat((history, profile, identifier, seen), -1))
        value = history + self._beta(self.beta_user_raw) * delta if self.residual else delta
        return {
            "final": F.normalize(value, dim=-1),
            "base": history,
            "delta": delta,
            "seen": seen,
        }

    def query(self, batch: dict) -> torch.Tensor:
        return self.user_components(batch)["final"]

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(batch), self.encode_item(batch, "target")

    def beta_values(self) -> dict[str, float]:
        return {
            "item": float(self._beta(self.beta_item_raw).detach()),
            "user": float(self._beta(self.beta_user_raw).detach()),
        }


class AddId64ControlTower(nn.Module):
    """Phase-7 H2 additive structure with only the ID width reduced to 64.

    The bias-free ID projections are required to add a 64-dimensional ID
    representation to the fixed 128-dimensional retrieval space. Padding row
    zero therefore remains exactly zero after projection.
    """

    residual = False
    additive = True

    def __init__(
        self,
        user_vocab: int,
        item_vocab: int,
        user_category_sizes: list[int],
        item_category_sizes: list[int],
        content_dim: int = 768,
        output_dim: int = 128,
        id_dim: int = 64,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.variant = "add_id64_control"
        self.output_dim = output_dim
        self.id_dim = id_dim
        self.content = nn.Linear(content_dim, output_dim, bias=False)
        self.item_meta = CategoricalNumericEncoder(
            item_category_sizes, 4, output_dim
        )
        self.user_profile = CategoricalNumericEncoder(
            user_category_sizes, 2, output_dim
        )
        self.item_id = nn.Embedding(item_vocab + 1, id_dim, padding_idx=0)
        self.user_id = nn.Embedding(user_vocab + 1, id_dim, padding_idx=0)
        self.item_id_projection = nn.Linear(id_dim, output_dim, bias=False)
        self.user_id_projection = nn.Linear(id_dim, output_dim, bias=False)
        self.attention = MaskedAttention(output_dim)
        self.history_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.alpha_meta_raw = _positive_parameter(alpha_init)
        self.alpha_profile_raw = _positive_parameter(alpha_init)
        self.alpha_item_id_raw = _positive_parameter(alpha_init)
        self.alpha_user_id_raw = _positive_parameter(alpha_init)
        nn.init.normal_(self.item_id.weight[1:], std=0.02)
        nn.init.normal_(self.user_id.weight[1:], std=0.02)
        with torch.no_grad():
            self.item_id.weight[0].zero_()
            self.user_id.weight[0].zero_()

    @staticmethod
    def _alpha(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)

    def encode_item_values(self, content, item_id_row, categorical, numeric):
        value = self.content(content)
        value = value + self._alpha(self.alpha_meta_raw) * self.item_meta(
            categorical, numeric
        )
        identifier = self.item_id_projection(self.item_id(item_id_row))
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
        history_items = self.encode_item(batch, "history")
        pooled = self.attention(history_items, batch["history_mask"])
        value = pooled + self.history_mlp(pooled)
        value = value + self._alpha(self.alpha_profile_raw) * self.user_profile(
            batch["user_categorical"], batch["user_numeric"]
        )
        identifier = self.user_id_projection(self.user_id(batch["user_id_row"]))
        value = value + self._alpha(self.alpha_user_id_raw) * identifier
        return F.normalize(value, dim=-1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(batch), self.encode_item(batch, "target")
