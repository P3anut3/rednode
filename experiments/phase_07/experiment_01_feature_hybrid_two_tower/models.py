"""Feature-enhanced ID and frozen-content hybrid two-tower models."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _alpha_parameter(initial: float) -> nn.Parameter:
    return nn.Parameter(
        torch.tensor(math.log(math.expm1(initial)), dtype=torch.float32)
    )


class MaskedAttention(nn.Module):
    def __init__(self, dim: int, attention_dim: int = 64):
        super().__init__()
        self.key = nn.Linear(dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (
            self.score(torch.tanh(self.key(values)))
            .squeeze(-1)
            .masked_fill(~mask, -1e4)
        )
        weights = torch.softmax(scores, dim=1) * mask.to(values.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        return (values * weights.unsqueeze(-1)).sum(1)


class CategoricalNumericEncoder(nn.Module):
    def __init__(
        self,
        category_sizes: list[int],
        numeric_dim: int,
        output_dim: int = 128,
        category_dim: int = 16,
        use_numeric: bool = True,
    ):
        super().__init__()
        self.use_numeric = use_numeric
        self.categories = nn.ModuleList(
            nn.Embedding(size, category_dim, padding_idx=0) for size in category_sizes
        )
        self.numeric = nn.Sequential(
            nn.Linear(numeric_dim, 32), nn.GELU(), nn.Linear(32, 32)
        )
        input_dim = len(category_sizes) * category_dim + (32 if use_numeric else 0)
        self.output = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )
        for embedding in self.categories:
            with torch.no_grad():
                embedding.weight[0].zero_()
        if not use_numeric:
            self.numeric.requires_grad_(False)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        values = [
            embedding(categorical[..., index])
            for index, embedding in enumerate(self.categories)
        ]
        if self.use_numeric:
            values.append(self.numeric(numeric))
        return self.output(torch.cat(values, dim=-1))


class FeatureItemEncoder(nn.Module):
    """Shared target/history encoder for Stage-B models."""

    def __init__(self, item_vocab: int, category_sizes: list[int], dim: int = 128):
        super().__init__()
        self.item_id = nn.Embedding(item_vocab + 1, dim, padding_idx=0)
        self.meta = CategoricalNumericEncoder(category_sizes, 4, dim)
        self.combine = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        nn.init.normal_(self.item_id.weight[1:], std=0.02)
        with torch.no_grad():
            self.item_id.weight[0].zero_()

    def forward(
        self,
        item_id_row: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        use_meta: bool,
    ) -> torch.Tensor:
        identifier = self.item_id(item_id_row)
        value = (
            self.combine(torch.cat((identifier, self.meta(categorical, numeric)), -1))
            if use_meta
            else identifier
        )
        return F.normalize(value, dim=-1)


class EnhancedIdTower(nn.Module):
    """Matched ID-B0/B1a/B1b/B2/B3 with tied target/history item encoder."""

    def __init__(
        self,
        variant: str,
        user_vocab: int,
        item_vocab: int,
        user_category_sizes: list[int],
        item_category_sizes: list[int],
        dim: int = 128,
    ):
        super().__init__()
        if variant not in {
            "id_b0_user_history_control",
            "id_b1a_stable_profile",
            "id_b1b_full_profile",
            "id_b2_item_features",
            "id_b3_all",
        }:
            raise ValueError(variant)
        self.variant = variant
        self.use_profile = variant in {
            "id_b1a_stable_profile",
            "id_b1b_full_profile",
            "id_b3_all",
        }
        self.use_profile_numeric = variant in {"id_b1b_full_profile", "id_b3_all"}
        self.use_item_features = variant in {"id_b2_item_features", "id_b3_all"}
        self.user_id = nn.Embedding(user_vocab + 1, dim, padding_idx=0)
        self.item = FeatureItemEncoder(item_vocab, item_category_sizes, dim)
        self.profile = CategoricalNumericEncoder(
            user_category_sizes, 2, dim, use_numeric=self.use_profile_numeric
        )
        self.attention = MaskedAttention(dim)
        inputs = 3 if self.use_profile else 2
        self.user_combine = nn.Sequential(
            nn.Linear(inputs * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        nn.init.normal_(self.user_id.weight[1:], std=0.02)
        with torch.no_grad():
            self.user_id.weight[0].zero_()
        if not self.use_profile:
            self.profile.requires_grad_(False)
        if not self.use_item_features:
            self.item.meta.requires_grad_(False)
            self.item.combine.requires_grad_(False)

    def encode_item(self, batch: dict, prefix: str) -> torch.Tensor:
        return self.item(
            batch[f"{prefix}_item_id_row"],
            batch[f"{prefix}_categorical"],
            batch[f"{prefix}_numeric"],
            self.use_item_features,
        )

    def query(self, batch: dict) -> torch.Tensor:
        history = self.encode_item(batch, "history")
        pooled = self.attention(history, batch["history_mask"])
        parts = [self.user_id(batch["user_id_row"]), pooled]
        if self.use_profile:
            parts.append(self.profile(batch["user_categorical"], batch["user_numeric"]))
        return F.normalize(self.user_combine(torch.cat(parts, -1)), dim=-1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(batch), self.encode_item(batch, "target")


class HybridTower(nn.Module):
    """H0/H1/H2/H3 single-index hybrid tower with frozen BGE inputs."""

    def __init__(
        self,
        variant: str,
        user_vocab: int,
        item_vocab: int,
        user_category_sizes: list[int],
        item_category_sizes: list[int],
        content_dim: int = 768,
        dim: int = 128,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        if variant not in {
            "h0_content_control",
            "h1_side_features",
            "h2_side_features_id",
            "h3_anonymous_dense",
        }:
            raise ValueError(variant)
        self.variant = variant
        self.use_side = variant != "h0_content_control"
        self.use_id = variant in {"h2_side_features_id", "h3_anonymous_dense"}
        self.content = nn.Linear(content_dim, dim, bias=False)
        self.item_meta = CategoricalNumericEncoder(item_category_sizes, 4, dim)
        self.user_profile = CategoricalNumericEncoder(user_category_sizes, 2, dim)
        self.item_id = nn.Embedding(item_vocab + 1, dim, padding_idx=0)
        self.user_id = nn.Embedding(user_vocab + 1, dim, padding_idx=0)
        self.attention = MaskedAttention(dim)
        self.history_projection = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.anonymous_dense = nn.Sequential(
            nn.Linear(40, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, dim)
        )
        self.alpha_meta_raw = _alpha_parameter(alpha_init)
        self.alpha_profile_raw = _alpha_parameter(alpha_init)
        self.alpha_item_id_raw = _alpha_parameter(alpha_init)
        self.alpha_user_id_raw = _alpha_parameter(alpha_init)
        self.alpha_dense_raw = _alpha_parameter(alpha_init)
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
            self.alpha_item_id_raw.requires_grad_(False)
            self.alpha_user_id_raw.requires_grad_(False)
        if variant != "h3_anonymous_dense":
            self.anonymous_dense.requires_grad_(False)
            self.alpha_dense_raw.requires_grad_(False)

    @staticmethod
    def _alpha(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)

    def encode_item_values(
        self,
        content: torch.Tensor,
        item_id_row: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        value = self.content(content)
        if self.use_side:
            value = value + self._alpha(self.alpha_meta_raw) * self.item_meta(
                categorical, numeric
            )
        if self.use_id:
            # OOV/cold item uses padding row 0, which is fixed to zero.
            value = value + self._alpha(self.alpha_item_id_raw) * self.item_id(
                item_id_row
            )
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
            value = value + self._alpha(self.alpha_user_id_raw) * self.user_id(
                batch["user_id_row"]
            )
        if self.variant == "h3_anonymous_dense":
            value = value + self._alpha(self.alpha_dense_raw) * self.anonymous_dense(
                batch["user_dense"]
            )
        return F.normalize(value, dim=-1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(batch), self.encode_item(batch, "target")

    def alpha_values(self) -> dict[str, float]:
        return {
            name: float(self._alpha(getattr(self, f"alpha_{name}_raw")).detach())
            for name in ("meta", "profile", "item_id", "user_id", "dense")
        }
