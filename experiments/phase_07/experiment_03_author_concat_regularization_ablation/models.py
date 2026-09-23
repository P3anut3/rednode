"""Author-style direct-concat towers and structured ID dropout.

The frozen BGE vectors are external inputs.  This module contains no encoder
weights and never updates the cached item embeddings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.models import (
    CategoricalNumericEncoder,
    MaskedAttention,
)
from experiments.phase_07.experiment_02_concat_fusion_ablation.models import (
    ConcatFusionTwoTower,
)


E0 = "e0_old_direct_concat"


@dataclass(frozen=True)
class Variant:
    name: str
    content_input_dim: int
    id_dim: int
    id_init_std: float
    id_dropout: float

    @property
    def item_fusion_input_dim(self) -> int:
        return self.content_input_dim + 128 + self.id_dim + 1

    @property
    def user_fusion_input_dim(self) -> int:
        return 128 + 128 + self.id_dim + 1

    def to_dict(self) -> dict:
        return asdict(self)


VARIANTS: dict[str, Variant] = {
    "e1_content768_only": Variant("e1_content768_only", 768, 64, 0.02, 0.0),
    "e2_id32_only": Variant("e2_id32_only", 128, 32, 0.02, 0.0),
    "e3_id32_small_init": Variant("e3_id32_small_init", 128, 32, 0.01, 0.0),
    "e4_dropout_only_p30": Variant("e4_dropout_only_p30", 128, 64, 0.02, 0.3),
    "e5_content768_id32_small_init": Variant(
        "e5_content768_id32_small_init", 768, 32, 0.01, 0.0
    ),
    "e6_full_p30": Variant("e6_full_p30", 768, 32, 0.01, 0.3),
    "e7_full_p50": Variant("e7_full_p50", 768, 32, 0.01, 0.5),
}

E0_VARIANT = Variant(E0, 128, 64, 0.02, 0.0)

FULL_VARIANTS = (
    "e5_content768_id32_small_init",
    "e6_full_p30",
    "e7_full_p50",
)


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input(value)
        value = F.gelu(value)
        value = self.norm(value)
        value = self.dropout(value)
        return self.output(value)


class AuthorConcatTower(nn.Module):
    """Direct concat model with optional raw-BGE input and route dropout."""

    _ROLES = ("target", "history", "user")

    def __init__(
        self,
        variant: Variant,
        user_vocab: int,
        item_vocab: int,
        user_category_sizes: list[int],
        item_category_sizes: list[int],
        bge_dim: int = 768,
        output_dim: int = 128,
    ):
        super().__init__()
        self.variant = variant
        self.output_dim = output_dim
        self.id_dim = variant.id_dim
        self.content = (
            nn.Linear(bge_dim, output_dim, bias=False)
            if variant.content_input_dim == output_dim
            else nn.Identity()
        )
        self.item_meta = CategoricalNumericEncoder(
            item_category_sizes, 4, output_dim
        )
        self.user_profile = CategoricalNumericEncoder(
            user_category_sizes, 2, output_dim
        )
        self.item_id = nn.Embedding(item_vocab + 1, variant.id_dim, padding_idx=0)
        self.user_id = nn.Embedding(user_vocab + 1, variant.id_dim, padding_idx=0)
        self.attention = MaskedAttention(output_dim)
        self.history_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.item_fusion = FusionMLP(
            variant.item_fusion_input_dim, output_dim=output_dim
        )
        self.user_fusion = FusionMLP(
            variant.user_fusion_input_dim, output_dim=output_dim
        )
        nn.init.normal_(self.item_id.weight[1:], std=variant.id_init_std)
        nn.init.normal_(self.user_id.weight[1:], std=variant.id_init_std)
        with torch.no_grad():
            self.item_id.weight[0].zero_()
            self.user_id.weight[0].zero_()
        for role in self._ROLES:
            for statistic in ("units", "seen", "oov", "dropped_seen"):
                self.register_buffer(
                    f"_drop_{role}_{statistic}",
                    torch.zeros((), dtype=torch.long),
                    persistent=False,
                )

    def reset_dropout_statistics(self) -> None:
        for role in self._ROLES:
            for statistic in ("units", "seen", "oov", "dropped_seen"):
                getattr(self, f"_drop_{role}_{statistic}").zero_()

    def dropout_statistics(self) -> dict:
        output = {}
        for role in self._ROLES:
            values = {
                statistic: int(
                    getattr(self, f"_drop_{role}_{statistic}").detach().cpu()
                )
                for statistic in ("units", "seen", "oov", "dropped_seen")
            }
            values["seen_drop_ratio"] = (
                values["dropped_seen"] / values["seen"] if values["seen"] else 0.0
            )
            output[role] = values
        return output

    def _record_dropout(
        self, role: str | None, seen: torch.Tensor, keep: torch.Tensor
    ) -> None:
        if not self.training or role not in self._ROLES:
            return
        with torch.no_grad():
            seen_bool = seen.bool()
            getattr(self, f"_drop_{role}_units").add_(seen.numel())
            getattr(self, f"_drop_{role}_seen").add_(seen_bool.sum())
            getattr(self, f"_drop_{role}_oov").add_((~seen_bool).sum())
            getattr(self, f"_drop_{role}_dropped_seen").add_(
                (seen_bool & ~keep.bool()).sum()
            )

    def _id_branch(
        self,
        embedding: nn.Embedding,
        rows: torch.Tensor,
        role: str | None,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Structured dropout intentionally simulates the serving-time cold
        # contract: both the ID vector and the seen flag disappear together.
        raw_seen = rows.ne(0).unsqueeze(-1)
        values = embedding(rows)
        keep = torch.ones_like(raw_seen, dtype=torch.bool)
        if self.training and self.variant.id_dropout > 0:
            keep = torch.rand_like(raw_seen, dtype=torch.float32).ge(
                self.variant.id_dropout
            )
        if not enabled:
            keep = torch.zeros_like(keep)
        effective_seen = raw_seen & keep
        self._record_dropout(role, raw_seen, keep)
        values = values * effective_seen.to(values.dtype)
        return values, raw_seen, effective_seen, keep

    def item_components(
        self,
        content: torch.Tensor,
        item_id_row: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        *,
        role: str | None = None,
        item_id_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        content_value = self.content(content)
        metadata = self.item_meta(categorical, numeric)
        identifier, raw_seen, effective_seen, keep = self._id_branch(
            self.item_id, item_id_row, role, item_id_enabled
        )
        fusion_input = torch.cat(
            (
                content_value,
                metadata,
                identifier,
                effective_seen.to(content_value.dtype),
            ),
            -1,
        )
        value = self.item_fusion(fusion_input)
        return {
            "final": F.normalize(value, dim=-1),
            "content": content_value,
            "metadata": metadata,
            "identifier": identifier,
            "seen": effective_seen,
            "raw_seen": raw_seen,
            "keep": keep,
            "fusion_input": fusion_input,
        }

    def encode_item_values(
        self,
        content: torch.Tensor,
        item_id_row: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        *,
        role: str | None = None,
        item_id_enabled: bool = True,
    ) -> torch.Tensor:
        return self.item_components(
            content,
            item_id_row,
            categorical,
            numeric,
            role=role,
            item_id_enabled=item_id_enabled,
        )["final"]

    def encode_item(
        self,
        batch: dict,
        prefix: str,
        *,
        item_id_enabled: bool = True,
    ) -> torch.Tensor:
        role = prefix if prefix in {"target", "history"} else None
        return self.encode_item_values(
            batch[f"{prefix}_content"],
            batch[f"{prefix}_item_id_row"],
            batch[f"{prefix}_categorical"],
            batch[f"{prefix}_numeric"],
            role=role,
            item_id_enabled=item_id_enabled,
        )

    def query_components(
        self,
        batch: dict,
        *,
        item_id_enabled: bool = True,
        user_id_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        history_items = self.encode_item(
            batch, "history", item_id_enabled=item_id_enabled
        )
        pooled = self.attention(history_items, batch["history_mask"])
        history = pooled + self.history_mlp(pooled)
        profile = self.user_profile(
            batch["user_categorical"], batch["user_numeric"]
        )
        identifier, raw_seen, effective_seen, keep = self._id_branch(
            self.user_id, batch["user_id_row"], "user", user_id_enabled
        )
        fusion_input = torch.cat(
            (history, profile, identifier, effective_seen.to(history.dtype)), -1
        )
        value = self.user_fusion(fusion_input)
        return {
            "final": F.normalize(value, dim=-1),
            "history": history,
            "profile": profile,
            "identifier": identifier,
            "seen": effective_seen,
            "raw_seen": raw_seen,
            "keep": keep,
            "fusion_input": fusion_input,
        }

    def query(
        self,
        batch: dict,
        *,
        item_id_enabled: bool = True,
        user_id_enabled: bool = True,
    ) -> torch.Tensor:
        return self.query_components(
            batch,
            item_id_enabled=item_id_enabled,
            user_id_enabled=user_id_enabled,
        )["final"]

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # Each target is encoded exactly once and becomes one shared column in
        # the in-batch logits matrix.  No per-query dropout mask is generated.
        query = self.query(batch)
        target = self.encode_item(batch, "target")
        return query, target


def make_e0_compatible_model(
    user_vocab: int,
    item_vocab: int,
    user_category_sizes: list[int],
    item_category_sizes: list[int],
) -> ConcatFusionTwoTower:
    """Construct the exact Experiment-02 E0 architecture for strict loading."""
    return ConcatFusionTwoTower(
        "m1_direct_concat_id64",
        user_vocab,
        item_vocab,
        user_category_sizes,
        item_category_sizes,
        content_dim=768,
        output_dim=128,
        id_dim=64,
        fusion_hidden_dim=512,
        dropout=0.1,
        beta_init=0.05,
    )
