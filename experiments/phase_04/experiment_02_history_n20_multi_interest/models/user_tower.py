"""Phase 4.1 history-length and collapse-controlled user towers."""

from __future__ import annotations

from torch import nn

from experiments.phase_04.experiment_01_learnable_user_tower_n10.models.user_tower import (
    Phase4MultiInterest,
    Phase4SingleAttention,
)


def phase4_1_make_model(name: str, dim: int, k: int = 4) -> nn.Module:
    """Build a Phase 4.1 tower while keeping Phase 4 checkpoints untouched."""
    if name == "single_attention_n20":
        return Phase4SingleAttention(dim)
    if name == "multi_interest_n20_div":
        return Phase4MultiInterest(dim, k=k)
    raise ValueError(name)
