#!/usr/bin/env python3
"""Synthetic-only checks for Experiment 03.

This script does not read Qilin data and does not start training.  It is kept
as an explicit post-review command rather than being executed on import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.losses import (  # noqa: E402
    tfidf_hard_loss,
)
from experiments.phase_07.experiment_03_author_concat_regularization_ablation.models import (  # noqa: E402
    E0_VARIANT,
    VARIANTS,
    AuthorConcatTower,
    Variant,
)


def model(variant: Variant) -> AuthorConcatTower:
    return AuthorConcatTower(
        variant,
        user_vocab=11,
        item_vocab=17,
        user_category_sizes=[4, 5, 6],
        item_category_sizes=[4, 5, 6, 3],
    )


def batch(size: int = 8, history: int = 20, negatives: int = 2) -> dict:
    torch.manual_seed(42)
    history_rows = torch.randint(1, 18, (size, history))
    target_rows = torch.randint(1, 18, (size,))
    negative_rows = torch.randint(1, 18, (size, negatives))
    return {
        "user_id": torch.arange(size),
        "user_id_row": torch.randint(1, 12, (size,)),
        "user_categorical": torch.stack(
            (
                torch.randint(0, 4, (size,)),
                torch.randint(0, 5, (size,)),
                torch.randint(0, 6, (size,)),
            ), -1,
        ),
        "user_numeric": torch.randn(size, 2),
        "history_note_ids": history_rows.clone(),
        "history_item_id_row": history_rows,
        "history_content": torch.randn(size, history, 768),
        "history_categorical": torch.stack(
            tuple(torch.randint(0, limit, (size, history)) for limit in (4, 5, 6, 3)), -1
        ),
        "history_numeric": torch.randn(size, history, 4),
        "history_mask": torch.ones(size, history, dtype=torch.bool),
        "target_note_id": target_rows.clone(),
        "target_item_id_row": target_rows,
        "target_content": torch.randn(size, 768),
        "target_categorical": torch.stack(
            tuple(torch.randint(0, limit, (size,)) for limit in (4, 5, 6, 3)), -1
        ),
        "target_numeric": torch.randn(size, 4),
        "negative_note_ids": negative_rows.clone(),
        "negative_item_id_row": negative_rows,
        "negative_content": torch.randn(size, negatives, 768),
        "negative_categorical": torch.stack(
            tuple(torch.randint(0, limit, (size, negatives)) for limit in (4, 5, 6, 3)), -1
        ),
        "negative_numeric": torch.randn(size, negatives, 4),
        "negative_mask": torch.ones(size, negatives, dtype=torch.bool),
    }


def check_shapes_and_initialization() -> None:
    for name, variant in VARIANTS.items():
        torch.manual_seed(42)
        tower = model(variant)
        values = batch()
        tower.eval()
        query, item = tower(values)
        assert query.shape == item.shape == (8, 128), name
        assert tower.item_fusion.input.in_features == variant.item_fusion_input_dim
        assert tower.user_fusion.input.in_features == variant.user_fusion_input_dim
        assert tower.item_id.embedding_dim == tower.user_id.embedding_dim == variant.id_dim
        assert abs(float(tower.item_id.weight[1:].std()) - variant.id_init_std) < 0.002
        assert torch.count_nonzero(tower.item_id.weight[0]) == 0
        assert torch.count_nonzero(tower.user_id.weight[0]) == 0
        assert torch.allclose(query.norm(dim=-1), torch.ones(8), atol=1e-5)
        assert torch.allclose(item.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert E0_VARIANT.item_fusion_input_dim == 321
    assert VARIANTS["e1_content768_only"].item_fusion_input_dim == 961
    assert VARIANTS["e5_content768_id32_small_init"].item_fusion_input_dim == 929
    assert VARIANTS["e5_content768_id32_small_init"].user_fusion_input_dim == 289


def check_structured_dropout() -> None:
    forced = Variant("forced_drop", 768, 32, 0.01, 1.0)
    tower = model(forced)
    values = batch()
    tower.train()
    target_parts = tower.item_components(
        values["target_content"], values["target_item_id_row"],
        values["target_categorical"], values["target_numeric"], role="target",
    )
    history_parts = tower.item_components(
        values["history_content"], values["history_item_id_row"],
        values["history_categorical"], values["history_numeric"], role="history",
    )
    user_parts = tower.query_components(values)
    for parts in (target_parts, history_parts, user_parts):
        assert parts["keep"].shape == (*parts["identifier"].shape[:-1], 1)
        assert torch.count_nonzero(parts["identifier"]) == 0
        assert torch.count_nonzero(parts["keep"]) == 0
        assert torch.count_nonzero(parts["seen"]) == 0
        assert torch.all(parts["raw_seen"])
    stats = tower.dropout_statistics()
    assert all(stats[role]["seen_drop_ratio"] == 1.0 for role in ("target", "history", "user"))
    (target_parts["final"].sum() + user_parts["final"].sum()).backward()
    assert tower.item_id.weight.grad is None or torch.count_nonzero(tower.item_id.weight.grad) == 0
    assert tower.user_id.weight.grad is None or torch.count_nonzero(tower.user_id.weight.grad) == 0

    tower.eval()
    target_eval = tower.item_components(
        values["target_content"], values["target_item_id_row"],
        values["target_categorical"], values["target_numeric"], role="target",
    )
    assert torch.all(target_eval["keep"])
    assert torch.equal(target_eval["seen"], target_eval["raw_seen"])
    expected = tower.item_id(values["target_item_id_row"])
    assert torch.equal(target_eval["identifier"], expected)


def check_gradients_and_id_off() -> None:
    tower = model(VARIANTS["e6_full_p30"])
    values = batch()
    # Deterministically exercise the unchanged in-batch false-negative masks:
    # duplicate target, same user with another target, and target in history.
    values["target_note_id"] = torch.arange(101, 109)
    values["target_note_id"][2] = values["target_note_id"][0]
    values["user_id"][1] = values["user_id"][0]
    values["history_note_ids"][0, 0] = values["target_note_id"][3]
    tower.train()
    query, target = tower(values)
    negative = tower.encode_item(values, "negative")
    loss, details = tfidf_hard_loss(
        query, target, negative, values["negative_mask"], values, 0.05, 0.5
    )
    assert torch.isfinite(loss)
    assert details["tfidf_pair_loss"] >= 0
    assert details["masked_duplicate"] > 0
    assert details["masked_same_user"] > 0
    assert details["masked_in_history"] > 0
    loss.backward()
    for prefix in (
        "item_meta", "user_profile", "item_id", "user_id", "attention",
        "history_mlp", "item_fusion", "user_fusion",
    ):
        assert any(
            name.startswith(prefix) and parameter.grad is not None
            for name, parameter in tower.named_parameters()
        ), prefix
    assert not any("bge" in name.lower() for name, _ in tower.named_parameters())
    assert torch.count_nonzero(tower.item_id.weight.grad[0]) == 0
    assert torch.count_nonzero(tower.user_id.weight.grad[0]) == 0

    tower.eval()
    with torch.inference_mode():
        cold_item = dict(values)
        cold_item["target_item_id_row"] = torch.zeros_like(values["target_item_id_row"])
        before = tower.encode_item(cold_item, "target")
        saved = tower.item_id.weight[1:].clone()
        tower.item_id.weight[1:].normal_()
        after = tower.encode_item(cold_item, "target")
        tower.item_id.weight[1:].copy_(saved)
        assert torch.equal(before, after)

        cold_user = dict(values)
        cold_user["user_id_row"] = torch.zeros_like(values["user_id_row"])
        before = tower.query(cold_user)
        saved = tower.user_id.weight[1:].clone()
        tower.user_id.weight[1:].normal_()
        after = tower.query(cold_user)
        tower.user_id.weight[1:].copy_(saved)
        assert torch.equal(before, after)


def main() -> None:
    check_shapes_and_initialization()
    check_structured_dropout()
    check_gradients_and_id_off()
    print("Experiment 03 synthetic self-check passed")


if __name__ == "__main__":
    main()
