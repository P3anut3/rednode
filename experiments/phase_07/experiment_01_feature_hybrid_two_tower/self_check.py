#!/usr/bin/env python3
"""Synthetic-only shape, mask, loss, and retrieval checks. Reads no Qilin data."""

from __future__ import annotations

import numpy as np
import torch

from .losses import logq_inbatch_loss, tfidf_hard_loss
from .models import EnhancedIdTower, HybridTower
from .dataset import Phase7Dataset
from .features import _category


def batch(batch_size=8, history=20, content_dim=768):
    values = {
        "history_item_id_row": torch.randint(0, 50, (batch_size, history)),
        "history_categorical": torch.randint(0, 3, (batch_size, history, 4)),
        "history_numeric": torch.randn(batch_size, history, 4),
        "history_content": torch.randn(batch_size, history, content_dim),
        "history_mask": torch.ones(batch_size, history, dtype=torch.bool),
        "history_note_ids": torch.arange(batch_size * history).reshape(
            batch_size, history
        ),
        "target_item_id_row": torch.randint(1, 50, (batch_size,)),
        "target_categorical": torch.randint(0, 3, (batch_size, 4)),
        "target_numeric": torch.randn(batch_size, 4),
        "target_content": torch.randn(batch_size, content_dim),
        "target_note_id": torch.arange(10_000, 10_000 + batch_size),
        "user_id": torch.arange(batch_size),
        "user_id_row": torch.arange(1, batch_size + 1),
        "user_categorical": torch.randint(0, 3, (batch_size, 3)),
        "user_numeric": torch.randn(batch_size, 2),
        "user_dense": torch.randn(batch_size, 40),
        "same_request_positive_ids": [tuple() for _ in range(batch_size)],
        "negative_item_id_row": torch.randint(0, 50, (batch_size, 2)),
        "negative_categorical": torch.randint(0, 3, (batch_size, 2, 4)),
        "negative_numeric": torch.randn(batch_size, 2, 4),
        "negative_content": torch.randn(batch_size, 2, content_dim),
        "negative_mask": torch.ones(batch_size, 2, dtype=torch.bool),
    }
    values["history_item_id_row"][:, -1] = 0
    values["history_mask"][:, -1] = False
    return values


def main():
    torch.manual_seed(42)
    value = batch()
    value["user_id"][1] = value["user_id"][0]
    assert all(
        _category(token) == "<MISSING>"
        for token in (None, "", "nan", "None", "NULL", "<NA>")
    )
    sampled = Phase7Dataset._rank_sample(
        [101, 102, 103], [15, 16, 17], np.random.default_rng(42)
    )
    assert len(sampled) == 2 and len(set(sampled)) == 2
    for name in (
        "id_b0_user_history_control",
        "id_b1a_stable_profile",
        "id_b1b_full_profile",
        "id_b2_item_features",
        "id_b3_all",
    ):
        model = EnhancedIdTower(name, 20, 50, [4, 4, 4], [4, 4, 4, 4])
        query, target = model(value)
        assert query.shape == target.shape == (8, 128)
        loss, details = logq_inbatch_loss(query, target, value, torch.zeros(8))
        loss.backward()
        assert torch.isfinite(loss)
        assert details["masked_same_user"] >= 2
        assert all(
            module.weight.grad is None
            or torch.count_nonzero(module.weight.grad[0]) == 0
            for module in model.modules()
            if isinstance(module, torch.nn.Embedding) and module.padding_idx == 0
        )
    for name in (
        "h0_content_control",
        "h1_side_features",
        "h2_side_features_id",
        "h3_anonymous_dense",
    ):
        model = HybridTower(name, 20, 50, [4, 4, 4], [4, 4, 4, 4])
        query, target = model(value)
        negative = model.encode_item(value, "negative")
        assert query.shape == target.shape == (8, 128) and negative.shape == (8, 2, 128)
        loss, _ = tfidf_hard_loss(
            query, target, negative, value["negative_mask"], value
        )
        loss.backward()
        assert torch.isfinite(loss)
        cold = model.encode_item_values(
            value["target_content"],
            torch.zeros(8, dtype=torch.long),
            value["target_categorical"],
            value["target_numeric"],
        )
        assert torch.isfinite(cold).all()
    print("Phase 7 synthetic self-check: PASS (no project data read, no training run)")


if __name__ == "__main__":
    main()
