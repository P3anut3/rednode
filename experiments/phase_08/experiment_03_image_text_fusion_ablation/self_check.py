#!/usr/bin/env python3
"""Synthetic, CPU-only contract checks; never reads corpus or test labels."""

from __future__ import annotations

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_08.experiment_02_image_hybrid_recall.models import ImageHybridH2256
from experiments.phase_08.experiment_03_image_text_fusion_ablation.models import (
    VARIANTS, FusionTower, copy_a0_start,
)


def constructors():
    return dict(user_vocab=6, item_vocab=8, user_category_sizes=[4, 4, 4],
                item_category_sizes=[4, 4, 4, 4], retrieval_dim=256,
                id_dim=128, alpha_image_init=0.05)


def batch():
    torch.manual_seed(7)
    value = {
        "history_content": torch.randn(2, 3, 768),
        "history_image": torch.randn(2, 3, 768),
        "history_image_available": torch.tensor([[True, False, True], [False, True, False]]),
        "history_item_id_row": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "history_categorical": torch.randint(0, 4, (2, 3, 4)),
        "history_numeric": torch.randn(2, 3, 4),
        "history_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "user_categorical": torch.randint(0, 4, (2, 3)),
        "user_numeric": torch.randn(2, 2),
        "user_id_row": torch.tensor([1, 0]),
        "target_content": torch.randn(2, 768),
        "target_image": torch.randn(2, 768),
        "target_image_available": torch.tensor([True, False]),
        "target_item_id_row": torch.tensor([2, 0]),
        "target_categorical": torch.randint(0, 4, (2, 4)),
        "target_numeric": torch.randn(2, 4),
    }
    return value


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    a0 = ImageHybridH2256(**constructors()).eval()
    values = batch()
    with torch.no_grad():
        base_item = a0.encode_item(values, "target")
        base_query = a0.query(values)
    for variant in VARIANTS:
        model = FusionTower(variant=variant, **constructors()).eval()
        copy_a0_start(model, a0)
        with torch.no_grad():
            candidate = model.encode_item(values, "target")
            query = model.query(values)
            assert torch.allclose(candidate, base_item, atol=1e-6), variant
            assert torch.allclose(query, base_query, atol=1e-6), variant
            assert torch.allclose(model.image_gate[-1].weight, torch.zeros_like(model.image_gate[-1].weight))
            if model.interaction is not None:
                assert torch.count_nonzero(model.interaction[-1].weight) == 0
            # A missing image must have exactly the same value regardless of
            # its stored image tensor, even though its normal ID/meta terms stay.
            changed = dict(values)
            changed["target_image"] = values["target_image"].clone()
            changed["target_image"][1] = torch.randn(768) * 100
            assert torch.allclose(model.encode_item(changed, "target")[1], candidate[1], atol=1e-7)
            assert torch.allclose(candidate.norm(dim=-1), torch.ones(2), atol=1e-6)
        # The zero-final-layer correction is not a dead branch: its output
        # layer learns on step one, earlier layers receive gradients by step two.
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            item = model.encode_item(values, "target")
            query = model.query(values)
            loss = -(item * query).sum(-1).mean()
            loss.backward()
            assert model.image_projection.weight.grad is not None
            assert model.image_gate[-1].weight.grad is not None
            if model.interaction is not None:
                assert model.interaction[-1].weight.grad is not None
                if step == 1:
                    assert torch.count_nonzero(model.interaction[0].weight.grad) > 0
            optimizer.step()
    print("phase8-03 synthetic self-check passed")


if __name__ == "__main__":
    main()
