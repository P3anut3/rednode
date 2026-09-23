"""Synthetic-only structural checks. This module never reads Qilin data."""

from __future__ import annotations

import torch

from .models import ConcatFusionTwoTower


def batch(batch_size: int = 4, history_n: int = 20) -> dict:
    output = {
        "history_content": torch.randn(batch_size, history_n, 768),
        "history_item_id_row": torch.randint(0, 20, (batch_size, history_n)),
        "history_categorical": torch.randint(0, 3, (batch_size, history_n, 4)),
        "history_numeric": torch.randn(batch_size, history_n, 4),
        "history_mask": torch.ones(batch_size, history_n, dtype=torch.bool),
        "target_content": torch.randn(batch_size, 768),
        "target_item_id_row": torch.tensor([0, 1, 2, 3]),
        "target_categorical": torch.randint(0, 3, (batch_size, 4)),
        "target_numeric": torch.randn(batch_size, 4),
        "user_id_row": torch.tensor([0, 1, 2, 3]),
        "user_categorical": torch.randint(0, 3, (batch_size, 3)),
        "user_numeric": torch.randn(batch_size, 2),
    }
    return output


def main() -> None:
    for variant in ("m1_direct_concat_id64", "m2_concat_residual_id64"):
        model = ConcatFusionTwoTower(variant, 10, 20, [4, 4, 4], [4, 4, 4, 4])
        value = batch()
        query, item = model(value)
        assert query.shape == item.shape == (4, 128)
        assert model.user_id.embedding_dim == model.item_id.embedding_dim == 64
        assert torch.allclose(query.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose(item.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.count_nonzero(model.user_id.weight[0]) == 0
        assert torch.count_nonzero(model.item_id.weight[0]) == 0
        if model.residual:
            parts = model.item_components(
                value["target_content"], value["target_item_id_row"],
                value["target_categorical"], value["target_numeric"],
            )
            expected = torch.nn.functional.normalize(parts["base"], dim=-1)
            assert torch.allclose(parts["final"], expected, atol=1e-6)
            assert model.beta_item_raw.requires_grad and model.beta_user_raw.requires_grad
    print("Phase 7 Experiment 02 synthetic self-check: PASS (no project data read)")


if __name__ == "__main__":
    main()
