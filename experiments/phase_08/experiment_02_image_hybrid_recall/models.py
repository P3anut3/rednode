"""H2-256 with exactly one frozen-image residual branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.models import _alpha_parameter
from experiments.phase_07.experiment_04_h2_retrieval_dimension_ablation.models import HybridTower256


class ImageHybridH2256(HybridTower256):
    def __init__(self, *args, image_dim: int = 768, alpha_image_init: float = 0.05, **kwargs):
        super().__init__("h2_256", *args, **kwargs)
        self.image_projection = nn.Linear(image_dim, self.retrieval_dim, bias=False)
        self.alpha_image_raw = _alpha_parameter(alpha_image_init)

    def encode_item_values(self, content, item_id_row, categorical, numeric, image, image_available):
        # Keep the H2 terms in their original order, and normalize only once.
        value = self.content(content)
        value = value + self._alpha(self.alpha_meta_raw) * self.item_meta(categorical, numeric)
        value = value + self._alpha(self.alpha_item_id_raw) * self.item_id_adapter(self.item_id(item_id_row))
        image_delta = self.image_projection(image)
        value = value + image_available.to(value.dtype).unsqueeze(-1) * self._alpha(self.alpha_image_raw) * image_delta
        return F.normalize(value, dim=-1)

    def encode_item(self, batch: dict, prefix: str) -> torch.Tensor:
        return self.encode_item_values(
            batch[f"{prefix}_content"],
            batch[f"{prefix}_item_id_row"],
            batch[f"{prefix}_categorical"],
            batch[f"{prefix}_numeric"],
            batch[f"{prefix}_image"],
            batch[f"{prefix}_image_available"],
        )

    def alpha_values(self) -> dict[str, float]:
        return {**super().alpha_values(), "image": float(self._alpha(self.alpha_image_raw).detach())}
