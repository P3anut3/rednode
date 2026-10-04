"""Matched-start gated image/text fusion over the Phase 8-02 A0 tower."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase_08.experiment_02_image_hybrid_recall.models import ImageHybridH2256


VARIANTS = ("f1_gate", "f2_projected_concat", "f3_raw_concat")


class FusionTower(ImageHybridH2256):
    def __init__(self, *args, variant: str, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.image_gate = nn.Sequential(
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.image_gate[-1].weight)
        nn.init.zeros_(self.image_gate[-1].bias)
        if variant == "f1_gate":
            self.interaction = None
        else:
            input_dim = 512 if variant == "f2_projected_concat" else 1536
            self.interaction = nn.Sequential(
                nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512),
                nn.Dropout(0.1), nn.Linear(512, 256),
            )
            nn.init.zeros_(self.interaction[-1].weight)
            nn.init.zeros_(self.interaction[-1].bias)
        # Inference-only mechanism interventions. Never enabled for training.
        self.disable_history_image = False
        self.disable_candidate_image = False
        self.zero_image_values = False

    def encode_item_values(
        self, content, item_id_row, categorical, numeric, image, image_available,
        *, image_off: bool | None = None,
    ):
        if image_off is None:
            image_off = self.disable_candidate_image
        if image_off:
            image_available = torch.zeros_like(image_available)
        if self.zero_image_values:
            image = torch.zeros_like(image)  # keep availability to test image semantics
        text = self.content(content)
        projected_image = self.image_projection(image)
        gate_input = torch.cat((text, projected_image), dim=-1)
        gate = 2.0 * torch.sigmoid(self.image_gate(gate_input))
        if self.interaction is None:
            correction = torch.zeros_like(projected_image)
        elif self.variant == "f2_projected_concat":
            correction = self.interaction(gate_input)
        else:
            correction = self.interaction(torch.cat((content, image), dim=-1))
        image_term = (
            image_available.to(text.dtype).unsqueeze(-1)
            * self._alpha(self.alpha_image_raw)
            * gate
            * (projected_image + correction)
        )
        value = text + image_term
        value = value + self._alpha(self.alpha_meta_raw) * self.item_meta(categorical, numeric)
        value = value + self._alpha(self.alpha_item_id_raw) * self.item_id_adapter(self.item_id(item_id_row))
        return F.normalize(value, dim=-1)

    def encode_item(self, batch: dict, prefix: str) -> torch.Tensor:
        return self.encode_item_values(
            batch[f"{prefix}_content"], batch[f"{prefix}_item_id_row"],
            batch[f"{prefix}_categorical"], batch[f"{prefix}_numeric"],
            batch[f"{prefix}_image"], batch[f"{prefix}_image_available"],
            image_off=self.disable_history_image and prefix == "history",
        )

    def query(self, batch: dict) -> torch.Tensor:
        return super().query(batch)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(batch), self.encode_item(batch, "target")

    def alpha_values(self) -> dict[str, float]:
        return super().alpha_values()


def copy_a0_start(fusion: FusionTower, a0: ImageHybridH2256) -> None:
    """Copy every shared tensor explicitly; RNG consumption is not a control."""
    source, target = a0.state_dict(), fusion.state_dict()
    for key, value in source.items():
        if key not in target or target[key].shape != value.shape:
            raise RuntimeError(f"A0/F shared parameter mismatch: {key}")
        target[key] = value.detach().clone()
    fusion.load_state_dict(target, strict=True)


class A0Intervention(ImageHybridH2256):
    """A0 with inference-only history/candidate image switches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_history_image = False
        self.disable_candidate_image = False

    def encode_item_values(
        self, content, item_id_row, categorical, numeric, image, image_available,
        *, image_off: bool | None = None,
    ):
        if image_off is None:
            image_off = self.disable_candidate_image
        if image_off:
            image_available = torch.zeros_like(image_available)
        return super().encode_item_values(
            content, item_id_row, categorical, numeric, image, image_available
        )

    def encode_item(self, batch: dict, prefix: str) -> torch.Tensor:
        return self.encode_item_values(
            batch[f"{prefix}_content"], batch[f"{prefix}_item_id_row"],
            batch[f"{prefix}_categorical"], batch[f"{prefix}_numeric"],
            batch[f"{prefix}_image"], batch[f"{prefix}_image_available"],
            image_off=self.disable_history_image and prefix == "history",
        )
