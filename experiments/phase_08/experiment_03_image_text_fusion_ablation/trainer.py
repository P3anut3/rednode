"""Thin experimental adapters over the unchanged Phase 8-02 data/loss protocol."""

from __future__ import annotations

import torch

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import PROTOCOL
from experiments.phase_08.experiment_02_image_hybrid_recall.trainer import (
    encode_items, encode_queries, pack_item_features, train_epoch,
)
from experiments.phase_08.experiment_02_image_hybrid_recall.trainer import make_model as make_a0

from .models import FusionTower, copy_a0_start


IMAGE_STRATEGY = "i3_all"


def make_model(store, device: str, variant: str) -> FusionTower:
    # A0 must be instantiated first: its full state is then copied, not merely
    # hoped to match because the two constructors happened to use one seed.
    a0 = make_a0(store, device)
    schema = store.schema()
    model = FusionTower(
        variant=variant,
        user_vocab=schema["user_count"],
        item_vocab=schema["item_id_count"],
        user_category_sizes=schema["user_category_sizes"],
        item_category_sizes=schema["item_category_sizes"],
        retrieval_dim=256,
        id_dim=128,
        alpha_image_init=PROTOCOL.alpha_init,
    ).to(device)
    copy_a0_start(model, a0)
    del a0
    return model


def parameter_report(model: FusionTower) -> dict:
    total = sum(p.numel() for p in model.parameters())
    identifier = model.item_id.weight.numel() + model.user_id.weight.numel()
    fusion = sum(p.numel() for p in model.image_gate.parameters())
    if model.interaction is not None:
        fusion += sum(p.numel() for p in model.interaction.parameters())
    return {
        "total_parameters": total,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "id_parameters": identifier,
        "non_id_parameters": total - identifier,
        "fusion_parameters": fusion + model.image_projection.weight.numel() + model.alpha_image_raw.numel(),
        "retrieval_dim": 256,
        "id_dim": 128,
        "frozen_bge_parameters": 0,
        "frozen_siglip_parameters": 0,
    }


__all__ = (
    "IMAGE_STRATEGY", "make_model", "parameter_report", "train_epoch",
    "encode_items", "encode_queries", "pack_item_features",
)
