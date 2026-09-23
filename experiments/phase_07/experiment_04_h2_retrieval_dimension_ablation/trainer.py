"""Training and 256d vector encoding for Experiment 04."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (
    BGE_PATH,
    PROTOCOL,
    check_deadline,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.features import FeatureStore
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.trainer import (
    encode_queries,
    pack_item_features,
    train_epoch,
)

from .models import HybridTower256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase_07/experiment_04_h2_retrieval_dimension_ablation"
DIM = 256
ID_DIM = 128


def make_model(name: str, store: FeatureStore, device: str):
    schema = store.schema()
    return HybridTower256(
        name,
        user_vocab=schema["user_count"],
        item_vocab=schema["item_id_count"],
        user_category_sizes=schema["user_category_sizes"],
        item_category_sizes=schema["item_category_sizes"],
        retrieval_dim=DIM,
        id_dim=ID_DIM,
    ).to(device)


def parameter_report(model) -> dict:
    total = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    identifier = model.item_id.weight.numel() + model.user_id.weight.numel()
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "id_parameters": identifier,
        "non_id_parameters": total - identifier,
        "retrieval_dim": DIM,
        "id_dim": ID_DIM,
        "estimated_checkpoint_bytes_fp32": trainable * 4,
    }


def encode_items(
    model,
    item_ids: np.ndarray,
    store: FeatureStore,
    device: str,
    batch_size: int = 8192,
    deadline: float | None = None,
    packed: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    bge = np.memmap(
        BGE_PATH,
        mode="r",
        dtype=np.float16,
        shape=(len(store.item_ids), PROTOCOL.content_dim),
    )
    output = np.empty((len(item_ids), DIM), dtype=np.float32)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(item_ids), batch_size):
            check_deadline(deadline, "Experiment-04 item encoding")
            ids = np.asarray(item_ids[start : start + batch_size], dtype=np.int64)
            rows = store.item_lookup[ids]
            id_rows = (
                packed["item_id_rows"][start : start + len(ids)]
                if packed is not None
                else store.item_id_lookup[ids] + 1
            )
            categorical = (
                packed["categorical"][start : start + len(ids)]
                if packed is not None else store.item_categorical[rows]
            )
            numeric = (
                packed["numeric"][start : start + len(ids)]
                if packed is not None else store.item_numeric[rows]
            )
            content = (
                packed["content"][start : start + len(ids)]
                if packed is not None else bge[rows]
            )
            vector = model.encode_item_values(
                torch.as_tensor(np.asarray(content, dtype=np.float32), device=device),
                torch.as_tensor(np.asarray(id_rows, dtype=np.int64), device=device),
                torch.as_tensor(np.asarray(categorical, dtype=np.int64), device=device),
                torch.as_tensor(np.asarray(numeric, dtype=np.float32), device=device),
            )
            output[start : start + len(ids)] = vector.float().cpu().numpy()
    return output


__all__ = [
    "DIM", "ID_DIM", "OUT", "encode_items", "encode_queries", "make_model",
    "pack_item_features", "parameter_report", "train_epoch",
]
