"""Training, encoding, ID-off and representation diagnostics for Experiment 03."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (
    BGE_PATH,
    DeadlineExceeded,
    check_deadline,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import (
    Phase7Collator,
    Phase7Dataset,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.features import (
    FeatureStore,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.losses import (
    tfidf_hard_loss,
)

from .models import E0, E0_VARIANT, AuthorConcatTower, VARIANTS


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase_07/experiment_03_author_concat_regularization_ablation"

HISTORY_N = 20
CONTENT_DIM = 768
OUTPUT_DIM = 128
TEMPERATURE = 0.05
PAIR_LAMBDA = 0.5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
MAX_EPOCHS = 6
PATIENCE = 2
PROXY_REQUESTS = 5_000
PROXY_CANDIDATES = 100_000
TOPK = 500
OVERFETCH = 600
CORPUS_ITEMS = 1_983_938


def to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_model(name: str, store: FeatureStore, device: str) -> AuthorConcatTower:
    variant = E0_VARIANT if name == E0 else VARIANTS[name]
    schema = store.schema()
    return AuthorConcatTower(
        variant,
        user_vocab=schema["user_count"],
        item_vocab=schema["item_id_count"],
        user_category_sizes=schema["user_category_sizes"],
        item_category_sizes=schema["item_category_sizes"],
        bge_dim=CONTENT_DIM,
        output_dim=OUTPUT_DIM,
    ).to(device)


def parameter_report(model: AuthorConcatTower) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    identifier = model.item_id.weight.numel() + model.user_id.weight.numel()
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "id_parameters": identifier,
        "non_id_parameters": total - identifier,
        "id_embedding_bytes_fp32": identifier * 4,
        "estimated_parameter_bytes_fp32": total * 4,
        "id_dim": model.id_dim,
        "output_dim": model.output_dim,
        "item_fusion_input_dim": model.item_fusion.input.in_features,
        "user_fusion_input_dim": model.user_fusion.input.in_features,
        "item_id_observed_std": float(model.item_id.weight[1:].detach().std()),
        "user_id_observed_std": float(model.user_id.weight[1:].detach().std()),
        "configured_id_init_std": model.variant.id_init_std,
        "configured_id_dropout": model.variant.id_dropout,
        "frozen_bge_parameters": 0,
    }


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    epoch: int = 0,
    workers: int = 2,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        persistent_workers=False,
        prefetch_factor=4 if workers else None,
        timeout=120 if workers else 0,
        pin_memory=True,
        collate_fn=Phase7Collator(True),
        generator=torch.Generator().manual_seed(42 + epoch),
    )


def train_epoch(
    model: AuthorConcatTower,
    dataset: Phase7Dataset,
    optimizer,
    device: str,
    epoch: int,
    batch_size: int = BATCH_SIZE,
    max_batches: int | None = None,
    deadline: float | None = None,
) -> dict:
    dataset.set_epoch(epoch)
    loader = make_loader(dataset, batch_size, True, epoch)
    amp = device.startswith("cuda")
    if amp:
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model.train()
    model.reset_dropout_statistics()
    losses, inbatch, hard = [], [], []
    started = time.perf_counter()
    data_wait = 0.0
    iterator = iter(loader)
    step = 0
    while True:
        check_deadline(deadline, "experiment03 training")
        wait_started = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            break
        data_wait += time.perf_counter() - wait_started
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(cpu_batch, device)
        context = (
            torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        )
        with context:
            query, target = model(batch)
            # Explicit negatives are encoded once per stored occurrence.  The
            # shared in-batch target matrix uses the single `target` encoding.
            negative = model.encode_item(batch, "negative")
            loss, details = tfidf_hard_loss(
                query,
                target,
                negative,
                batch["negative_mask"],
                batch,
                TEMPERATURE,
                PAIR_LAMBDA,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Experiment-03 loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        inbatch.append(details["inbatch_loss"])
        hard.append(details["tfidf_pair_loss"])
        step += 1
    elapsed = time.perf_counter() - started
    return {
        "loss": float(np.mean(losses)),
        "inbatch_loss": float(np.mean(inbatch)),
        "tfidf_pair_loss": float(np.mean(hard)),
        "batches": len(losses),
        "seconds": elapsed,
        "data_wait_seconds": data_wait,
        "data_wait_fraction": data_wait / max(elapsed, 1e-9),
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if amp else 0
        ),
        "structured_id_dropout": model.dropout_statistics(),
    }


def _request_frame(requests) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(len(requests)),
            "request_id": [request.request_idx for request in requests],
            "user_id": [request.user_idx for request in requests],
            "history_item_ids": [request.history for request in requests],
            "positive_item_id": [min(request.ground_truth) for request in requests],
            "same_request_positive_ids": [
                tuple(request.ground_truth) for request in requests
            ],
        }
    )
    frame["tfidf_ids"] = [tuple() for _ in range(len(frame))]
    frame["tfidf_ranks"] = [tuple() for _ in range(len(frame))]
    return frame


def encode_queries(
    model: AuthorConcatTower,
    requests,
    store: FeatureStore,
    device: str,
    *,
    item_id_enabled: bool = True,
    user_id_enabled: bool = True,
    batch_size: int = 512,
    deadline: float | None = None,
) -> np.ndarray:
    dataset = Phase7Dataset(_request_frame(requests), store, True, HISTORY_N)
    output = []
    model.eval()
    with torch.inference_mode():
        for cpu_batch in make_loader(dataset, batch_size, False):
            check_deadline(deadline, "Experiment-03 query encoding")
            batch = to_device(cpu_batch, device)
            output.append(
                model.query(
                    batch,
                    item_id_enabled=item_id_enabled,
                    user_id_enabled=user_id_enabled,
                )
                .float()
                .cpu()
                .numpy()
            )
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


def pack_item_features(item_ids, store: FeatureStore) -> dict[str, np.ndarray]:
    ids = np.asarray(item_ids, dtype=np.int64)
    rows = store.item_lookup[ids]
    bge = np.memmap(
        BGE_PATH,
        mode="r",
        dtype=np.float16,
        shape=(len(store.item_ids), CONTENT_DIM),
    )
    return {
        "item_id_rows": np.asarray(store.item_id_lookup[ids] + 1, dtype=np.int64),
        "categorical": np.asarray(store.item_categorical[rows], dtype=np.int32).copy(),
        "numeric": np.asarray(store.item_numeric[rows], dtype=np.float32).copy(),
        "content": np.asarray(bge[rows], dtype=np.float16).copy(),
    }


def encode_items(
    model: AuthorConcatTower,
    item_ids,
    store: FeatureStore,
    device: str,
    *,
    item_id_enabled: bool = True,
    batch_size: int = 8192,
    deadline: float | None = None,
    packed: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    ids_all = np.asarray(item_ids, dtype=np.int64)
    bge = np.memmap(
        BGE_PATH,
        mode="r",
        dtype=np.float16,
        shape=(len(store.item_ids), CONTENT_DIM),
    )
    output = np.empty((len(ids_all), OUTPUT_DIM), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids_all), batch_size):
            check_deadline(deadline, "Experiment-03 item encoding")
            ids = ids_all[start : start + batch_size]
            rows = store.item_lookup[ids]
            id_rows = (
                store.item_id_lookup[ids] + 1
                if packed is None
                else packed["item_id_rows"][start : start + len(ids)]
            )
            categorical = (
                store.item_categorical[rows]
                if packed is None
                else packed["categorical"][start : start + len(ids)]
            )
            numeric = (
                store.item_numeric[rows]
                if packed is None
                else packed["numeric"][start : start + len(ids)]
            )
            content = (
                bge[rows]
                if packed is None
                else packed["content"][start : start + len(ids)]
            )
            vector = model.encode_item_values(
                torch.as_tensor(np.asarray(content, dtype=np.float32), device=device),
                torch.as_tensor(np.asarray(id_rows, dtype=np.int64), device=device),
                torch.as_tensor(
                    np.asarray(categorical, dtype=np.int64), device=device
                ),
                torch.as_tensor(np.asarray(numeric, dtype=np.float32), device=device),
                item_id_enabled=item_id_enabled,
            )
            output[start : start + len(ids)] = vector.float().cpu().numpy()
    return output


def _sample(values, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(sorted(set(map(int, values))), dtype=np.int64)
    if len(values) <= limit:
        return values
    chosen = np.sort(
        np.random.default_rng(seed).choice(len(values), limit, replace=False)
    )
    return values[chosen]


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def representation_stability(
    model: AuthorConcatTower,
    store: FeatureStore,
    train_frame: pd.DataFrame,
    device: str,
    *,
    seed: int = 42,
    max_per_bucket: int = 5_000,
    deadline: float | None = None,
) -> dict:
    """Compare item vectors with the ID route enabled and disabled."""
    frequencies = train_frame.positive_item_id.value_counts()
    targets = set(map(int, frequencies.index))
    vocabulary = set(map(int, store.train_item_id_vocab))
    history_only = vocabulary - targets
    corpus = set(map(int, store.item_ids))
    buckets = {
        "train_target_seen": targets,
        "train_history_only": history_only,
        "completely_unseen": corpus - vocabulary,
        "frequency_1": set(map(int, frequencies[frequencies == 1].index)),
        "frequency_2_3": set(
            map(int, frequencies[(frequencies >= 2) & (frequencies <= 3)].index)
        ),
        "frequency_4_10": set(
            map(int, frequencies[(frequencies >= 4) & (frequencies <= 10)].index)
        ),
        "frequency_11_plus": set(map(int, frequencies[frequencies > 10].index)),
    }
    output = {}
    for offset, (name, values) in enumerate(buckets.items()):
        check_deadline(deadline, "Experiment-03 representation stability")
        ids = _sample(values, max_per_bucket, seed + offset)
        packed = pack_item_features(ids, store)
        enabled = encode_items(
            model,
            ids,
            store,
            device,
            item_id_enabled=True,
            packed=packed,
            deadline=deadline,
        )
        disabled = encode_items(
            model,
            ids,
            store,
            device,
            item_id_enabled=False,
            packed=packed,
            deadline=deadline,
        )
        cosine = np.sum(enabled * disabled, axis=1) / np.maximum(
            np.linalg.norm(enabled, axis=1) * np.linalg.norm(disabled, axis=1),
            1e-12,
        )
        id_rows = packed["item_id_rows"]
        with torch.inference_mode():
            id_norm = (
                model.item_id(
                    torch.as_tensor(id_rows, dtype=torch.long, device=device)
                )
                .norm(dim=-1)
                .cpu()
                .numpy()
            )
        output[name] = {
            "cosine_id_on_off": _distribution(cosine),
            "id_on_vector_norm": _distribution(np.linalg.norm(enabled, axis=1)),
            "id_off_vector_norm": _distribution(np.linalg.norm(disabled, axis=1)),
            "raw_id_embedding_norm": _distribution(id_norm),
        }
    return output


__all__ = [
    "BATCH_SIZE",
    "CORPUS_ITEMS",
    "DeadlineExceeded",
    "HISTORY_N",
    "MAX_EPOCHS",
    "OVERFETCH",
    "OUT",
    "PATIENCE",
    "PROXY_CANDIDATES",
    "PROXY_REQUESTS",
    "TOPK",
    "encode_items",
    "encode_queries",
    "make_loader",
    "make_model",
    "pack_item_features",
    "parameter_report",
    "representation_stability",
    "to_device",
    "train_epoch",
]
