"""Training, encoding and diagnostics for concat-fusion models."""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import BGE_PATH
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import (
    Phase7Collator,
    Phase7Dataset,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.features import FeatureStore

from .config import CONDITIONAL_MODEL, PROTOCOL, check_deadline
from .losses import tfidf_hard_loss
from .models import AddId64ControlTower, ConcatFusionTwoTower


def to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_model(name: str, store: FeatureStore, device: str):
    schema = store.schema()
    if name == CONDITIONAL_MODEL:
        return AddId64ControlTower(
            user_vocab=schema["user_count"],
            item_vocab=schema["item_id_count"],
            user_category_sizes=schema["user_category_sizes"],
            item_category_sizes=schema["item_category_sizes"],
            content_dim=PROTOCOL.content_dim,
            output_dim=PROTOCOL.output_dim,
            id_dim=PROTOCOL.id_dim,
            alpha_init=PROTOCOL.beta_init,
        ).to(device)
    return ConcatFusionTwoTower(
        name,
        user_vocab=schema["user_count"],
        item_vocab=schema["item_id_count"],
        user_category_sizes=schema["user_category_sizes"],
        item_category_sizes=schema["item_category_sizes"],
        content_dim=PROTOCOL.content_dim,
        output_dim=PROTOCOL.output_dim,
        id_dim=PROTOCOL.id_dim,
        fusion_hidden_dim=PROTOCOL.fusion_hidden_dim,
        dropout=PROTOCOL.dropout,
        beta_init=PROTOCOL.beta_init,
    ).to(device)


def parameter_report(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    id_parameters = model.item_id.weight.numel() + model.user_id.weight.numel()
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": trainable,
        "id_embedding_parameters": id_parameters,
        "id_embedding_bytes_fp32": id_parameters * 4,
        "estimated_checkpoint_bytes_fp32": trainable * 4,
        "id_dim": model.id_dim,
        "output_dim": model.output_dim,
        "frozen_bge_parameters": 0,
    }


def make_loader(dataset, batch_size: int, shuffle: bool, epoch: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=2,
        persistent_workers=False,
        prefetch_factor=4,
        timeout=120,
        pin_memory=True,
        collate_fn=Phase7Collator(True),
        generator=torch.Generator().manual_seed(PROTOCOL.seed + epoch),
    )


def train_epoch(
    model,
    dataset: Phase7Dataset,
    optimizer,
    device: str,
    epoch: int,
    batch_size: int = 512,
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
    loss_values, inbatch_values, hard_values = [], [], []
    started = time.perf_counter()
    data_wait, iterator, step = 0.0, iter(loader), 0
    while True:
        check_deadline(deadline, "training")
        wait_started = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            break
        data_wait += time.perf_counter() - wait_started
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(cpu_batch, device)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            query, target = model(batch)
            negative = model.encode_item(batch, "negative")
            loss, details = tfidf_hard_loss(
                query,
                target,
                negative,
                batch["negative_mask"],
                batch,
                PROTOCOL.temperature,
                PROTOCOL.pair_lambda,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite concat-fusion loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_values.append(float(loss.detach()))
        inbatch_values.append(details["inbatch_loss"])
        hard_values.append(details["tfidf_pair_loss"])
        step += 1
    elapsed = time.perf_counter() - started
    return {
        "loss": float(np.mean(loss_values)),
        "inbatch_loss": float(np.mean(inbatch_values)),
        "tfidf_pair_loss": float(np.mean(hard_values)),
        "batches": len(loss_values),
        "seconds": elapsed,
        "data_wait_seconds": data_wait,
        "data_wait_fraction": data_wait / max(elapsed, 1e-9),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if amp else 0,
    }


def _request_frame(requests) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(len(requests)),
            "request_id": [r.request_idx for r in requests],
            "user_id": [r.user_idx for r in requests],
            "history_item_ids": [r.history for r in requests],
            "positive_item_id": [min(r.ground_truth) for r in requests],
            "same_request_positive_ids": [tuple(r.ground_truth) for r in requests],
        }
    )
    frame["tfidf_ids"] = [tuple() for _ in range(len(frame))]
    frame["tfidf_ranks"] = [tuple() for _ in range(len(frame))]
    return frame


def encode_queries(model, requests, store, device, batch_size=512, deadline=None):
    dataset = Phase7Dataset(_request_frame(requests), store, True, PROTOCOL.history_n)
    loader = make_loader(dataset, batch_size, False)
    output = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            check_deadline(deadline, "query encoding")
            output.append(model.query(to_device(batch, device)).float().cpu().numpy())
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


def encode_items(model, item_ids, store, device, batch_size=8192, deadline=None, packed=None):
    bge = np.memmap(
        BGE_PATH,
        mode="r",
        dtype=np.float16,
        shape=(len(store.item_ids), PROTOCOL.content_dim),
    )
    output = np.empty((len(item_ids), PROTOCOL.output_dim), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(item_ids), batch_size):
            check_deadline(deadline, "item encoding")
            ids = np.asarray(item_ids[start : start + batch_size], dtype=np.int64)
            rows = store.item_lookup[ids]
            id_rows = store.item_id_lookup[ids] + 1 if packed is None else packed["item_id_rows"][start:start+len(ids)]
            categorical = store.item_categorical[rows] if packed is None else packed["categorical"][start:start+len(ids)]
            numeric = store.item_numeric[rows] if packed is None else packed["numeric"][start:start+len(ids)]
            content = bge[rows] if packed is None else packed["content"][start:start+len(ids)]
            vector = model.encode_item_values(
                torch.as_tensor(np.asarray(content, dtype=np.float32), device=device),
                torch.as_tensor(np.asarray(id_rows, dtype=np.int64), device=device),
                torch.as_tensor(np.asarray(categorical, dtype=np.int64), device=device),
                torch.as_tensor(np.asarray(numeric, dtype=np.float32), device=device),
            )
            output[start : start + len(ids)] = vector.float().cpu().numpy()
    return output


def pack_item_features(item_ids, store) -> dict[str, np.ndarray]:
    ids = np.asarray(item_ids, dtype=np.int64)
    rows = store.item_lookup[ids]
    bge = np.memmap(BGE_PATH, mode="r", dtype=np.float16, shape=(len(store.item_ids), 768))
    return {
        "item_id_rows": np.asarray(store.item_id_lookup[ids] + 1, dtype=np.int64),
        "categorical": np.asarray(store.item_categorical[rows], dtype=np.int32).copy(),
        "numeric": np.asarray(store.item_numeric[rows], dtype=np.float32).copy(),
        "content": np.asarray(bge[rows], dtype=np.float16).copy(),
    }


def _sample(values, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(sorted(set(map(int, values))), dtype=np.int64)
    if len(values) <= limit:
        return values
    chosen = np.sort(np.random.default_rng(seed).choice(len(values), limit, replace=False))
    return values[chosen]


def _finalize_component_metrics(parts: list[dict[str, torch.Tensor]], beta: float) -> dict:
    if not parts:
        return {"count": 0}
    base = torch.cat([part["base"] for part in parts])
    delta = torch.cat([part["delta"] for part in parts])
    final = torch.cat([part["final"] for part in parts])
    residual = beta * delta
    return {
        "count": len(base),
        "residual_base_norm_ratio": float(
            (residual.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-8)).mean()
        ),
        "delta_base_cosine": float(F.cosine_similarity(delta, base).mean()),
        "final_base_cosine": float(F.cosine_similarity(final, base).mean()),
    }


def validation_residual_diagnostics(
    model,
    requests,
    store,
    train_targets: set[int],
    train_vocab: set[int],
    train_users: set[int],
    device: str,
    max_per_bucket: int = 5_000,
    seed: int = 42,
    deadline: float | None = None,
) -> dict:
    """Measure residual behavior on temporal validation item/user buckets."""
    if not model.residual:
        return {}
    positives = set().union(*(set(request.ground_truth) for request in requests))
    item_buckets = {
        "train_target_seen": positives & train_targets,
        "train_history_only": positives & (train_vocab - train_targets),
        "completely_unseen": positives - train_vocab,
    }
    bge = np.memmap(
        BGE_PATH,
        mode="r",
        dtype=np.float16,
        shape=(len(store.item_ids), PROTOCOL.content_dim),
    )
    item_output = {}
    model.eval()
    with torch.inference_mode():
        for offset, (name, values) in enumerate(item_buckets.items()):
            check_deadline(deadline, "validation item residual diagnostics")
            ids = _sample(values, max_per_bucket, seed + offset)
            parts = []
            for start in range(0, len(ids), 1024):
                check_deadline(deadline, "validation item residual diagnostics")
                selected = ids[start : start + 1024]
                rows = store.item_lookup[selected]
                parts.append(
                    {
                        key: value.cpu()
                        for key, value in model.item_components(
                            torch.as_tensor(np.asarray(bge[rows], dtype=np.float32), device=device),
                            torch.as_tensor(np.asarray(store.item_id_lookup[selected] + 1, dtype=np.int64), device=device),
                            torch.as_tensor(np.asarray(store.item_categorical[rows], dtype=np.int64), device=device),
                            torch.as_tensor(np.asarray(store.item_numeric[rows], dtype=np.float32), device=device),
                        ).items()
                        if key in {"base", "delta", "final"}
                    }
                )
            item_output[name] = _finalize_component_metrics(parts, model.beta_values()["item"])

        user_output = {}
        for offset, (name, selected_requests) in enumerate(
            {
                "seen_user": [r for r in requests if int(r.user_idx) in train_users],
                "unseen_user": [r for r in requests if int(r.user_idx) not in train_users],
            }.items()
        ):
            check_deadline(deadline, "validation user residual diagnostics")
            if len(selected_requests) > max_per_bucket:
                chosen = np.sort(
                    np.random.default_rng(seed + 10 + offset).choice(
                        len(selected_requests), max_per_bucket, replace=False
                    )
                )
                selected_requests = [selected_requests[int(index)] for index in chosen]
            dataset = Phase7Dataset(
                _request_frame(selected_requests), store, True, PROTOCOL.history_n
            )
            parts = []
            for cpu_batch in make_loader(dataset, 256, False):
                check_deadline(deadline, "validation user residual diagnostics")
                component = model.user_components(to_device(cpu_batch, device))
                parts.append(
                    {
                        key: value.cpu()
                        for key, value in component.items()
                        if key in {"base", "delta", "final"}
                    }
                )
            user_output[name] = _finalize_component_metrics(parts, model.beta_values()["user"])
    return {
        "beta_item": model.beta_values()["item"],
        "beta_user": model.beta_values()["user"],
        "item_buckets": item_output,
        "user_buckets": user_output,
    }
