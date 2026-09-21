"""Bounded Phase 7 training and vector encoding utilities."""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import BGE_PATH, ID_MODELS, PROTOCOL, check_deadline
from .dataset import Phase7Collator, Phase7Dataset
from .features import FeatureStore
from .losses import logq_inbatch_loss, tfidf_hard_loss
from .models import EnhancedIdTower, HybridTower


def make_model(name: str, store: FeatureStore, device: str):
    schema = store.schema()
    common = dict(
        user_vocab=schema["user_count"],
        item_vocab=schema["item_id_count"],
        user_category_sizes=schema["user_category_sizes"],
        item_category_sizes=schema["item_category_sizes"],
        dim=PROTOCOL.output_dim,
    )
    model = (
        EnhancedIdTower(name, **common)
        if name in ID_MODELS
        else HybridTower(
            name,
            content_dim=PROTOCOL.content_dim,
            alpha_init=PROTOCOL.alpha_init,
            **common,
        )
    )
    return model.to(device)


def parameter_report(model: torch.nn.Module) -> dict:
    trainable = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    return {
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": trainable,
        "frozen_bge_parameters": 0,
        "estimated_checkpoint_bytes_fp32": trainable * 4,
    }


def smoke_gradient_and_cold_audit(model, dataset: Phase7Dataset, device: str) -> dict:
    variant = model.variant
    required = ["attention"]
    if isinstance(model, EnhancedIdTower):
        required += ["user_id", "item.item_id", "user_combine"]
        if model.use_profile:
            required += ["profile.categories", "profile.output"]
        if model.use_profile_numeric:
            required += ["profile.numeric"]
        if model.use_item_features:
            required += ["item.meta", "item.combine"]
    else:
        required += ["content", "history_projection"]
        if model.use_side:
            required += ["item_meta", "user_profile", "alpha_meta", "alpha_profile"]
        if model.use_id:
            required += ["item_id", "user_id", "alpha_item_id", "alpha_user_id"]
        if variant == "h3_anonymous_dense":
            required += ["anonymous_dense", "alpha_dense"]
    named = dict(model.named_parameters())
    branch_gradients = {
        prefix: any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad).item() > 0
            for name, parameter in named.items()
            if name.startswith(prefix)
        )
        for prefix in required
    }
    padding_gradient_zero = all(
        module.weight.grad is None
        or torch.count_nonzero(module.weight.grad[0]).item() == 0
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding) and module.padding_idx == 0
    )

    cpu = Phase7Collator(dataset.hybrid)(
        [dataset[index] for index in range(min(4, len(dataset)))]
    )
    batch = to_device(cpu, device)
    batch["target_item_id_row"] = torch.zeros_like(batch["target_item_id_row"])
    batch["user_id_row"] = torch.zeros_like(batch["user_id_row"])
    model.eval()
    with torch.inference_mode():
        item_before = model.encode_item(batch, "target")
        user_before = model.query(batch)
        item_embedding = (
            model.item.item_id if isinstance(model, EnhancedIdTower) else model.item_id
        )
        user_embedding = model.user_id
        item_saved = item_embedding.weight[1:].clone()
        user_saved = user_embedding.weight[1:].clone()
        item_embedding.weight[1:].add_(torch.randn_like(item_embedding.weight[1:]))
        item_after = model.encode_item(batch, "target")
        item_embedding.weight[1:].copy_(item_saved)
        user_embedding.weight[1:].add_(torch.randn_like(user_embedding.weight[1:]))
        user_after = model.query(batch)
        user_embedding.weight[1:].copy_(user_saved)
    return {
        "required_branch_gradients": branch_gradients,
        "all_required_branches_have_gradient": all(branch_gradients.values()),
        "padding_embedding_gradient_zero": padding_gradient_zero,
        "cold_item_id_invariance_max_abs": float(
            (item_before - item_after).abs().max()
        ),
        "cold_user_id_invariance_max_abs": float(
            (user_before - user_after).abs().max()
        ),
    }


def to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_epoch(
    model,
    dataset: Phase7Dataset,
    optimizer,
    device: str,
    batch_size: int,
    epoch: int,
    target_logq: dict[int, float],
    max_batches: int | None = None,
    deadline: float | None = None,
) -> dict:
    dataset.set_epoch(epoch)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4,
        timeout=120,
        pin_memory=True,
        collate_fn=Phase7Collator(dataset.hybrid),
        generator=torch.Generator().manual_seed(PROTOCOL.seed + epoch),
    )
    amp = device.startswith("cuda")
    if amp:
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model.train()
    losses, inbatch, hard = [], [], []
    started = time.perf_counter()
    data_wait_seconds = 0.0
    iterator = iter(loader)
    step = 0
    while True:
        check_deadline(deadline, "training")
        wait_started = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            break
        data_wait_seconds += time.perf_counter() - wait_started
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(cpu_batch, device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp
            else nullcontext()
        )
        with context:
            query, target = model(batch)
            if isinstance(model, EnhancedIdTower):
                logq = torch.as_tensor(
                    [target_logq[int(note)] for note in cpu_batch["target_note_id"]],
                    device=device,
                    dtype=query.dtype,
                )
                loss, details = logq_inbatch_loss(
                    query, target, batch, logq, PROTOCOL.temperature
                )
            else:
                negative = model.encode_item(batch, "negative")
                loss, details = tfidf_hard_loss(
                    query,
                    target,
                    negative,
                    batch["negative_mask"],
                    batch,
                    PROTOCOL.temperature,
                    0.5,
                )
                inbatch.append(details["inbatch_loss"])
                hard.append(details["tfidf_pair_loss"])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Phase 7 loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        step += 1
    return {
        "loss": float(np.mean(losses)),
        "batches": len(losses),
        "inbatch_loss": float(np.mean(inbatch)) if inbatch else float("nan"),
        "tfidf_pair_loss": float(np.mean(hard)) if hard else float("nan"),
        "seconds": time.perf_counter() - started,
        "data_wait_seconds": data_wait_seconds,
        "data_wait_fraction": data_wait_seconds
        / max(time.perf_counter() - started, 1e-9),
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if amp else 0
        ),
    }


def _request_frame(requests) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": np.arange(len(requests)),
            "request_id": [r.request_idx for r in requests],
            "user_id": [r.user_idx for r in requests],
            "history_item_ids": [r.history for r in requests],
            "positive_item_id": [min(r.ground_truth) for r in requests],
            "same_request_positive_ids": [tuple(r.ground_truth) for r in requests],
        }
    )


def encode_queries(
    model,
    requests,
    store: FeatureStore,
    device: str,
    batch_size: int = 512,
    deadline: float | None = None,
) -> np.ndarray:
    hybrid = isinstance(model, HybridTower)
    frame = _request_frame(requests)
    if hybrid:
        # Query-only inference has no training hard negatives.  Empty pools
        # preserve the shared dataset/collator contract without affecting
        # model.query(), which reads history and user features only.
        frame["tfidf_ids"] = [tuple() for _ in range(len(frame))]
        frame["tfidf_ranks"] = [tuple() for _ in range(len(frame))]
    dataset = Phase7Dataset(
        frame,
        store,
        hybrid=hybrid,
        history_n=PROTOCOL.history_n,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4,
        timeout=120,
        pin_memory=True,
        collate_fn=Phase7Collator(dataset.hybrid),
    )
    model.eval()
    output = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            check_deadline(deadline, "query encoding")
            output.append(model.query(to_device(batch, device)).float().cpu().numpy())
            if batch_index and batch_index % 20 == 0:
                print(
                    f"query encoding: {batch_index * batch_size}/{len(dataset)}",
                    flush=True,
                )
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


def encode_items(
    model,
    item_ids: np.ndarray,
    store: FeatureStore,
    device: str,
    batch_size: int = 8192,
    deadline: float | None = None,
    packed: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    bge = None
    if isinstance(model, HybridTower):
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
            id_rows = (
                packed["item_id_rows"][start : start + len(ids)]
                if packed is not None
                else store.item_id_lookup[ids] + 1
            )
            categorical = torch.from_numpy(
                np.asarray(
                    packed["categorical"][start : start + len(ids)]
                    if packed is not None
                    else store.item_categorical[rows],
                    dtype=np.int64,
                )
            ).to(device)
            numeric = torch.from_numpy(
                np.asarray(
                    packed["numeric"][start : start + len(ids)]
                    if packed is not None
                    else store.item_numeric[rows],
                    dtype=np.float32,
                )
            ).to(device)
            id_tensor = torch.from_numpy(id_rows.astype(np.int64)).to(device)
            if isinstance(model, EnhancedIdTower):
                vector = model.item(
                    id_tensor, categorical, numeric, model.use_item_features
                )
            else:
                content = torch.from_numpy(
                    np.asarray(
                        packed["content"][start : start + len(ids)]
                        if packed is not None
                        else bge[rows],
                        dtype=np.float32,
                    )
                ).to(device)
                vector = model.encode_item_values(
                    content, id_tensor, categorical, numeric
                )
            output[start : start + len(ids)] = vector.float().cpu().numpy()
            if start and start % (batch_size * 25) == 0:
                print(f"item encoding: {start}/{len(item_ids)}", flush=True)
    return output


def pack_item_features(
    item_ids: np.ndarray, store: FeatureStore, hybrid: bool
) -> dict[str, np.ndarray]:
    """Pack a bounded proxy catalog once; the trainable encoder still reruns each epoch."""
    ids = np.asarray(item_ids, dtype=np.int64)
    rows = store.item_lookup[ids]
    packed = {
        "item_id_rows": np.asarray(store.item_id_lookup[ids] + 1, dtype=np.int64),
        "categorical": np.asarray(store.item_categorical[rows], dtype=np.int32).copy(),
        "numeric": np.asarray(store.item_numeric[rows], dtype=np.float32).copy(),
    }
    if hybrid:
        bge = np.memmap(
            BGE_PATH,
            mode="r",
            dtype=np.float16,
            shape=(len(store.item_ids), PROTOCOL.content_dim),
        )
        packed["content"] = np.asarray(bge[rows], dtype=np.float16).copy()
    return packed
