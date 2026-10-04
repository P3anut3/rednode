"""Phase-7 H2 training objective and retrieval, with read-only pooled images."""

from __future__ import annotations

import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import BGE_PATH, PROTOCOL, check_deadline
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import Phase7Dataset
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.features import FeatureStore
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.losses import tfidf_hard_loss
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.trainer import pack_item_features, to_device

from .data import ImageCollator, image_paths
from .models import ImageHybridH2256


DIM = 256
ID_DIM = 128


def process_tree_rss() -> int:
    parent = psutil.Process()
    total = 0
    for process in (parent, *parent.children(recursive=True)):
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def check_resources(limits: dict | None, device: str, check_disk: bool = False) -> dict:
    if limits is None:
        return {}
    rss = process_tree_rss()
    available = psutil.virtual_memory().available
    gpu_allocated = torch.cuda.memory_allocated(device)
    gpu_peak = torch.cuda.max_memory_allocated(device)
    if rss > limits["max_process_tree_rss_bytes"]:
        raise RuntimeError(f"process-tree RSS safety limit exceeded: {rss / 2**30:.2f} GiB")
    if available < limits["min_host_available_bytes"]:
        raise RuntimeError(f"host available-memory safety limit crossed: {available / 2**30:.2f} GiB")
    if max(gpu_allocated, gpu_peak) > limits["max_gpu_allocated_bytes"]:
        raise RuntimeError(f"GPU allocation safety limit exceeded: {gpu_peak / 2**30:.2f} GiB peak")
    free = shutil.disk_usage(Path(limits["disk_path"])).free if check_disk else None
    if free is not None and free < limits["min_disk_free_bytes"]:
        raise RuntimeError(f"disk free-space safety limit crossed: {free / 2**30:.2f} GiB")
    return {"process_tree_rss_bytes": rss, "host_available_bytes": available,
            "gpu_allocated_bytes": gpu_allocated, "disk_free_bytes": free}


def make_model(store: FeatureStore, device: str) -> ImageHybridH2256:
    schema = store.schema()
    return ImageHybridH2256(
        user_vocab=schema["user_count"],
        item_vocab=schema["item_id_count"],
        user_category_sizes=schema["user_category_sizes"],
        item_category_sizes=schema["item_category_sizes"],
        retrieval_dim=DIM,
        id_dim=ID_DIM,
        alpha_image_init=PROTOCOL.alpha_init,
    ).to(device)


def parameter_report(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    identifier = model.item_id.weight.numel() + model.user_id.weight.numel()
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "id_parameters": identifier,
        "non_id_parameters": total - identifier,
        "image_projection_parameters": model.image_projection.weight.numel(),
        "frozen_bge_parameters": 0,
        "frozen_siglip_parameters": 0,
        "retrieval_dim": DIM,
        "id_dim": ID_DIM,
    }


def train_epoch(model, dataset: Phase7Dataset, optimizer, device: str, batch_size: int,
                epoch: int, strategy: str, max_batches: int | None = None,
                deadline: float | None = None, num_workers: int = 2,
                prefetch_factor: int = 1, limits: dict | None = None) -> dict:
    dataset.set_epoch(epoch)
    # Worker/prefetch changes are resource controls only; dataset sampling,
    # batch size, loss and the dedicated shuffle generator remain unchanged.
    loader_options = dict(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, persistent_workers=False,
        timeout=120 if num_workers else 0, pin_memory=True,
        collate_fn=ImageCollator(strategy),
        generator=torch.Generator().manual_seed(PROTOCOL.seed + epoch),
    )
    if num_workers:
        loader_options["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_options)
    torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.train()
    losses, inbatch, hard = [], [], []
    peak_process_tree_rss = 0
    started, wait_seconds = time.perf_counter(), 0.0
    iterator = iter(loader)
    try:
        for step in range(len(loader)):
            check_deadline(deadline, "image-hybrid training")
            sample = check_resources(limits, device, check_disk=(step % 20 == 0))
            peak_process_tree_rss = max(peak_process_tree_rss, sample.get("process_tree_rss_bytes", 0))
            wait_started = time.perf_counter()
            cpu = next(iterator)
            wait_seconds += time.perf_counter() - wait_started
            if max_batches is not None and step >= max_batches:
                break
            batch = to_device(cpu, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                query, target = model(batch)
                negative = model.encode_item(batch, "negative")
                loss, parts = tfidf_hard_loss(
                    query, target, negative, batch["negative_mask"], batch,
                    PROTOCOL.temperature, 0.5,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite image-hybrid loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sample = check_resources(limits, device, check_disk=(step % 20 == 0))
            peak_process_tree_rss = max(peak_process_tree_rss, sample.get("process_tree_rss_bytes", 0))
            losses.append(float(loss.detach()))
            inbatch.append(parts["inbatch_loss"])
            hard.append(parts["tfidf_pair_loss"])
            if step % 20 == 0:
                print(f"image-hybrid epoch={epoch} batch={step + 1}/{len(loader)} "
                      f"loss={losses[-1]:.4f} rss={peak_process_tree_rss / 2**30:.2f}GiB "
                      f"gpu={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB", flush=True)
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    elapsed = time.perf_counter() - started
    return {
        "loss": float(np.mean(losses)),
        "batches": len(losses),
        "inbatch_loss": float(np.mean(inbatch)),
        "tfidf_pair_loss": float(np.mean(hard)),
        "seconds": elapsed,
        "samples_per_second": len(losses) * batch_size / max(elapsed, 1e-9),
        "data_wait_seconds": wait_seconds,
        "data_wait_fraction": wait_seconds / max(elapsed, 1e-9),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "process_tree_peak_rss_bytes_sampled": peak_process_tree_rss,
    }


def encode_items(model, item_ids: np.ndarray, store: FeatureStore, device: str,
                 strategy: str, batch_size: int = 8192, deadline: float | None = None,
                 packed: dict[str, np.ndarray] | None = None,
                 limits: dict | None = None) -> np.ndarray:
    bge = np.memmap(BGE_PATH, mode="r", dtype=np.float16,
                    shape=(len(store.item_ids), PROTOCOL.content_dim))
    image_path, mask_path, _ = image_paths(strategy)
    image = np.load(image_path, mmap_mode="r")
    image_available = np.load(mask_path, mmap_mode="r")
    output = np.empty((len(item_ids), DIM), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(item_ids), batch_size):
            check_deadline(deadline, "image-hybrid item encoding")
            check_resources(limits, device, check_disk=(start % (batch_size * 25) == 0))
            ids = np.asarray(item_ids[start:start + batch_size], dtype=np.int64)
            rows = store.item_lookup[ids]
            if np.any(rows < 0):
                raise RuntimeError("item outside canonical corpus")
            choose = lambda key, fallback: packed[key][start:start + len(ids)] if packed is not None else fallback
            content = choose("content", bge[rows])
            id_rows = choose("item_id_rows", store.item_id_lookup[ids] + 1)
            categorical = choose("categorical", store.item_categorical[rows])
            numeric = choose("numeric", store.item_numeric[rows])
            vector = model.encode_item_values(
                torch.as_tensor(np.asarray(content, np.float32), device=device),
                torch.as_tensor(np.asarray(id_rows, np.int64), device=device),
                torch.as_tensor(np.asarray(categorical, np.int64), device=device),
                torch.as_tensor(np.asarray(numeric, np.float32), device=device),
                torch.as_tensor(np.asarray(image[rows], np.float32), device=device),
                torch.as_tensor(np.asarray(image_available[rows], np.bool_), device=device),
            )
            output[start:start + len(ids)] = vector.float().cpu().numpy()
            if start and start % (batch_size * 25) == 0:
                print(f"image-hybrid item encoding: {start}/{len(item_ids)}", flush=True)
    return output


def encode_queries(model, requests, store: FeatureStore, device: str, strategy: str,
                   batch_size: int = 512, deadline: float | None = None,
                   num_workers: int = 2, prefetch_factor: int = 1,
                   limits: dict | None = None) -> np.ndarray:
    # Deliberately use dummy targets: neither validation nor test clicked item
    # can enter user representation or even be looked up by this loader.
    frame = pd.DataFrame({
        "sample_id": np.arange(len(requests)),
        "request_id": [r.request_idx for r in requests],
        "user_id": [r.user_idx for r in requests],
        "history_item_ids": [r.history for r in requests],
        "positive_item_id": [-1] * len(requests),
        "same_request_positive_ids": [tuple()] * len(requests),
        "tfidf_ids": [tuple()] * len(requests),
        "tfidf_ranks": [tuple()] * len(requests),
    })
    dataset = Phase7Dataset(frame, store, hybrid=True, history_n=PROTOCOL.history_n)
    loader_options = dict(dataset=dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, persistent_workers=False,
                          timeout=120 if num_workers else 0, pin_memory=True,
                          collate_fn=ImageCollator(strategy))
    if num_workers:
        loader_options["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_options)
    model.eval()
    output = []
    with torch.inference_mode():
        for index, cpu in enumerate(loader):
            check_deadline(deadline, "image-hybrid query encoding")
            check_resources(limits, device, check_disk=(index % 20 == 0))
            output.append(model.query(to_device(cpu, device)).float().cpu().numpy())
            if index and index % 20 == 0:
                print(f"image-hybrid query encoding: {index * batch_size}/{len(dataset)}", flush=True)
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


__all__ = ["DIM", "ID_DIM", "make_model", "parameter_report", "train_epoch", "encode_items", "encode_queries", "pack_item_features"]
