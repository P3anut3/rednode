"""Faiss exact and IVF-PQ indices for normalized dense item embeddings."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import faiss
import numpy as np


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def embedding_memmap(path: Path, items: int, dim: int) -> np.memmap:
    return np.memmap(path, mode="r", dtype=np.float16, shape=(items, dim))


def build_flat_ip(embeddings: np.ndarray, index_path: Path, add_batch: int = 20_000) -> tuple[faiss.Index, dict]:
    faiss.omp_set_num_threads(min(64, faiss.omp_get_max_threads()))
    start = time.perf_counter()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    for offset in range(0, len(embeddings), add_batch):
        index.add(np.asarray(embeddings[offset:offset + add_batch], dtype=np.float32))
    build_seconds = time.perf_counter() - start
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    stats = {"type": "IndexFlatIP", "items": index.ntotal, "dimension": index.d,
             "build_seconds": build_seconds, "index_file_bytes": index_path.stat().st_size,
             "peak_rss_gib": peak_rss_gib()}
    return index, stats


def build_ivfpq(embeddings: np.ndarray, index_path: Path, nlist: int = 4096, m: int = 96,
                nbits: int = 8, train_size: int = 200_000, seed: int = 42,
                add_batch: int = 20_000) -> tuple[faiss.IndexIVFPQ, dict]:
    faiss.omp_set_num_threads(min(64, faiss.omp_get_max_threads()))
    start = time.perf_counter()
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFPQ(quantizer, embeddings.shape[1], nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(seed)
    sample_ids = np.sort(rng.choice(len(embeddings), size=min(train_size, len(embeddings)), replace=False))
    training = np.asarray(embeddings[sample_ids], dtype=np.float32)
    train_start = time.perf_counter(); index.train(training); train_seconds = time.perf_counter() - train_start
    del training
    add_start = time.perf_counter()
    for offset in range(0, len(embeddings), add_batch):
        index.add(np.asarray(embeddings[offset:offset + add_batch], dtype=np.float32))
    add_seconds = time.perf_counter() - add_start
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    stats = {"type": "IndexIVFPQ", "items": index.ntotal, "dimension": index.d,
             "nlist": nlist, "m": m, "nbits": nbits, "train_size": len(sample_ids),
             "train_seconds": train_seconds, "add_seconds": add_seconds,
             "build_seconds": time.perf_counter() - start,
             "index_file_bytes": index_path.stat().st_size, "peak_rss_gib": peak_rss_gib()}
    return index, stats


def load_index(path: Path, nprobe: int | None = None) -> faiss.Index:
    index = faiss.read_index(str(path))
    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = nprobe
    return index


def search_index(index: faiss.Index, queries: np.ndarray, topk: int = 600,
                 batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, dict]:
    all_scores, all_rows, normalized_batch_latency = [], [], []
    start_all = time.perf_counter()
    for offset in range(0, len(queries), batch_size):
        batch = np.ascontiguousarray(queries[offset:offset + batch_size], dtype=np.float32)
        start = time.perf_counter(); scores, rows = index.search(batch, topk); elapsed = time.perf_counter() - start
        all_scores.append(scores); all_rows.append(rows)
        normalized_batch_latency.extend([elapsed / len(batch)] * len(batch))
    total = time.perf_counter() - start_all
    lat = np.asarray(normalized_batch_latency) * 1000
    stats = {"query_seconds": total, "requests": len(queries), "mean_latency_ms": total / len(queries) * 1000,
             "p50_batch_normalized_latency_ms": float(np.quantile(lat, .5)),
             "p95_batch_normalized_latency_ms": float(np.quantile(lat, .95)),
             "search_batch_size": batch_size, "topk_searched": topk, "peak_rss_gib": peak_rss_gib()}
    return np.vstack(all_scores), np.vstack(all_rows), stats

