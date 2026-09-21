"""Proxy and one-pass GPU exact retrieval for Phase 7."""

from __future__ import annotations

import time

import numpy as np
import torch

from .config import DeadlineExceeded, check_deadline


def filter_history(
    raw_rows: np.ndarray,
    candidate_ids: np.ndarray,
    histories,
    topk: int = 500,
    deadline: float | None = None,
) -> list[list[int]]:
    output = []
    for request_index, (rows, history) in enumerate(zip(raw_rows, histories)):
        if request_index % 250 == 0:
            check_deadline(deadline, "history filtering")
        blocked, seen, selected = set(map(int, history)), set(), []
        for row in rows:
            note = int(candidate_ids[int(row)])
            if note in blocked or note in seen:
                continue
            seen.add(note)
            selected.append(note)
            if len(selected) == topk:
                break
        if len(selected) != topk:
            raise AssertionError(f"retrieval underflow: {len(selected)}")
        output.append(selected)
    return output


def gpu_exact_search(
    corpus: np.ndarray,
    queries: np.ndarray,
    topk: int,
    device: str,
    query_batch: int = 256,
    deadline: float | None = None,
):
    """Prefer FAISS GPU FlatIP; otherwise keep the full corpus on one GPU."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal Phase 7 exact retrieval requires CUDA")
    try:
        import faiss

        if hasattr(faiss, "StandardGpuResources"):
            gpu = int(device.split(":", 1)[1]) if ":" in device else 0
            resources = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(corpus.shape[1])
            index = faiss.index_cpu_to_gpu(resources, gpu, cpu_index)
            started = time.perf_counter()
            index.add(np.ascontiguousarray(corpus, dtype=np.float32))
            check_deadline(deadline, "FAISS GPU index build")
            index_build_seconds = time.perf_counter() - started
            search_started = time.perf_counter()
            score_parts, row_parts = [], []
            for start in range(0, len(queries), query_batch):
                check_deadline(deadline, "FAISS GPU query search")
                score, row = index.search(
                    np.ascontiguousarray(
                        queries[start : start + query_batch], dtype=np.float32
                    ),
                    topk,
                )
                score_parts.append(score)
                row_parts.append(row)
            query_search_seconds = time.perf_counter() - search_started
            return (
                np.vstack(score_parts),
                np.vstack(row_parts),
                index_build_seconds + query_search_seconds,
                "faiss_gpu_flat_ip",
                {
                    "index_build_seconds": index_build_seconds,
                    "query_search_seconds": query_search_seconds,
                },
            )
    except DeadlineExceeded:
        raise
    except (ImportError, AttributeError, RuntimeError):
        pass
    # The 1.98M x 128 float32 Phase-7 corpus is about 0.95 GiB, so keep it
    # resident on one 24-GiB GPU.  The previous chunked fallback retransferred
    # every corpus chunk for every query batch and was PCIe-bound.
    build_started = time.perf_counter()
    resident_corpus = torch.from_numpy(
        np.ascontiguousarray(corpus, dtype=np.float32)
    ).to(device=device, non_blocking=False)
    torch.cuda.synchronize(device)
    index_build_seconds = time.perf_counter() - build_started
    check_deadline(deadline, "GPU resident corpus upload")
    search_started = time.perf_counter()
    all_scores, all_rows = [], []
    with torch.inference_mode():
        for q_start in range(0, len(queries), query_batch):
            check_deadline(deadline, "GPU exact query search")
            query = torch.as_tensor(
                queries[q_start : q_start + query_batch],
                device=device,
                dtype=torch.float32,
            )
            score = query @ resident_corpus.T
            best_score, best_row = torch.topk(score, topk, dim=1)
            all_scores.append(best_score.cpu().numpy())
            all_rows.append(best_row.cpu().numpy())
            del score, best_score, best_row, query
    torch.cuda.synchronize(device)
    query_search_seconds = time.perf_counter() - search_started
    del resident_corpus
    return (
        np.vstack(all_scores),
        np.vstack(all_rows),
        query_search_seconds,
        "torch_gpu_resident_exact_ip",
        {
            "index_build_seconds": index_build_seconds,
            "query_search_seconds": query_search_seconds,
        },
    )


def deterministic_proxy_candidates(
    catalog: np.ndarray, positive_ids: set[int], size: int, seed: int = 42
) -> np.ndarray:
    positive = np.asarray(sorted(positive_ids), dtype=np.int64)
    if len(positive) > size:
        raise ValueError("proxy positives exceed candidate budget")
    remaining = np.setdiff1d(catalog, positive, assume_unique=False)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        remaining, size=min(size - len(positive), len(remaining)), replace=False
    )
    return np.sort(np.concatenate((positive, sampled))).astype(np.int64)


def validate_topk(
    rankings: list[list[int]],
    corpus_ids: np.ndarray,
    histories,
    deadline: float | None = None,
) -> None:
    corpus = set(map(int, corpus_ids))
    for request_index, (ranking, history) in enumerate(zip(rankings, histories)):
        if request_index % 250 == 0:
            check_deadline(deadline, "TopK validation")
        if len(ranking) != 500 or len(set(ranking)) != 500:
            raise AssertionError("Top500 must contain 500 unique items")
        if not set(ranking).issubset(corpus):
            raise AssertionError("candidate outside corpus")
        if set(ranking) & set(map(int, history)):
            raise AssertionError("history was not filtered")
