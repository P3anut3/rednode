"""Fixed-encoder user representations and history-diversity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from experiments.common.data import TestRequest


@dataclass(frozen=True)
class HistoryView:
    note_ids: tuple[int, ...]
    rows: np.ndarray
    vectors: np.ndarray


def history_view(request: TestRequest, embeddings: np.ndarray, row_lookup: np.ndarray,
                 history_n: int = 10) -> HistoryView:
    """Return the last N valid history items, preserving their source order."""
    notes, rows = [], []
    for note in request.history[-history_n:]:
        note = int(note)
        if 0 <= note < len(row_lookup) and row_lookup[note] >= 0:
            notes.append(note)
            rows.append(int(row_lookup[note]))
    row_array = np.asarray(rows, dtype=np.int64)
    vectors = (np.asarray(embeddings[row_array], dtype=np.float32)
               if len(row_array) else np.empty((0, embeddings.shape[1]), dtype=np.float32))
    return HistoryView(tuple(notes), row_array, vectors)


def normalized_mean(vectors: np.ndarray) -> np.ndarray:
    if not len(vectors):
        return np.zeros(vectors.shape[1], dtype=np.float32)
    value = vectors.mean(axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else np.zeros_like(value)


def history_query_matrix(requests: Sequence[TestRequest], embeddings: np.ndarray,
                         row_lookup: np.ndarray, history_n: int = 10,
                         mode: str = "mean") -> tuple[np.ndarray, dict]:
    queries = np.zeros((len(requests), embeddings.shape[1]), dtype=np.float32)
    lengths, empty = [], 0
    for idx, request in enumerate(requests):
        view = history_view(request, embeddings, row_lookup, history_n)
        lengths.append(len(view.rows))
        if not len(view.rows):
            empty += 1
        elif mode == "mean":
            queries[idx] = normalized_mean(view.vectors)
        elif mode == "recent":
            queries[idx] = view.vectors[-1]
        else:
            raise ValueError(mode)
    return queries, {
        "mode": mode, "history_n": history_n, "empty_requests": empty,
        "mean_valid_history": float(np.mean(lengths)),
    }


def pairwise_cosine_stats(vectors: np.ndarray) -> tuple[float, float, float]:
    if len(vectors) < 2:
        return float("nan"), float("nan"), float("nan")
    similarity = vectors @ vectors.T
    values = similarity[np.triu_indices(len(vectors), 1)]
    return float(values.mean()), float(values.std()), float(values.min())


def fit_normalized_kmeans(vectors: np.ndarray, k: int, seed: int = 42,
                          n_init: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster one request's history and return normalized centroids/labels/weights."""
    if not len(vectors):
        return (np.empty((0, vectors.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
    actual_k = min(int(k), len(vectors))
    if actual_k == 1:
        labels = np.zeros(len(vectors), dtype=np.int32)
        centroids = normalized_mean(vectors)[None, :]
    else:
        model = KMeans(n_clusters=actual_k, random_state=seed, n_init=n_init,
                       max_iter=100, algorithm="lloyd")
        labels = model.fit_predict(vectors).astype(np.int32)
        centroids = model.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids /= np.maximum(norms, 1e-12)
    sizes = np.bincount(labels, minlength=actual_k).astype(np.float32)
    return centroids, labels, sizes / sizes.sum()


def diversity_record(request: TestRequest, embeddings: np.ndarray, row_lookup: np.ndarray,
                     history_n: int = 10, seed: int = 42,
                     silhouette_threshold: float = 0.25) -> dict:
    view = history_view(request, embeddings, row_lookup, history_n)
    mean, std, minimum = pairwise_cosine_stats(view.vectors)
    scores: dict[int, float] = {}
    upper = min(4, len(view.vectors) - 1)
    for k in range(2, upper + 1):
        _, labels, _ = fit_normalized_kmeans(view.vectors, k, seed)
        if len(np.unique(labels)) > 1:
            scores[k] = float(silhouette_score(view.vectors, labels, metric="cosine"))
    best_k = max(scores, key=scores.get) if scores else 1
    best_score = scores.get(best_k, float("nan"))
    operational_k = best_k if scores and best_score >= silhouette_threshold else 1
    return {
        "request_idx": request.request_idx, "user_idx": request.user_idx,
        "history_length": len(view.rows), "pairwise_cosine_mean": mean,
        "pairwise_cosine_std": std, "pairwise_cosine_min": minimum,
        **{f"silhouette_k{k}": scores.get(k, float("nan")) for k in (2, 3, 4)},
        "best_silhouette_k": best_k, "best_silhouette": best_score,
        "operational_interest_count": operational_k,
    }
