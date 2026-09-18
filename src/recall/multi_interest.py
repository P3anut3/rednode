"""Per-request unsupervised multi-interest construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import TestRequest
from .user_representation import fit_normalized_kmeans, history_view


def build_interest_queries(requests: Sequence[TestRequest], embeddings: np.ndarray,
                           row_lookup: np.ndarray, k: int, history_n: int = 10,
                           seed: int = 42) -> tuple[np.ndarray, list[np.ndarray],
                                                   list[np.ndarray], list[np.ndarray], dict]:
    """Return flattened centroids plus per-request labels/weights and offsets."""
    flat, labels_by_request, weights_by_request, offsets = [], [], [], []
    empty = 0
    for request in requests:
        view = history_view(request, embeddings, row_lookup, history_n)
        centroids, labels, weights = fit_normalized_kmeans(view.vectors, k, seed)
        if not len(centroids):
            empty += 1
        start = len(flat)
        flat.extend(centroids)
        offsets.append(np.arange(start, start + len(centroids), dtype=np.int64))
        labels_by_request.append(labels)
        weights_by_request.append(weights)
    matrix = (np.asarray(flat, dtype=np.float32)
              if flat else np.empty((0, embeddings.shape[1]), dtype=np.float32))
    stats = {"requested_k": k, "centroid_queries": len(matrix), "empty_requests": empty,
             "history_n": history_n, "seed": seed}
    return matrix, offsets, labels_by_request, weights_by_request, stats

