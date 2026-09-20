"""Dense user-vector construction, history filtering, and shared evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import TestRequest


def make_row_lookup(note_ids: np.ndarray) -> np.ndarray:
    lookup = np.full(int(note_ids.max()) + 1, -1, dtype=np.int64)
    lookup[note_ids] = np.arange(len(note_ids), dtype=np.int64)
    return lookup


def build_user_embeddings(requests: Sequence[TestRequest], item_embeddings: np.ndarray,
                          row_lookup: np.ndarray, history_n: int = 20,
                          weighting: str = "mean", decay: float = 0.8) -> tuple[np.ndarray, dict]:
    dim = item_embeddings.shape[1]
    queries = np.zeros((len(requests), dim), dtype=np.float32)
    empty = 0
    used_lengths = []
    for idx, request in enumerate(requests):
        history = request.history[-history_n:]
        rows = [int(row_lookup[note]) for note in history if 0 <= note < len(row_lookup) and row_lookup[note] >= 0]
        if not rows:
            empty += 1
            continue
        vectors = np.asarray(item_embeddings[rows], dtype=np.float32)
        if weighting == "mean":
            query = vectors.mean(axis=0)
        elif weighting == "recency":
            # History order is assumed oldest -> newest; newest receives weight 1.
            weights = decay ** np.arange(len(rows) - 1, -1, -1, dtype=np.float32)
            query = (vectors * weights[:, None]).sum(axis=0) / weights.sum()
        else:
            raise ValueError(weighting)
        norm = np.linalg.norm(query)
        if norm > 0:
            queries[idx] = query / norm
        else:
            empty += 1
        used_lengths.append(len(rows))
    return queries, {"history_n": history_n, "weighting": weighting, "recency_decay": decay,
                     "empty_query_requests": empty, "mean_history_items_used": float(np.mean(used_lengths))}


def rows_to_filtered_rankings(rows: np.ndarray, note_ids: np.ndarray,
                              requests: Sequence[TestRequest], k: int = 500) -> list[list[int]]:
    rankings: list[list[int]] = []
    for candidate_rows, request in zip(rows, requests):
        blocked, seen, ranking = set(request.history), set(), []
        for row in candidate_rows:
            if row < 0:
                continue
            note = int(note_ids[row])
            if note in blocked or note in seen:
                continue
            seen.add(note); ranking.append(note)
            if len(ranking) == k:
                break
        rankings.append(ranking)
    return rankings


def validate_rankings(rankings: Sequence[Sequence[int]], requests: Sequence[TestRequest],
                      catalog_set: set[int], k: int = 500) -> dict:
    short = 0
    for ranking, request in zip(rankings, requests):
        if len(ranking) < k: short += 1
        if len(ranking) != len(set(ranking)): raise AssertionError("duplicate retrieved item")
        if not set(ranking).issubset(catalog_set): raise AssertionError("retrieved item outside catalog")
        if set(ranking) & set(request.history): raise AssertionError("history filtering failed")
    return {"requests": len(rankings), "shorter_than_topk": short, "validated_topk": k}
