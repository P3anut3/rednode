"""Sparse cosine-normalized ItemCF without a dense item-item matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from experiments.common.data import TestRequest


@dataclass
class ItemCFBuildStats:
    users: int
    active_items: int
    raw_undirected_pairs: int
    coalesced_directed_edges: int
    retained_directed_edges: int


class ItemCF:
    def __init__(self, neighbors_per_item: int = 200):
        self.neighbors_per_item = int(neighbors_per_item)
        self.item_ids: np.ndarray | None = None
        self.lookup: np.ndarray | None = None
        self.neighbors: sparse.csr_matrix | None = None
        self.stats: ItemCFBuildStats | None = None

    def fit(self, interactions: Iterable[set[int]]) -> "ItemCF":
        user_sets = [np.asarray(sorted(items), dtype=np.int64) for items in interactions if items]
        if not user_sets:
            raise ValueError("No ItemCF interactions")
        item_ids = np.unique(np.concatenate(user_sets))
        lookup = np.full(int(item_ids.max()) + 1, -1, dtype=np.int32)
        lookup[item_ids] = np.arange(len(item_ids), dtype=np.int32)
        compact_sets = [lookup[items] for items in user_sets]
        frequencies = np.zeros(len(item_ids), dtype=np.float32)
        for items in compact_sets:
            frequencies[items] += 1.0

        pair_count = sum(len(items) * (len(items) - 1) // 2 for items in compact_sets)
        rows = np.empty(pair_count, dtype=np.int32)
        cols = np.empty(pair_count, dtype=np.int32)
        cursor = 0
        for items in tqdm(compact_sets, desc="ItemCF co-occurrence", leave=False):
            n = len(items)
            if n < 2:
                continue
            left, right = np.triu_indices(n, 1)
            size = len(left)
            rows[cursor:cursor + size] = items[left]
            cols[cursor:cursor + size] = items[right]
            cursor += size
        values = np.ones(cursor, dtype=np.float32)
        upper = sparse.coo_matrix(
            (values, (rows[:cursor], cols[:cursor])), shape=(len(item_ids), len(item_ids)), dtype=np.float32
        ).tocsr()
        upper.sum_duplicates()
        adjacency = upper + upper.T
        del rows, cols, values, upper
        inv_sqrt = np.zeros_like(frequencies)
        positive = frequencies > 0
        inv_sqrt[positive] = 1.0 / np.sqrt(frequencies[positive])
        adjacency = adjacency.multiply(inv_sqrt[:, None]).multiply(inv_sqrt[None, :]).tocsr()
        adjacency.sort_indices()
        coalesced_edges = int(adjacency.nnz)

        indptr = np.zeros(len(item_ids) + 1, dtype=np.int64)
        kept_indices: list[np.ndarray] = []
        kept_data: list[np.ndarray] = []
        for row in tqdm(range(len(item_ids)), desc="ItemCF top-neighbor pruning", leave=False):
            start, end = adjacency.indptr[row], adjacency.indptr[row + 1]
            data = adjacency.data[start:end]
            indices = adjacency.indices[start:end]
            if len(data) > self.neighbors_per_item:
                take = np.argpartition(data, -self.neighbors_per_item)[-self.neighbors_per_item:]
                take = take[np.argsort(data[take])[::-1]]
                data, indices = data[take], indices[take]
            elif len(data):
                order = np.argsort(data)[::-1]
                data, indices = data[order], indices[order]
            kept_data.append(data.astype(np.float32, copy=True))
            kept_indices.append(indices.astype(np.int32, copy=True))
            indptr[row + 1] = indptr[row] + len(data)
        data = np.concatenate(kept_data) if kept_data else np.array([], dtype=np.float32)
        indices = np.concatenate(kept_indices) if kept_indices else np.array([], dtype=np.int32)
        neighbors = sparse.csr_matrix((data, indices, indptr), shape=adjacency.shape)
        self.item_ids, self.lookup, self.neighbors = item_ids, lookup, neighbors
        self.stats = ItemCFBuildStats(
            users=len(user_sets), active_items=len(item_ids), raw_undirected_pairs=pair_count,
            coalesced_directed_edges=coalesced_edges, retained_directed_edges=int(neighbors.nnz),
        )
        return self

    def _native(self, history: Sequence[int], k: int, exclude_history: bool) -> list[int]:
        assert self.lookup is not None and self.neighbors is not None and self.item_ids is not None
        scores: dict[int, float] = {}
        for note in dict.fromkeys(map(int, history)):
            if note < 0 or note >= len(self.lookup):
                continue
            row = int(self.lookup[note])
            if row < 0:
                continue
            start, end = self.neighbors.indptr[row], self.neighbors.indptr[row + 1]
            for compact, score in zip(self.neighbors.indices[start:end], self.neighbors.data[start:end]):
                candidate = int(self.item_ids[compact])
                scores[candidate] = scores.get(candidate, 0.0) + float(score)
        if exclude_history:
            for note in history:
                scores.pop(int(note), None)
        return [note for note, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]

    def recommend(self, requests: Sequence[TestRequest], k: int = 500, exclude_history: bool = True,
                  fallback: Sequence[int] | None = None) -> list[list[int]]:
        outputs = []
        for request in tqdm(requests, desc="ItemCF recall", leave=False):
            ranking = self._native(request.history, k, exclude_history)
            if fallback is not None and len(ranking) < k:
                seen = set(ranking)
                blocked = set(request.history) if exclude_history else set()
                for note in fallback:
                    if note not in seen and note not in blocked:
                        ranking.append(int(note))
                        seen.add(int(note))
                        if len(ranking) >= k:
                            break
            outputs.append(ranking)
        return outputs
