"""Deterministic random full-corpus recall sanity baseline."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from .data import TestRequest


class RandomRecall:
    def __init__(self, catalog_ids: np.ndarray, seed: int = 42):
        self.catalog_ids = np.asarray(catalog_ids, dtype=np.int64)
        self.seed = int(seed)

    def _sample(self, request_idx: int, size: int) -> list[int]:
        # Sampling a few hundred integer positions avoids materializing a 2M permutation.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(request_idx)]))
        chosen: set[int] = set()
        while len(chosen) < size:
            need = size - len(chosen)
            chosen.update(map(int, rng.integers(0, len(self.catalog_ids), size=need * 2)))
        positions = list(chosen)[:size]
        return self.catalog_ids[np.asarray(positions, dtype=np.int64)].astype(int).tolist()

    def recommend(self, requests: Sequence[TestRequest], k: int = 500, exclude_history: bool = True) -> list[list[int]]:
        result = []
        for request in tqdm(requests, desc="Random recall", leave=False):
            history = set(request.history) if exclude_history else set()
            candidates = self._sample(request.request_idx, k + 64)
            result.append([x for x in candidates if x not in history][:k])
        return result
