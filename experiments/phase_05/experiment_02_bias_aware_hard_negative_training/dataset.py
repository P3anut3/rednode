"""Frozen-vector dataset with deterministic per-epoch hard-negative sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


SOURCE_DENSE = 0
SOURCE_TFIDF = 1
SOURCE_IMPRESSION = 2


class Phase5HardNegativeDataset(Dataset):
    def __init__(self, parquet_path: Path, embedding_path: Path, note_ids: np.ndarray,
                 lookup: np.ndarray, dim: int = 768, history_n: int = 20,
                 sources: frozenset[str] = frozenset(("dense", "tfidf", "impression")),
                 position_aware: bool = False, seed: int = 42):
        self.table = pq.read_table(parquet_path).combine_chunks()
        self.embedding_path = Path(embedding_path)
        self.note_ids, self.lookup = note_ids, lookup
        self.dim, self.history_n = dim, history_n
        self.sources, self.position_aware, self.seed = sources, position_aware, seed
        self.epoch = 0
        self._embedding = None

    def __len__(self) -> int:
        return self.table.num_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _list(self, column: str, index: int):
        value = self.table[column][index].as_py()
        return value if value is not None else []

    @staticmethod
    def _rank_sample(ids: list[int], ranks: list[int], rng: np.random.Generator) -> list[int]:
        ids_array, ranks_array = np.asarray(ids, dtype=np.int64), np.asarray(ranks, dtype=np.int64)
        selected = []
        for lo, hi in ((11, 50), (51, 100)):
            candidates = ids_array[(ranks_array >= lo) & (ranks_array <= hi)]
            if len(candidates): selected.append(int(rng.choice(candidates)))
        fallback = ids_array[(ranks_array >= 11) & ~np.isin(ids_array, selected)]
        while len(selected) < 2 and len(fallback):
            choice = int(rng.choice(fallback)); selected.append(choice)
            fallback = fallback[fallback != choice]
        return selected

    def _impression_sample(self, index: int, rng: np.random.Generator):
        ids = np.asarray(self._list("impression_ids", index), dtype=np.int64)
        kinds = np.asarray(self._list("impression_types", index), dtype=object)
        simple = np.asarray(self._list("impression_simple_weights", index), dtype=np.float32)
        position = np.asarray(self._list("impression_position_weights", index), dtype=np.float32)
        if not len(ids): return []
        order = []
        for kind in ("before_any_positive", "between_positives", "after_all_positives"):
            candidates = np.flatnonzero(kinds == kind)
            if len(candidates):
                candidates = rng.permutation(candidates)
                order.extend(map(int, candidates))
            if len(order) >= 2: break
        weights = position if self.position_aware else simple
        return [(int(ids[i]), float(weights[i])) for i in order[:2]]

    def __getitem__(self, index: int) -> dict:
        if self._embedding is None:
            self._embedding = np.memmap(self.embedding_path, mode="r", dtype=np.float16,
                                        shape=(len(self.note_ids), self.dim))
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        history_ids_value = list(map(int, self._list("history_item_ids", index)[-self.history_n:]))
        history = np.zeros((self.history_n, self.dim), dtype=np.float32)
        history_ids = np.full(self.history_n, -1, dtype=np.int64)
        history_mask = np.zeros(self.history_n, dtype=np.bool_)
        valid_history = [note for note in history_ids_value
                         if 0 <= note < len(self.lookup) and self.lookup[note] >= 0]
        if valid_history:
            rows = self.lookup[np.asarray(valid_history, dtype=np.int64)]
            history[:len(rows)] = np.asarray(self._embedding[rows], dtype=np.float32)
            history_ids[:len(rows)] = valid_history
            history_mask[:len(rows)] = True
        target_id = int(self.table["positive_item_id"][index].as_py())
        target_row = int(self.lookup[target_id])
        if target_row < 0: raise ValueError(f"target {target_id} absent from corpus")
        target = np.asarray(self._embedding[target_row], dtype=np.float32).copy()
        sampled: list[tuple[int, int, float]] = []
        if "dense" in self.sources:
            for note in self._rank_sample(self._list("dense_ids", index),
                                          self._list("dense_ranks", index), rng):
                sampled.append((note, SOURCE_DENSE, 1.0))
        if "tfidf" in self.sources:
            for note in self._rank_sample(self._list("tfidf_ids", index),
                                          self._list("tfidf_ranks", index), rng):
                sampled.append((note, SOURCE_TFIDF, 1.0))
        if "impression" in self.sources:
            for note, weight in self._impression_sample(index, rng):
                sampled.append((note, SOURCE_IMPRESSION, weight))
        negatives = np.zeros((6, self.dim), dtype=np.float32)
        negative_mask = np.zeros(6, dtype=np.bool_)
        negative_sources = np.full(6, -1, dtype=np.int64)
        negative_weights = np.zeros(6, dtype=np.float32)
        negative_ids = np.full(6, -1, dtype=np.int64)
        for slot, (note, source, weight) in enumerate(sampled[:6]):
            row = int(self.lookup[note]) if 0 <= note < len(self.lookup) else -1
            if row < 0: continue
            negatives[slot] = self._embedding[row]
            negative_mask[slot] = True; negative_sources[slot] = source
            negative_weights[slot] = weight; negative_ids[slot] = note
        return {
            "history": torch.from_numpy(history), "mask": torch.from_numpy(history_mask),
            "history_ids": torch.from_numpy(history_ids), "target": torch.from_numpy(target),
            "target_id": target_id, "user_id": int(self.table["user_id"][index].as_py()),
            "negatives": torch.from_numpy(negatives),
            "negative_mask": torch.from_numpy(negative_mask),
            "negative_sources": torch.from_numpy(negative_sources),
            "negative_weights": torch.from_numpy(negative_weights),
            "negative_ids": torch.from_numpy(negative_ids),
        }
