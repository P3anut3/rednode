"""Leakage-audited Phase 4 temporal train/validation examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from experiments.common.data import TestRequest


@dataclass(frozen=True)
class Phase4Example:
    request_idx: int
    user_idx: int
    history: tuple[int, ...]
    target: int
    timestamp: float


def phase4_load_train_examples(root: Path, valid_fraction: float = 0.15,
                               history_n: int = 10) -> tuple[list[Phase4Example], list[Phase4Example], dict]:
    files = sorted((root / "data/recommendation_train").glob("*.parquet"))
    if not files: raise FileNotFoundError("data/recommendation_train/*.parquet")
    rows, overlap, duplicate, empty_hist, timestamp_spread = [], 0, 0, 0, 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=[
                "request_idx", "user_idx", "recent_clicked_note_idxs", "rec_result_details_with_idx"]):
            fields = [batch.column(i).to_pylist() for i in range(4)]
            for request_idx, user_idx, history, details in zip(*fields):
                history = tuple(map(int, (history or [])[-history_n:]))
                empty_hist += not bool(history)
                positive = [row for row in (details or []) if int(row.get("click") or 0) == 1]
                if not positive: continue
                times = [float(row["request_timestamp"]) for row in positive
                         if row.get("request_timestamp") is not None]
                if not times: continue
                timestamp_spread += bool(max(times) != min(times))
                request_time = min(times)
                seen = set()
                for row in positive:
                    target = int(row["note_idx"])
                    if target in seen:
                        duplicate += 1; continue
                    seen.add(target)
                    if target in history:
                        overlap += 1; continue
                    rows.append(Phase4Example(int(request_idx), int(user_idx), history,
                                               target, request_time))
    request_times = np.asarray(sorted({row.timestamp for row in rows}), dtype=np.float64)
    cutoff = float(np.quantile(request_times, 1 - valid_fraction))
    train = [row for row in rows if row.timestamp < cutoff]
    valid = [row for row in rows if row.timestamp >= cutoff]
    train_requests = {row.request_idx for row in train}
    valid_requests = {row.request_idx for row in valid}
    if train_requests & valid_requests: raise AssertionError("request crossed temporal split")
    if max(row.timestamp for row in train) >= min(row.timestamp for row in valid):
        raise AssertionError("validation is not strictly later")
    train_users, valid_users = {row.user_idx for row in train}, {row.user_idx for row in valid}
    stats = {
        "raw_positive_interactions_after_dedup_and_overlap_filter": len(rows),
        "history_target_overlap_removed": overlap, "duplicate_target_removed": duplicate,
        "empty_history_requests": empty_hist, "positive_timestamp_spread_requests": timestamp_spread,
        "temporal_cutoff_epoch_seconds": cutoff, "train_samples": len(train),
        "valid_samples": len(valid), "train_requests": len(train_requests),
        "valid_requests": len(valid_requests), "train_users": len(train_users),
        "valid_users": len(valid_users), "overlap_users": len(train_users & valid_users),
        "train_timestamp_min": min(row.timestamp for row in train),
        "train_timestamp_max": max(row.timestamp for row in train),
        "valid_timestamp_min": min(row.timestamp for row in valid),
        "valid_timestamp_max": max(row.timestamp for row in valid),
        "history_temporal_assumption": "recent_clicked_note_idxs is pre-request history; individual history timestamps unavailable",
    }
    return train, valid, stats


class Phase4UserTowerDataset(Dataset):
    def __init__(self, examples: list[Phase4Example], embedding_path: Path,
                 note_ids: np.ndarray, lookup: np.ndarray, dim: int, history_n: int = 10):
        self.examples = examples
        self.embedding_path = embedding_path
        self.note_ids = note_ids
        self.lookup = lookup
        self.dim = dim
        self.history_n = history_n
        self._embedding = None

    def __len__(self) -> int: return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        if self._embedding is None:
            self._embedding = np.memmap(self.embedding_path, mode="r", dtype=np.float16,
                                        shape=(len(self.note_ids), self.dim))
        example = self.examples[index]
        history = np.zeros((self.history_n, self.dim), dtype=np.float32)
        mask = np.zeros(self.history_n, dtype=np.bool_)
        hist_ids = np.full(self.history_n, -1, dtype=np.int64)
        valid = [n for n in example.history[-self.history_n:]
                 if 0 <= n < len(self.lookup) and self.lookup[n] >= 0]
        for i, note in enumerate(valid):
            hist_ids[i] = note
            history[i] = self._embedding[int(self.lookup[note])]
            mask[i] = True
        target_row = int(self.lookup[example.target])
        if target_row < 0: raise ValueError("target not in frozen item corpus")
        target = np.asarray(self._embedding[target_row], dtype=np.float32).copy()
        return {"history": torch.from_numpy(history), "mask": torch.from_numpy(mask),
                "target": torch.from_numpy(target), "target_id": example.target,
                "user_id": example.user_idx, "history_ids": torch.from_numpy(hist_ids)}
