"""Leakage-safe Phase 6 data contract.

Only the Phase 4/5 temporal train and validation artifacts are opened here.
Test is deliberately implemented in the terminal-test path in ``run.py`` so
ordinary audit/training/selection commands cannot accidentally inspect it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.common.data import TestRequest


@dataclass(frozen=True)
class IdContract:
    vocab_ids: np.ndarray
    candidate_ids: np.ndarray
    train_users: np.ndarray
    item_to_vocab: np.ndarray
    user_to_row: dict[int, int]
    target_frequency: np.ndarray


def load_temporal_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "results/phase_05/experiment_01_request_hard_negative_mining"
    columns = ["sample_id", "request_id", "user_id", "timestamp", "history_item_ids",
               "positive_item_id", "same_request_positive_ids"]
    train = pd.read_parquet(base / "train_samples_base.parquet", columns=columns)
    valid = pd.read_parquet(base / "valid_samples_base.parquet", columns=columns)
    if len(train) != 254_583 or len(valid) != 47_733:
        raise AssertionError(f"temporal split drift: {len(train)=}, {len(valid)=}")
    # Membership was frozen by Phase 4 using each request's minimum positive
    # timestamp.  Phase 5 retained the individual-positive timestamp in this
    # column, so multi-positive requests can visually cross the cutoff even
    # though no request is present in both partitions.
    if set(map(int, train.request_id.unique())) & set(map(int, valid.request_id.unique())):
        raise AssertionError("a request crossed the frozen temporal split")
    return train, valid


def build_contract(train: pd.DataFrame, catalog_ids: np.ndarray) -> IdContract:
    targets = train.positive_item_id.to_numpy(np.int64, copy=False)
    target_ids, counts = np.unique(targets, return_counts=True)
    histories = np.unique(np.concatenate([
        np.asarray(value, dtype=np.int64) for value in train.history_item_ids if len(value)
    ]))
    vocab = np.union1d(histories, target_ids).astype(np.int64, copy=False)
    max_id = int(max(catalog_ids.max(), vocab.max()))
    lookup = np.full(max_id + 1, -1, dtype=np.int32)
    lookup[vocab] = np.arange(len(vocab), dtype=np.int32)
    frequency = np.zeros(len(vocab), dtype=np.int64)
    frequency[lookup[target_ids]] = counts
    users = np.sort(train.user_id.unique().astype(np.int64))
    return IdContract(vocab, target_ids, users, lookup,
                      {int(user): row for row, user in enumerate(users)}, frequency)


def grouped_requests(frame: pd.DataFrame) -> list[TestRequest]:
    grouped: dict[int, list] = {}
    for row in frame.itertuples(index=False):
        key = int(row.request_id)
        if key not in grouped:
            grouped[key] = [int(row.user_id), tuple(map(int, row.history_item_ids)), set()]
        grouped[key][2].add(int(row.positive_item_id))
    return [TestRequest(key, value[0], value[1], frozenset(value[2]))
            for key, value in sorted(grouped.items())]


def item_status(note: int, target_items: set[int], history_items: set[int]) -> str:
    if note in target_items:
        return "train_target_seen"
    if note in history_items:
        return "train_history_only"
    return "completely_unseen"


def positive_status(frame: pd.DataFrame, target_items: set[int], history_items: set[int]) -> dict:
    counts = Counter(item_status(int(note), target_items, history_items)
                     for note in frame.positive_item_id)
    total = len(frame)
    return {name: {"count": int(counts.get(name, 0)),
                   "rate": float(counts.get(name, 0) / total)}
            for name in ("train_target_seen", "train_history_only", "completely_unseen")}


def audit_train_valid(train: pd.DataFrame, valid: pd.DataFrame,
                      contract: IdContract) -> dict:
    targets = set(map(int, contract.candidate_ids))
    vocab = set(map(int, contract.vocab_ids))
    all_history = set(map(int, np.unique(np.concatenate([
        np.asarray(value, dtype=np.int64) for value in train.history_item_ids if len(value)
    ]))))
    histories = all_history - targets
    counts = train.positive_item_id.value_counts()
    buckets = {
        "1": int((counts == 1).sum()),
        "2-3": int(((counts >= 2) & (counts <= 3)).sum()),
        "4-10": int(((counts >= 4) & (counts <= 10)).sum()),
        "10+_overlapping_definition": int((counts >= 10).sum()),
        "11+_mutually_exclusive": int((counts > 10).sum()),
    }
    train_users = set(map(int, contract.train_users))
    valid_users = set(map(int, valid.user_id.unique()))
    return {
        "protocol": "Phase 4/5 global temporal split; test not opened",
        "train_positive_samples": int(len(train)),
        "valid_positive_samples": int(len(valid)),
        "train_requests": int(train.request_id.nunique()),
        "valid_requests": int(valid.request_id.nunique()),
        "train_unique_users": len(train_users),
        "valid_unique_users": len(valid_users),
        "valid_unseen_users": len(valid_users - train_users),
        "valid_unseen_user_rate": len(valid_users - train_users) / len(valid_users),
        "train_target_unique_items": len(targets),
        "train_history_unique_items": len(all_history),
        "history_union_target_items": len(vocab),
        "history_only_items": len(histories),
        "pure_id_candidate_universe": len(targets),
        "target_frequency_item_buckets": buckets,
        "valid_positive_status": positive_status(valid, targets, histories),
        "stored_positive_timestamp_train_min": float(train.timestamp.min()),
        "stored_positive_timestamp_train_max": float(train.timestamp.max()),
        "stored_positive_timestamp_valid_min": float(valid.timestamp.min()),
        "stored_positive_timestamp_valid_max": float(valid.timestamp.max()),
        "timestamp_audit_note": "membership follows Phase 4 request-min timestamp split; stored per-positive timestamps overlap for multi-positive requests",
    }


class Phase6Dataset(Dataset):
    """Shared Pure-ID/M2 dataset; BGE vectors are opened lazily per worker."""

    def __init__(self, frame: pd.DataFrame, contract: IdContract, note_ids: np.ndarray,
                 bge_path: Path | None = None, bge_dim: int = 768, history_n: int = 20):
        frame = frame.reset_index(drop=True)
        self.histories = frame.history_item_ids.tolist()
        self.targets = frame.positive_item_id.to_numpy(np.int64, copy=True)
        self.users = frame.user_id.to_numpy(np.int64, copy=True)
        self.requests = frame.request_id.to_numpy(np.int64, copy=True)
        self.same_request = frame.same_request_positive_ids.tolist()
        self.contract = contract
        self.history_n = history_n
        self.note_ids = note_ids
        max_id = int(note_ids.max())
        self.note_lookup = np.full(max_id + 1, -1, dtype=np.int32)
        self.note_lookup[note_ids] = np.arange(len(note_ids), dtype=np.int32)
        self.bge_path, self.bge_dim, self._bge = bge_path, bge_dim, None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        history = list(map(int, self.histories[index]))[-self.history_n:]
        hist_note = np.full(self.history_n, -1, dtype=np.int64)
        hist_vocab = np.zeros(self.history_n, dtype=np.int64)  # 0 is padding after +1.
        mask = np.zeros(self.history_n, dtype=np.bool_)
        for pos, note in enumerate(history):
            hist_note[pos] = note
            if 0 <= note < len(self.contract.item_to_vocab):
                mapped = int(self.contract.item_to_vocab[note])
                if mapped >= 0:
                    hist_vocab[pos] = mapped + 1
                    mask[pos] = True
        target = int(self.targets[index])
        target_vocab = int(self.contract.item_to_vocab[target]) + 1
        user_id = int(self.users[index])
        user_row = self.contract.user_to_row.get(user_id, -1) + 1
        output = {
            "history_vocab": torch.from_numpy(hist_vocab),
            "history_note_ids": torch.from_numpy(hist_note),
            "mask": torch.from_numpy(mask),
            "target_vocab": target_vocab,
            "target_note_id": target,
            "user_row": user_row,
            "user_id": user_id,
            "request_id": int(self.requests[index]),
            "same_request_positive_ids": tuple(map(int, self.same_request[index])),
        }
        if self.bge_path is not None:
            if self._bge is None:
                self._bge = np.memmap(self.bge_path, mode="r", dtype=np.float16,
                                      shape=(len(self.note_ids), self.bge_dim))
            vectors = np.zeros((self.history_n, self.bge_dim), dtype=np.float32)
            content_mask = np.zeros(self.history_n, dtype=np.bool_)
            for pos, note in enumerate(history):
                note_row = int(self.note_lookup[note]) if 0 <= note < len(self.note_lookup) else -1
                if note_row >= 0:
                    vectors[pos] = self._bge[note_row]
                    content_mask[pos] = True
            target_row = int(self.note_lookup[target])
            output.update({"history_content": torch.from_numpy(vectors),
                           "content_mask": torch.from_numpy(content_mask),
                           "target_content": torch.from_numpy(
                               np.asarray(self._bge[target_row], dtype=np.float32).copy())})
        return output


def collate_phase6(rows: list[dict]) -> dict:
    tensors = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if torch.is_tensor(values[0]):
            tensors[key] = torch.stack(values)
        elif key == "same_request_positive_ids":
            tensors[key] = values
        else:
            tensors[key] = torch.as_tensor(values, dtype=torch.long)
    return tensors
