"""Memory-mapped Phase 7 datasets; no Parquet parsing inside workers."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .config import BGE_PATH, HARD_NEGATIVE_PATH, PROTOCOL
from .features import FeatureStore


def load_training_frame(hybrid: bool) -> pd.DataFrame:
    columns = [
        "sample_id",
        "request_id",
        "user_id",
        "history_item_ids",
        "positive_item_id",
        "same_request_positive_ids",
    ]
    if hybrid:
        columns += ["tfidf_ids", "tfidf_ranks"]
        return pq.read_table(HARD_NEGATIVE_PATH, columns=columns).to_pandas()
    from .config import TRAIN_PATH

    return pq.read_table(TRAIN_PATH, columns=columns).to_pandas()


class Phase7Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        store: FeatureStore,
        hybrid: bool,
        history_n: int = 20,
        seed: int = 42,
        limit: int | None = None,
    ):
        if limit is not None:
            chosen = np.sort(
                np.random.default_rng(seed).choice(
                    len(frame), min(limit, len(frame)), replace=False
                )
            )
            frame = frame.iloc[chosen]
        self.frame = frame.reset_index(drop=True)
        self.store, self.hybrid, self.history_n, self.seed = (
            store,
            hybrid,
            history_n,
            seed,
        )
        self.epoch = 0
        count = len(self.frame)
        self.history_note_ids = np.full((count, self.history_n), -1, dtype=np.int64)
        for row_index, values in enumerate(self.frame.history_item_ids.to_numpy()):
            selected = np.asarray(values, dtype=np.int64)[-self.history_n :]
            self.history_note_ids[row_index, : len(selected)] = selected
        self.history_rows = self._item_rows(self.history_note_ids)
        self.history_item_id_rows = self._item_id_rows(self.history_note_ids)
        self.targets = self.frame.positive_item_id.to_numpy(np.int64, copy=True)
        self.target_rows = self._item_rows(self.targets)
        self.target_item_id_rows = self._item_id_rows(self.targets)
        self.users = self.frame.user_id.to_numpy(np.int64, copy=True)
        self.requests = self.frame.request_id.to_numpy(np.int64, copy=True)
        self.same_request = self.frame.same_request_positive_ids.to_list()
        safe_users = np.clip(self.users, 0, len(store.user_lookup) - 1)
        in_user_range = (self.users >= 0) & (self.users < len(store.user_lookup))
        self.user_rows = np.where(
            in_user_range, store.user_lookup[safe_users], -1
        ).astype(np.int64)
        safe_id_users = np.clip(self.users, 0, len(store.user_id_lookup) - 1)
        in_id_range = (self.users >= 0) & (self.users < len(store.user_id_lookup))
        self.user_id_rows = (
            np.where(in_id_range, store.user_id_lookup[safe_id_users], -1) + 1
        ).astype(np.int64)
        self.tfidf_ids = self.frame.tfidf_ids.to_list() if hybrid else None
        self.tfidf_ranks = self.frame.tfidf_ranks.to_list() if hybrid else None

    def __len__(self):
        return len(self.targets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _item_rows(self, note_ids: np.ndarray) -> np.ndarray:
        safe = np.clip(note_ids, 0, len(self.store.item_lookup) - 1)
        rows = self.store.item_lookup[safe]
        in_range = (note_ids >= 0) & (note_ids < len(self.store.item_lookup))
        rows = np.where(in_range, rows, -1)
        return rows.astype(np.int64)

    def _item_id_rows(self, note_ids: np.ndarray) -> np.ndarray:
        safe = np.clip(note_ids, 0, len(self.store.item_id_lookup) - 1)
        rows = self.store.item_id_lookup[safe]
        in_range = (note_ids >= 0) & (note_ids < len(self.store.item_id_lookup))
        rows = np.where(in_range, rows, -1)
        return (rows + 1).astype(np.int64)  # unknown -1 -> padding 0

    def _user(self, index: int):
        feature_row = int(self.user_rows[index])
        id_row = int(self.user_id_rows[index])
        if feature_row < 0:
            categorical = np.zeros(self.store.user_categorical.shape[1], dtype=np.int64)
            numeric = np.zeros(self.store.user_numeric.shape[1], dtype=np.float32)
        else:
            categorical = np.asarray(
                self.store.user_categorical[feature_row], dtype=np.int64
            )
            numeric = np.asarray(self.store.user_numeric[feature_row], dtype=np.float32)
        dense = (
            np.zeros(40, dtype=np.float32)
            if feature_row < 0 or self.store.user_dense is None
            else np.asarray(self.store.user_dense[feature_row], dtype=np.float32)
        )
        return id_row, categorical, numeric, dense, feature_row >= 0

    def _features(
        self,
        note_ids: np.ndarray,
        content: bool,
        rows: np.ndarray | None = None,
        item_id_rows: np.ndarray | None = None,
    ):
        rows = self._item_rows(note_ids) if rows is None else rows
        valid = rows >= 0
        categorical = np.zeros(
            note_ids.shape + (self.store.item_categorical.shape[1],), dtype=np.int64
        )
        numeric = np.zeros(
            note_ids.shape + (self.store.item_numeric.shape[1],), dtype=np.float32
        )
        categorical[valid] = self.store.item_categorical[rows[valid]]
        numeric[valid] = self.store.item_numeric[rows[valid]]
        output = {
            "item_id_row": (
                self._item_id_rows(note_ids) if item_id_rows is None else item_id_rows
            ),
            "categorical": categorical,
            "numeric": numeric,
            "mask": valid,
        }
        if content:
            output["corpus_row"] = rows
        return output

    @staticmethod
    def _rank_sample(ids, ranks, rng: np.random.Generator) -> list[int]:
        ids = np.asarray(ids if ids is not None else (), dtype=np.int64)
        ranks = np.asarray(ranks if ranks is not None else (), dtype=np.int64)
        chosen = []
        for lo, hi in ((11, 50), (51, 100)):
            options = ids[(ranks >= lo) & (ranks <= hi)]
            if len(options):
                chosen.append(int(rng.choice(options)))
        fallback = ids[(ranks >= 11) & ~np.isin(ids, chosen)]
        while len(chosen) < 2 and len(fallback):
            choice = int(rng.choice(fallback))
            chosen.append(choice)
            fallback = fallback[fallback != choice]
        return chosen

    def __getitem__(self, index: int) -> dict:
        history_ids = self.history_note_ids[index]
        target_id = int(self.targets[index])
        history = self._features(
            history_ids,
            self.hybrid,
            self.history_rows[index],
            self.history_item_id_rows[index],
        )
        target = self._features(
            np.asarray(target_id, dtype=np.int64),
            self.hybrid,
            np.asarray(self.target_rows[index]),
            np.asarray(self.target_item_id_rows[index]),
        )
        user_id = int(self.users[index])
        user_id_row, user_cat, user_num, user_dense, profile_available = self._user(
            index
        )
        output = {
            "history_note_ids": history_ids,
            "history_mask": history["mask"],
            "history_item_id_row": history["item_id_row"],
            "history_categorical": history["categorical"],
            "history_numeric": history["numeric"],
            "target_note_id": target_id,
            "target_item_id_row": target["item_id_row"],
            "target_categorical": target["categorical"],
            "target_numeric": target["numeric"],
            "user_id": user_id,
            "user_id_row": user_id_row,
            "user_categorical": user_cat,
            "user_numeric": user_num,
            "user_dense": user_dense,
            "profile_available": profile_available,
            "request_id": int(self.requests[index]),
            "same_request_positive_ids": tuple(map(int, self.same_request[index])),
        }
        if self.hybrid:
            output["history_corpus_row"] = history["corpus_row"]
            output["target_corpus_row"] = target["corpus_row"]
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            negative_ids = self._rank_sample(
                self.tfidf_ids[index], self.tfidf_ranks[index], rng
            )
            negative_note_ids = np.full(2, -1, dtype=np.int64)
            negative_note_ids[: len(negative_ids)] = negative_ids
            negative = self._features(negative_note_ids, True)
            output.update(
                {
                    "negative_note_ids": negative_note_ids,
                    "negative_mask": negative["mask"],
                    "negative_item_id_row": negative["item_id_row"],
                    "negative_categorical": negative["categorical"],
                    "negative_numeric": negative["numeric"],
                    "negative_corpus_row": negative["corpus_row"],
                }
            )
        return output


def collate(rows: list[dict]) -> dict:
    output = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key == "same_request_positive_ids":
            output[key] = values
        elif isinstance(values[0], (bool, np.bool_)):
            output[key] = torch.as_tensor(values, dtype=torch.bool)
        elif isinstance(values[0], np.ndarray):
            output[key] = torch.from_numpy(np.stack(values))
        else:
            output[key] = torch.as_tensor(values, dtype=torch.long)
    return output


class Phase7Collator:
    """Vectorized BGE gather once per batch instead of per Dataset sample."""

    def __init__(self, hybrid: bool):
        self.hybrid = hybrid
        self._bge = None

    def __call__(self, rows: list[dict]) -> dict:
        output = collate(rows)
        if not self.hybrid:
            return output
        if self._bge is None:
            row_bytes = np.dtype(np.float16).itemsize * PROTOCOL.content_dim
            note_count = BGE_PATH.stat().st_size // row_bytes
            self._bge = np.memmap(
                BGE_PATH,
                mode="r",
                dtype=np.float16,
                shape=(note_count, PROTOCOL.content_dim),
            )
        for prefix in ("history", "target", "negative"):
            corpus_rows = output.pop(f"{prefix}_corpus_row").numpy()
            valid = corpus_rows >= 0
            vectors = np.zeros(
                corpus_rows.shape + (PROTOCOL.content_dim,), dtype=np.float32
            )
            vectors[valid] = self._bge[corpus_rows[valid]]
            output[f"{prefix}_content"] = torch.from_numpy(vectors)
        return output
