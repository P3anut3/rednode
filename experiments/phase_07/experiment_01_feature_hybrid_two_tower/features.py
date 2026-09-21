"""Leakage-safe feature cache builder.

Vocabulary and normalization statistics are fitted only on temporal-train
users and temporal-train history/target items.  The final arrays retain every
corpus item and every user_feat row; unseen values map to OOV index 0.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import (
    BGE_PATH,
    FORBIDDEN_ITEM_COLUMNS,
    ITEM_CATEGORICAL,
    ITEM_NUMERIC,
    NOTE_IDS_PATH,
    OUT,
    PROTOCOL,
    ROOT,
    USER_CATEGORICAL,
    USER_NUMERIC,
)
from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import (
    build_contract,
    load_temporal_frames,
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _category(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "<MISSING>"
    value = html.unescape(str(value)).strip()
    return (
        value
        if value and value.lower() not in {"nan", "none", "null", "<na>"}
        else "<MISSING>"
    )


def _vocab(values) -> dict[str, int]:
    unique = sorted({_category(value) for value in values})
    return {value: index + 1 for index, value in enumerate(unique)}  # 0 = OOV


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.clip(values, 0.0, None))


def _fit_normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _notes_frame(columns: list[str]) -> pd.DataFrame:
    frames = []
    for path in sorted((ROOT / "data/notes").glob("*.parquet")):
        frames.append(pq.read_table(path, columns=columns).to_pandas())
    result = pd.concat(frames, ignore_index=True)
    if result.note_idx.duplicated().any():
        raise AssertionError("notes contains duplicate note_idx")
    return result


def build_feature_cache() -> dict:
    """Build Stage-A arrays. This function performs no model training."""
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    train, valid = load_temporal_frames(ROOT)
    note_ids = np.load(NOTE_IDS_PATH, mmap_mode="r")
    if len(note_ids) != PROTOCOL.corpus_items:
        raise AssertionError(f"corpus drift: {len(note_ids)}")
    contract = build_contract(train, note_ids)
    temporal_items = set(map(int, contract.vocab_ids))
    temporal_users = set(map(int, contract.train_users))

    # User cache is small enough for one read. Fit all transforms on train users.
    user_columns = ["user_idx", *USER_CATEGORICAL, *USER_NUMERIC]
    user = (
        pq.read_table(
            sorted((ROOT / "data/user_feat").glob("*.parquet"))[0], columns=user_columns
        )
        .to_pandas()
        .sort_values("user_idx")
    )
    if user.user_idx.duplicated().any():
        raise AssertionError("user_feat contains duplicate user_idx")
    train_user_mask = user.user_idx.isin(temporal_users).to_numpy()
    user_vocabs = {
        column: _vocab(user.loc[train_user_mask, column]) for column in USER_CATEGORICAL
    }
    user_cat = np.zeros((len(user), len(USER_CATEGORICAL)), dtype=np.int32)
    for index, column in enumerate(USER_CATEGORICAL):
        user_cat[:, index] = [
            user_vocabs[column].get(_category(value), 0) for value in user[column]
        ]
    user_raw = np.column_stack(
        [_safe_log1p(user[column].to_numpy()) for column in USER_NUMERIC]
    ).astype(np.float32)
    user_mean, user_std = _fit_normalization(user_raw[train_user_mask])
    user_num = ((user_raw - user_mean) / user_std).astype(np.float32)

    # Only approved structural fields are read. Behavior aggregates never enter memory.
    raw_columns = [
        "note_idx",
        "note_type",
        "taxonomy1_id",
        "taxonomy2_id",
        "commercial_flag",
        "video_duration",
        "video_height",
        "video_width",
        "image_num",
        "content_length",
    ]
    notes_raw = _notes_frame(raw_columns)
    if set(map(int, notes_raw.note_idx)) != set(map(int, note_ids)):
        raise AssertionError("note mapping and feature rows are not one-to-one")
    notes = notes_raw.set_index("note_idx").reindex(note_ids)
    train_item_mask = np.fromiter(
        (int(note) in temporal_items for note in note_ids),
        dtype=np.bool_,
        count=len(note_ids),
    )
    item_vocabs = {
        column: _vocab(notes.loc[train_item_mask, column])
        for column in ITEM_CATEGORICAL
    }
    item_cat = np.zeros((len(notes), len(ITEM_CATEGORICAL)), dtype=np.int32)
    for index, column in enumerate(ITEM_CATEGORICAL):
        item_cat[:, index] = [
            item_vocabs[column].get(_category(value), 0) for value in notes[column]
        ]
    height = np.nan_to_num(notes.video_height.to_numpy(np.float64), nan=0.0)
    width = np.nan_to_num(notes.video_width.to_numpy(np.float64), nan=0.0)
    aspect = np.divide(width, height, out=np.zeros_like(width), where=height > 0)
    aspect = np.clip(np.nan_to_num(aspect, nan=0.0, posinf=0.0), 0.0, 10.0)
    item_raw = np.column_stack(
        (
            _safe_log1p(notes.video_duration.to_numpy()),
            aspect,
            _safe_log1p(notes.image_num.to_numpy()),
            _safe_log1p(notes.content_length.to_numpy()),
        )
    ).astype(np.float32)
    item_mean, item_std = _fit_normalization(item_raw[train_item_mask])
    item_num = ((item_raw - item_mean) / item_std).astype(np.float32)

    np.save(cache / "user_ids.npy", user.user_idx.to_numpy(np.int64))
    np.save(cache / "user_categorical.npy", user_cat)
    np.save(cache / "user_numeric.npy", user_num)
    np.save(cache / "item_ids.npy", np.asarray(note_ids, dtype=np.int64))
    np.save(cache / "item_categorical.npy", item_cat)
    np.save(cache / "item_numeric.npy", item_num)
    np.save(cache / "train_item_id_vocab.npy", contract.vocab_ids)
    np.save(cache / "train_target_item_ids.npy", contract.candidate_ids)
    np.save(cache / "train_user_ids.npy", contract.train_users)

    vocab_payload = {
        "oov_index": 0,
        "user": user_vocabs,
        "item": item_vocabs,
        "item_id_vocabulary": "temporal-train history union positive targets",
        "user_id_vocabulary": "temporal-train users",
    }
    normalization = {
        "fit_scope": "temporal-train only",
        "user_columns": list(USER_NUMERIC),
        "user_transform": "log1p_nonnegative_then_zscore",
        "user_mean": user_mean.tolist(),
        "user_std": user_std.tolist(),
        "item_columns": list(ITEM_NUMERIC),
        "item_transform": [
            "log1p_nonnegative",
            "width/height clipped[0,10]",
            "log1p_nonnegative",
            "log1p_nonnegative",
        ],
        "item_mean": item_mean.tolist(),
        "item_std": item_std.tolist(),
    }
    valid_users = set(map(int, valid.user_id.unique()))
    audit = {
        "status": "cache_complete",
        "train_samples": len(train),
        "valid_samples": len(valid),
        "corpus_items": len(note_ids),
        "feature_users": len(user),
        "train_users": len(temporal_users),
        "train_item_vocabulary": len(temporal_items),
        "train_target_items": len(contract.candidate_ids),
        "valid_unique_users": len(valid_users),
        "valid_unseen_users": len(valid_users - temporal_users),
        "valid_unseen_user_rate": len(valid_users - temporal_users) / len(valid_users),
        "user_profile_coverage": {
            "train_users_with_row": int(user.user_idx.isin(temporal_users).sum()),
            "train_users_total": len(temporal_users),
            "valid_users_with_row": len(valid_users & set(map(int, user.user_idx))),
        },
        "user_vocab_sizes": {name: len(value) for name, value in user_vocabs.items()},
        "item_vocab_sizes": {name: len(value) for name, value in item_vocabs.items()},
        "user_oov_counts": {
            column: int((user_cat[:, i] == 0).sum())
            for i, column in enumerate(USER_CATEGORICAL)
        },
        "item_oov_counts": {
            column: int((item_cat[:, i] == 0).sum())
            for i, column in enumerate(ITEM_CATEGORICAL)
        },
        "cache_arrays": {
            "user_categorical": {
                "shape": list(user_cat.shape),
                "dtype": str(user_cat.dtype),
            },
            "user_numeric": {
                "shape": list(user_num.shape),
                "dtype": str(user_num.dtype),
            },
            "item_categorical": {
                "shape": list(item_cat.shape),
                "dtype": str(item_cat.dtype),
            },
            "item_numeric": {
                "shape": list(item_num.shape),
                "dtype": str(item_num.dtype),
            },
        },
        "forbidden_item_fields_read": [],
        "forbidden_item_fields": list(FORBIDDEN_ITEM_COLUMNS),
        "bge_exists": BGE_PATH.exists(),
        "bge_bytes": BGE_PATH.stat().st_size,
        "test_opened": False,
        "trained": False,
    }
    _json(cache / "train_vocab.json", vocab_payload)
    _json(cache / "normalization.json", normalization)
    _json(cache / "feature_audit.json", audit)
    _json(OUT / "audit/feature_audit.json", audit)
    return audit


def build_h3_dense_cache() -> dict:
    """Conditional H3-only cache; intentionally excluded from Stage A/H0-H2."""
    cache = OUT / "cache"
    if not (OUT / "locks/h3_eligible.json").exists() or not json.loads(
        (OUT / "locks/h3_eligible.json").read_text()
    ).get("eligible"):
        raise RuntimeError(
            "H3 cache is forbidden until H2 has a positive validation decision"
        )
    train, _ = load_temporal_frames(ROOT)
    train_users = set(map(int, train.user_id.unique()))
    columns = ["user_idx", *[f"dense_feat{i}" for i in range(1, 41)]]
    frame = (
        pq.read_table(
            sorted((ROOT / "data/user_feat").glob("*.parquet"))[0], columns=columns
        )
        .to_pandas()
        .sort_values("user_idx")
    )
    values = frame[columns[1:]].to_numpy(np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    train_mask = frame.user_idx.isin(train_users).to_numpy()
    lower = np.quantile(values[train_mask], 0.005, axis=0)
    upper = np.quantile(values[train_mask], 0.995, axis=0)
    clipped = np.clip(values, lower, upper)
    transformed = np.sign(clipped) * np.log1p(np.abs(clipped))
    mean, std = _fit_normalization(transformed[train_mask])
    normalized = ((transformed - mean) / std).astype(np.float32)
    np.save(cache / "user_dense.npy", normalized)
    payload = {
        "fit_scope": "temporal-train users",
        "columns": columns[1:],
        "transform": "train-p0.5/p99.5 clip + signed_log1p + zscore",
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "test_opened": False,
    }
    _json(cache / "dense_normalization.json", payload)
    return payload


class FeatureStore:
    """Memory-mapped feature arrays plus constant-time ID-to-row lookup."""

    def __init__(self, cache: Path | None = None):
        cache = cache or OUT / "cache"
        required = (
            "user_ids",
            "user_categorical",
            "user_numeric",
            "item_ids",
            "item_categorical",
            "item_numeric",
            "train_item_id_vocab",
            "train_target_item_ids",
            "train_user_ids",
        )
        values = {
            name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in required
        }
        self.__dict__.update(values)
        self.vocab = json.loads((cache / "train_vocab.json").read_text())
        self.user_dense = (
            np.load(cache / "user_dense.npy", mmap_mode="r")
            if (cache / "user_dense.npy").exists()
            else None
        )
        max_item = int(self.item_ids.max())
        self.item_lookup = np.full(max_item + 1, -1, dtype=np.int32)
        self.item_lookup[self.item_ids] = np.arange(len(self.item_ids), dtype=np.int32)
        max_user = int(max(self.user_ids.max(), self.train_user_ids.max()))
        self.user_lookup = np.full(max_user + 1, -1, dtype=np.int32)
        self.user_lookup[self.user_ids] = np.arange(len(self.user_ids), dtype=np.int32)
        self.item_id_lookup = np.full(max_item + 1, -1, dtype=np.int32)
        self.item_id_lookup[self.train_item_id_vocab] = np.arange(
            len(self.train_item_id_vocab), dtype=np.int32
        )
        self.user_id_lookup = np.full(max_user + 1, -1, dtype=np.int32)
        self.user_id_lookup[self.train_user_ids] = np.arange(
            len(self.train_user_ids), dtype=np.int32
        )

    def schema(self) -> dict:
        return {
            "user_category_sizes": [
                len(self.vocab["user"][name]) + 1 for name in USER_CATEGORICAL
            ],
            "item_category_sizes": [
                len(self.vocab["item"][name]) + 1 for name in ITEM_CATEGORICAL
            ],
            "user_count": len(self.train_user_ids),
            "item_id_count": len(self.train_item_id_vocab),
            "corpus_count": len(self.item_ids),
        }
