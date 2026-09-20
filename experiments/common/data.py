"""Shared data preparation for all Qilin recall baselines.

Only this module reads the recommendation parquet structures.  Test candidate
details are reduced immediately to clicked note ids and are never exposed to a
retriever as candidate-generation features.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class TestRequest:
    request_idx: int
    user_idx: int
    history: tuple[int, ...]
    ground_truth: frozenset[int]
    query: str = ""
    positive_interaction_count: int = 0


def _files(root: Path, dataset: str) -> list[Path]:
    files = sorted((root / "data" / dataset).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under data/{dataset}")
    return files


def _valid_category(value: object) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "nan", "none", "null", "<na>"}


class QilinData:
    """Lazy, shared view of train interactions, test labels, and the catalog."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._catalog_ids: np.ndarray | None = None
        self._category_codes: np.ndarray | None = None
        self._category_names: list[str] | None = None
        self._train_user_clicks: dict[int, set[int]] | None = None
        self._train_user_histories: dict[int, set[int]] | None = None
        self._train_exposure: dict[int, int] | None = None
        self._train_click: dict[int, int] | None = None
        self._train_users: set[int] | None = None

    def load_test_requests(self, limit: int | None = None) -> list[TestRequest]:
        requests: list[TestRequest] = []
        columns = ["request_idx", "user_idx", "recent_clicked_note_idxs", "query", "rec_result_details_with_idx"]
        for file in _files(self.root, "recommendation_test"):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=2_000, columns=columns):
                values = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
                for request_idx, user_idx, history, query, details in zip(*(values[c] for c in columns)):
                    # Evaluation-only extraction: discard all unclicked test exposure records.
                    clicked_rows = [int(x["note_idx"]) for x in (details or []) if (x.get("click") or 0) == 1]
                    clicked = frozenset(clicked_rows)
                    requests.append(TestRequest(
                        request_idx=int(request_idx), user_idx=int(user_idx),
                        history=tuple(map(int, history or ())), ground_truth=clicked,
                        query=query or "", positive_interaction_count=len(clicked_rows),
                    ))
                    if limit is not None and len(requests) >= limit:
                        return requests
        return requests

    def _load_train(self) -> None:
        if self._train_user_clicks is not None:
            return
        user_clicks: dict[int, set[int]] = defaultdict(set)
        user_histories: dict[int, set[int]] = defaultdict(set)
        exposure: dict[int, int] = defaultdict(int)
        clicks: dict[int, int] = defaultdict(int)
        train_users: set[int] = set()
        columns = ["user_idx", "recent_clicked_note_idxs", "rec_result_details_with_idx"]
        for file in _files(self.root, "recommendation_train"):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=5_000, columns=columns):
                values = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
                for user, history, details in zip(*(values[c] for c in columns)):
                    user = int(user)
                    train_users.add(user)
                    user_histories[user].update(map(int, history or ()))
                    for row in details or ():
                        note = int(row["note_idx"])
                        exposure[note] += 1
                        if (row.get("click") or 0) == 1:
                            clicks[note] += 1
                            user_clicks[user].add(note)
        self._train_user_clicks = dict(user_clicks)
        self._train_user_histories = dict(user_histories)
        self._train_exposure = dict(exposure)
        self._train_click = dict(clicks)
        self._train_users = train_users

    @property
    def train_user_clicks(self) -> dict[int, set[int]]:
        self._load_train()
        return self._train_user_clicks or {}

    @property
    def train_user_histories(self) -> dict[int, set[int]]:
        self._load_train()
        return self._train_user_histories or {}

    @property
    def train_exposure_counts(self) -> dict[int, int]:
        self._load_train()
        return self._train_exposure or {}

    @property
    def train_click_counts(self) -> dict[int, int]:
        self._load_train()
        return self._train_click or {}

    @property
    def train_users(self) -> set[int]:
        self._load_train()
        return self._train_users or set()

    @property
    def train_exposed_items(self) -> set[int]:
        return set(self.train_exposure_counts)

    def load_catalog_ids(self) -> np.ndarray:
        if self._catalog_ids is None:
            parts = []
            for file in _files(self.root, "notes"):
                for batch in pq.ParquetFile(file).iter_batches(batch_size=200_000, columns=["note_idx"]):
                    parts.append(batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
            ids = np.concatenate(parts)
            if len(np.unique(ids)) != len(ids):
                raise ValueError("notes.note_idx is not unique")
            self._catalog_ids = ids
        return self._catalog_ids

    def load_categories(self) -> tuple[np.ndarray, list[str]]:
        """Return category code by note id; taxonomy2 falls back to taxonomy1."""
        if self._category_codes is not None:
            return self._category_codes, self._category_names or []
        ids = self.load_catalog_ids()
        max_id = int(ids.max())
        if max_id > len(ids) * 2:
            raise ValueError("note_idx is too sparse for a direct category lookup array")
        codes = np.full(max_id + 1, -1, dtype=np.int32)
        category_to_code: dict[str, int] = {}
        category_names: list[str] = []
        for file in _files(self.root, "notes"):
            columns = ["note_idx", "taxonomy1_id", "taxonomy2_id"]
            for batch in pq.ParquetFile(file).iter_batches(batch_size=100_000, columns=columns):
                note_ids, tax1, tax2 = (batch.column(i).to_pylist() for i in range(3))
                for note, c1, c2 in zip(note_ids, tax1, tax2):
                    category = str(c2) if _valid_category(c2) else (str(c1) if _valid_category(c1) else "")
                    if not category:
                        continue
                    code = category_to_code.get(category)
                    if code is None:
                        code = len(category_names)
                        category_to_code[category] = code
                        category_names.append(category)
                    codes[int(note)] = code
        self._category_codes = codes
        self._category_names = category_names
        return codes, category_names

    def iter_note_text_batches(self, batch_size: int = 50_000) -> Iterator[tuple[np.ndarray, list[str]]]:
        """Yield catalog ids and title+content, never behavior aggregate columns."""
        columns = ["note_idx", "note_title", "note_content"]
        for file in _files(self.root, "notes"):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=batch_size, columns=columns):
                ids, titles, contents = (batch.column(i).to_pylist() for i in range(3))
                texts = [f"{title or ''} {content or ''}".strip() for title, content in zip(titles, contents)]
                yield np.asarray(ids, dtype=np.int64), texts

    def load_note_texts(self, limit: int | None = None,
                        required_ids: set[int] | None = None) -> tuple[np.ndarray, list[str], dict[int, int]]:
        """Load text records, optionally retaining a bounded smoke corpus.

        A bounded corpus always includes required_ids (normally smoke-request
        histories), then fills remaining slots in catalog order. The parquet is
        still scanned so high note ids can be found, but unused text is not kept.
        """
        id_parts: list[np.ndarray] = []
        texts: list[str] = []
        required_ids = set(required_ids or ())
        required_found: dict[int, str] = {}
        filler_ids: list[int] = []
        filler_texts: list[str] = []
        for ids, batch_texts in self.iter_note_text_batches():
            if limit is None:
                id_parts.append(ids)
                texts.extend(batch_texts)
                continue
            for note, text in zip(ids, batch_texts):
                note = int(note)
                if note in required_ids:
                    required_found[note] = text
                elif len(filler_ids) < limit:
                    filler_ids.append(note)
                    filler_texts.append(text)
        if limit is None:
            ids = np.concatenate(id_parts)
        else:
            missing = required_ids - set(required_found)
            if missing:
                raise ValueError(f"Required smoke note ids absent from catalog: {sorted(missing)[:10]}")
            required_order = sorted(required_found)
            room = max(0, limit - len(required_order))
            ids = np.asarray(required_order + filler_ids[:room], dtype=np.int64)
            texts = [required_found[note] for note in required_order] + filler_texts[:room]
        row_by_id = {int(note): row for row, note in enumerate(ids)}
        return ids, texts, row_by_id
