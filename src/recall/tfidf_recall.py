"""Exact sparse TF-IDF cosine retrieval over the complete notes corpus."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from .data import TestRequest


@dataclass
class TfidfStats:
    corpus_docs: int
    vocabulary_size: int
    matrix_nnz: int
    matrix_bytes: int
    fit_seconds: float
    retrieval_seconds: float = 0.0


class TfidfRecall:
    def __init__(self, max_features: int = 30_000, min_df: int = 5, max_df: float = 0.8,
                 ngram_range: tuple[int, int] = (2, 2), history_n: int = 20, batch_size: int = 4,
                 n_jobs: int = 8):
        self.max_features = int(max_features)
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.history_n = int(history_n)
        self.batch_size = int(batch_size)
        self.n_jobs = int(n_jobs)
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=ngram_range, max_features=max_features,
            min_df=min_df, max_df=max_df, sublinear_tf=True, norm="l2", dtype=np.float32,
        )
        self.note_ids: np.ndarray | None = None
        self.texts: list[str] | None = None
        self.row_by_id: dict[int, int] | None = None
        self.matrix: sparse.csr_matrix | None = None
        self.stats: TfidfStats | None = None

    def fit(self, note_ids: np.ndarray, texts: list[str], row_by_id: dict[int, int]) -> "TfidfRecall":
        start = time.perf_counter()
        matrix = self.vectorizer.fit_transform(texts).tocsr()
        matrix.sort_indices()
        matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        self.note_ids, self.texts, self.row_by_id, self.matrix = note_ids, texts, row_by_id, matrix
        self.stats = TfidfStats(
            corpus_docs=len(note_ids), vocabulary_size=len(self.vectorizer.vocabulary_),
            matrix_nnz=int(matrix.nnz), matrix_bytes=int(matrix_bytes),
            fit_seconds=time.perf_counter() - start,
        )
        return self

    def _history_text(self, request: TestRequest) -> str:
        assert self.texts is not None and self.row_by_id is not None
        rows = [self.row_by_id[note] for note in request.history[-self.history_n:] if note in self.row_by_id]
        return " ".join(self.texts[row] for row in rows)

    @staticmethod
    def _top_sparse(row: sparse.csr_matrix, note_ids: np.ndarray, k: int) -> list[int]:
        data, indices = row.data, row.indices
        if not len(data):
            return []
        if len(data) > k:
            take = np.argpartition(data, -k)[-k:]
            take = take[np.argsort(data[take], kind="stable")[::-1]]
        else:
            take = np.argsort(data, kind="stable")[::-1]
        return note_ids[indices[take]].astype(int).tolist()

    def recommend(self, requests: Sequence[TestRequest], k: int = 500, exclude_history: bool = True,
                  fallback: Sequence[int] | None = None, use_query_field: bool = False) -> list[list[int]]:
        assert self.matrix is not None and self.note_ids is not None
        start_time = time.perf_counter()
        outputs: list[list[int]] = []
        overfetch = k + self.history_n + 8
        starts = list(range(0, len(requests), self.batch_size))

        def retrieve_batch(start: int) -> list[list[int]]:
            batch = requests[start:start + self.batch_size]
            query_texts = [r.query if use_query_field else self._history_text(r) for r in batch]
            query_matrix = self.vectorizer.transform(query_texts)
            similarities = (query_matrix @ self.matrix.T).tocsr()
            similarities.sum_duplicates()
            batch_outputs = []
            for offset, request in enumerate(batch):
                ranking = self._top_sparse(similarities.getrow(offset), self.note_ids, overfetch)
                if exclude_history:
                    blocked = set(request.history)
                    ranking = [note for note in ranking if note not in blocked]
                ranking = ranking[:k]
                if fallback is not None and len(ranking) < k:
                    seen = set(ranking)
                    blocked = set(request.history) if exclude_history else set()
                    for note in fallback:
                        if note not in seen and note not in blocked:
                            ranking.append(int(note))
                            seen.add(int(note))
                            if len(ranking) >= k:
                                break
                batch_outputs.append(ranking)
            return batch_outputs

        if self.n_jobs == 1:
            batches = map(retrieve_batch, starts)
            for batch_outputs in tqdm(batches, total=len(starts), desc="TF-IDF exact full-corpus"):
                outputs.extend(batch_outputs)
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                batches = executor.map(retrieve_batch, starts)
                for batch_outputs in tqdm(batches, total=len(starts), desc="TF-IDF exact full-corpus"):
                    outputs.extend(batch_outputs)
        if self.stats is not None:
            self.stats.retrieval_seconds = time.perf_counter() - start_time
        return outputs

    def stats_dict(self) -> dict[str, int | float]:
        return asdict(self.stats) if self.stats is not None else {}
