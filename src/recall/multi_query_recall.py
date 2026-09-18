"""Candidate generation and exact re-scoring for multi-query user recall."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from .data import TestRequest
from .user_representation import history_view


def search_cached(index: faiss.Index, queries: np.ndarray, topk: int, cache_dir: Path,
                  batch_size: int = 128) -> tuple[np.memmap, np.memmap, dict]:
    """Exact search with resumable score/row memmaps."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_path, row_path = cache_dir / "scores.f32", cache_dir / "rows.i32"
    meta_path = cache_dir / "metadata.json"
    shape = (len(queries), topk)
    expected_scores = int(np.prod(shape)) * 4
    if meta_path.exists() and score_path.exists() and row_path.exists():
        old = json.loads(meta_path.read_text())
        if (old.get("complete") and old.get("shape") == list(shape)
                and score_path.stat().st_size == expected_scores
                and row_path.stat().st_size == expected_scores):
            return (np.memmap(score_path, mode="r", dtype=np.float32, shape=shape),
                    np.memmap(row_path, mode="r", dtype=np.int32, shape=shape), old)
    scores = np.memmap(score_path, mode="w+", dtype=np.float32, shape=shape)
    rows = np.memmap(row_path, mode="w+", dtype=np.int32, shape=shape)
    start = time.perf_counter()
    for offset in range(0, len(queries), batch_size):
        batch = np.ascontiguousarray(queries[offset:offset + batch_size], dtype=np.float32)
        batch_scores, batch_rows = index.search(batch, topk)
        scores[offset:offset + len(batch)] = batch_scores
        rows[offset:offset + len(batch)] = batch_rows.astype(np.int32, copy=False)
        if offset % (batch_size * 20) == 0:
            scores.flush(); rows.flush()
    scores.flush(); rows.flush()
    meta = {"complete": True, "shape": list(shape), "topk": topk,
            "queries": len(queries), "batch_size": batch_size,
            "query_seconds": time.perf_counter() - start}
    meta_path.write_text(json.dumps(meta, indent=2))
    return scores, rows, meta


def unique_history_queries(requests: Sequence[TestRequest], embeddings: np.ndarray,
                           row_lookup: np.ndarray, history_n: int = 10
                           ) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    note_ids = sorted({int(note) for request in requests for note in request.history[-history_n:]
                       if 0 <= int(note) < len(row_lookup) and row_lookup[int(note)] >= 0})
    rows = np.asarray([row_lookup[note] for note in note_ids], dtype=np.int64)
    return (np.asarray(embeddings[rows], dtype=np.float32), rows,
            {note: idx for idx, note in enumerate(note_ids)})


def _rank_candidates(candidate_rows: np.ndarray, score: np.ndarray, note_ids: np.ndarray,
                     blocked: set[int], k: int, fallback: Sequence[int]) -> list[int]:
    order = np.argsort(-score, kind="stable")
    ranking, seen = [], set()
    for pos in order:
        note = int(note_ids[int(candidate_rows[pos])])
        if note in blocked or note in seen:
            continue
        ranking.append(note); seen.add(note)
        if len(ranking) == k:
            return ranking
    for note in fallback:
        note = int(note)
        if note not in blocked and note not in seen:
            ranking.append(note); seen.add(note)
            if len(ranking) == k:
                return ranking
    return ranking


def history_aggregate_rankings(
    requests: Sequence[TestRequest], embeddings: np.ndarray, row_lookup: np.ndarray,
    note_ids: np.ndarray, searched_rows: np.ndarray, query_by_note: dict[int, int],
    aggregation_m: int, per_history_budget: int = 500, k: int = 500,
    fallback_rankings: Sequence[Sequence[int]] = (),
) -> list[list[int]]:
    rankings = []
    for request_idx, request in enumerate(requests):
        view = history_view(request, embeddings, row_lookup, 10)
        fallback = fallback_rankings[request_idx] if fallback_rankings else ()
        if not len(view.rows):
            rankings.append(list(fallback[:k])); continue
        raw = [searched_rows[query_by_note[note], :per_history_budget] for note in view.note_ids]
        candidates = np.unique(np.concatenate(raw))
        candidates = candidates[candidates >= 0]
        similarities = np.asarray(embeddings[candidates], dtype=np.float32) @ view.vectors.T
        use_m = min(aggregation_m, similarities.shape[1])
        top = np.partition(similarities, similarities.shape[1] - use_m, axis=1)[:, -use_m:]
        score = top.mean(axis=1)
        rankings.append(_rank_candidates(candidates, score, note_ids, set(request.history), k, fallback))
    return rankings


def recent_rankings(requests: Sequence[TestRequest], note_ids: np.ndarray,
                    searched_rows: np.ndarray, query_by_note: dict[int, int],
                    k: int = 500, fallback_rankings: Sequence[Sequence[int]] = ()) -> list[list[int]]:
    rankings = []
    for idx, request in enumerate(requests):
        valid = [int(note) for note in request.history[-10:] if int(note) in query_by_note]
        fallback = fallback_rankings[idx] if fallback_rankings else ()
        if not valid:
            rankings.append(list(fallback[:k])); continue
        blocked, seen, ranking = set(request.history), set(), []
        for row in searched_rows[query_by_note[valid[-1]]]:
            note = int(note_ids[int(row)])
            if note not in blocked and note not in seen:
                ranking.append(note); seen.add(note)
                if len(ranking) == k: break
        if len(ranking) < k:
            for note in fallback:
                if note not in blocked and note not in seen:
                    ranking.append(int(note)); seen.add(int(note))
                    if len(ranking) == k: break
        rankings.append(ranking)
    return rankings


def _candidate_union(rows_by_interest: list[np.ndarray], budget_mode: str,
                     per_interest_budget: int, final_k: int, note_ids: np.ndarray,
                     blocked: set[int]) -> np.ndarray:
    if budget_mode == "practical":
        return np.unique(np.concatenate([rows[:per_interest_budget] for rows in rows_by_interest]))
    if budget_mode != "equal":
        raise ValueError(budget_mode)
    # Exactly ~final_k unique raw candidates, selected round-robin across interests.
    selected, seen = [], set()
    depth = 0
    while len(selected) < final_k and any(depth < len(rows) for rows in rows_by_interest):
        for rows in rows_by_interest:
            if depth >= len(rows): continue
            row = int(rows[depth])
            if row >= 0 and int(note_ids[row]) not in blocked and row not in seen:
                selected.append(row); seen.add(row)
                if len(selected) == final_k: break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def interest_rankings(
    requests: Sequence[TestRequest], embeddings: np.ndarray, note_ids: np.ndarray,
    centroid_queries: np.ndarray, offsets: Sequence[np.ndarray], weights: Sequence[np.ndarray],
    searched_rows: np.ndarray, budget_mode: str, score_mode: str,
    per_interest_budget: int = 500, k: int = 500,
    fallback_rankings: Sequence[Sequence[int]] = (),
) -> list[list[int]]:
    rankings = []
    for request_idx, request in enumerate(requests):
        query_indices = offsets[request_idx]
        fallback = fallback_rankings[request_idx] if fallback_rankings else ()
        if not len(query_indices):
            rankings.append(list(fallback[:k])); continue
        blocked = set(request.history)
        candidates = _candidate_union([searched_rows[int(q)] for q in query_indices],
                                      budget_mode, per_interest_budget, k, note_ids, blocked)
        candidates = candidates[candidates >= 0]
        similarity = np.asarray(embeddings[candidates], dtype=np.float32) @ centroid_queries[query_indices].T
        if score_mode == "max":
            score = similarity.max(axis=1)
        elif score_mode == "weighted":
            score = (similarity * np.sqrt(weights[request_idx])[None, :]).max(axis=1)
        else:
            raise ValueError(score_mode)
        rankings.append(_rank_candidates(candidates, score, note_ids, blocked, k, fallback))
    return rankings
