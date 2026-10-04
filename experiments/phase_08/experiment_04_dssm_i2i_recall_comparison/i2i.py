"""Exact, sharded KNN cache and recency-weighted request-level I2I merge."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_corpus(path: Path, count: int, dimension: int, dtype: str) -> np.ndarray:
    """Read existing vectors without changing the source asset."""
    if dtype == "float16_raw":
        source = np.memmap(path, mode="r", dtype=np.float16, shape=(count, dimension))
    else:
        source = np.load(path, mmap_mode="r")
        if source.shape != (count, dimension):
            raise ValueError("item vector shape differs from canonical mapping")
    output = np.asarray(source, dtype=np.float32).copy()
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    np.divide(output, np.maximum(norm, 1e-12), out=output)
    return output


def exact_knn_shards(
    corpus: np.ndarray,
    history_rows: np.ndarray,
    output: Path,
    *,
    source_hash: str,
    depth: int,
    device: str,
    shard_size: int = 2048,
    query_batch: int = 32,
    deadline: float | None = None,
) -> dict:
    """One GPU-resident exact IP index, resumable via per-shard file hashes."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal I2I exact retrieval requires CUDA")
    output.mkdir(parents=True, exist_ok=True)
    if corpus.dtype != np.float32 or len(corpus) == 0:
        raise ValueError("expected nonempty float32 corpus")
    if depth < 500 or depth > len(corpus):
        raise ValueError("invalid KNN depth")
    if not np.all(np.isfinite(corpus)):
        raise ValueError("nonfinite corpus vectors")
    if len(np.unique(history_rows)) != len(history_rows):
        raise ValueError("history rows must be unique")
    if (history_rows < 0).any() or (history_rows >= len(corpus)).any():
        raise ValueError("history row outside corpus")
    history_hash = hashlib.sha256(history_rows.tobytes()).hexdigest()
    started = time.perf_counter()
    resident = torch.from_numpy(np.ascontiguousarray(corpus)).to(device)
    shard_stats = []
    with torch.inference_mode():
        for start in range(0, len(history_rows), shard_size):
            if deadline and time.monotonic() > deadline:
                raise TimeoutError("I2I KNN deadline")
            stop = min(start + shard_size, len(history_rows))
            base = output / f"rows_{start:07d}_{stop:07d}"
            rows_path = Path(str(base) + ".i32.npy")
            scores_path = Path(str(base) + ".f32.npy")
            marker = Path(str(base) + ".json")
            expected = {"start": start, "stop": stop, "depth": depth,
                        "source_hash": source_hash, "history_hash": history_hash}
            if marker.exists():
                old = json.loads(marker.read_text())
                if any(old.get(k) != v for k, v in expected.items()):
                    raise RuntimeError(f"KNN cache contract changed: {marker}")
                if (not rows_path.exists() or not scores_path.exists()
                    or sha256(rows_path) != old["rows_sha256"]
                    or sha256(scores_path) != old["scores_sha256"]):
                    raise RuntimeError(f"completed KNN shard corrupted: {marker}")
                shard_stats.append(old)
                continue
            if rows_path.exists() or scores_path.exists():
                raise RuntimeError(f"incomplete KNN shard needs explicit review: {base}")
            all_rows, all_scores = [], []
            for q_start in range(start, stop, query_batch):
                if deadline and time.monotonic() > deadline:
                    raise TimeoutError("I2I KNN deadline")
                ids = history_rows[q_start:min(q_start + query_batch, stop)]
                query = resident[torch.as_tensor(ids, device=device, dtype=torch.long)]
                score = query @ resident.T
                values, indices = torch.topk(score, depth, dim=1)
                all_rows.append(indices.cpu().numpy().astype(np.int32))
                all_scores.append(values.cpu().numpy().astype(np.float32))
            rows = np.concatenate(all_rows)
            scores = np.concatenate(all_scores)
            for path, value in ((rows_path, rows), (scores_path, scores)):
                temporary = Path(str(path) + ".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, value, allow_pickle=False)
                temporary.replace(path)
            value = {**expected, "rows_sha256": sha256(rows_path),
                     "scores_sha256": sha256(scores_path), "complete": True}
            temporary_marker = Path(str(marker) + ".tmp")
            temporary_marker.write_text(json.dumps(value, indent=2))
            temporary_marker.replace(marker)
            shard_stats.append(value)
            print(f"KNN {stop}/{len(history_rows)}", flush=True)
    del resident
    return {"unique_history_items": len(history_rows), "depth": depth,
            "source_hash": source_hash, "history_hash": history_hash,
            "shards": len(shard_stats), "seconds": time.perf_counter() - started,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}


def load_knn(cache: Path, history_rows: np.ndarray, depth: int, source_hash: str):
    """Validate and load a completed KNN cache once for all requests."""
    expected_history = hashlib.sha256(history_rows.tobytes()).hexdigest()
    rows, scores = [], []
    start = 0
    while start < len(history_rows):
        matches = sorted(cache.glob(f"rows_{start:07d}_*.json"))
        if len(matches) != 1:
            raise RuntimeError(f"missing/ambiguous KNN shard at row {start}")
        marker = matches[0]
        value = json.loads(marker.read_text())
        stop = int(value["stop"])
        base = marker.with_suffix("")
        row_path = Path(str(base) + ".i32.npy")
        score_path = Path(str(base) + ".f32.npy")
        if (value["start"] != start or value["depth"] != depth
            or value["source_hash"] != source_hash
            or value["history_hash"] != expected_history
            or sha256(row_path) != value["rows_sha256"]
            or sha256(score_path) != value["scores_sha256"]):
            raise RuntimeError(f"KNN shard verification failed: {marker}")
        one_rows = np.load(row_path, mmap_mode="r")
        one_scores = np.load(score_path, mmap_mode="r")
        if one_rows.shape != (stop - start, depth) or one_scores.shape != one_rows.shape:
            raise RuntimeError("KNN shard shape mismatch")
        rows.append(one_rows)
        scores.append(one_scores)
        start = stop
    return np.concatenate(rows), np.concatenate(scores)


def merge_request(history, note_to_row, catalog, history_row_to_knn,
                  knn_rows, knn_scores, depth, topk=500):
    """Most recent history first; 1/(1+idx) weighted sum of per-item scores."""
    blocked = np.asarray(tuple(map(int, history)), dtype=np.int64)
    source_rows = []
    weights = []
    for index, note in enumerate(reversed(tuple(history)[-20:])):
        row = note_to_row.get(int(note))
        if row is None:
            continue
        source = history_row_to_knn.get(row)
        if source is None:
            raise RuntimeError("history KNN mapping incomplete")
        source_rows.append(source)
        weights.append(1.0 / (1.0 + index))
    if not source_rows:
        return []
    candidate_rows = np.asarray(knn_rows[source_rows, :depth]).reshape(-1)
    values = (np.asarray(knn_scores[source_rows, :depth], dtype=np.float64)
              * np.asarray(weights, dtype=np.float64)[:, None]).reshape(-1)
    unique_rows, inverse = np.unique(candidate_rows, return_inverse=True)
    summed = np.bincount(inverse, weights=values, minlength=len(unique_rows))
    notes = np.asarray(catalog[unique_rows], dtype=np.int64)
    valid = ~np.isin(notes, blocked)
    notes, summed = notes[valid], summed[valid]
    if len(notes) > topk:
        chosen = np.argpartition(-summed, topk - 1)[:topk]
        # Include exact cutoff ties so the final deterministic note-ID tie break
        # cannot depend on argpartition's unspecified tie order.
        cutoff = float(summed[chosen].min())
        chosen = np.flatnonzero(summed >= cutoff)
        notes, summed = notes[chosen], summed[chosen]
    order = np.lexsort((notes, -summed))[:topk]
    return notes[order].astype(int).tolist()
