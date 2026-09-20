"""Leakage-audited request-level hard-negative mining utilities for Phase 5A."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import faiss
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy import sparse
from tqdm.auto import tqdm

from experiments.common.data import QilinData
from experiments.phase_01.experiment_01_full_corpus_baselines.recall.tfidf_recall import TfidfRecall
from experiments.phase_04.experiment_01_learnable_user_tower_n10.models.user_tower import Phase4SingleAttention
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.dataset import (
    phase4_load_train_examples,
)


DIM = 768
HISTORY_N = 20
SEED = 42
RAW_TOPK = 200
FINAL_TOPK = 100
RANK_BUCKETS = ((1, 10), (11, 20), (21, 50), (51, 100))


@dataclass(frozen=True)
class MiningPaths:
    root: Path
    out: Path
    frozen: Path

    @property
    def embedding(self) -> Path:
        return self.frozen / "embeddings/embeddings.f16"

    @property
    def note_ids(self) -> Path:
        return self.frozen / "mapping/note_ids.npy"

    @property
    def checkpoint(self) -> Path:
        return self.frozen / "model/single_attention_n20_best.pt"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def _train_files(root: Path) -> list[Path]:
    files = sorted((root / "data/recommendation_train").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("data/recommendation_train/*.parquet")
    return files


def _impression_type(position: int, positive_positions: list[int]) -> str:
    if position < min(positive_positions):
        return "before_any_positive"
    if position > max(positive_positions):
        return "after_all_positives"
    return "between_positives"


def prepare_samples(paths: MiningPaths, limit_requests: int | None = None) -> dict:
    """Preserve Phase-4 samples/split and attach same-timestamp impression context."""
    train, valid, split_stats = phase4_load_train_examples(paths.root, history_n=HISTORY_N)
    temporal_train_requests = {row.request_idx for row in train}
    split_by_key = {(row.request_idx, row.target): "train" for row in train}
    split_by_key.update({(row.request_idx, row.target): "valid" for row in valid})
    expected = len(split_by_key)
    position_exposure: Counter[int] = Counter()
    position_click: Counter[int] = Counter()
    user_positives: dict[int, set[int]] = defaultdict(set)
    request_rows: dict[int, dict] = {}
    sample_rows: list[dict] = []
    matched: set[tuple[int, int]] = set()
    timestamp_spread = 0
    impression_filter = Counter()
    columns = ["request_idx", "session_idx", "user_idx", "recent_clicked_note_idxs",
               "rec_result_details_with_idx"]

    # First pass creates the complete train-known-positive filter and position statistics.
    for file in _train_files(paths.root):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=4_000, columns=columns):
            values = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
            for request_idx, user, details in zip(values["request_idx"], values["user_idx"],
                                                  values["rec_result_details_with_idx"]):
                user = int(user)
                for item in details or ():
                    position = int(item.get("position") or 0)
                    # Position proxy is estimated on the temporal training partition,
                    # never on held-out validation requests.
                    if int(request_idx) in temporal_train_requests and position > 0:
                        position_exposure[position] += 1
                        if int(item.get("click") or 0) == 1:
                            position_click[position] += 1
                    if int(item.get("click") or 0) == 1:
                        user_positives[user].add(int(item["note_idx"]))

    position_rows = []
    q_by_position: dict[int, float] = {}
    for position in sorted(position_exposure):
        exposure = position_exposure[position]
        clicks = position_click[position]
        ctr = clicks / exposure
        # Beta(1,1) smoothing is used only to avoid zero proxy values.
        q = (clicks + 1.0) / (exposure + 2.0)
        q_by_position[position] = q
        position_rows.append({"position": position, "exposure": exposure, "click": clicks,
                              "ctr": ctr, "smoothed_ctr_proxy": q})
    q_ref = q_by_position.get(1, float(np.mean(list(q_by_position.values()))))
    for row in position_rows:
        row["normalized_inverse_proxy"] = q_ref / row["smoothed_ctr_proxy"]
        row["clipped_position_weight"] = float(np.clip(row["normalized_inverse_proxy"], 0.5, 1.5))
    weight_by_position = {int(row["position"]): float(row["clipped_position_weight"])
                          for row in position_rows}

    # Second pass reconstructs each Phase-4 sample without changing its split membership.
    for file in _train_files(paths.root):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=2_000, columns=columns):
            values = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
            for request_idx, session_idx, user_idx, history, details in zip(*(values[c] for c in columns)):
                request_idx, user_idx = int(request_idx), int(user_idx)
                history = tuple(map(int, (history or [])[-HISTORY_N:]))
                details = list(details or ())
                positives = []
                seen_targets = set()
                for item in details:
                    if int(item.get("click") or 0) != 1:
                        continue
                    target = int(item["note_idx"])
                    if target in seen_targets or target in history:
                        continue
                    seen_targets.add(target)
                    if (request_idx, target) in split_by_key:
                        positives.append(item)
                if not positives:
                    continue
                all_positive_ids = sorted({int(item["note_idx"]) for item in details
                                           if int(item.get("click") or 0) == 1})
                all_positive_set = set(all_positive_ids)
                split_values = {split_by_key[(request_idx, int(item["note_idx"]))]
                                for item in positives}
                if len(split_values) != 1:
                    raise AssertionError(f"request {request_idx} crosses the Phase-4 split")
                split = split_values.pop()
                request_rows[request_idx] = {
                    "request_idx": request_idx, "session_idx": int(session_idx),
                    "user_idx": user_idx, "split": split, "history_item_ids": list(history),
                    "same_request_positive_ids": all_positive_ids,
                }
                distinct_times = {float(item["request_timestamp"]) for item in positives
                                  if item.get("request_timestamp") is not None}
                timestamp_spread += len(distinct_times) > 1
                for positive in positives:
                    target = int(positive["note_idx"])
                    key = (request_idx, target)
                    if key in matched:
                        continue
                    matched.add(key)
                    timestamp = float(positive["request_timestamp"])
                    positive_position = int(positive["position"])
                    same_timestamp = [item for item in details
                                      if item.get("request_timestamp") is not None
                                      and float(item["request_timestamp"]) == timestamp]
                    positive_positions = [int(item["position"]) for item in same_timestamp
                                          if int(item.get("click") or 0) == 1]
                    if not positive_positions:
                        positive_positions = [positive_position]
                    impression_ids, positions, types, deltas, simple_weights, position_weights = [], [], [], [], [], []
                    seen_impressions = set()
                    for item in same_timestamp:
                        note = int(item["note_idx"])
                        if int(item.get("click") or 0) == 1:
                            continue
                        impression_filter["raw_unclicked"] += 1
                        if note in seen_impressions:
                            impression_filter["duplicate"] += 1; continue
                        seen_impressions.add(note)
                        if note in user_positives[user_idx]:
                            impression_filter["user_train_known_positive"] += 1; continue
                        if note in history:
                            impression_filter["history_overlap"] += 1; continue
                        if note in all_positive_set:
                            impression_filter["same_request_positive"] += 1; continue
                        position = int(item["position"])
                        kind = _impression_type(position, positive_positions)
                        simple = 1.0 if kind == "before_any_positive" else (0.75 if kind == "between_positives" else 0.5)
                        impression_ids.append(note); positions.append(position); types.append(kind)
                        deltas.append(positive_position - position); simple_weights.append(simple)
                        position_weights.append(simple * weight_by_position.get(position, 1.0))
                        impression_filter["kept"] += 1
                    sample_rows.append({
                        "sample_id": f"{split}:{request_idx}:{target}", "split": split,
                        "request_id": request_idx, "session_id": int(session_idx), "user_id": user_idx,
                        "timestamp": timestamp, "history_item_ids": list(history),
                        "positive_item_id": target, "positive_position": positive_position,
                        "same_request_positive_ids": all_positive_ids,
                        "impression_ids": impression_ids, "impression_positions": positions,
                        "impression_types": types, "impression_delta_positions": deltas,
                        "impression_simple_weights": simple_weights,
                        "impression_position_weights": position_weights,
                    })

    if len(matched) != expected:
        missing = list(set(split_by_key) - matched)[:10]
        raise AssertionError(f"matched {len(matched)}/{expected} Phase-4 samples; missing={missing}")
    full_request_count, full_sample_count = len(request_rows), len(sample_rows)
    if limit_requests is not None:
        selected_requests = set(sorted(request_rows)[:limit_requests])
        request_rows = {key: value for key, value in request_rows.items() if key in selected_requests}
        sample_rows = [row for row in sample_rows if int(row["request_id"]) in selected_requests]
    paths.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(position_rows).to_csv(paths.out / "position_confidence_weights.csv", index=False)
    request_table = pa.Table.from_pylist(sorted(request_rows.values(), key=lambda row: row["request_idx"]))
    pq.write_table(request_table, paths.out / "request_contexts.parquet", compression="zstd")
    for split in ("train", "valid"):
        rows = [row for row in sample_rows if row["split"] == split]
        pq.write_table(pa.Table.from_pylist(rows), paths.out / f"{split}_samples_base.parquet", compression="zstd")
    user_rows = [{"user_id": user, "known_positive_ids": sorted(items)}
                 for user, items in sorted(user_positives.items())]
    pq.write_table(pa.Table.from_pylist(user_rows), paths.out / "user_train_known_positives.parquet",
                   compression="zstd")
    stats = {**split_stats, "history_n": HISTORY_N, "samples": len(sample_rows),
             "requests": len(request_rows), "matched_phase4_samples": len(matched),
             "full_requests_before_optional_smoke_limit": full_request_count,
             "full_samples_before_optional_smoke_limit": full_sample_count,
             "request_rows_with_multiple_positive_timestamps": timestamp_spread,
             "impression_grouping": "same request_idx and exact request_timestamp as target",
             "position_proxy": "Beta(1,1)-smoothed CTR(position) on temporal training partition only",
             "position_weight": "clip(q_position1/q_position,0.5,1.5) * relative confidence",
             "impression_filter_counts": dict(impression_filter),
             "test_data_used": False}
    save_json(paths.out / "prepare_stats.json", stats)
    return stats


def _load_contexts(path: Path, split: str | None = None) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return rows


def _dense_worker(rank: int, world_size: int, request_rows: list[dict], paths_dict: dict,
                  batch_size: int, raw_topk: int) -> None:
    paths = MiningPaths(Path(paths_dict["root"]), Path(paths_dict["out"]), Path(paths_dict["frozen"]))
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    note_ids = np.load(paths.note_ids, mmap_mode="r")
    lookup = np.full(int(note_ids.max()) + 1, -1, dtype=np.int32)
    lookup[note_ids] = np.arange(len(note_ids), dtype=np.int32)
    embeddings = np.memmap(paths.embedding, mode="r", dtype=np.float16, shape=(len(note_ids), DIM))
    # Float32 matches the numerical contract of the frozen IndexFlatIP artifact;
    # the multi-GPU matmul is an exact full-corpus implementation, not ANN.
    corpus = torch.from_numpy(np.asarray(embeddings)).to(device=device, dtype=torch.float32)
    model = Phase4SingleAttention(DIM).to(device)
    checkpoint = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    indices = np.array_split(np.arange(len(request_rows)), world_size)[rank]
    out_ids = np.empty((len(indices), raw_topk), dtype=np.int32)
    out_scores = np.empty((len(indices), raw_topk), dtype=np.float32)
    request_ids = np.empty(len(indices), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            history = np.zeros((len(selected), HISTORY_N, DIM), dtype=np.float32)
            mask = np.zeros((len(selected), HISTORY_N), dtype=np.bool_)
            for local, source_index in enumerate(selected):
                row = request_rows[int(source_index)]
                valid = [int(note) for note in row["history_item_ids"]
                         if 0 <= int(note) < len(lookup) and lookup[int(note)] >= 0]
                rows = lookup[np.asarray(valid, dtype=np.int64)] if valid else np.empty(0, dtype=np.int32)
                if len(rows):
                    history[local, :len(rows)] = np.asarray(embeddings[rows], dtype=np.float32)
                    mask[local, :len(rows)] = True
                request_ids[start + local] = int(row["request_idx"])
            query = model(torch.from_numpy(history).to(device), torch.from_numpy(mask).to(device))[:, 0]
            scores = torch.matmul(query.to(torch.float32), corpus.T)
            top_scores, top_rows = torch.topk(scores, k=raw_topk, dim=1, sorted=True)
            out_ids[start:start + len(selected)] = top_rows.cpu().numpy().astype(np.int32)
            out_scores[start:start + len(selected)] = top_scores.cpu().numpy().astype(np.float32)
            del scores, top_scores, top_rows, query
            if start % (batch_size * 50) == 0:
                print(f"dense gpu={rank} {start}/{len(indices)}", flush=True)
    part = paths.out / "cache/dense_raw" / f"part_{rank}.npz"
    part.parent.mkdir(parents=True, exist_ok=True)
    np.savez(part, request_ids=request_ids, rows=out_ids, scores=out_scores)


def mine_dense(paths: MiningPaths, gpus: int = 4, batch_size: int = 32,
               raw_topk: int = RAW_TOPK, limit_requests: int | None = None) -> dict:
    request_rows = _load_contexts(paths.out / "request_contexts.parquet")
    if limit_requests is not None:
        request_rows = request_rows[:limit_requests]
    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
        raise RuntimeError(f"Dense mining requires {gpus} visible CUDA GPUs")
    args = {"root": str(paths.root), "out": str(paths.out), "frozen": str(paths.frozen)}
    started = time.perf_counter()
    torch.multiprocessing.spawn(_dense_worker,
                               args=(gpus, request_rows, args, batch_size, raw_topk),
                               nprocs=gpus, join=True)
    stats = {"source": "dense", "dense_hn_source": "frozen_bge_base_zh_single_attention_n20_v1",
             "retrieval": "exact full-corpus float32 normalized inner product (IndexFlatIP-equivalent)",
             "requests": len(request_rows), "raw_topk": raw_topk, "gpus": gpus,
             "batch_size_per_gpu": batch_size, "seconds": time.perf_counter() - started}
    save_json(paths.out / "dense_mining_stats.json", stats)
    return stats


def _load_user_positives(path: Path) -> dict[int, set[int]]:
    return {int(row["user_id"]): set(map(int, row["known_positive_ids"]))
            for row in pq.read_table(path).to_pylist()}


def _filter_candidates(note_ids: np.ndarray, candidate_rows: np.ndarray, scores: np.ndarray,
                       context: dict, user_positive: set[int], topk: int = FINAL_TOPK):
    history = set(map(int, context["history_item_ids"]))
    request_positive = set(map(int, context["same_request_positive_ids"]))
    kept_ids, kept_scores, kept_ranks = [], [], []
    audit = {f"{lo}-{hi}": {"candidates": 0, "known_positive": 0, "history": 0,
                             "same_request_positive": 0} for lo, hi in RANK_BUCKETS}
    seen = set()
    for rank0, (row, score) in enumerate(zip(candidate_rows, scores)):
        rank = rank0 + 1
        note = int(note_ids[int(row)])
        for lo, hi in RANK_BUCKETS:
            if lo <= rank <= hi:
                bucket = audit[f"{lo}-{hi}"]; bucket["candidates"] += 1
                bucket["known_positive"] += note in user_positive
                bucket["history"] += note in history
                bucket["same_request_positive"] += note in request_positive
                break
        if note in seen or note in user_positive or note in history or note in request_positive:
            continue
        seen.add(note); kept_ids.append(note); kept_scores.append(float(score)); kept_ranks.append(rank)
        if len(kept_ids) == topk:
            break
    return kept_ids, kept_scores, kept_ranks, audit


def _score_distribution(chunks: list[np.ndarray]) -> dict:
    if not chunks:
        return {key: float("nan") for key in
                ("score_mean", "score_std", "score_p50", "score_p95", "score_min", "score_max")}
    values = np.concatenate(chunks).astype(np.float32, copy=False)
    return {"score_mean": float(values.mean()), "score_std": float(values.std()),
            "score_p50": float(np.quantile(values, 0.50)),
            "score_p95": float(np.quantile(values, 0.95)),
            "score_min": float(values.min()), "score_max": float(values.max())}


def finalize_dense(paths: MiningPaths) -> dict:
    contexts = {int(row["request_idx"]): row
                for row in _load_contexts(paths.out / "request_contexts.parquet")}
    user_positives = _load_user_positives(paths.out / "user_train_known_positives.parquet")
    note_ids = np.load(paths.note_ids, mmap_mode="r")
    aggregate = defaultdict(Counter); score_chunks = defaultdict(list)
    output_rows = []
    for part in sorted((paths.out / "cache/dense_raw").glob("part_*.npz")):
        values = np.load(part)
        part_scores = np.asarray(values["scores"], dtype=np.float32)
        for lo, hi in RANK_BUCKETS:
            score_chunks[f"{lo}-{hi}"].append(part_scores[:, lo - 1:hi].reshape(-1))
        for request_id, rows, scores in zip(values["request_ids"], values["rows"], values["scores"]):
            context = contexts[int(request_id)]
            ids, kept_scores, ranks, audit = _filter_candidates(
                note_ids, rows, scores, context, user_positives[int(context["user_idx"])])
            for bucket, counts in audit.items():
                aggregate[bucket].update(counts)
            output_rows.append({"request_idx": int(request_id), "dense_ids": ids,
                                "dense_scores": kept_scores, "dense_ranks": ranks})
    request_ids = [row["request_idx"] for row in output_rows]
    if len(request_ids) != len(contexts) or len(set(request_ids)) != len(contexts):
        missing = sorted(set(contexts) - set(request_ids))[:10]
        raise AssertionError(
            f"incomplete/duplicate Dense parts: rows={len(request_ids)} expected={len(contexts)} "
            f"unique={len(set(request_ids))} missing={missing}")
    output_rows.sort(key=lambda row: row["request_idx"])
    pq.write_table(pa.Table.from_pylist(output_rows), paths.out / "dense_request_negatives.parquet",
                   compression="zstd")
    audit_rows = []
    for bucket, counts in aggregate.items():
        denominator = counts["candidates"] or 1
        audit_rows.append({"source": "dense", "rank_bucket": bucket, **counts,
                           **_score_distribution(score_chunks[bucket]),
                           "known_positive_rate": counts["known_positive"] / denominator,
                           "history_overlap_rate": counts["history"] / denominator,
                           "same_request_positive_rate": counts["same_request_positive"] / denominator})
    pd.DataFrame(audit_rows).to_csv(paths.out / "dense_rank_audit.csv", index=False)
    stats = {"requests": len(output_rows), "mean_available": float(np.mean([len(r["dense_ids"]) for r in output_rows])),
             "min_available": int(min(len(r["dense_ids"]) for r in output_rows)),
             "requests_below_100": int(sum(len(r["dense_ids"]) < FINAL_TOPK for r in output_rows))}
    save_json(paths.out / "dense_finalize_stats.json", stats)
    return stats


def validate_dense_faiss_equivalence(paths: MiningPaths, requests: int = 100,
                                     threads: int = 64) -> dict:
    """Compare GPU float32 exact mining with the frozen CPU IndexFlatIP."""
    contexts = _load_contexts(paths.out / "request_contexts.parquet")[:requests]
    note_ids = np.load(paths.note_ids, mmap_mode="r")
    lookup = np.full(int(note_ids.max()) + 1, -1, dtype=np.int32)
    lookup[note_ids] = np.arange(len(note_ids), dtype=np.int32)
    embeddings = np.memmap(paths.embedding, mode="r", dtype=np.float16,
                           shape=(len(note_ids), DIM))
    model = Phase4SingleAttention(DIM)
    checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    history = np.zeros((len(contexts), HISTORY_N, DIM), dtype=np.float32)
    mask = np.zeros((len(contexts), HISTORY_N), dtype=np.bool_)
    for offset, context in enumerate(contexts):
        valid = [int(note) for note in context["history_item_ids"]
                 if 0 <= int(note) < len(lookup) and lookup[int(note)] >= 0]
        if valid:
            rows = lookup[np.asarray(valid, dtype=np.int64)]
            history[offset, :len(rows)] = np.asarray(embeddings[rows], dtype=np.float32)
            mask[offset, :len(rows)] = True
    with torch.inference_mode():
        query = model(torch.from_numpy(history), torch.from_numpy(mask))[:, 0].numpy()
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(str(paths.frozen / "index/items_flat_ip.faiss"))
    faiss_scores, faiss_rows = index.search(np.ascontiguousarray(query, dtype=np.float32), RAW_TOPK)
    gpu_raw = {}
    for part in sorted((paths.out / "cache/dense_raw").glob("part_*.npz")):
        values = np.load(part)
        for request_id, rows, scores in zip(values["request_ids"], values["rows"], values["scores"]):
            if int(request_id) in {int(row["request_idx"]) for row in contexts}:
                gpu_raw[int(request_id)] = (rows, scores)
    top100_overlap, top200_overlap, score_error = [], [], []
    exact_prefix = 0
    for offset, context in enumerate(contexts):
        gpu_rows, gpu_scores = gpu_raw[int(context["request_idx"])]
        top100_overlap.append(len(set(map(int, gpu_rows[:100])) & set(map(int, faiss_rows[offset, :100]))) / 100)
        top200_overlap.append(len(set(map(int, gpu_rows)) & set(map(int, faiss_rows[offset]))) / RAW_TOPK)
        exact_prefix += bool(np.array_equal(gpu_rows[:100], faiss_rows[offset, :100]))
        score_error.append(float(np.max(np.abs(np.asarray(gpu_scores[:100], dtype=np.float32) -
                                               faiss_scores[offset, :100]))))
    result = {"requests": len(contexts), "top100_mean_set_overlap": float(np.mean(top100_overlap)),
              "top200_mean_set_overlap": float(np.mean(top200_overlap)),
              "exact_top100_order_requests": int(exact_prefix),
              "max_top100_aligned_score_absolute_error": float(max(score_error)),
              "gpu_method": "float32 exact normalized inner product",
              "reference": "frozen CPU Faiss IndexFlatIP"}
    save_json(paths.out / "dense_faiss_equivalence.json", result)
    return result


def _required_history_texts(root: Path, required: set[int]) -> dict[int, str]:
    result = {}
    for ids, texts in QilinData(root).iter_note_text_batches():
        for note, text in zip(ids, texts):
            note = int(note)
            if note in required:
                result[note] = text
    missing = required - set(result)
    if missing:
        raise AssertionError(f"history notes missing from catalog: {list(missing)[:10]}")
    return result


def _top_sparse_with_scores(row: sparse.csr_matrix, note_ids: np.ndarray, k: int):
    data, indices = row.data, row.indices
    if not len(data):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    if len(data) > k:
        take = np.argpartition(data, -k)[-k:]
        take = take[np.argsort(data[take], kind="stable")[::-1]]
    else:
        take = np.argsort(data, kind="stable")[::-1]
    return note_ids[indices[take]].astype(np.int64), data[take].astype(np.float32)


def mine_tfidf(paths: MiningPaths, batch_size: int = 4, threads: int = 8,
               raw_topk: int = RAW_TOPK, part_requests: int = 1000,
               limit_requests: int | None = None, corpus_limit: int | None = None,
               worker_index: int = 0, worker_count: int = 1) -> dict:
    contexts = _load_contexts(paths.out / "request_contexts.parquet")
    if limit_requests is not None:
        contexts = contexts[:limit_requests]
    cache = paths.out / "cache/tfidf"
    cache.mkdir(parents=True, exist_ok=True)
    vectorizer_path, matrix_path = cache / "vectorizer.joblib", cache / "matrix.npz"
    note_ids_path = cache / "note_ids.npy"
    required = {int(note) for row in contexts for note in row["history_item_ids"]}
    started = time.perf_counter()
    if vectorizer_path.exists() and matrix_path.exists() and note_ids_path.exists():
        vectorizer = joblib.load(vectorizer_path)
        matrix = sparse.load_npz(matrix_path).tocsr()
        note_ids = np.load(note_ids_path, mmap_mode="r")
        history_texts = _required_history_texts(paths.root, required)
        fit_seconds = 0.0
    else:
        data = QilinData(paths.root)
        note_ids, texts, _ = data.load_note_texts(limit=corpus_limit, required_ids=required)
        model = TfidfRecall(max_features=30_000, min_df=5, max_df=0.8,
                            ngram_range=(2, 2), history_n=20, batch_size=batch_size, n_jobs=threads)
        model.fit(note_ids, texts, {int(note): i for i, note in enumerate(note_ids)})
        vectorizer, matrix = model.vectorizer, model.matrix
        fit_seconds = model.stats.fit_seconds if model.stats else 0.0
        history_texts = {note: texts[i] for i, note in enumerate(note_ids) if int(note) in required}
        joblib.dump(vectorizer, vectorizer_path)
        sparse.save_npz(matrix_path, matrix, compressed=False)
        np.save(note_ids_path, note_ids)
        del texts, model
    parts = paths.out / "cache/tfidf_parts"; parts.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(contexts), part_requests))
    for part_index, start in enumerate(tqdm(starts, desc="TF-IDF request parts")):
        if part_index % worker_count != worker_index:
            continue
        part_path = parts / f"part_{part_index:05d}.parquet"
        if part_path.exists():
            continue
        subset = contexts[start:start + part_requests]
        output_rows = []

        def retrieve(batch_start: int):
            rows = subset[batch_start:batch_start + batch_size]
            query_text = [" ".join(history_texts[int(note)] for note in row["history_item_ids"]
                                   if int(note) in history_texts) for row in rows]
            similarities = (vectorizer.transform(query_text) @ matrix.T).tocsr()
            similarities.sum_duplicates()
            found = []
            for offset, context in enumerate(rows):
                ids, scores = _top_sparse_with_scores(similarities.getrow(offset), note_ids, raw_topk)
                found.append((context, ids, scores))
            return found

        batch_starts = list(range(0, len(subset), batch_size))
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for found in executor.map(retrieve, batch_starts):
                output_rows.extend(found)
        pq.write_table(pa.Table.from_pylist([
            {"request_idx": int(context["request_idx"]), "raw_ids": ids.tolist(),
             "raw_scores": scores.tolist()} for context, ids, scores in output_rows
        ]), part_path, compression="zstd")
    stats = {"source": "tfidf", "requests": len(contexts), "raw_topk": raw_topk,
             "configuration": {"analyzer": "char", "ngram_range": [2, 2], "max_features": 30000,
                               "min_df": 5, "max_df": 0.8, "sublinear_tf": True,
                               "history_n": 20, "similarity": "exact sparse cosine"},
             "corpus_limit": corpus_limit,
             "worker_index": worker_index, "worker_count": worker_count,
             "fit_seconds_this_run": fit_seconds, "total_seconds": time.perf_counter() - started,
             "matrix_shape": list(matrix.shape), "matrix_nnz": int(matrix.nnz)}
    save_json(paths.out / "tfidf_mining_stats.json", stats)
    return stats


def finalize_tfidf(paths: MiningPaths) -> dict:
    contexts = {int(row["request_idx"]): row
                for row in _load_contexts(paths.out / "request_contexts.parquet")}
    user_positives = _load_user_positives(paths.out / "user_train_known_positives.parquet")
    aggregate = defaultdict(Counter); score_chunks = defaultdict(list); output_rows = []
    for part in sorted((paths.out / "cache/tfidf_parts").glob("part_*.parquet")):
        raw_rows = pq.read_table(part).to_pylist()
        for lo, hi in RANK_BUCKETS:
            chunks = [np.asarray(row["raw_scores"][lo - 1:hi], dtype=np.float32)
                      for row in raw_rows if len(row["raw_scores"]) >= lo]
            if chunks:
                score_chunks[f"{lo}-{hi}"].append(np.concatenate(chunks))
        for raw in raw_rows:
            request_id = int(raw["request_idx"]); context = contexts[request_id]
            # raw_ids are already note ids, so use an identity row mapping for the shared filter.
            raw_ids = np.asarray(raw["raw_ids"], dtype=np.int64)
            identity = np.arange(len(raw_ids), dtype=np.int64)
            ids, scores, ranks, audit = _filter_candidates(
                raw_ids, identity, np.asarray(raw["raw_scores"], dtype=np.float32), context,
                user_positives[int(context["user_idx"])])
            for bucket, counts in audit.items(): aggregate[bucket].update(counts)
            output_rows.append({"request_idx": request_id, "tfidf_ids": ids,
                                "tfidf_scores": scores, "tfidf_ranks": ranks})
    request_ids = [row["request_idx"] for row in output_rows]
    if len(request_ids) != len(contexts) or len(set(request_ids)) != len(contexts):
        missing = sorted(set(contexts) - set(request_ids))[:10]
        raise AssertionError(
            f"incomplete/duplicate TF-IDF parts: rows={len(request_ids)} expected={len(contexts)} "
            f"unique={len(set(request_ids))} missing={missing}")
    output_rows.sort(key=lambda row: row["request_idx"])
    pq.write_table(pa.Table.from_pylist(output_rows), paths.out / "tfidf_request_negatives.parquet",
                   compression="zstd")
    audit_rows = []
    for bucket, counts in aggregate.items():
        denominator = counts["candidates"] or 1
        audit_rows.append({"source": "tfidf", "rank_bucket": bucket, **counts,
                           **_score_distribution(score_chunks[bucket]),
                           "known_positive_rate": counts["known_positive"] / denominator,
                           "history_overlap_rate": counts["history"] / denominator,
                           "same_request_positive_rate": counts["same_request_positive"] / denominator})
    pd.DataFrame(audit_rows).to_csv(paths.out / "tfidf_rank_audit.csv", index=False)
    stats = {"requests": len(output_rows), "mean_available": float(np.mean([len(r["tfidf_ids"]) for r in output_rows])),
             "min_available": int(min(len(r["tfidf_ids"]) for r in output_rows)),
             "requests_below_100": int(sum(len(r["tfidf_ids"]) < FINAL_TOPK for r in output_rows))}
    save_json(paths.out / "tfidf_finalize_stats.json", stats)
    return stats


def merge_sample_pools(paths: MiningPaths) -> dict:
    dense = {int(row["request_idx"]): row for row in pq.read_table(
        paths.out / "dense_request_negatives.parquet").to_pylist()}
    tfidf = {int(row["request_idx"]): row for row in pq.read_table(
        paths.out / "tfidf_request_negatives.parquet").to_pylist()}
    counts = {}
    for split in ("train", "valid"):
        base = pq.read_table(paths.out / f"{split}_samples_base.parquet").to_pylist()
        destination = paths.out / f"{split}_hard_negatives.parquet"
        writer = None
        try:
            for start in range(0, len(base), 2_000):
                rows = []
                for sample in base[start:start + 2_000]:
                    request_id = int(sample["request_id"])
                    if request_id not in dense or request_id not in tfidf:
                        raise KeyError(f"missing mined request {request_id}")
                    rows.append({**sample,
                                 "dense_ids": dense[request_id]["dense_ids"],
                                 "dense_scores": dense[request_id]["dense_scores"],
                                 "dense_ranks": dense[request_id]["dense_ranks"],
                                 "tfidf_ids": tfidf[request_id]["tfidf_ids"],
                                 "tfidf_scores": tfidf[request_id]["tfidf_scores"],
                                 "tfidf_ranks": tfidf[request_id]["tfidf_ranks"]})
                table = pa.Table.from_pylist(rows)
                if writer is None: writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None: writer.close()
        counts[split] = len(base)
    save_json(paths.out / "merge_stats.json", counts)
    return counts


def audit_pools(paths: MiningPaths) -> dict:
    rows = []
    overlap_rows = []
    impression_type = Counter()
    sample_records = []
    request_pools: dict[tuple[str, int], dict[str, set[int]]] = {}
    user_positives = _load_user_positives(paths.out / "user_train_known_positives.parquet")
    blocked_overlap = Counter({f"{source}_{kind}": 0
                               for source in ("dense", "tfidf", "impression")
                               for kind in ("explicit", "known_positive")})
    for split in ("train", "valid"):
        table = pq.read_table(paths.out / f"{split}_hard_negatives.parquet")
        for row in table.to_pylist():
            sample_records.append(row)
            dense, tfidf, impression = map(set, (row["dense_ids"], row["tfidf_ids"], row["impression_ids"]))
            explicit_blocked = (set(map(int, row["history_item_ids"])) |
                                set(map(int, row["same_request_positive_ids"])))
            known_blocked = user_positives[int(row["user_id"])]
            for source, candidates in (("dense", dense), ("tfidf", tfidf),
                                       ("impression", impression)):
                blocked_overlap[f"{source}_explicit"] += len(candidates & explicit_blocked)
                blocked_overlap[f"{source}_known_positive"] += len(candidates & known_blocked)
            pooled = request_pools.setdefault((split, int(row["request_id"])),
                                              {"dense": set(), "tfidf": set(), "impression": set()})
            pooled["dense"].update(dense); pooled["tfidf"].update(tfidf)
            pooled["impression"].update(impression)
            intersection_dt = dense & tfidf
            intersection_di = dense & impression
            intersection_ti = tfidf & impression
            triple = dense & tfidf & impression
            overlap_rows.append({"split": split, "sample_id": row["sample_id"],
                                 "request_id": int(row["request_id"]),
                                 "dense_only": len(dense - tfidf - impression),
                                 "tfidf_only": len(tfidf - dense - impression),
                                 "impression_only": len(impression - dense - tfidf),
                                 "dense_tfidf": len(intersection_dt), "dense_impression": len(intersection_di),
                                 "tfidf_impression": len(intersection_ti), "triple": len(triple),
                                 "dense_count": len(dense), "tfidf_count": len(tfidf),
                                 "impression_count": len(impression)})
            impression_type.update(row["impression_types"])
    frame = pd.DataFrame(overlap_rows)
    frame.to_csv(paths.out / "overlap_analysis.csv", index=False)
    request_audit_frames = []
    for split, group in frame.groupby("split"):
        numeric = [column for column in group.columns
                   if column not in {"split", "sample_id", "request_id"}]
        rows.append({"unit": "sample", "split": split,
                     **{f"mean_{column}": float(group[column].mean()) for column in numeric}})
        request_rows = []
        for (pool_split, request_id), pools in request_pools.items():
            if pool_split != split:
                continue
            dense, tfidf, impression = pools["dense"], pools["tfidf"], pools["impression"]
            request_rows.append({
                "request_id": request_id,
                "dense_only": len(dense - tfidf - impression),
                "tfidf_only": len(tfidf - dense - impression),
                "impression_only": len(impression - dense - tfidf),
                "dense_tfidf": len(dense & tfidf),
                "dense_impression": len(dense & impression),
                "tfidf_impression": len(tfidf & impression),
                "triple": len(dense & tfidf & impression),
                "dense_count": len(dense), "tfidf_count": len(tfidf),
                "impression_count": len(impression),
            })
        request_frame = pd.DataFrame(request_rows)
        request_frame.insert(0, "split", split)
        request_audit_frames.append(request_frame)
        rows.append({"unit": "request", "split": split,
                     **{f"mean_{column}": float(request_frame[column].mean()) for column in numeric}})
    pd.concat(request_audit_frames, ignore_index=True).to_csv(
        paths.out / "request_overlap_analysis.csv", index=False)
    rank_audit = pd.concat([pd.read_csv(paths.out / "dense_rank_audit.csv"),
                            pd.read_csv(paths.out / "tfidf_rank_audit.csv")], ignore_index=True)
    rank_audit.to_csv(paths.out / "rank_audit.csv", index=False)
    position = pd.read_csv(paths.out / "position_confidence_weights.csv")
    coverage = []
    for split, group in frame.groupby("split"):
        coverage.append({"split": split, "samples": int(len(group)),
                         "dense_nonempty_rate": float((group["dense_count"] > 0).mean()),
                         "tfidf_nonempty_rate": float((group["tfidf_count"] > 0).mean()),
                         "impression_nonempty_rate": float((group["impression_count"] > 0).mean())})
    _write_false_negative_diagnostic(paths, sample_records)
    if any(blocked_overlap.values()):
        raise AssertionError(f"hard-negative leakage audit failed: {dict(blocked_overlap)}")
    stats = {"overlap_means": rows, "sample_overlap_means":
             [row for row in rows if row["unit"] == "sample"],
             "request_overlap_means": [row for row in rows if row["unit"] == "request"],
             "impression_types": dict(impression_type),
             "impression_total": int(sum(impression_type.values())),
             "sample_source_coverage": coverage,
             "rank_audit": rank_audit.to_dict("records"),
             "position_weight_min": float(position["clipped_position_weight"].min()),
             "position_weight_max": float(position["clipped_position_weight"].max()),
             "blocked_overlap_after_filter": dict(blocked_overlap),
             "test_data_used": False}
    save_json(paths.out / "mining_stats.json", stats)
    return stats


def _write_false_negative_diagnostic(paths: MiningPaths, samples: list[dict],
                                     per_source: int = 150) -> None:
    """Fixed-seed structural sample; no relevance labels or manual selection."""
    rng = np.random.default_rng(SEED)
    selected = []
    specifications = (
        ("dense", "dense_ids", "dense_scores", "dense_ranks"),
        ("tfidf", "tfidf_ids", "tfidf_scores", "tfidf_ranks"),
        ("impression", "impression_ids", None, "impression_positions"),
    )
    for source, ids_column, score_column, rank_column in specifications:
        available = [index for index, row in enumerate(samples) if row[ids_column]]
        if not available:
            continue
        chosen = rng.choice(available, min(per_source, len(available)), replace=False)
        for sample_index in chosen:
            row = samples[int(sample_index)]
            candidate_index = int(rng.integers(len(row[ids_column])))
            negative = int(row[ids_column][candidate_index])
            dense_lookup = {int(note): float(score) for note, score in
                            zip(row["dense_ids"], row["dense_scores"])}
            tfidf_lookup = {int(note): float(score) for note, score in
                            zip(row["tfidf_ids"], row["tfidf_scores"])}
            selected.append({
                "sample_id": row["sample_id"], "request_id": int(row["request_id"]),
                "user_id": int(row["user_id"]), "negative_source": source,
                "history_item_ids": list(map(int, row["history_item_ids"])),
                "positive_item_id": int(row["positive_item_id"]),
                "negative_item_id": negative,
                "source_rank_or_position": int(row[rank_column][candidate_index]),
                "negative_subtype": (str(row["impression_types"][candidate_index])
                                     if source == "impression" else
                                     f"rank_{int(row[rank_column][candidate_index])}"),
                "dense_score": dense_lookup.get(negative),
                "tfidf_score": tfidf_lookup.get(negative),
            })

    required = set()
    for row in selected:
        required.update(row["history_item_ids"])
        required.add(row["positive_item_id"]); required.add(row["negative_item_id"])
    titles = {}
    if required:
        for file in sorted((paths.root / "data/notes").glob("*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=100_000,
                                                            columns=["note_idx", "note_title"]):
                note_ids, values = (batch.column(i).to_pylist() for i in range(2))
                for note, title in zip(note_ids, values):
                    note = int(note)
                    if note in required:
                        titles[note] = str(title or "")
    for row in selected:
        row["history_titles"] = [titles.get(note, "") for note in row["history_item_ids"]]
        row["positive_title"] = titles.get(row["positive_item_id"], "")
        row["negative_title"] = titles.get(row["negative_item_id"], "")
    destination = paths.out / "false_negative_diagnostic.parquet"
    pq.write_table(pa.Table.from_pylist(selected), destination, compression="zstd")
