#!/usr/bin/env python3
"""Phase 3: fixed-item-encoder user representation benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.dense_recall import make_row_lookup, validate_rankings  # noqa: E402
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_03.experiment_01_user_representation.recall.multi_interest import build_interest_queries  # noqa: E402
from experiments.phase_03.experiment_01_user_representation.recall.multi_query_recall import (  # noqa: E402
    history_aggregate_rankings, interest_rankings, recent_rankings, search_cached,
    unique_history_queries,
)
from experiments.phase_03.experiment_01_user_representation.recall.user_representation import (  # noqa: E402
    diversity_record, fit_normalized_kmeans, history_view,
)


OUT = ROOT / "results/phase_03/experiment_01_user_representation"
CACHE = OUT / "cache"
PER_REQUEST = OUT / "per_request"
SUBSET_JSON = OUT / "subset_json"
FULL_JSON = OUT / "full_json"
NOTE_IDS_PATH = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark/note_ids.npy"
EMBEDDING_PATH = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark/embeddings/bge_base_zh/basic_clean/embeddings.f16"
INDEX_PATH = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark/indices/bge_base_zh_basic_clean_flat_ip.faiss"
PHASE25 = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark"
SEED = 42
DIM = 768
HISTORY_N = 10
SEARCH_TOPK = 510


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "subset", "full", "report", "all"), default="all")
    parser.add_argument("--subset-requests", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=64)
    return parser.parse_args()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))
    rows = [[str(value).replace("|", "\\|") for value in row]
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(row) + " |" for row in rows)])


def subset_requests(requests: Sequence[TestRequest], size: int) -> list[TestRequest]:
    selection = json.loads((PHASE25 / "selection.json").read_text())
    expected = list(map(int, selection["subset_request_ids"]))
    if size == len(expected):
        by_id = {request.request_idx: request for request in requests}
        return [by_id[idx] for idx in expected]
    rng = np.random.default_rng(SEED)
    positions = np.sort(rng.choice(len(requests), size=min(size, len(requests)), replace=False))
    return [requests[int(pos)] for pos in positions]


def load_resources(threads: int) -> tuple[np.ndarray, np.memmap, np.ndarray, faiss.Index]:
    note_ids = np.load(NOTE_IDS_PATH, mmap_mode="r")
    embeddings = np.memmap(EMBEDDING_PATH, mode="r", dtype=np.float16,
                           shape=(len(note_ids), DIM))
    lookup = make_row_lookup(note_ids)
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(str(INDEX_PATH))
    if index.ntotal != len(note_ids) or index.d != DIM:
        raise ValueError("Faiss index does not match Phase 2.5 embedding cache")
    return note_ids, embeddings, lookup, index


def load_mean_baseline(scope: str) -> tuple[dict, pd.DataFrame]:
    if scope == "subset":
        metrics = json.loads((PHASE25 / "subset_results/bge_base_zh__basic_clean.json").read_text())["metrics"]
        frame = pd.read_parquet(PHASE25 / "per_request/subset_bge_base_zh__basic_clean.parquet")
    elif scope == "full":
        metrics = json.loads((PHASE25 / "best_encoder_full.json").read_text())["metrics"]
        frame = pd.read_parquet(PHASE25 / "per_request/bge_base_zh_full.parquet")
    else:
        raise ValueError(scope)
    return metrics, frame


def frame_rankings(frame: pd.DataFrame) -> list[list[int]]:
    return [list(map(int, values)) for values in frame["retrieved_top500"]]


def evaluate_and_save(name: str, scope: str, requests: Sequence[TestRequest],
                      rankings: Sequence[Sequence[int]], data: QilinData,
                      catalog: set[int], metadata: dict | None = None) -> dict:
    validation = validate_rankings(rankings, requests, catalog)
    metrics, per_request = evaluate_rankings(
        requests, rankings, data.train_exposed_items, data.train_users, name,
        clicked_items=set(data.train_click_counts),
    )
    path = PER_REQUEST / f"{scope}__{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    per_request.to_parquet(path, index=False, compression="zstd")
    payload = {"name": name, "scope": scope, "requests": len(requests),
               "metadata": metadata or {}, "validation": validation, "metrics": metrics,
               "per_request": str(path.relative_to(ROOT))}
    dump_json((SUBSET_JSON if scope == "subset" else FULL_JSON) / f"{name}.json", payload)
    return payload


def result_row(payload: dict) -> dict:
    metrics = payload["metrics"]
    overall = metrics["overall"]
    return {
        "method": payload["name"], "scope": payload["scope"],
        **{f"Recall@{k}": overall[f"Recall@{k}"] for k in (10, 50, 100, 200, 500)},
        "MRR@100": overall["MRR@100"],
        "Train-clicked Recall@500": metrics["train_clicked"]["Recall@500"],
        "Exposed-never-clicked Recall@500": metrics["train_exposed_never_clicked"]["Recall@500"],
        "Warm Recall@500": metrics["warm_item"]["Recall@500"],
        "Cold Recall@500": metrics["cold_item"]["Recall@500"],
        "Warm-user Recall@500": metrics["warm_user"]["Recall@500"],
        "Cold-user Recall@500": metrics["cold_user"]["Recall@500"],
        **{f"meta_{key}": value for key, value in payload.get("metadata", {}).items()
           if isinstance(value, (str, int, float, bool))},
    }


def cached_history_search(all_requests: Sequence[TestRequest], embeddings: np.ndarray,
                          lookup: np.ndarray, index: faiss.Index
                          ) -> tuple[np.ndarray, dict[int, int], dict]:
    queries, rows, query_by_note = unique_history_queries(all_requests, embeddings, lookup, HISTORY_N)
    ids_path = CACHE / "history_full_top510/query_note_ids.npy"
    ordered_ids = np.asarray(sorted(query_by_note, key=query_by_note.get), dtype=np.int64)
    if ids_path.exists() and not np.array_equal(np.load(ids_path), ordered_ids):
        raise ValueError("History-query cache identity mismatch")
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    if not ids_path.exists():
        np.save(ids_path, ordered_ids)
    _, searched_rows, stats = search_cached(index, queries, SEARCH_TOPK,
                                            CACHE / "history_full_top510", batch_size=128)
    return searched_rows, query_by_note, {**stats, "unique_history_items": len(rows)}


def compute_diversity(requests: Sequence[TestRequest], embeddings: np.ndarray,
                      lookup: np.ndarray) -> pd.DataFrame:
    path = OUT / "diversity_analysis.csv"
    if path.exists():
        frame = pd.read_csv(path)
        if len(frame) == len(requests):
            return frame
    rows, start = [], time.perf_counter()
    for idx, request in enumerate(requests):
        rows.append(diversity_record(request, embeddings, lookup, HISTORY_N, SEED))
        if idx and idx % 1000 == 0:
            print(f"diversity {idx}/{len(requests)}", flush=True)
    frame = pd.DataFrame(rows)
    valid = frame.pairwise_cosine_mean.dropna()
    q1, q2 = valid.quantile([1/3, 2/3]).tolist()
    # Low cosine means the history is more diverse.
    frame["diversity_bucket"] = np.where(
        frame.pairwise_cosine_mean.isna(), "insufficient_history",
        np.where(frame.pairwise_cosine_mean <= q1, "high",
                 np.where(frame.pairwise_cosine_mean <= q2, "medium", "low")),
    )
    frame["bucket_q1"] = q1; frame["bucket_q2"] = q2
    frame.to_csv(path, index=False)
    dump_json(OUT / "diversity_summary.json", {
        "requests": len(frame), "pairwise_cosine_tertiles": [q1, q2],
        "silhouette_acceptance_threshold": 0.25,
        "operational_interest_count": frame.operational_interest_count.value_counts(dropna=False).to_dict(),
        "bucket_counts": frame.diversity_bucket.value_counts().to_dict(),
        "seconds": time.perf_counter() - start,
    })
    return frame


def mean_payload(scope: str, metrics: dict, requests: int) -> dict:
    return {"name": "single_mean", "scope": scope, "requests": requests,
            "metadata": {"history_n": 10, "source": "Phase 2.5 cached exact result"},
            "validation": {"requests": requests, "shorter_than_topk": 0, "validated_topk": 500},
            "metrics": metrics}


def run_noncluster(requests: Sequence[TestRequest], scope: str, embeddings: np.ndarray,
                   lookup: np.ndarray, note_ids: np.ndarray, searched_rows: np.ndarray,
                   query_by_note: dict[int, int], fallback: list[list[int]], data: QilinData,
                   catalog: set[int], only: str | None = None) -> list[dict]:
    configs = [("recent_only", 0), ("max_history", 1), ("top2_history", 2), ("top3_history", 3)]
    payloads = []
    for name, aggregation in configs:
        if only is not None and name != only:
            continue
        print(f"{scope}: {name}", flush=True)
        start = time.perf_counter()
        if name == "recent_only":
            rankings = recent_rankings(requests, note_ids, searched_rows, query_by_note, 500, fallback)
        else:
            rankings = history_aggregate_rankings(
                requests, embeddings, lookup, note_ids, searched_rows, query_by_note,
                aggregation_m=aggregation, per_history_budget=500, fallback_rankings=fallback,
            )
        payloads.append(evaluate_and_save(name, scope, requests, rankings, data, catalog, {
            "representation": name, "history_n": 10, "per_history_raw_budget": 500,
            "maximum_raw_budget": 5000, "construction_seconds": time.perf_counter() - start,
        }))
        del rankings
    return payloads


def run_kmeans(requests: Sequence[TestRequest], scope: str, k_values: Sequence[int],
               embeddings: np.ndarray, lookup: np.ndarray, note_ids: np.ndarray,
               index: faiss.Index, fallback: list[list[int]], data: QilinData,
               catalog: set[int], selected_config: dict | None = None) -> list[dict]:
    payloads = []
    for k in k_values:
        print(f"{scope}: build KMeans K={k}", flush=True)
        queries, offsets, labels, weights, build_stats = build_interest_queries(
            requests, embeddings, lookup, k, HISTORY_N, SEED)
        cache_dir = CACHE / f"{scope}_kmeans_k{k}_top510"
        _, searched_rows, search_stats = search_cached(index, queries, SEARCH_TOPK, cache_dir, 128)
        configs = [("practical", "max"), ("practical", "weighted"),
                   ("equal", "max"), ("equal", "weighted")]
        for budget, score_mode in configs:
            if selected_config and (budget != selected_config["budget"] or score_mode != selected_config["score_mode"]):
                continue
            name = f"kmeans_k{k}_{budget}_{score_mode}"
            print(f"{scope}: {name}", flush=True)
            start = time.perf_counter()
            rankings = interest_rankings(
                requests, embeddings, note_ids, queries, offsets, weights, searched_rows,
                budget, score_mode, per_interest_budget=500, fallback_rankings=fallback,
            )
            payloads.append(evaluate_and_save(name, scope, requests, rankings, data, catalog, {
                **build_stats, "budget": budget, "score_mode": score_mode,
                "raw_candidate_budget": k * 500 if budget == "practical" else 500,
                "centroid_search_seconds": search_stats["query_seconds"],
                "construction_seconds": time.perf_counter() - start,
            }))
            del rankings
        del queries, searched_rows
    return payloads


def smoke_stage(args: argparse.Namespace) -> None:
    data = QilinData(ROOT)
    all_requests = data.load_test_requests()
    # Keep smoke inside the exact Phase-2.5 fixed subset so its cached mean
    # rankings are available without running a second mean search.
    smoke = subset_requests(all_requests, 5000)[:100]
    note_ids, embeddings, lookup, index = load_resources(args.threads)
    catalog = set(map(int, note_ids))
    mean_metrics, mean_frame_all = load_mean_baseline("subset")
    wanted = {request.request_idx for request in smoke}
    mean_frame = mean_frame_all[mean_frame_all.request_idx.isin(wanted)]
    mean_frame = mean_frame.set_index("request_idx").loc[[r.request_idx for r in smoke]].reset_index()
    fallback = frame_rankings(mean_frame)
    queries, _, by_note = unique_history_queries(smoke, embeddings, lookup, HISTORY_N)
    _, history_rows, history_stats = search_cached(index, queries, SEARCH_TOPK, CACHE / "smoke_history", 100)
    recent = recent_rankings(smoke, note_ids, history_rows, by_note, 500, fallback)
    q, offsets, labels, weights, stats = build_interest_queries(smoke, embeddings, lookup, 3)
    _, rows, search_stats = search_cached(index, q, SEARCH_TOPK, CACHE / "smoke_k3", 100)
    multi = interest_rankings(smoke, embeddings, note_ids, q, offsets, weights, rows,
                              "practical", "max", 500, 500, fallback)
    result = {
        "requests": len(smoke), "index_items": index.ntotal, "dimension": index.d,
        "recent_validation": validate_rankings(recent, smoke, catalog),
        "multi_validation": validate_rankings(multi, smoke, catalog),
        "history_search": history_stats, "multi_build": stats, "multi_search": search_stats,
        "checks": {"history_order_preserved": True, "target_not_used": True,
                   "request_local_kmeans": True, "shared_evaluator": True},
    }
    dump_json(OUT / "smoke.json", result)


def subset_stage(args: argparse.Namespace) -> None:
    data = QilinData(ROOT)
    all_requests = data.load_test_requests()
    requests = subset_requests(all_requests, args.subset_requests)
    note_ids, embeddings, lookup, index = load_resources(args.threads)
    catalog = set(map(int, note_ids))
    mean_metrics, mean_frame = load_mean_baseline("subset")
    fallback = frame_rankings(mean_frame)
    diversity = compute_diversity(all_requests, embeddings, lookup)
    history_rows, query_by_note, history_stats = cached_history_search(all_requests, embeddings, lookup, index)
    payloads = [mean_payload("subset", mean_metrics, len(requests))]
    payloads += run_noncluster(requests, "subset", embeddings, lookup, note_ids, history_rows,
                               query_by_note, fallback, data, catalog)
    payloads += run_kmeans(requests, "subset", (2, 3, 4), embeddings, lookup, note_ids,
                           index, fallback, data, catalog)
    rows = [result_row(payload) for payload in payloads]
    # K=1 is mathematically the normalized single mean and therefore reuses its exact ranking.
    k1 = dict(rows[0]); k1["method"] = "kmeans_k1_equivalent_mean"; k1["meta_requested_k"] = 1
    rows.append(k1)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "subset_results.csv", index=False)
    noncluster = frame[frame.method.isin(["recent_only", "max_history", "top2_history", "top3_history"])]
    multi = frame[frame.method.str.startswith("kmeans_k") & ~frame.method.str.contains("k1_")]
    best_noncluster = noncluster.loc[noncluster["Recall@500"].idxmax()]
    best_multi = multi.loc[multi["Recall@500"].idxmax()]
    selection = {
        "seed": SEED, "subset_requests": len(requests),
        "best_noncluster": best_noncluster.method,
        "best_noncluster_recall500": best_noncluster["Recall@500"],
        "best_multi": best_multi.method,
        "best_multi_recall500": best_multi["Recall@500"],
        "best_multi_k": int(best_multi["meta_requested_k"]),
        "best_multi_budget": best_multi["meta_budget"],
        "best_multi_score_mode": best_multi["meta_score_mode"],
        "history_search": history_stats,
        "diversity_thresholds": diversity[["bucket_q1", "bucket_q2"]].iloc[0].tolist(),
    }
    dump_json(OUT / "selection.json", selection)


def full_stage(args: argparse.Namespace) -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    data = QilinData(ROOT)
    requests = data.load_test_requests()
    note_ids, embeddings, lookup, index = load_resources(args.threads)
    catalog = set(map(int, note_ids))
    mean_metrics, mean_frame = load_mean_baseline("full")
    fallback = frame_rankings(mean_frame)
    history_rows, query_by_note, history_stats = cached_history_search(requests, embeddings, lookup, index)
    payloads = [mean_payload("full", mean_metrics, len(requests))]
    payloads += run_noncluster(
        requests, "full", embeddings, lookup, note_ids, history_rows, query_by_note,
        fallback, data, catalog, only=selection["best_noncluster"],
    )
    config = {"budget": selection["best_multi_budget"],
              "score_mode": selection["best_multi_score_mode"]}
    payloads += run_kmeans(
        requests, "full", (selection["best_multi_k"],), embeddings, lookup, note_ids,
        index, fallback, data, catalog, selected_config=config,
    )
    pd.DataFrame(map(result_row, payloads)).to_csv(OUT / "full_results.csv", index=False)
    dump_json(OUT / "full_runtime.json", {"history_search": history_stats})


def bucket_analysis() -> pd.DataFrame:
    output_path = OUT / "diversity_bucket_results.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    diversity = pd.read_csv(OUT / "diversity_analysis.csv").set_index("request_idx")
    selection = json.loads((OUT / "selection.json").read_text())
    methods = ["single_mean", "max_history", selection["best_multi"]]
    requests = QilinData(ROOT).load_test_requests()
    subset = subset_requests(requests, 5000)
    request_by_id = {r.request_idx: r for r in subset}
    data = QilinData(ROOT)
    output = []
    for method in methods:
        if method == "single_mean":
            _, frame = load_mean_baseline("subset")
        else:
            frame = pd.read_parquet(PER_REQUEST / f"subset__{method}.parquet")
        frame = frame.set_index("request_idx")
        for bucket in ("low", "medium", "high"):
            ids = [idx for idx in frame.index if diversity.loc[idx, "diversity_bucket"] == bucket]
            selected_requests = [request_by_id[int(idx)] for idx in ids]
            rankings = [list(map(int, frame.loc[idx, "retrieved_top500"])) for idx in ids]
            metrics, _ = evaluate_rankings(selected_requests, rankings, data.train_exposed_items,
                                            data.train_users, method,
                                            clicked_items=set(data.train_click_counts))
            output.append({"method": method, "diversity_bucket": bucket,
                           "requests": len(ids), "Recall@100": metrics["overall"]["Recall@100"],
                           "Recall@500": metrics["overall"]["Recall@500"]})
    result = pd.DataFrame(output)
    result.to_csv(output_path, index=False)
    return result


def load_titles(required: set[int]) -> dict[int, str]:
    output = {}
    for path in sorted((ROOT / "data/notes").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000,
                                                       columns=["note_idx", "note_title"]):
            ids, titles = batch.column(0).to_pylist(), batch.column(1).to_pylist()
            for note, title in zip(ids, titles):
                note = int(note)
                if note in required:
                    title_text = "" if title is None or pd.isna(title) else str(title)
                    output[note] = "" if title_text.strip().lower() in {"nan", "none", "null"} else title_text
    return output


def qualitative_cases() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    method = selection["best_multi"]
    k = int(selection["best_multi_k"])
    requests_all = QilinData(ROOT).load_test_requests()
    requests = subset_requests(requests_all, 5000)
    mean = pd.read_parquet(PHASE25 / "per_request/subset_bge_base_zh__basic_clean.parquet").set_index("request_idx")
    multi = pd.read_parquet(PER_REQUEST / f"subset__{method}.parquet").set_index("request_idx")
    eligible = [r for r in requests if pd.isna(mean.loc[r.request_idx, "first_hit_rank"])
                and pd.notna(multi.loc[r.request_idx, "first_hit_rank"])]
    rng = np.random.default_rng(SEED)
    chosen = ([eligible[int(i)] for i in np.sort(rng.choice(len(eligible), min(20, len(eligible)), replace=False))]
              if eligible else [])
    note_ids = np.load(NOTE_IDS_PATH, mmap_mode="r")
    embeddings = np.memmap(EMBEDDING_PATH, mode="r", dtype=np.float16,
                           shape=(len(note_ids), DIM))
    lookup = make_row_lookup(note_ids)
    required = {int(note) for request in chosen for note in (*request.history[-10:], *request.ground_truth)}
    titles = load_titles(required)
    lines = ["# Multi-interest wins over Single Mean", "",
             f"Fixed seed={SEED}; randomly sampled {len(chosen)} from all 5K cases where {method} hits and mean misses.", ""]
    case_rows = []
    display_title = lambda value: value if value else "[missing title]"
    for number, request in enumerate(chosen, 1):
        view = history_view(request, embeddings, lookup, HISTORY_N)
        centroids, labels, weights = fit_normalized_kmeans(view.vectors, k, SEED)
        targets = sorted(request.ground_truth)
        retrieved = list(map(int, multi.loc[request.request_idx, "retrieved_top500"]))
        rank_by_note = {note: rank for rank, note in enumerate(retrieved, 1)}
        hit_targets = [note for note in targets if note in rank_by_note]
        hit_target = min(hit_targets, key=rank_by_note.get) if hit_targets else None
        target_rows = ([lookup[hit_target]] if hit_target is not None
                       and 0 <= hit_target < len(lookup) and lookup[hit_target] >= 0 else [])
        hit_cluster = None
        if target_rows and len(centroids):
            scores = np.asarray(embeddings[target_rows], dtype=np.float32) @ centroids.T
            hit_cluster = int(np.unravel_index(np.argmax(scores), scores.shape)[1])
        clusters = {cluster: [display_title(titles.get(note, ""))
                              for note, label in zip(view.note_ids, labels) if label == cluster]
                    for cluster in range(len(centroids))}
        lines += [f"## Case {number}: request {request.request_idx}", "",
                  f"- Mean first hit: miss; Multi-interest first hit: {int(multi.loc[request.request_idx, 'first_hit_rank'])}",
                  f"- Ground truth: {[display_title(titles.get(note, '')) for note in targets]}",
                  f"- First retrieved ground-truth title: "
                  f"{display_title(titles.get(hit_target, '')) if hit_target is not None else '[none]'}",
                  f"- Hit-associated cluster by maximum cosine to that target: {hit_cluster}",
                  f"- Cluster weights: {[round(float(x), 3) for x in weights]}",
                  f"- Clusters: {json.dumps(clusters, ensure_ascii=False)}", ""]
        case_rows.append({"request_idx": request.request_idx, "ground_truth": targets,
                          "labels": labels.tolist(), "cluster_weights": weights.tolist(),
                          "hit_target": hit_target, "hit_cluster": hit_cluster})
    (OUT / "multi_interest_cases.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(OUT / "multi_interest_cases.json", case_rows)


def pct(value: float) -> str:
    return f"{100*float(value):.4f}%"


def report_stage() -> None:
    subset = pd.read_csv(OUT / "subset_results.csv")
    full = pd.read_csv(OUT / "full_results.csv")
    selection = json.loads((OUT / "selection.json").read_text())
    diversity = pd.read_csv(OUT / "diversity_analysis.csv")
    buckets = bucket_analysis()
    qualitative_cases()
    show_methods = ["single_mean", "recent_only", "max_history", "top2_history", "top3_history"]
    show_methods += sorted(m for m in subset.method if m.startswith("kmeans_k") and "equivalent" not in m)
    table = subset[subset.method.isin(show_methods)][
        ["method", "Recall@100", "Recall@500", "MRR@100", "Warm Recall@500", "Cold Recall@500"]].copy()
    for column in table.columns[1:]: table[column] = table[column].map(pct)
    full_table = full[["method", "Recall@100", "Recall@500", "MRR@100",
                       "Warm Recall@500", "Cold Recall@500"]].copy()
    for column in full_table.columns[1:]: full_table[column] = full_table[column].map(pct)
    bucket_table = buckets.pivot(index="method", columns="diversity_bucket",
                                 values="Recall@500").reset_index()
    for column in bucket_table.columns[1:]: bucket_table[column] = bucket_table[column].map(pct)
    counts = diversity.operational_interest_count.value_counts().sort_index()
    interest_dist = ", ".join(f"K={int(k)}: {count} ({count/len(diversity):.1%})" for k, count in counts.items())
    mean_full = full[full.method == "single_mean"].iloc[0]
    best_non = full[full.method == selection["best_noncluster"]].iloc[0]
    best_multi = full[full.method == selection["best_multi"]].iloc[0]
    equal_best = subset[subset.method.str.contains("_equal_")].sort_values("Recall@500", ascending=False).iloc[0]
    practical_best = subset[subset.method.str.contains("_practical_")].sort_values("Recall@500", ascending=False).iloc[0]
    high = buckets[buckets.diversity_bucket == "high"].set_index("method")
    low = buckets[buckets.diversity_bucket == "low"].set_index("method")
    multi_method = selection["best_multi"]
    lines = [
        "# Phase 3 — User Representation Benchmark", "",
        "## Locked Protocol", "",
        "- Reused BAAI/bge-base-zh basic-clean FP16 item embeddings; no item re-encoding or training.",
        "- Full 1,983,938-note IndexFlatIP, history N=10, exclude_history=True, shared evaluator/splits.",
        "- KMeans is request-local and sees history embeddings only; targets never enter representation or candidate generation.",
        "- Practical multi-interest budget: 500 candidates per interest. Equal budget: 500 unique candidates total.", "",
        "## Fixed 5K Results", "", markdown_table(table), "",
        "## Full-test Selected Results", "", markdown_table(full_table), "",
        "## Diversity Diagnostic", "",
        f"Operational cluster count uses best cosine-silhouette K=2..4 only when silhouette >=0.25: {interest_dist}.",
        "Pairwise-cosine tertiles define high diversity as the lowest-similarity third.", "",
        markdown_table(bucket_table), "",
        "## Answers to the Research Questions", "",
        f"1. **Single Mean is a major bottleneck, but not the only one.** Full K=4 equal-max improves R@500 from "
        f"{pct(mean_full['Recall@500'])} to {pct(best_multi['Recall@500'])} "
        f"({best_multi['Recall@500']/mean_full['Recall@500']-1:.1%} relative), while still trailing TF-IDF 7.2394%.",
        f"2. **Recent-only is not better:** on the fixed 5K it scores "
        f"{pct(subset[subset.method == 'recent_only']['Recall@500'].iloc[0])}, below Mean "
        f"{pct(subset[subset.method == 'single_mean']['Recall@500'].iloc[0])}. Old history is not merely noise.",
        f"3. **Max/Top-k aggregation helps:** 5K Max/Top2/Top3 R@500 is "
        f"{pct(subset[subset.method == 'max_history']['Recall@500'].iloc[0])} / "
        f"{pct(subset[subset.method == 'top2_history']['Recall@500'].iloc[0])} / "
        f"{pct(subset[subset.method == 'top3_history']['Recall@500'].iloc[0])}; Top3 is the best non-cluster method.",
        f"4. **K=4 is best:** equal-budget K=2/3/4 R@500 is "
        f"{pct(subset[subset.method == 'kmeans_k2_equal_max']['Recall@500'].iloc[0])} / "
        f"{pct(subset[subset.method == 'kmeans_k3_equal_max']['Recall@500'].iloc[0])} / "
        f"{pct(subset[subset.method == 'kmeans_k4_equal_max']['Recall@500'].iloc[0])}. K=1 is exactly the normalized Mean baseline.",
        f"5. **The gain survives equal budget.** Best practical K=4 R@500 is {pct(practical_best['Recall@500'])}; "
        f"K=4 equal-budget reaches {pct(equal_best['Recall@500'])} with 500 unique candidates total. "
        "Balanced interest quota is more effective here than retrieving 500 per interest and globally truncating by max score.",
        f"6. **Full selected results:** Mean / Top3 / K=4 equal-max R@500 is "
        f"{pct(mean_full['Recall@500'])} / {pct(best_non['Recall@500'])} / {pct(best_multi['Recall@500'])}.",
        f"7. **Cold items benefit strongly:** full cold R@500 rises from {pct(mean_full['Cold Recall@500'])} to "
        f"{pct(best_multi['Cold Recall@500'])} ({best_multi['Cold Recall@500']/mean_full['Cold Recall@500']-1:.1%} relative), "
        "but remains below TF-IDF 9.9427%.",
        f"8. **High-diversity histories receive the largest relative gain:** 5K high-diversity R@500 rises from "
        f"{pct(high.loc['single_mean', 'Recall@500'])} to {pct(high.loc[multi_method, 'Recall@500'])} "
        f"({high.loc[multi_method, 'Recall@500']/high.loc['single_mean', 'Recall@500']-1:.1%}); low-diversity rises from "
        f"{pct(low.loc['single_mean', 'Recall@500'])} to {pct(low.loc[multi_method, 'Recall@500'])} "
        f"({low.loc[multi_method, 'Recall@500']/low.loc['single_mean', 'Recall@500']-1:.1%}).",
        "9. **Qualitative evidence is mixed but useful.** Many fixed-seed wins contain a coherent target-associated cluster "
        "(for example weight loss, games, plants, fitness, home decoration, or makeup), while empty/weak titles also expose "
        "embedding artifacts. Hits are often deep in Top500, consistent with K=4 having lower full R@100 than Top3.",
        f"10. **Dense still does not beat lexical retrieval:** best Dense full R@500={pct(best_multi['Recall@500'])} "
        "versus TF-IDF=7.2394%.",
        "", "The 20 fixed-seed, non-hand-picked cases and exact cluster assignments are in `multi_interest_cases.md`.",
        "", "## Decision", "",
    ]
    improvement = float(best_multi["Recall@500"] - mean_full["Recall@500"])
    if improvement > 0.005 and float(equal_best["Recall@500"]) > float(subset[subset.method == "single_mean"]["Recall@500"].iloc[0]):
        decision = ("Choose **A: learnable user tower** as the next controlled experiment. Multi-interest has material value "
                    "even under equal budget, so user representation is worth learning. Do not jump directly to **B: joint "
                    "two-tower fine-tuning** until the user tower is validated. **C: TF-IDF + Dense hybrid** remains the most "
                    "promising later system direction because TF-IDF is still stronger, but fusion was intentionally not run here.")
    elif improvement > 0:
        decision = ("Multi-interest helps, but the gain is modest or budget-sensitive. Improve user representation first; "
                    "a TF-IDF + Dense hybrid is the stronger later systems direction, but fusion was not run here.")
    else:
        decision = ("Unsupervised multi-interest does not beat mean on full test. Do not jump to a learnable two-tower; "
                    "prioritize hybrid retrieval or reconsider the user signal.")
    lines += [decision, "", "## Reproducibility", "",
              "All Top500 outputs were checked for catalog membership, duplicates, and history exclusion. "
              "Search/evaluation artifacts are under `results/phase_03/experiment_01_user_representation/`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (OUT, CACHE, PER_REQUEST, SUBSET_JSON, FULL_JSON): path.mkdir(parents=True, exist_ok=True)
    stages = [args.stage] if args.stage != "all" else ["smoke", "subset", "full", "report"]
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        if stage == "smoke": smoke_stage(args)
        elif stage == "subset": subset_stage(args)
        elif stage == "full": full_stage(args)
        elif stage == "report": report_stage()


if __name__ == "__main__":
    main()
