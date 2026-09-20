#!/usr/bin/env python3
"""Run leakage-safe Qilin full-corpus recall baselines.

Examples:
  python experiments/phase_01/experiment_01_full_corpus_baselines/run.py --mode smoke
  python experiments/phase_01/experiment_01_full_corpus_baselines/run.py --mode full
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.metrics import KS, evaluate_rankings  # noqa: E402
from experiments.phase_01.experiment_01_full_corpus_baselines.recall.category_popularity import CategoryPopularity  # noqa: E402
from experiments.phase_01.experiment_01_full_corpus_baselines.recall.itemcf import ItemCF  # noqa: E402
from experiments.phase_01.experiment_01_full_corpus_baselines.recall.popularity import GlobalPopularity  # noqa: E402
from experiments.phase_01.experiment_01_full_corpus_baselines.recall.random_recall import RandomRecall  # noqa: E402
from experiments.phase_01.experiment_01_full_corpus_baselines.recall.tfidf_recall import TfidfRecall  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--methods", nargs="+", choices=("random", "popularity", "category_popularity", "itemcf", "tfidf"),
                        default=["random", "popularity", "category_popularity", "itemcf", "tfidf"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--itemcf-neighbors", type=int, default=200)
    parser.add_argument("--tfidf-max-features", type=int, default=30_000)
    parser.add_argument("--tfidf-min-df", type=int, default=5)
    parser.add_argument("--tfidf-batch-size", type=int, default=4)
    parser.add_argument("--tfidf-jobs", type=int, default=8)
    parser.add_argument("--tfidf-ngram-max", type=int, choices=(2, 3), default=2)
    parser.add_argument("--tfidf-corpus-limit", type=int, default=None,
                        help="Smoke-only vocabulary/corpus cap; full results must leave this unset.")
    return parser.parse_args()


def peak_rss_gib() -> float:
    # Linux ru_maxrss is KiB. psutil is included as a current-RSS sanity floor.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return max(peak, psutil.Process().memory_info().rss) / (1024 ** 3)


def derive_filter_variants(raw: Sequence[Sequence[int]], requests: Sequence[TestRequest], k: int = 500) -> dict[str, list[list[int]]]:
    without = [list(map(int, ranking[:k])) for ranking in raw]
    with_filter = []
    for ranking, request in zip(raw, requests):
        blocked = set(request.history)
        with_filter.append([int(note) for note in ranking if note not in blocked][:k])
    return {"without_history_filter": without, "with_history_filter": with_filter}


def family_name(method: str) -> str:
    if method.startswith("random"):
        return "random"
    if method.startswith("popularity"):
        return "popularity"
    if method.startswith("category_popularity"):
        return "category_popularity"
    if method.startswith("itemcf"):
        return "itemcf"
    if method.startswith("tfidf"):
        return "tfidf"
    raise ValueError(method)


class Recorder:
    def __init__(self, output_dir: Path, requests: Sequence[TestRequest], warm_items: set[int], warm_users: set[int]):
        self.output_dir = output_dir
        self.per_request_dir = output_dir / "per_request"
        self.per_request_dir.mkdir(parents=True, exist_ok=True)
        self.requests = requests
        self.warm_items = warm_items
        self.warm_users = warm_users
        self.families: dict[str, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []

    def record(self, method: str, rankings: Sequence[Sequence[int]], seconds: float,
               config: dict[str, Any] | None = None) -> None:
        metrics, per_request = evaluate_rankings(
            self.requests, rankings, self.warm_items, self.warm_users, method=method
        )
        per_request.to_parquet(self.per_request_dir / f"{method}.parquet", index=False, compression="zstd")
        family = family_name(method)
        payload = self.families.setdefault(family, {"variants": {}})
        payload["variants"][method] = {
            "runtime_seconds": seconds,
            "peak_rss_gib": peak_rss_gib(),
            "config": config or {},
            "metrics": metrics,
        }
        for segment, row in metrics.items():
            self.rows.append({**row, "runtime_seconds": seconds, "peak_rss_gib": peak_rss_gib()})
        self.flush_family(family)

    def flush_family(self, family: str) -> None:
        (self.output_dir / f"{family}.json").write_text(
            json.dumps(self.families[family], ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
        )


def check_ground_truth(requests: Sequence[TestRequest], mode: str) -> dict[str, Any]:
    positives = np.asarray([len(r.ground_truth) for r in requests], dtype=np.int64)
    interactions = np.asarray([r.positive_interaction_count for r in requests], dtype=np.int64)
    overlap = sum(bool(set(r.history) & set(r.ground_truth)) for r in requests)
    stats = {
        "requests": len(requests), "positive_interactions": int(interactions.sum()),
        "unique_request_positive_items": int(positives.sum()),
        "duplicate_positive_rows_within_request": int(interactions.sum() - positives.sum()),
        "positive_mean": float(interactions.mean()), "positive_min": int(interactions.min()),
        "positive_p25": float(np.quantile(interactions, 0.25)), "positive_p50": float(np.quantile(interactions, 0.50)),
        "positive_p75": float(np.quantile(interactions, 0.75)), "positive_p90": float(np.quantile(interactions, 0.90)),
        "positive_p95": float(np.quantile(interactions, 0.95)), "positive_p99": float(np.quantile(interactions, 0.99)),
        "positive_max": int(interactions.max()), "requests_with_ground_truth_in_history": overlap,
    }
    if mode == "full":
        if (stats["requests"] != 11_115 or stats["positive_interactions"] != 44_685
                or stats["unique_request_positive_items"] != 44_683):
            raise AssertionError(f"Ground truth mismatch against Phase-0 analysis: {stats}")
    return stats


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    labels = {
        "method": "Method", "Recall@10": "Recall@10", "Recall@50": "Recall@50",
        "Recall@100": "Recall@100", "Recall@500": "Recall@500", "HitRate@100": "Hit@100",
        "MRR@100": "MRR@100", "median_first_hit_rank_on_hits": "MedianHitRank",
        "eligible_requests": "EligibleRequests", "overall_miss_rate@500": "MissRate@500",
    }
    lines = ["| " + " | ".join(labels.get(c, c) for c in columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col == "method":
                cells.append(str(value))
            elif col == "eligible_requests":
                cells.append(f"{int(value):,}")
            elif "Rank" in labels.get(col, col):
                cells.append("NA" if pd.isna(value) else f"{float(value):.1f}")
            else:
                cells.append("NA" if pd.isna(value) else f"{100*float(value):.4f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(output_dir: Path, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "summary.csv", index=False)
    display_columns = ["method", "Recall@10", "Recall@50", "Recall@100", "Recall@500",
                       "HitRate@100", "MRR@100", "median_first_hit_rank_on_hits"]
    sections = [
        "# Qilin Full-Corpus Recall Baselines", "",
        "## Experiment Protocol", "",
        "- Train recommendation logs are the only source of popularity and collaborative supervision.",
        "- Test feedback is read only to create `ground_truth = {note_idx | click == 1}` for evaluation.",
        "- The candidate corpus is `data/notes`; no cumulative behavior columns from notes are used.",
        "- Main variants filter recent history from retrieval, but ground truth is never filtered.",
        "- Recall is request-level macro recall. HitRate is request-level any-hit rate.",
        "- MRR assigns zero to misses. Mean/median first-hit rank use hit requests only; misses are reported separately.",
        "- Warm item means exposed at least once in recommendation_train; warm user means present in recommendation_train.",
        "- Warm/cold item macro metrics include only requests having at least one positive in that item stratum.",
        "",
        "```json", json.dumps(protocol, ensure_ascii=False, indent=2), "```", "",
        "## Baseline Implementations", "",
        "- **Random:** deterministic per-request sampling from all catalog note ids with seed 42; no 2M-item permutation is materialized.",
        "- **Global Popularity:** descending click count recomputed only from recommendation_train; exposure then note id break ties. "
        "Exposure and CTR sorting are implemented alternatives, but CTR is not a main result because of position/selection bias.",
        "- **Category-aware Popularity:** taxonomy2 (fallback taxonomy1) frequency in request history selects Top-1 or Top-3 categories; "
        "category-local train click popularity is round-robin merged and global popularity backfills TopK.",
        "- **ItemCF-A:** per-user train clicked-item sets. **ItemCF-B:** the union of train recent histories and train clicked items. "
        "Both use `cooccur(i,j)/sqrt(user_freq(i)*user_freq(j))`, sparse COO/CSR construction, and Top-200 neighbors per item. "
        "Native and train-popularity-backfilled rankings are reported separately.",
        "- **TF-IDF:** 30K float32 character-bigram vocabulary over `title + content`; the query concatenates the latest 20 history item texts. "
        "Retrieval is exact batched sparse cosine over all catalog rows. Test `query` and note behavior aggregates are unused.",
        "- BM25 was not added: no BM25 package is present, and adding another large inverted-index implementation/dependency was outside the phase constraint.",
        "",
    ]
    for segment, title in (("overall", "Overall"), ("warm_item", "Warm-item"),
                           ("cold_item", "Cold-item"), ("warm_user", "Warm-user"),
                           ("cold_user", "Cold-user")):
        part = frame.loc[frame["segment"] == segment]
        sections += [f"## {title}", "", markdown_table(part, display_columns), ""]
    performance = frame.loc[frame["segment"] == "overall", ["method", "runtime_seconds", "peak_rss_gib"]].copy()
    performance["runtime_seconds"] = performance["runtime_seconds"].map(lambda x: f"{x:.3f}")
    performance["peak_rss_gib"] = performance["peak_rss_gib"].map(lambda x: f"{x:.3f}")
    sections += [
        "## Runtime and Peak Memory", "",
        "Runtime is retrieval time for the shared unfiltered Top-520 pass; filtered/unfiltered variants therefore share it. "
        "ItemCF build time and TF-IDF fit/matrix details are recorded in their family JSON files.", "",
        performance.to_markdown(index=False), "",
    ]
    main = frame[(frame["segment"] == "overall") & frame["method"].str.endswith("__with_history_filter")]
    cold = frame[(frame["segment"] == "cold_item") & frame["method"].str.endswith("__with_history_filter")]
    best_overall = main.loc[main["Recall@500"].idxmax()]
    best_cold = cold.loc[cold["Recall@500"].idxmax()]
    gt = protocol["test_ground_truth"]
    tfidf = protocol.get("tfidf", {})
    sections += [
        "## Findings and Evaluation Caveats", "",
        f"- Ground truth contains {gt['positive_interactions']:,} positive rows but {gt['unique_request_positive_items']:,} "
        f"request-level unique items; {gt['duplicate_positive_rows_within_request']:,} duplicate positive rows are de-duplicated by set semantics.",
        f"- Ground truth/history overlap occurs in {gt['requests_with_ground_truth_in_history']:,} test requests, so history filtering does not remove a positive in this split.",
        "- Phase-0 showed every test request has at least one click. The benchmark is therefore positive-request-conditioned and is not an online CTR estimate.",
        "- Popularity, Category Popularity, and ItemCF-A cannot natively retrieve train-unexposed cold items. "
        "ItemCF-B can retrieve a small number because train histories legitimately contain items absent from train recommendation exposure.",
        f"- Best main Overall Recall@500 is `{best_overall['method']}` at {100*best_overall['Recall@500']:.4f}%.",
        f"- Best Cold-item Recall@500 is `{best_cold['method']}` at {100*best_cold['Recall@500']:.4f}%.",
        f"- TF-IDF fit took {tfidf.get('fit_seconds', float('nan')):.1f}s; exact retrieval took "
        f"{tfidf.get('retrieval_seconds', float('nan')):.1f}s. The sparse item matrix contains "
        f"{int(tfidf.get('matrix_nnz', 0)):,} nonzeros and occupies {tfidf.get('matrix_bytes', 0)/(1024**3):.3f}GiB; "
        f"end-to-end peak RSS was {protocol.get('final_peak_rss_gib', float('nan')):.3f}GiB.",
        "- Exact TF-IDF is the only baseline here with meaningful train-unexposed cold-item coverage, but full evaluation is expensive. "
        "The result supports a next-phase content bi-encoder plus ANN index, while these lexical/collaborative methods remain mandatory controls.",
        "",
    ]
    sections += [
        "## Metric Definitions", "",
        "- `Recall@K`: for each eligible request, `|TopK ∩ positives| / |positives|`, then macro average.",
        "- `HitRate@K`: fraction of eligible requests with at least one positive in TopK.",
        "- `MRR@K`: reciprocal rank of the first positive if its rank is at most K, otherwise zero.",
        "- `Mean/MedianHitRank`: first-hit rank among hit requests only. `overall_miss_rate@500` in summary.csv retains misses.",
    ]
    (output_dir / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_results = ROOT / "results/phase_01/experiment_01_full_corpus_baselines"
    output_dir = args.output_dir or experiment_results / ("smoke" if args.mode == "smoke" else "full")
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = QilinData(ROOT)
    request_limit = 100 if args.mode == "smoke" else None
    requests = data.load_test_requests(limit=request_limit)
    ground_truth_stats = check_ground_truth(requests, args.mode)
    catalog_ids = data.load_catalog_ids()
    warm_items, warm_users = data.train_exposed_items, data.train_users
    global_model = GlobalPopularity(data.train_click_counts, data.train_exposure_counts, mode="click")
    recorder = Recorder(output_dir, requests, warm_items, warm_users)
    protocol: dict[str, Any] = {
        "mode": args.mode, "seed": args.seed, "candidate_corpus_unique_notes": len(catalog_ids),
        "test_ground_truth": ground_truth_stats, "warm_item_count": len(warm_items),
        "warm_user_count": len(warm_users), "history_filter_main": True,
        "test_query_used_in_main_experiments": False,
    }

    def record_pair(base_name: str, raw: Sequence[Sequence[int]], seconds: float, config: dict[str, Any]) -> None:
        for filter_name, rankings in derive_filter_variants(raw, requests).items():
            recorder.record(f"{base_name}__{filter_name}", rankings, seconds, config)

    if "random" in args.methods:
        start = time.perf_counter()
        raw = RandomRecall(catalog_ids, seed=args.seed).recommend(requests, k=520, exclude_history=False)
        record_pair("random", raw, time.perf_counter() - start, {"seed": args.seed, "corpus": "all notes"})
        del raw

    if "popularity" in args.methods:
        start = time.perf_counter()
        raw = global_model.recommend(requests, k=520, exclude_history=False)
        record_pair("popularity_click", raw, time.perf_counter() - start,
                    {"source": "recommendation_train", "primary_sort": "click_count DESC",
                     "tie_break": "exposure_count DESC, note_idx ASC",
                     "alternate_rankings_prepared": ["exposure", "ctr"], "ctr_reported_as_main": False})
        del raw

    if "category_popularity" in args.methods:
        category_codes, category_names = data.load_categories()
        protocol["category_count_taxonomy2_fallback_taxonomy1"] = len(category_names)
        for top_n in (1, 3):
            model = CategoryPopularity(data.train_click_counts, data.train_exposure_counts,
                                       category_codes, global_model.ranking, top_categories=top_n)
            start = time.perf_counter()
            raw = model.recommend(requests, k=520, exclude_history=False)
            record_pair(f"category_popularity_top{top_n}", raw, time.perf_counter() - start,
                        {"top_categories": top_n, "category": "taxonomy2 fallback taxonomy1",
                         "backfill": "train global click popularity"})
            del model, raw

    if "itemcf" in args.methods:
        itemcf_variants = ("itemcf_a_train_click", "itemcf_b_history_plus_train_click")
        for name in itemcf_variants:
            if name == "itemcf_a_train_click":
                interactions = [set(v) for v in data.train_user_clicks.values()]
            else:
                interactions = [
                    set(data.train_user_clicks.get(user, set())) | set(history)
                    for user, history in data.train_user_histories.items()
                ]
            build_start = time.perf_counter()
            model = ItemCF(neighbors_per_item=args.itemcf_neighbors).fit(interactions)
            build_seconds = time.perf_counter() - build_start
            config = {"neighbors_per_item": args.itemcf_neighbors, "similarity": "cooccur/sqrt(user_freq_i*user_freq_j)",
                      "build_seconds": build_seconds, "build_stats": asdict(model.stats) if model.stats else {}}
            for suffix, fallback in (("native", None), ("popularity_fallback", global_model.ranking)):
                start = time.perf_counter()
                raw = model.recommend(requests, k=520, exclude_history=False, fallback=fallback)
                record_pair(f"{name}_{suffix}", raw, time.perf_counter() - start, config)
                del raw
            del model, interactions
            gc.collect()

    if "tfidf" in args.methods:
        if args.tfidf_corpus_limit is not None and args.mode != "smoke":
            raise ValueError("--tfidf-corpus-limit is allowed only in smoke mode")
        required_ids = {note for request in requests for note in request.history}
        note_ids, texts, row_by_id = data.load_note_texts(
            limit=args.tfidf_corpus_limit, required_ids=required_ids if args.tfidf_corpus_limit else None
        )
        model = TfidfRecall(max_features=args.tfidf_max_features, min_df=args.tfidf_min_df,
                            batch_size=args.tfidf_batch_size, history_n=20,
                            ngram_range=(2, args.tfidf_ngram_max), n_jobs=args.tfidf_jobs)
        model.fit(note_ids, texts, row_by_id)
        start = time.perf_counter()
        raw = model.recommend(requests, k=520, exclude_history=False, fallback=global_model.ranking)
        retrieval_seconds = time.perf_counter() - start
        config = {
            "item_text": "note_title + space + note_content", "user_text": "last 20 history item texts",
            "analyzer": "char", "ngram_range": [2, args.tfidf_ngram_max], "max_features": args.tfidf_max_features,
            "min_df": args.tfidf_min_df, "max_df": 0.8, "dtype": "float32", "similarity": "exact sparse cosine",
            "query_batch_size": args.tfidf_batch_size, "query_threads": args.tfidf_jobs,
            "query_field_used": False, "popularity_fallback_only_when_fewer_than_k_nonzero": True,
            **model.stats_dict(), "measured_retrieval_seconds": retrieval_seconds,
        }
        record_pair("tfidf_history20", raw, retrieval_seconds, config)
        protocol["tfidf"] = config
        del model, raw, note_ids, texts, row_by_id
        gc.collect()

    protocol["final_peak_rss_gib"] = peak_rss_gib()
    protocol["methods"] = args.methods
    protocol["wall_clock_completed_unix"] = time.time()
    (output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(output_dir, recorder.rows, protocol)
    print(f"Results written to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
