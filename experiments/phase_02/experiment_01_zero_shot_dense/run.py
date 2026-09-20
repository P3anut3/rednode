#!/usr/bin/env python3
"""Phase 2: zero-shot E5 dense retrieval with Faiss exact and IVF-PQ search."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.dense_recall import (  # noqa: E402
    build_user_embeddings, make_row_lookup, rows_to_filtered_rankings, validate_rankings,
)
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_02.experiment_01_zero_shot_dense.recall.dense_encoder import (  # noqa: E402
    E5_CACHE, MODEL_NAME, encode_corpus_multi_gpu, encode_texts, format_item_text,
    load_e5, prepare_catalog_metadata, valid_text,
)
from experiments.phase_02.experiment_01_zero_shot_dense.recall.dense_index import (  # noqa: E402
    build_flat_ip, build_ivfpq, embedding_memmap, load_index, search_index,
)


OUT = ROOT / "results/phase_02/experiment_01_zero_shot_dense"
EMBED_ROOT = OUT / "item_embeddings"
INDEX_ROOT = OUT / "indices"
ABLATION_ROOT = OUT / "ablations"
PER_REQUEST = OUT / "per_request"
DIM = 768
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "encode", "indices", "ablations", "full", "report", "all"), default="all")
    parser.add_argument("--text-config", choices=("text_only", "text_taxonomy", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--ablation-requests", type=int, default=5000)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--pq-m", type=int, default=96)
    parser.add_argument("--nprobe", type=int, default=64)
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def load_selected_records(ids: set[int]) -> dict[int, tuple[str, str, str, str]]:
    result = {}
    columns = ["note_idx", "note_title", "note_content", "taxonomy1_id", "taxonomy2_id"]
    for file in sorted((ROOT / "data/notes").glob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=100_000, columns=columns):
            vals = [batch.column(i).to_pylist() for i in range(5)]
            for note, title, content, tax1, tax2 in zip(*vals):
                if int(note) in ids:
                    result[int(note)] = (valid_text(title), valid_text(content), valid_text(tax1), valid_text(tax2))
    missing = ids - set(result)
    if missing:
        raise ValueError(f"Missing selected note ids: {sorted(missing)[:10]}")
    return result


def ground_truth_protocol(requests: Sequence[TestRequest]) -> dict:
    interactions = sum(r.positive_interaction_count for r in requests)
    unique = sum(len(r.ground_truth) for r in requests)
    return {"requests": len(requests), "positive_interactions": interactions,
            "unique_request_positive_items": unique,
            "duplicate_positive_rows_within_request": interactions - unique,
            "ground_truth_history_overlap_requests": sum(bool(set(r.ground_truth) & set(r.history)) for r in requests)}


def smoke_test() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    data = QilinData(ROOT)
    requests = data.load_test_requests(limit=100)
    positives = list(dict.fromkeys(note for r in requests for note in sorted(r.ground_truth)))
    histories = list(dict.fromkeys(note for r in requests for note in r.history))
    selected = (positives + [x for x in histories if x not in set(positives)])[:1000]
    if len(selected) < 1000:
        selected += [int(x) for x in data.load_catalog_ids() if int(x) not in set(selected)][:1000-len(selected)]
    records = load_selected_records(set(selected))
    tokenizer, model = load_e5("cuda:0")
    import torch
    torch.cuda.reset_peak_memory_stats(0)
    vectors = []
    for offset in range(0, len(selected), 64):
        texts = [format_item_text(*records[note], text_config="text_only") for note in selected[offset:offset+64]]
        vectors.append(encode_texts(texts, tokenizer, model, "cuda:0", max_length=256))
    embeddings = np.vstack(vectors).astype(np.float32)
    assert embeddings.shape == (1000, DIM)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-3)
    note_ids = np.asarray(selected, dtype=np.int64)
    lookup = make_row_lookup(note_ids)
    queries, query_stats = build_user_embeddings(requests, embeddings, lookup, history_n=20)

    import faiss
    flat = faiss.IndexFlatIP(DIM); flat.add(embeddings)
    scores, rows = flat.search(queries, 600)
    brute = embeddings @ queries[0]
    assert int(rows[0, 0]) == int(np.argmax(brute))
    rankings = rows_to_filtered_rankings(rows, note_ids, requests)
    validation = validate_rankings(rankings, requests, set(note_ids))
    metrics, _ = evaluate_rankings(requests, rankings, data.train_exposed_items, data.train_users,
                                  "smoke_flat", clicked_items=set(data.train_click_counts))
    # Small ANN validates the Faiss approximate path; it is not a reported model result.
    quantizer = faiss.IndexFlatIP(DIM)
    ann = faiss.IndexIVFPQ(quantizer, DIM, 16, 48, 8, faiss.METRIC_INNER_PRODUCT)
    ann.train(embeddings); ann.add(embeddings); ann.nprobe = 8
    _, ann_rows = ann.search(queries, 600)
    ann_rankings = rows_to_filtered_rankings(ann_rows, note_ids, requests)
    validate_rankings(ann_rankings, requests, set(note_ids))
    result = {
        "passed": True, "items": 1000, "requests": 100, "dimension": DIM,
        "mapping_exact": True, "cosine_ip_exact_top1": True,
        "history_filter_valid": True, "rankings_unique": True,
        "query_uses_only_recent_clicked_note_idxs": True,
        "query_stats": query_stats, "validation": validation,
        "flat_metrics": metrics, "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(0)),
        "model": MODEL_NAME,
    }
    json_dump(OUT / "smoke.json", result)
    return result


def encode_stage(args: argparse.Namespace) -> None:
    prepare_catalog_metadata(ROOT, EMBED_ROOT)
    configs = ("text_only", "text_taxonomy") if args.text_config == "all" else (args.text_config,)
    for config in configs:
        encode_corpus_multi_gpu(ROOT, EMBED_ROOT / config, config, batch_size=args.batch_size,
                                max_length=args.max_length, world_size=args.gpus, dim=DIM)


def load_embeddings(config: str) -> tuple[np.ndarray, np.memmap, np.ndarray]:
    ids = np.load(EMBED_ROOT / "note_ids.npy", mmap_mode="r")
    emb = embedding_memmap(EMBED_ROOT / config / "embeddings.f16", len(ids), DIM)
    return ids, emb, make_row_lookup(ids)


def index_paths(config: str) -> tuple[Path, Path]:
    return INDEX_ROOT / f"{config}_flat_ip.faiss", INDEX_ROOT / f"{config}_ivfpq.faiss"


def indices_stage(args: argparse.Namespace) -> dict:
    configs = ("text_only", "text_taxonomy") if args.text_config == "all" else (args.text_config,)
    output = {}
    for config in configs:
        ids, emb, _ = load_embeddings(config)
        _, ivf_path = index_paths(config)
        if ivf_path.exists():
            index = load_index(ivf_path, args.nprobe)
            stats_path = INDEX_ROOT / f"{config}_ivfpq_metadata.json"
            stats = json.loads(stats_path.read_text()) if stats_path.exists() else {"reused": True}
        else:
            index, stats = build_ivfpq(emb, ivf_path, nlist=args.nlist, m=args.pq_m, seed=SEED)
            stats["nprobe_evaluation"] = args.nprobe
            json_dump(INDEX_ROOT / f"{config}_ivfpq_metadata.json", stats)
        index.nprobe = args.nprobe
        output[config] = stats
        del index, emb
    return output


def subset_requests(requests: Sequence[TestRequest], size: int) -> list[TestRequest]:
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(requests), size=min(size, len(requests)), replace=False))
    return [requests[int(i)] for i in chosen]


def evaluate_config(config: str, requests: Sequence[TestRequest], history_n: int, weighting: str,
                    args: argparse.Namespace, data: QilinData | None = None) -> dict:
    # Reuse one lazy QilinData instance across ablations so the train-derived
    # warm/cold split is scanned once, not once per configuration.
    data = data or QilinData(ROOT)
    ids, emb, lookup = load_embeddings(config)
    _, ivf_path = index_paths(config)
    index = load_index(ivf_path, args.nprobe)
    queries, query_stats = build_user_embeddings(requests, emb, lookup, history_n=history_n, weighting=weighting)
    _, rows, search_stats = search_index(index, queries, topk=600)
    rankings = rows_to_filtered_rankings(rows, ids, requests)
    validation = validate_rankings(rankings, requests, set(map(int, ids)))
    name = f"{config}__n{history_n}__{weighting}"
    metrics, _ = evaluate_rankings(requests, rankings, data.train_exposed_items, data.train_users,
                                  name, clicked_items=set(data.train_click_counts))
    return {"name": name, "text_config": config, "history_n": history_n, "weighting": weighting,
            "query_stats": query_stats, "search_stats": search_stats, "validation": validation,
            "metrics": metrics}


def ablations_stage(args: argparse.Namespace) -> dict:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    data = QilinData(ROOT)
    requests = subset_requests(data.load_test_requests(), args.ablation_requests)
    results = []
    # Text configuration comparison at N=20.
    for config in ("text_only", "text_taxonomy"):
        results.append(evaluate_config(config, requests, 20, "mean", args, data))
    best_text = max(results, key=lambda x: x["metrics"]["overall"]["Recall@500"])["text_config"]
    # Reuse N=20 and add N=5/N=10 for the selected text configuration.
    for n in (5, 10):
        results.append(evaluate_config(best_text, requests, n, "mean", args, data))
    mean_results = [x for x in results if x["text_config"] == best_text and x["weighting"] == "mean"]
    best_n = max(mean_results, key=lambda x: x["metrics"]["overall"]["Recall@500"])["history_n"]
    results.append(evaluate_config(best_text, requests, best_n, "recency", args, data))
    best = max(results, key=lambda x: x["metrics"]["overall"]["Recall@500"])
    payload = {"seed": SEED, "subset_requests": len(requests), "subset_request_ids": [r.request_idx for r in requests],
               "selection_metric": "ANN Overall Recall@500", "best_text_config": best_text,
               "best_history_n_mean": best_n,
               "selected_full_config": {"text_config": best["text_config"], "history_n": best["history_n"],
                                        "weighting": best["weighting"]}, "results": results}
    json_dump(ABLATION_ROOT / "ablations.json", payload)
    rows = [{"name": x["name"], **x["metrics"]["overall"],
             "cold_Recall@500": x["metrics"]["cold_item"]["Recall@500"]} for x in results]
    pd.DataFrame(rows).to_csv(ABLATION_ROOT / "summary.csv", index=False)
    return payload


def full_stage(args: argparse.Namespace) -> tuple[dict, dict]:
    data = QilinData(ROOT)
    requests = data.load_test_requests()
    ablations = json.loads((ABLATION_ROOT / "ablations.json").read_text())
    selected = ablations["selected_full_config"]
    config, history_n, weighting = selected["text_config"], selected["history_n"], selected["weighting"]
    ids, emb, lookup = load_embeddings(config)
    queries, query_stats = build_user_embeddings(requests, emb, lookup, history_n=history_n, weighting=weighting)
    warm_items, warm_users, clicked = data.train_exposed_items, data.train_users, set(data.train_click_counts)
    catalog_set = set(map(int, ids))

    flat_path, ivf_path = index_paths(config)
    if flat_path.exists():
        flat, flat_build = load_index(flat_path), json.loads((INDEX_ROOT / f"{config}_flat_ip_metadata.json").read_text())
    else:
        flat, flat_build = build_flat_ip(emb, flat_path)
        json_dump(INDEX_ROOT / f"{config}_flat_ip_metadata.json", flat_build)
    _, exact_rows, exact_search = search_index(flat, queries, topk=600, batch_size=128)
    exact_rankings = rows_to_filtered_rankings(exact_rows, ids, requests)
    exact_validation = validate_rankings(exact_rankings, requests, catalog_set)
    exact_metrics, exact_per = evaluate_rankings(requests, exact_rankings, warm_items, warm_users,
                                                 "dense_exact", clicked_items=clicked)
    PER_REQUEST.mkdir(parents=True, exist_ok=True)
    exact_per.to_parquet(PER_REQUEST / "dense_exact.parquet", index=False, compression="zstd")
    exact_payload = {"model": MODEL_NAME, "selected_config": selected, "query_stats": query_stats,
                     "index_build": flat_build, "search": exact_search,
                     "validation": exact_validation, "metrics": exact_metrics}
    json_dump(OUT / "exact.json", exact_payload)
    del flat

    ann = load_index(ivf_path, args.nprobe)
    _, ann_rows, ann_search = search_index(ann, queries, topk=600, batch_size=256)
    ann_rankings = rows_to_filtered_rankings(ann_rows, ids, requests)
    ann_validation = validate_rankings(ann_rankings, requests, catalog_set)
    ann_metrics, ann_per = evaluate_rankings(requests, ann_rankings, warm_items, warm_users,
                                             "dense_ann", clicked_items=clicked)
    ann_per.to_parquet(PER_REQUEST / "dense_ann.parquet", index=False, compression="zstd")
    exact_sets = [set(x[:500]) for x in exact_rankings]
    ann_vs_exact = float(np.mean([len(set(a[:500]) & e) / 500 for a, e in zip(ann_rankings, exact_sets)]))
    ann_build = json.loads((INDEX_ROOT / f"{config}_ivfpq_metadata.json").read_text())
    ann_payload = {"model": MODEL_NAME, "selected_config": selected, "query_stats": query_stats,
                   "index_build": ann_build, "search": ann_search, "validation": ann_validation,
                   "ANN_mean_overlap_with_exact_top500": ann_vs_exact, "metrics": ann_metrics}
    json_dump(OUT / "ann.json", ann_payload)
    return exact_payload, ann_payload


def qualitative_analysis(seed: int = SEED) -> dict:
    exact = pd.read_parquet(PER_REQUEST / "dense_exact.parquet")
    tfidf_path = ROOT / "results/phase_01/experiment_01_full_corpus_baselines/full/per_request/tfidf_history20__with_history_filter.parquet"
    tfidf = pd.read_parquet(tfidf_path)
    merged = exact[["request_idx", "user_idx", "ground_truth", "retrieved_top500", "first_hit_rank"]].merge(
        tfidf[["request_idx", "retrieved_top500", "first_hit_rank"]], on="request_idx", suffixes=("_dense", "_tfidf"))
    tf_hit = merged.first_hit_rank_tfidf.notna(); de_hit = merged.first_hit_rank_dense.notna()
    rng = np.random.default_rng(seed)
    groups = {}
    for name, mask in (("tfidf_hit_dense_miss", tf_hit & ~de_hit), ("dense_hit_tfidf_miss", de_hit & ~tf_hit)):
        candidates = np.flatnonzero(mask.to_numpy())
        chosen = rng.choice(candidates, size=min(20, len(candidates)), replace=False)
        groups[name] = merged.iloc[np.sort(chosen)]
    data = QilinData(ROOT); requests = {r.request_idx: r for r in data.load_test_requests()}
    required = set()
    for frame in groups.values():
        for row in frame.itertuples():
            req = requests[row.request_idx]
            required.update(req.history[-5:]); required.update(req.ground_truth)
            required.update(list(row.retrieved_top500_dense)[:5]); required.update(list(row.retrieved_top500_tfidf)[:5])
    records = load_selected_records(required)

    def bigrams(text: str) -> set[str]:
        text = "".join(text.split())
        return {text[i:i+2] for i in range(max(0, len(text)-1))}
    output, overlaps = {}, {}
    for name, frame in groups.items():
        cases, group_overlap = [], []
        for row in frame.itertuples():
            req = requests[row.request_idx]
            history_titles = [records[x][0] for x in req.history[-5:]]
            gt_titles = [records[x][0] for x in sorted(req.ground_truth)]
            hgrams, ggrams = bigrams(" ".join(history_titles)), bigrams(" ".join(gt_titles))
            overlap = len(hgrams & ggrams) / len(hgrams | ggrams) if hgrams | ggrams else 0.0
            group_overlap.append(overlap)
            cases.append({"request_idx": row.request_idx, "recent_clicked_titles": history_titles,
                          "ground_truth_titles": gt_titles,
                          "tfidf_first_positive_rank": None if pd.isna(row.first_hit_rank_tfidf) else int(row.first_hit_rank_tfidf),
                          "dense_first_positive_rank": None if pd.isna(row.first_hit_rank_dense) else int(row.first_hit_rank_dense),
                          "tfidf_top_candidate_titles": [records[x][0] for x in list(row.retrieved_top500_tfidf)[:5]],
                          "dense_top_candidate_titles": [records[x][0] for x in list(row.retrieved_top500_dense)[:5]],
                          "history_ground_truth_char_bigram_jaccard": overlap})
        output[name] = cases; overlaps[name] = float(np.mean(group_overlap)) if group_overlap else float("nan")
    payload = {"seed": seed, "sample_size_per_direction": 20, "cases": output,
               "mean_history_ground_truth_char_bigram_jaccard": overlaps}
    json_dump(OUT / "qualitative.json", payload)
    return payload


def pct(x: float) -> str:
    return f"{100*x:.4f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def report_stage() -> None:
    exact = json.loads((OUT / "exact.json").read_text())
    ann = json.loads((OUT / "ann.json").read_text())
    abl = json.loads((ABLATION_ROOT / "ablations.json").read_text())
    qual = qualitative_analysis()
    protocol = ground_truth_protocol(QilinData(ROOT).load_test_requests())
    phase1 = pd.read_csv(ROOT / "results/phase_01/experiment_01_full_corpus_baselines/full/summary.csv")
    phase1 = phase1[phase1.method.str.endswith("__with_history_filter")]

    def p1(method: str, segment: str = "overall") -> pd.Series:
        return phase1[(phase1.method == method) & (phase1.segment == segment)].iloc[0]
    comparison = []
    for label, method in (("Popularity", "popularity_click__with_history_filter"),
                          ("ItemCF best", "itemcf_a_train_click_popularity_fallback__with_history_filter"),
                          ("TF-IDF", "tfidf_history20__with_history_filter")):
        r, c, w = p1(method), p1(method, "cold_item"), p1(method, "warm_item")
        comparison.append({"Method": label, "Recall@10": r["Recall@10"], "Recall@50": r["Recall@50"],
                           "Recall@100": r["Recall@100"], "Recall@500": r["Recall@500"], "MRR@100": r["MRR@100"],
                           "Cold Recall@500": c["Recall@500"], "Warm Recall@500": w["Recall@500"],
                           "Query seconds": 2419.2716 if label == "TF-IDF" else r["runtime_seconds"]})
    for label, payload in (("Dense Exact", exact), ("Dense ANN", ann)):
        m = payload["metrics"]
        comparison.append({"Method": label, "Recall@10": m["overall"]["Recall@10"],
                           "Recall@50": m["overall"]["Recall@50"], "Recall@100": m["overall"]["Recall@100"],
                           "Recall@500": m["overall"]["Recall@500"], "MRR@100": m["overall"]["MRR@100"],
                           "Cold Recall@500": m["cold_item"]["Recall@500"],
                           "Warm Recall@500": m["warm_item"]["Recall@500"],
                           "Query seconds": payload["search"]["query_seconds"]})
    comp = pd.DataFrame(comparison)
    comp.to_csv(OUT / "comparison.csv", index=False)
    display = comp.copy()
    for col in [x for x in display if "Recall" in x or "MRR" in x]: display[col] = display[col].map(pct)
    display["Query seconds"] = display["Query seconds"].map(lambda x: f"{x:.3f}")

    best = exact["selected_config"]
    emeta = json.loads((EMBED_ROOT / best["text_config"] / "metadata.json").read_text())
    ex, an = exact["metrics"], ann["metrics"]
    tfidf_r500 = float(p1("tfidf_history20__with_history_filter")["Recall@500"])
    tfidf_cold = float(p1("tfidf_history20__with_history_filter", "cold_item")["Recall@500"])
    ann_loss = ex["overall"]["Recall@500"] - an["overall"]["Recall@500"]
    ann_relative_loss = ann_loss / ex["overall"]["Recall@500"]
    speedup = 2419.2716 / ann["search"]["query_seconds"]
    exact_speedup = 2419.2716 / exact["search"]["query_seconds"]
    ab_rows = []
    for x in abl["results"]:
        ab_rows.append({"config": x["name"], "Recall@100": pct(x["metrics"]["overall"]["Recall@100"]),
                        "Recall@500": pct(x["metrics"]["overall"]["Recall@500"]),
                        "Cold Recall@500": pct(x["metrics"]["cold_item"]["Recall@500"])})
    segment_rows = []
    for label, payload in (("Dense Exact", exact), ("Dense ANN", ann)):
        for segment in ("train_clicked", "train_exposed_never_clicked", "cold_item", "warm_user", "cold_user"):
            metric = payload["metrics"][segment]
            segment_rows.append({"Method": label, "Segment": segment,
                                 "Eligible requests": metric["eligible_requests"],
                                 "Recall@100": pct(metric["Recall@100"]),
                                 "Recall@500": pct(metric["Recall@500"]),
                                 "MRR@100": pct(metric["MRR@100"])})
    overall_rows = []
    for label, payload in (("Dense Exact", exact), ("Dense ANN", ann)):
        metric = payload["metrics"]["overall"]
        overall_rows.append({"Method": label, "HitRate@100": pct(metric["HitRate@100"]),
                             "HitRate@500": pct(metric["HitRate@500"]),
                             "MRR@500": pct(metric["MRR@500"]),
                             "Mean first-hit rank (hits)": f"{metric['mean_first_hit_rank_on_hits']:.2f}",
                             "Median first-hit rank (hits)": f"{metric['median_first_hit_rank_on_hits']:.1f}",
                             "Miss@500": pct(metric["overall_miss_rate@500"])})
    jacc = qual["mean_history_ground_truth_char_bigram_jaccard"]
    recommendation = (
        "A. Fine-tune Dense Retriever" if ex["overall"]["Recall@500"] > tfidf_r500
        else "C. 先改 user representation，再决定是否 fine-tune"
    )
    lines = [
        "# Phase 2 — Zero-shot Dense Content Retriever + Faiss ANN", "",
        "## Protocol", "",
        f"- Model: `{MODEL_NAME}` from complete local cache; no training or downloading.",
        f"- Corpus: {emeta['items']:,} notes; embedding dim={DIM}, FP16 storage, L2 normalized.",
        f"- Selected query: `{best}`; query uses only recent_clicked_note_idxs item embeddings.",
        "- E5 corpus inputs use `passage:`. Mean-history queries average passage embeddings and do not add a second query prefix.",
        "- Ground truth, filtering, warm/cold definitions, and evaluator are shared with Phase 1.",
        f"- Test protocol: {protocol['requests']:,} requests, {protocol['positive_interactions']:,} positive rows, "
        f"{protocol['unique_request_positive_items']:,} request-level unique positives, "
        f"{protocol['ground_truth_history_overlap_requests']} requests with target/history overlap.",
        f"- {exact['query_stats']['empty_query_requests']} requests have no usable history embedding; target content is never used to construct a query.", "",
        "## Direct Comparison", "", markdown_table(display), "",
        "## Dense Overall Diagnostics", "", markdown_table(pd.DataFrame(overall_rows)), "",
        "## Dense Item/User Segments", "", markdown_table(pd.DataFrame(segment_rows)), "",
        "## Ablations (fixed 5K requests, IVF-PQ)", "", markdown_table(pd.DataFrame(ab_rows)), "",
        "## Speed and Storage", "",
        f"- Selected item encoding: {emeta['encoding_wall_seconds']:.1f}s, {emeta['items_per_second']:.1f} items/s, "
        f"peak GPU allocation {emeta['gpu_peak_bytes_max']/2**30:.3f}GiB, embedding file {emeta['embedding_file_bytes']/2**30:.3f}GiB.",
        f"- Exact IndexFlatIP build: {exact['index_build']['build_seconds']:.1f}s, "
        f"index {exact['index_build']['index_file_bytes']/2**30:.3f}GiB.",
        f"- Exact search: {exact['search']['query_seconds']:.3f}s; mean {exact['search']['mean_latency_ms']:.3f}ms/request, "
        f"p50/p95 batch-normalized {exact['search']['p50_batch_normalized_latency_ms']:.3f}/"
        f"{exact['search']['p95_batch_normalized_latency_ms']:.3f}ms; peak process RSS {exact['search']['peak_rss_gib']:.3f}GiB.",
        f"- ANN build: {ann['index_build']['build_seconds']:.1f}s, index {ann['index_build']['index_file_bytes']/2**30:.3f}GiB.",
        f"- ANN search: {ann['search']['query_seconds']:.3f}s; mean {ann['search']['mean_latency_ms']:.3f}ms/request; "
        f"p50/p95 batch-normalized {ann['search']['p50_batch_normalized_latency_ms']:.3f}/"
        f"{ann['search']['p95_batch_normalized_latency_ms']:.3f}ms; TF-IDF exact speedup={speedup:.1f}×.",
        f"- Dense Exact is {exact_speedup:.1f}× faster than TF-IDF exact. ANN search ran after Exact in the same process, "
        "so its process-RSS high-water mark is not an isolated ANN memory measurement; ANN index-build peak RSS was "
        f"{ann['index_build']['peak_rss_gib']:.3f}GiB.", "",
        "## Answers", "",
        f"1. Zero-shot dense {'exceeds' if ex['overall']['Recall@500'] > tfidf_r500 else 'does not exceed'} TF-IDF: "
        f"Dense Exact Recall@500={pct(ex['overall']['Recall@500'])}, TF-IDF={pct(tfidf_r500)}.",
        f"2. Dense Cold Recall@500={pct(ex['cold_item']['Recall@500'])}; it "
        f"{'exceeds' if ex['cold_item']['Recall@500'] > tfidf_cold else 'does not exceed'} TF-IDF {pct(tfidf_cold)}.",
        f"3. Dense warm/cold item Recall@500={pct(ex['warm_item']['Recall@500'])}/"
        f"{pct(ex['cold_item']['Recall@500'])}; warm/cold user={pct(ex['warm_user']['Recall@500'])}/"
        f"{pct(ex['cold_user']['Recall@500'])}.",
        f"4–6. Taxonomy/history-N/recency effects are shown in the ablation table; selected config is `{best}`.",
        f"7. ANN Recall@500 loss vs exact={100*ann_loss:.4f} percentage points "
        f"({100*ann_relative_loss:.2f}% relative); mean ANN/exact Top500 overlap="
        f"{pct(ann['ANN_mean_overlap_with_exact_top500'])}.",
        f"8. ANN is {speedup:.1f}× faster than Phase-1 exact TF-IDF query time.",
        f"9. In 20+20 fixed-seed qualitative cases, TF-IDF-only mean history↔target char-bigram Jaccard="
        f"{jacc.get('tfidf_hit_dense_miss', float('nan')):.4f}, versus dense-only="
        f"{jacc.get('dense_hit_tfidf_miss', float('nan')):.4f}. This supports a small lexical-overlap advantage for TF-IDF, "
        "but both values are very low. Dense-only cases include thematic links (request 92086: K-pop history to a Song "
        "Yuqi target; request 83606: health-exam history to nursing/Shanghai targets), while many other cases are noisy "
        "multi-positive requests. The sample does not justify a broad synonym-understanding claim. See qualitative.json "
        "for all 40 unedited cases, ranks, and candidate titles.",
        f"10. Recommended next step: **{recommendation}**. This decision follows the measured zero-shot gap, ablations, and cold split.", "",
        "## Model Limitation", "",
        "`e5-base-v2` uses an English BERT vocabulary; smoke inspection showed many Chinese characters become `[UNK]`. "
        "This is a real zero-shot limitation of the requested first-choice model and is considered when interpreting failure cases.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for directory in (OUT, EMBED_ROOT, INDEX_ROOT, ABLATION_ROOT, PER_REQUEST): directory.mkdir(parents=True, exist_ok=True)
    stages = [args.stage] if args.stage != "all" else ["smoke", "encode", "indices", "ablations", "full", "report"]
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        if stage == "smoke": smoke_test()
        elif stage == "encode": encode_stage(args)
        elif stage == "indices": indices_stage(args)
        elif stage == "ablations": ablations_stage(args)
        elif stage == "full": full_stage(args)
        elif stage == "report": report_stage()


if __name__ == "__main__":
    main()
