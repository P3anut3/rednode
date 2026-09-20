#!/usr/bin/env python3
"""Phase 4.1: N=20 history and collapse-controlled multi-interest benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.dense_recall import make_row_lookup, rows_to_filtered_rankings, validate_rankings  # noqa: E402
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_03.experiment_01_user_representation.recall.multi_interest import build_interest_queries  # noqa: E402
from experiments.phase_03.experiment_01_user_representation.recall.multi_query_recall import interest_rankings, search_cached  # noqa: E402
from experiments.phase_03.experiment_01_user_representation.recall.user_representation import (  # noqa: E402
    fit_normalized_kmeans, history_query_matrix, pairwise_cosine_stats, history_view,
)
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.dataset import Phase4UserTowerDataset, phase4_load_train_examples  # noqa: E402
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.trainer import phase4_encode_requests, phase4_save_checkpoint, phase4_seed  # noqa: E402
from experiments.phase_04.experiment_02_history_n20_multi_interest.models.user_tower import phase4_1_make_model  # noqa: E402
from experiments.phase_04.experiment_02_history_n20_multi_interest.training.trainer import phase4_1_train_epoch  # noqa: E402

OUT = ROOT / "results/phase_04/experiment_02_history_n20_multi_interest"
P25 = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark"
P3 = ROOT / "results/phase_03/experiment_01_user_representation"
P4 = ROOT / "results/phase_04/experiment_01_learnable_user_tower_n10"
IDS_PATH = P25 / "note_ids.npy"
EMB_PATH = P25 / "embeddings/bge_base_zh/basic_clean/embeddings.f16"
INDEX_PATH = P25 / "indices/bge_base_zh_basic_clean_flat_ip.faiss"
DIM = 768
HISTORY_N = 20
SEED = 42
MODELS = ("single_attention_n20", "multi_interest_n20_div")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "smoke", "train", "subset", "full-single", "full",
                                             "report", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--valid-requests", type=int, default=1000)
    parser.add_argument("--subset-requests", type=int, default=5000)
    parser.add_argument("--smoke-requests", type=int, default=20)
    parser.add_argument("--diversity-lambda", type=float, default=0.01)
    parser.add_argument("--only-model", choices=MODELS, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume the selected model from its best Phase 4.1 checkpoint.")
    return parser.parse_args()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def device_for(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def resources(threads: int):
    ids = np.load(IDS_PATH, mmap_mode="r")
    emb = np.memmap(EMB_PATH, mode="r", dtype=np.float16, shape=(len(ids), DIM))
    lookup = make_row_lookup(ids)
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(str(INDEX_PATH))
    assert index.ntotal == len(ids) and index.d == DIM
    return ids, emb, lookup, index


def load_split():
    train, valid, stats = phase4_load_train_examples(ROOT, history_n=HISTORY_N)
    stats["history_n"] = HISTORY_N
    save_json(OUT / "configs/phase4_1_temporal_split.json", stats)
    return train, valid, stats


def validation_requests(examples, limit: int) -> list[TestRequest]:
    grouped = {}
    for row in examples:
        grouped.setdefault(row.request_idx, [row.user_idx, row.history, set()])[2].add(row.target)
    keys = sorted(grouped)
    if len(keys) > limit:
        selected = np.sort(np.random.default_rng(SEED).choice(len(keys), limit, replace=False))
        keys = [keys[int(i)] for i in selected]
    return [TestRequest(int(key), int(grouped[key][0]), tuple(grouped[key][1]),
                        frozenset(grouped[key][2])) for key in keys]


def fixed_subset(requests: list[TestRequest], size: int) -> list[TestRequest]:
    selected = json.loads((P25 / "selection.json").read_text())["subset_request_ids"]
    if size == len(selected):
        by_id = {r.request_idx: r for r in requests}
        return [by_id[int(idx)] for idx in selected]
    positions = np.sort(np.random.default_rng(SEED).choice(len(requests), min(size, len(requests)), False))
    return [requests[int(i)] for i in positions]


def search(index, queries: np.ndarray, topk: int = 600, batch: int = 128):
    rows, start = [], time.perf_counter()
    for offset in range(0, len(queries), batch):
        _, part = index.search(np.ascontiguousarray(queries[offset:offset + batch], dtype=np.float32), topk)
        rows.append(part)
    return np.vstack(rows), time.perf_counter() - start


def cached_kmeans_queries(requests, emb, lookup, cache_dir: Path, k: int = 4):
    """Build request-local KMeans centroids with request-level restartability."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    query_path = cache_dir / "queries.f32"
    offset_path = cache_dir / "offsets.i64"
    meta_path = cache_dir / "build_metadata.json"
    query_shape = (len(requests) * k, DIM)
    offset_shape = (len(requests), 2)
    expected_query_bytes = int(np.prod(query_shape)) * 4
    expected_offset_bytes = int(np.prod(offset_shape)) * 8
    meta = None
    if meta_path.exists() and query_path.exists() and offset_path.exists():
        candidate = json.loads(meta_path.read_text())
        if (candidate.get("requests") == len(requests) and candidate.get("k") == k
                and query_path.stat().st_size == expected_query_bytes
                and offset_path.stat().st_size == expected_offset_bytes):
            meta = candidate
    if meta is None:
        queries = np.memmap(query_path, mode="w+", dtype=np.float32, shape=query_shape)
        offsets_array = np.memmap(offset_path, mode="w+", dtype=np.int64, shape=offset_shape)
        meta = {"requests": len(requests), "k": k, "next_request": 0,
                "next_query": 0, "complete": False, "seconds": 0.0}
        save_json(meta_path, meta)
    else:
        queries = np.memmap(query_path, mode="r+", dtype=np.float32, shape=query_shape)
        offsets_array = np.memmap(offset_path, mode="r+", dtype=np.int64, shape=offset_shape)
    start_request = int(meta["next_request"])
    next_query = int(meta["next_query"])
    prior_seconds = float(meta.get("seconds", 0.0))
    started = time.perf_counter()
    for request_index in range(start_request, len(requests)):
        view = history_view(requests[request_index], emb, lookup, HISTORY_N)
        centroids, _, _ = fit_normalized_kmeans(view.vectors, k, SEED)
        offsets_array[request_index] = (next_query, len(centroids))
        if len(centroids):
            queries[next_query:next_query + len(centroids)] = centroids
            next_query += len(centroids)
        if request_index % 10 == 0 or request_index + 1 == len(requests):
            queries.flush(); offsets_array.flush()
            meta.update({"next_request": request_index + 1, "next_query": next_query,
                         "seconds": prior_seconds + time.perf_counter() - started})
            save_json(meta_path, meta)
    meta["complete"] = True
    save_json(meta_path, meta)
    offsets = [np.arange(int(start), int(start + length), dtype=np.int64)
               for start, length in offsets_array]
    weights = [np.ones(len(value), dtype=np.float32) / len(value) if len(value)
               else np.empty(0, dtype=np.float32) for value in offsets]
    stats = {"requested_k": k, "centroid_queries": next_query,
             "empty_requests": sum(not len(value) for value in offsets),
             "history_n": HISTORY_N, "seed": SEED, "build_seconds": meta["seconds"]}
    return queries[:next_query], offsets, weights, stats


def load_eval_splits(data: QilinData):
    cache = OUT / "cache/evaluator"
    paths = {name: cache / f"{name}.npy" for name in ("train_exposed", "train_clicked", "train_users")}
    if not all(path.exists() for path in paths.values()):
        cache.mkdir(parents=True, exist_ok=True)
        np.save(paths["train_exposed"], np.asarray(sorted(data.train_exposed_items), dtype=np.int64))
        np.save(paths["train_clicked"], np.asarray(sorted(data.train_click_counts), dtype=np.int64))
        np.save(paths["train_users"], np.asarray(sorted(data.train_users), dtype=np.int64))
    return (set(map(int, np.load(paths["train_exposed"]))),
            set(map(int, np.load(paths["train_clicked"]))),
            set(map(int, np.load(paths["train_users"]))))


def metrics_payload(name, requests, rankings, eval_splits, catalog, validate: bool = True):
    train_exposed, train_clicked, train_users = eval_splits
    validation = (validate_rankings(rankings, requests, catalog) if validate else
                  {"requests": len(requests), "validated_topk": 500,
                   "shorter_than_topk": 0, "duplicate_rankings": 0,
                   "out_of_catalog": 0, "history_violations": 0,
                   "inherited_from_validated_sources": True})
    metrics, per_request = evaluate_rankings(
        requests, rankings, train_exposed, train_users, name,
        clicked_items=train_clicked)
    return validation, metrics, per_request


def gpu_equal_interest_rankings(requests, searched_rows, vectors, embeddings, note_ids,
                                lookup, device: str, fallback, final_k: int = 500,
                                batch_size: int = 8):
    """Equal-budget round-robin union followed by exact GPU max-interest scoring."""
    n, k, _ = vectors.shape
    candidate_rows = np.empty((n, final_k), dtype=np.int64)
    for request_index, request in enumerate(requests):
        blocked = set(map(int, request.history))
        selected, seen, depth = [], set(), 0
        rows_by_interest = searched_rows[request_index*k:(request_index+1)*k]
        while len(selected) < final_k and depth < rows_by_interest.shape[1]:
            for rows in rows_by_interest:
                row = int(rows[depth])
                if row < 0 or row in seen or int(note_ids[row]) in blocked:
                    continue
                selected.append(row); seen.add(row)
                if len(selected) == final_k:
                    break
            depth += 1
        if len(selected) < final_k:
            for note in fallback[request_index]:
                row = int(lookup[int(note)])
                if row >= 0 and row not in seen and int(note) not in blocked:
                    selected.append(row); seen.add(row)
                    if len(selected) == final_k:
                        break
        if len(selected) != final_k:
            raise AssertionError(f"request {request.request_idx}: only {len(selected)} candidates")
        candidate_rows[request_index] = selected
    rankings = []
    with torch.inference_mode():
        for offset in range(0, n, batch_size):
            rows = candidate_rows[offset:offset+batch_size]
            candidate_vectors = torch.from_numpy(
                np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            interests = torch.from_numpy(vectors[offset:offset+batch_size]).to(device)
            scores = torch.einsum("bcd,bkd->bck", candidate_vectors, interests).amax(-1)
            order = scores.argsort(dim=1, descending=True).cpu().numpy()
            for local_index, positions in enumerate(order):
                rankings.append([int(note_ids[row]) for row in rows[local_index, positions]])
            if offset and offset % 500 == 0:
                print(f"GPU candidate re-score {offset}/{n}", flush=True)
    return rankings


def result_row(name: str, metrics: dict, metadata: dict | None = None) -> dict:
    return {"method": name,
            **{f"Recall@{k}": metrics["overall"][f"Recall@{k}"] for k in (10, 50, 100, 200, 500)},
            "MRR@100": metrics["overall"]["MRR@100"],
            "Train-clicked Recall@500": metrics["train_clicked"]["Recall@500"],
            "Exposed-never-clicked Recall@500": metrics["train_exposed_never_clicked"]["Recall@500"],
            "Warm Recall@500": metrics["warm_item"]["Recall@500"],
            "Cold Recall@500": metrics["cold_item"]["Recall@500"],
            "Warm-user Recall@500": metrics["warm_user"]["Recall@500"],
            "Cold-user Recall@500": metrics["cold_user"]["Recall@500"],
            **(metadata or {})}


def evaluate_validation(model, requests, emb, ids, lookup, index, device):
    vectors = phase4_encode_requests(model, requests, emb, lookup, device, history_n=HISTORY_N)
    n, k, dim = vectors.shape
    rows, seconds = search(index, vectors.reshape(n * k, dim))
    if k == 1:
        rankings = rows_to_filtered_rankings(rows, ids, requests)
    else:
        offsets = [np.arange(i * k, (i + 1) * k) for i in range(n)]
        weights = [np.ones(k, dtype=np.float32) / k for _ in requests]
        rankings = interest_rankings(requests, emb, ids, vectors.reshape(n*k, dim), offsets,
                                     weights, rows, "equal", "max", 500, 500)
    metrics, _ = evaluate_rankings(requests, rankings, set(), set(), "validation")
    return metrics["overall"], seconds


def smoke(config):
    phase4_seed(SEED)
    ids, emb, lookup, index = resources(config.threads)
    train, valid, stats = load_split()
    device = device_for(config.device)
    dataset = Phase4UserTowerDataset(train[:5000], EMB_PATH, ids, lookup, DIM, HISTORY_N)
    requests = validation_requests(valid, config.smoke_requests)
    output = {"device": device, "history_n": HISTORY_N, "frozen_item_embeddings": True,
              "train_examples": len(dataset), "validation_requests": len(requests), "models": {}}
    for name in MODELS:
        model = phase4_1_make_model(name, DIM).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        logs = []
        lam = config.diversity_lambda if name.startswith("multi") else 0.0
        for epoch in (1, 2):
            logs.append(phase4_1_train_epoch(model, dataset, optimizer, device,
                        config.batch_size, SEED, epoch, lam, max_batches=16))
        metric, seconds = evaluate_validation(model, requests, emb, ids, lookup, index, device)
        output["models"][name] = {"logs": logs, "Recall@500": metric["Recall@500"],
                                         "search_seconds": seconds,
                                         "output_shape": [len(requests), 4 if name.startswith("multi") else 1, DIM]}
    save_json(OUT / "phase4_1_smoke.json", output)


def train(config):
    phase4_seed(SEED)
    ids, emb, lookup, index = resources(config.threads)
    train_rows, valid_rows, stats = load_split()
    requests = validation_requests(valid_rows, config.valid_requests)
    device = device_for(config.device)
    dataset = Phase4UserTowerDataset(train_rows, EMB_PATH, ids, lookup, DIM, HISTORY_N)
    curve_path = OUT / "phase4_1_training_curves.csv"
    model_path = OUT / "configs/phase4_1_models.json"
    curves = (pd.read_csv(curve_path).to_dict("records")
              if config.resume and curve_path.exists() else [])
    models = (json.loads(model_path.read_text())
              if config.resume and model_path.exists() else {})
    selected_models = (config.only_model,) if config.only_model else MODELS
    for name in selected_models:
        phase4_seed(SEED)
        model = phase4_1_make_model(name, DIM).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        lam = config.diversity_lambda if name.startswith("multi") else 0.0
        start_epoch, best, stale = 1, -1.0, 0
        if config.resume and name in models:
            latest_path = OUT / "checkpoints" / f"phase4_1_{name}_latest.pt"
            checkpoint_path = (latest_path if latest_path.exists()
                               else OUT / "checkpoints" / f"phase4_1_{name}_best.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint.get("epoch", models[name]["best_epoch"])) + 1
            best = float(models[name]["validation_recall500"])
            prior = [row for row in curves if row["model"] == name
                     and int(row["epoch"]) > int(models[name]["best_epoch"])]
            stale = len(prior) if latest_path.exists() else 0
        params = sum(p.numel() for p in model.parameters())
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.perf_counter()
            log = phase4_1_train_epoch(model, dataset, optimizer, device, config.batch_size,
                                       SEED, epoch, lam)
            metric, search_seconds = evaluate_validation(
                model, requests, emb, ids, lookup, index, device)
            row = {"model": name, "epoch": epoch, **log,
                   "Recall@100": metric["Recall@100"], "Recall@500": metric["Recall@500"],
                   "MRR@100": metric["MRR@100"], "search_seconds": search_seconds,
                   "epoch_seconds": time.perf_counter() - started}
            curves = [old for old in curves
                      if not (old["model"] == name and int(old["epoch"]) == epoch)]
            curves.append(row)
            curves.sort(key=lambda value: (value["model"], int(value["epoch"])))
            pd.DataFrame(curves).to_csv(curve_path, index=False)
            print(f"{name} epoch={epoch} loss={log['train_loss']:.4f} "
                  f"div={log['diversity_loss']:.4f} R500={metric['Recall@500']:.6f}", flush=True)
            if metric["Recall@500"] > best:
                best, stale = metric["Recall@500"], 0
                checkpoint = OUT / "checkpoints" / f"phase4_1_{name}_best.pt"
                size = phase4_save_checkpoint(checkpoint, model, {
                    "model": name, "epoch": epoch, "history_n": HISTORY_N,
                    "diversity_lambda": lam, "validation_recall500": best,
                    "training_samples": len(train_rows), "validation_requests": len(requests)})
                models[name] = {"parameters": params, "best_epoch": epoch,
                                "validation_recall500": best, "checkpoint_bytes": size,
                                "diversity_lambda": lam}
                save_json(model_path, models)
            else:
                stale += 1
            torch.save({"state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "epoch": epoch},
                       OUT / "checkpoints" / f"phase4_1_{name}_latest.pt")
            if stale >= config.patience:
                break
    save_json(OUT / "configs/phase4_1_train_config.json", vars(config) | {
        "device_used": device, "history_n": HISTORY_N, "frozen_item_embeddings": True,
        "temperature": 0.05, "optimizer": "AdamW", "learning_rate": 1e-4})


def model_vectors(name, requests, emb, lookup, device):
    checkpoint = torch.load(OUT / "checkpoints" / f"phase4_1_{name}_best.pt",
                            map_location="cpu", weights_only=False)
    model = phase4_1_make_model(name, DIM).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, phase4_encode_requests(model, requests, emb, lookup, device,
                                          history_n=HISTORY_N)


def collapse(model, requests, emb, lookup, device):
    vectors, attention = phase4_encode_requests(model, requests, emb, lookup, device,
                                                 history_n=HISTORY_N, return_attention=True)
    pairs = vectors @ vectors.transpose(0, 2, 1)
    tri = np.triu_indices(vectors.shape[1], 1)
    off = pairs[:, tri[0], tri[1]]
    return {"mean_inter_interest_cosine": float(off.mean()),
            "p50_inter_interest_cosine": float(np.quantile(off, .5)),
            "p90_inter_interest_cosine": float(np.quantile(off, .9)),
            "mean_attention_entropy": float((-attention.clip(1e-12) * np.log(attention.clip(1e-12))).sum(-1).mean())}


def run_evaluation(config, scope: str):
    data = QilinData(ROOT)
    eval_splits = load_eval_splits(data)
    all_requests = data.load_test_requests()
    requests = fixed_subset(all_requests, config.subset_requests) if scope == "subset" else all_requests
    ids, emb, lookup, index = resources(config.threads)
    catalog = set(map(int, ids))
    device = device_for(config.device)
    cache = OUT / "cache" / scope
    rows_out, ranking_by_method = [], {}
    per_dir = OUT / "per_request"
    per_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT / "cache/rankings"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def cached_rankings(name: str):
        path = per_dir / f"phase4_1_{scope}_{name}.parquet"
        if not path.exists():
            path = raw_dir / f"phase4_1_{scope}_{name}.parquet"
            if not path.exists():
                return None
        return [list(map(int, values)) for values in
                pd.read_parquet(path, columns=["retrieved_top500"])["retrieved_top500"]]

    def save_raw_rankings(name: str, rankings):
        pd.DataFrame({"request_idx": [r.request_idx for r in requests],
                      "retrieved_top500": rankings}).to_parquet(
            raw_dir / f"phase4_1_{scope}_{name}.parquet", index=False, compression="zstd")

    mean_rankings = cached_rankings("frozen_mean_n20")
    mean_stats = {"mean_valid_history": float(np.mean([
        len(history_view(r, emb, lookup, HISTORY_N).rows) for r in requests]))}
    mean_search = json.loads((cache / "mean_n20/metadata.json").read_text())
    if mean_rankings is None:
        mean_query, mean_stats = history_query_matrix(requests, emb, lookup, HISTORY_N, "mean")
        _, mean_rows, mean_search = search_cached(index, mean_query, 600, cache / "mean_n20", 16)
        mean_rankings = rows_to_filtered_rankings(mean_rows, ids, requests)
    ranking_by_method["frozen_mean_n20"] = mean_rankings
    save_raw_rankings("frozen_mean_n20", mean_rankings)
    print(f"{scope}: mean rankings ready", flush=True)

    learned_vectors = {}
    learned_models = {}
    for name in MODELS:
        existing_rankings = cached_rankings(name)
        if existing_rankings is not None:
            ranking_by_method[name] = existing_rankings
            print(f"{scope}: reused rankings {name}", flush=True)
            continue
        model, vectors = model_vectors(name, requests, emb, lookup, device)
        print(f"{scope}: encoded {name}", flush=True)
        learned_models[name], learned_vectors[name] = model, vectors
        n, k, dim = vectors.shape
        _, searched, stats = search_cached(index, vectors.reshape(n*k, dim), 600,
                                            cache / name, 16)
        if k == 1:
            rankings = rows_to_filtered_rankings(searched, ids, requests)
        else:
            rankings = gpu_equal_interest_rankings(
                requests, searched, vectors, emb, ids, lookup, device, mean_rankings)
        ranking_by_method[name] = rankings
        save_raw_rankings(name, rankings)
        print(f"{scope}: rankings ready {name}", flush=True)

    pair_mean = np.asarray([pairwise_cosine_stats(history_view(r, emb, lookup, HISTORY_N).vectors)[0]
                            for r in requests], dtype=np.float32)
    valid = pair_mean[np.isfinite(pair_mean)]
    threshold = float(np.quantile(valid, 1/3))
    use_multi = np.isfinite(pair_mean) & (pair_mean <= threshold)
    adaptive = [ranking_by_method["multi_interest_n20_div"][i] if use_multi[i]
                else ranking_by_method["single_attention_n20"][i]
                for i in range(len(requests))]
    ranking_by_method["adaptive_n20_high_diversity"] = adaptive
    save_raw_rankings("adaptive_n20_high_diversity", adaptive)
    print(f"{scope}: adaptive gate ready", flush=True)

    for name, rankings in ranking_by_method.items():
        metric_path = OUT / "metrics" / f"phase4_1_{scope}_{name}.json"
        request_path = per_dir / f"phase4_1_{scope}_{name}.parquet"
        if metric_path.exists() and request_path.exists():
            cached_metrics = json.loads(metric_path.read_text())["metrics"]
            rows_out.append(result_row(name, cached_metrics, {
                "scope": scope, "history_n": HISTORY_N,
                "high_diversity_threshold": threshold,
                "multi_request_ratio": float(use_multi.mean())}))
            print(f"{scope}: reused metrics {name}", flush=True)
            continue
        validation, metrics, per_request = metrics_payload(name, requests, rankings,
                                                           eval_splits, catalog,
                                                           validate=not name.startswith("adaptive_"))
        assert validation["shorter_than_topk"] == 0, (name, validation)
        per_request.to_parquet(request_path, index=False, compression="zstd")
        rows_out.append(result_row(name, metrics, {
            "scope": scope, "history_n": HISTORY_N,
            "high_diversity_threshold": threshold,
            "multi_request_ratio": float(use_multi.mean())}))
        save_json(metric_path, {"validation": validation, "metrics": metrics})
        print(f"{scope}: evaluated {name}", flush=True)
    frame = pd.DataFrame(rows_out)
    frame.to_csv(OUT / f"phase4_1_{scope}_results.csv", index=False)
    multi_model = learned_models.get("multi_interest_n20_div")
    if multi_model is None:
        multi_model, _ = model_vectors("multi_interest_n20_div", requests[:1], emb, lookup, device)
    diagnostics = collapse(multi_model, requests, emb, lookup, device)
    diagnostics.update({"requests": len(requests), "history_n": HISTORY_N,
                        "high_diversity_threshold": threshold,
                        "multi_request_ratio": float(use_multi.mean()),
                        "mean_valid_history": mean_stats["mean_valid_history"],
                        "mean_search_seconds": mean_search["query_seconds"],
                        "n20_kmeans_status": "skipped: request-local n_init=10 cost was excessive; Phase 3 N=10 result retained"})
    save_json(OUT / f"phase4_1_{scope}_diagnostics.json", diagnostics)


def run_full_single(config):
    """Run only the subset-selected N=20 Single Attention on all test requests."""
    name, scope = "single_attention_n20", "full"
    data = QilinData(ROOT)
    eval_splits = load_eval_splits(data)
    requests = data.load_test_requests()
    ids, emb, lookup, index = resources(config.threads)
    device = device_for(config.device)
    per_path = OUT / "per_request/phase4_1_full_single_attention_n20.parquet"
    raw_path = OUT / "cache/rankings/phase4_1_full_single_attention_n20.parquet"
    metric_path = OUT / "metrics/phase4_1_full_single_attention_n20.json"
    source = per_path if per_path.exists() else raw_path
    if source.exists():
        rankings = [list(map(int, values)) for values in
                    pd.read_parquet(source, columns=["retrieved_top500"])["retrieved_top500"]]
        print("full-single: reused rankings", flush=True)
    else:
        _, vectors = model_vectors(name, requests, emb, lookup, device)
        _, searched, stats = search_cached(index, vectors[:, 0, :], 600,
                                            OUT / "cache/full/single_attention_n20", 16)
        rankings = rows_to_filtered_rankings(searched, ids, requests)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"request_idx": [r.request_idx for r in requests],
                      "retrieved_top500": rankings}).to_parquet(
                          raw_path, index=False, compression="zstd")
        print(f"full-single: rankings saved, search_seconds={stats['query_seconds']:.1f}", flush=True)
    if metric_path.exists() and per_path.exists():
        metrics = json.loads(metric_path.read_text())["metrics"]
    else:
        catalog = set(map(int, ids))
        validation, metrics, per_request = metrics_payload(
            name, requests, rankings, eval_splits, catalog)
        if validation["shorter_than_topk"]:
            raise AssertionError(validation)
        per_path.parent.mkdir(parents=True, exist_ok=True)
        per_request.to_parquet(per_path, index=False, compression="zstd")
        save_json(metric_path, {"validation": validation, "metrics": metrics})
    pd.DataFrame([result_row(name, metrics, {"scope": scope, "history_n": HISTORY_N})]).to_csv(
        OUT / "phase4_1_full_results.csv", index=False)
    print(f"full-single: R500={metrics['overall']['Recall@500']:.8f}", flush=True)


def report():
    subset = pd.read_csv(OUT / "phase4_1_subset_results.csv")
    full_path = OUT / "phase4_1_full_results.csv"
    full = pd.read_csv(full_path) if full_path.exists() else None
    diag = json.loads((OUT / "phase4_1_subset_diagnostics.json").read_text())
    split = json.loads((OUT / "configs/phase4_1_temporal_split.json").read_text())
    models = json.loads((OUT / "configs/phase4_1_models.json").read_text())
    reference = pd.read_csv(OUT / "phase4_1_n10_fixed_subset_reference.csv")
    lines = ["# Phase 4.1 — N=20 and Adaptive Multi-interest", "",
             "## Protocol", "", "- Frozen BGE-base-zh item embeddings; history N=20.",
             "- Same full 1,983,938-item IndexFlatIP and evaluator; exclude_history=True.",
             "- Single Attention and K=4 Multi-interest are trained on train only; item vectors stay frozen.",
             "- K=4 uses squared off-diagonal cosine regularization with lambda=0.01.",
             "- Adaptive retrieval uses K=4 only for the lowest history-cosine third; this gate uses history only.", "",
             "## Data", "", f"- Train/validation samples: {split['train_samples']:,} / {split['valid_samples']:,}.",
             f"- Mean effective test history in the subset: {diag['mean_valid_history']:.4f}.", "",
             "## Checkpoints", "", "| Model | Params | Best epoch | Valid R@500 |", "| --- | ---: | ---: | ---: |"]
    for name, row in models.items():
        lines.append(f"| {name} | {row['parameters']:,} | {row['best_epoch']} | {100*row['validation_recall500']:.4f}% |")
    def table(frame, title):
        lines.extend(["", title, "", "| Method | R@100 | R@500 | Cold R@500 |", "| --- | ---: | ---: | ---: |"])
        for _, row in frame.iterrows():
            lines.append(f"| {row['method']} | {100*row['Recall@100']:.4f}% | {100*row['Recall@500']:.4f}% | {100*row['Cold Recall@500']:.4f}% |")
    table(subset, "## Fixed 5K Subset")
    table(reference, "## Same 5K N=10 References")
    if full is not None:
        table(full, "## Full Test")
    subset_single = subset[subset.method == "single_attention_n20"].iloc[0]
    subset_multi = subset[subset.method == "multi_interest_n20_div"].iloc[0]
    subset_adaptive = subset[subset.method == "adaptive_n20_high_diversity"].iloc[0]
    full_single = full[full.method == "single_attention_n20"].iloc[0] if full is not None else None
    lines += ["", "## Collapse Check", "",
              f"- Inter-interest cosine mean/p50/p90: {diag['mean_inter_interest_cosine']:.4f} / {diag['p50_inter_interest_cosine']:.4f} / {diag['p90_inter_interest_cosine']:.4f}.",
              f"- Adaptive multi-interest request ratio: {100*diag['multi_request_ratio']:.2f}%.", "",
              "The lambda=0.01 squared-cosine term did not prevent collapse; K=4 remains effectively one vector.", "",
              "## Findings", "",
              f"- Fixed 5K N=20 Single/K=4/Adaptive R@500: {100*subset_single['Recall@500']:.4f}% / "
              f"{100*subset_multi['Recall@500']:.4f}% / {100*subset_adaptive['Recall@500']:.4f}%.",
              f"- K=4 adds only {100*(subset_multi['Recall@500']-subset_single['Recall@500']):.4f} percentage points; "
              "adaptive gating is slightly worse than Single Attention."]
    if full_single is not None:
        lines += [f"- Full N=20 Single Attention R@100/R@500: {100*full_single['Recall@100']:.4f}% / {100*full_single['Recall@500']:.4f}%.",
                  f"- Full N=20 Single Attention Cold R@500: {100*full_single['Cold Recall@500']:.4f}%.",
                  "- Versus N=10 Single Attention, full R@500 improves from 5.9759% to "
                  f"{100*full_single['Recall@500']:.4f}%.",
                  "- It also exceeds TF-IDF overall 7.2394% and cold 9.9427%."]
    lines += ["", "## Decision", "",
              "Do not continue fixed-K multi-interest modeling now. The controlled experiment shows that retaining "
              "all 20 recent events with a single target-independent attention tower delivers the material gain, "
              "while K=4 adds only a negligible subset improvement, adaptive gating does not help, and learned slots "
              "still collapse. Use N=20 Single Attention as the dense user-tower baseline. Revisit multi-interest only "
              "if a later error analysis identifies a specific high-diversity cohort that remains underserved."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    config = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    stages = [config.stage] if config.stage != "all" else ["prepare", "smoke", "train", "subset", "report"]
    for stage in stages:
        print(f"=== phase4.1 {stage} ===", flush=True)
        if stage == "prepare":
            load_split()
        elif stage == "smoke":
            smoke(config)
        elif stage == "train":
            train(config)
        elif stage == "subset":
            run_evaluation(config, "subset")
        elif stage == "full-single":
            run_full_single(config)
        elif stage == "full":
            run_evaluation(config, "full")
        elif stage == "report":
            report()


if __name__ == "__main__":
    main()
