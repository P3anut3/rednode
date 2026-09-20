#!/usr/bin/env python3
"""Phase 4: train only frozen-item user towers and evaluate exact full-corpus recall."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.dense_recall import make_row_lookup, rows_to_filtered_rankings, validate_rankings  # noqa: E402
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_03.experiment_01_user_representation.recall.multi_query_recall import interest_rankings  # noqa: E402
from experiments.phase_04.experiment_01_learnable_user_tower_n10.models.user_tower import phase4_make_model  # noqa: E402
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.dataset import (  # noqa: E402
    Phase4UserTowerDataset, phase4_load_train_examples,
)
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.trainer import (  # noqa: E402
    phase4_encode_requests, phase4_save_checkpoint, phase4_seed, phase4_train_epoch,
)

OUT = ROOT / "results/phase_04/experiment_01_learnable_user_tower_n10"
PHASE25 = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark"
PHASE3 = ROOT / "results/phase_03/experiment_01_user_representation"
EMBEDDING_PATH = PHASE25 / "embeddings/bge_base_zh/basic_clean/embeddings.f16"
INDEX_PATH = PHASE25 / "indices/bge_base_zh_basic_clean_flat_ip.faiss"
NOTE_IDS_PATH = PHASE25 / "note_ids.npy"
MODELS = ("mean_mlp", "single_attention", "multi_interest")
SEED = 42


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "smoke", "train", "test", "repair-practical",
                                             "diagnostics", "report", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--valid-requests", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--only-model", choices=MODELS, default=None,
                        help="For a targeted test-stage rerun; existing other model rows are preserved.")
    return parser.parse_args()


def json_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def resources(threads: int):
    ids = np.load(NOTE_IDS_PATH, mmap_mode="r")
    dim = int(json.loads((PHASE25 / "embeddings/bge_base_zh/basic_clean/metadata.json").read_text())["dim"])
    emb = np.memmap(EMBEDDING_PATH, mode="r", dtype=np.float16, shape=(len(ids), dim))
    lookup = make_row_lookup(ids)
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(str(INDEX_PATH))
    assert index.ntotal == len(ids) and index.d == dim
    return ids, emb, lookup, index


def validation_requests(examples, max_requests: int | None = None) -> list[TestRequest]:
    by_request = {}
    for row in examples:
        if row.request_idx not in by_request:
            by_request[row.request_idx] = [row.user_idx, row.history, set()]
        by_request[row.request_idx][2].add(row.target)
    ids = sorted(by_request)
    if max_requests is not None and len(ids) > max_requests:
        chosen = np.sort(np.random.default_rng(SEED).choice(len(ids), max_requests, replace=False))
        ids = [ids[int(i)] for i in chosen]
    return [TestRequest(request_idx=i, user_idx=by_request[i][0],
                        history=by_request[i][1], ground_truth=frozenset(by_request[i][2])) for i in ids]


def search_vectors(index, vectors: np.ndarray, topk: int = 600, batch_size: int = 128):
    all_rows = []
    start = time.perf_counter()
    for offset in range(0, len(vectors), batch_size):
        _, rows = index.search(np.ascontiguousarray(vectors[offset:offset + batch_size],
                                                   dtype=np.float32), topk)
        all_rows.append(rows)
    return np.vstack(all_rows), time.perf_counter() - start


def evaluate_model(name: str, model, requests: list[TestRequest], note_ids, embeddings, lookup,
                   index, device: str, scope: str, data: QilinData | None = None,
                   budget: str = "equal", save: bool = False):
    vectors = phase4_encode_requests(model, requests, embeddings, lookup, device)
    n, k, dim = vectors.shape
    rows, search_seconds = search_vectors(index, vectors.reshape(n * k, dim))
    if k == 1:
        rankings = rows_to_filtered_rankings(rows, note_ids, requests)
    else:
        offsets = [np.arange(i * k, (i + 1) * k) for i in range(n)]
        weights = [np.ones(k, dtype=np.float32) / k for _ in range(n)]
        rankings = interest_rankings(requests, embeddings, note_ids, vectors.reshape(n * k, dim),
                                     offsets, weights, rows, budget, "max", 500, 500)
    catalog = set(map(int, note_ids))
    validation = validate_rankings(rankings, requests, catalog)
    if data is None:
        metrics, per_request = evaluate_rankings(requests, rankings, set(), set(), name)
    else:
        metrics, per_request = evaluate_rankings(requests, rankings, data.train_exposed_items,
                                                  data.train_users, name,
                                                  clicked_items=set(data.train_click_counts))
    payload = {"model": name, "scope": scope, "budget": budget if k > 1 else "single",
               "requests": n, "interests": k, "search_seconds": search_seconds,
               "validation": validation, "metrics": metrics}
    if save:
        output = OUT / "per_request" / f"phase4_{scope}_{name}_{payload['budget']}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        per_request.to_parquet(output, index=False, compression="zstd")
        payload["per_request"] = str(output.relative_to(ROOT))
    return payload


def collapse_diagnostic(model, requests, note_ids, embeddings, lookup, device: str):
    if not requests: return {}
    vectors, attention = phase4_encode_requests(model, requests, embeddings, lookup, device,
                                                 return_attention=True)
    k = vectors.shape[1]
    pair = np.einsum("bkd,bjd->bkj", vectors, vectors)
    off = pair[:, np.triu_indices(k, 1)[0], np.triu_indices(k, 1)[1]]
    winners = np.zeros(k, dtype=np.int64)
    for request, interests in zip(requests, vectors):
        for target in request.ground_truth:
            if 0 <= target < len(lookup) and lookup[target] >= 0:
                target_vec = np.asarray(embeddings[int(lookup[target])], dtype=np.float32)
                winners[int(np.argmax(interests @ target_vec))] += 1
    usage = winners / max(1, winners.sum())
    return {"mean_inter_interest_cosine": float(off.mean()),
            "p50_inter_interest_cosine": float(np.quantile(off, .5)),
            "p90_inter_interest_cosine": float(np.quantile(off, .9)),
            "positive_winner_counts": winners.tolist(),
            "positive_winner_rates": usage.tolist(),
            "max_interest_usage": float(usage.max()),
            "attention_shape": list(attention.shape)}


def load_split():
    train, valid, stats = phase4_load_train_examples(ROOT)
    json_save(OUT / "configs/phase4_temporal_split.json", stats)
    return train, valid, stats


def smoke(config):
    phase4_seed()
    ids, emb, lookup, index = resources(config.threads)
    train, valid, stats = load_split()
    train = train[:min(5000, len(train))]
    requests = validation_requests(valid, 100)
    device = resolve_device(config.device)
    logs = []
    for name in MODELS:
        model = phase4_make_model(name, emb.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        dataset = Phase4UserTowerDataset(train, EMBEDDING_PATH, ids, lookup, emb.shape[1])
        for epoch in (1, 2):
            output = phase4_train_epoch(model, dataset, optimizer, device, config.batch_size,
                                        SEED, epoch, max_batches=16)
            logs.append({"model": name, "epoch": epoch, **output})
        result = evaluate_model(name, model, requests, ids, emb, lookup, index, device, "smoke")
        assert result["validation"]["shorter_than_topk"] == 0
        assert result["interests"] == (4 if name == "multi_interest" else 1)
    json_save(OUT / "phase4_smoke.json", {"device": device, "train_examples": len(train),
                                           "validation_requests": len(requests), "logs": logs,
                                           "temporal_split": stats,
                                           "item_embeddings_trainable": False})


def train_models(config):
    phase4_seed(); torch.set_num_threads(config.threads)
    ids, emb, lookup, index = resources(config.threads)
    train, valid, stats = load_split()
    requests = validation_requests(valid, config.valid_requests)
    device = resolve_device(config.device)
    dataset = Phase4UserTowerDataset(train, EMBEDDING_PATH, ids, lookup, emb.shape[1])
    curves, validation_rows, model_info = [], [], {}
    for name in MODELS:
        phase4_seed()
        model = phase4_make_model(name, emb.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        param_count = sum(p.numel() for p in model.parameters())
        model_info[name] = {"total_parameters": param_count, "trainable_parameters": param_count,
                            "frozen_item_representation_parameters": 0, "dim": emb.shape[1],
                            "interests": 4 if name == "multi_interest" else 1}
        best, stale = -1.0, 0
        for epoch in range(1, config.epochs + 1):
            start = time.perf_counter()
            output = phase4_train_epoch(model, dataset, optimizer, device, config.batch_size,
                                        SEED, epoch)
            valid_result = evaluate_model(name, model, requests, ids, emb, lookup, index,
                                          device, "validation", budget="equal")
            score = valid_result["metrics"]["overall"]["Recall@500"]
            row = {"model": name, "epoch": epoch, "seconds": time.perf_counter() - start,
                   **output, "Recall@50": valid_result["metrics"]["overall"]["Recall@50"],
                   "Recall@100": valid_result["metrics"]["overall"]["Recall@100"],
                   "Recall@500": score, "MRR@100": valid_result["metrics"]["overall"]["MRR@100"]}
            curves.append(row); validation_rows.append(row)
            pd.DataFrame(curves).to_csv(OUT / "training_curves.csv", index=False)
            pd.DataFrame(validation_rows).to_csv(OUT / "validation_results.csv", index=False)
            print(f"{name} epoch={epoch} loss={output['train_loss']:.4f} valid_R500={score:.6f}", flush=True)
            if score > best:
                best, stale = score, 0
                checkpoint = OUT / "checkpoints" / f"phase4_{name}_best.pt"
                size = phase4_save_checkpoint(checkpoint, model, {"name": name, "epoch": epoch,
                    "validation_recall500": score, "seed": SEED, "temperature": 0.05,
                    "training_samples": len(train), "validation_requests": len(requests)})
                model_info[name].update({"best_epoch": epoch, "best_validation_recall500": score,
                                         "checkpoint_bytes": size})
            else:
                stale += 1
            if stale >= config.patience: break
    json_save(OUT / "configs/phase4_models.json", model_info)
    json_save(OUT / "configs/phase4_train_config.json", vars(config) | {"device_used": device,
               "validation_request_count": len(requests), "item_embedding_path": str(EMBEDDING_PATH.relative_to(ROOT)),
               "index_path": str(INDEX_PATH.relative_to(ROOT)), "temperature": 0.05,
               "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
               "gradient_clip": 1.0, "history_n": 10, "frozen_item_embeddings": True})


def test_models(config):
    phase4_seed(); torch.set_num_threads(config.threads)
    ids, emb, lookup, index = resources(config.threads)
    data = QilinData(ROOT); requests = data.load_test_requests()
    device = resolve_device(config.device)
    selected_models = (config.only_model,) if config.only_model else MODELS
    rows, diagnostics = [], {}
    existing_path = OUT / "full_test_results.csv"
    if config.only_model and existing_path.exists():
        existing = pd.read_csv(existing_path)
        rows = existing[existing.model != config.only_model].to_dict("records")
    for name in selected_models:
        checkpoint = torch.load(OUT / "checkpoints" / f"phase4_{name}_best.pt", map_location="cpu",
                                weights_only=False)
        model = phase4_make_model(name, emb.shape[1]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        modes = ("equal", "practical") if name == "multi_interest" else ("single",)
        for mode in modes:
            result = evaluate_model(name, model, requests, ids, emb, lookup, index,
                                    device, "test", data, mode, save=True)
            json_save(OUT / "test_results" / f"phase4_{name}_{mode}.json", result)
            m = result["metrics"]
            rows.append({"model": name, "budget": mode,
                         **{f"Recall@{k}": m["overall"][f"Recall@{k}"] for k in (10,50,100,200,500)},
                         "MRR@100": m["overall"]["MRR@100"],
                         **{f"{segment}_Recall@500": m[segment]["Recall@500"] for segment in (
                             "train_clicked", "train_exposed_never_clicked", "warm_item", "cold_item", "warm_user", "cold_user")},
                         "search_seconds": result["search_seconds"]})
            pd.DataFrame(rows).to_csv(OUT / "full_test_results.csv", index=False)
        if name == "multi_interest":
            diagnostics = collapse_diagnostic(model, requests, ids, emb, lookup, device)
            json_save(OUT / "interest_usage.json", diagnostics)
            pd.DataFrame({"interest": range(4), "winner_count": diagnostics["positive_winner_counts"],
                          "winner_rate": diagnostics["positive_winner_rates"]}).to_csv(
                              OUT / "interest_usage.csv", index=False)


def repair_practical_result(config):
    """Repair legacy practical rankings that predated the post-filter search buffer."""
    ids, emb, lookup, _ = resources(config.threads)
    data = QilinData(ROOT)
    requests = data.load_test_requests()
    checkpoint = torch.load(OUT / "checkpoints/phase4_multi_interest_best.pt",
                            map_location="cpu", weights_only=False)
    device = resolve_device(config.device)
    model = phase4_make_model("multi_interest", emb.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    vectors = phase4_encode_requests(model, requests, emb, lookup, device)
    practical_path = OUT / "per_request/phase4_test_multi_interest_practical.parquet"
    equal_path = OUT / "per_request/phase4_test_multi_interest_equal.parquet"
    practical = pd.read_parquet(practical_path).set_index("request_idx")
    equal = pd.read_parquet(equal_path).set_index("request_idx")
    rankings, repaired = [], 0
    for i, request in enumerate(requests):
        ranking = list(map(int, practical.at[request.request_idx, "retrieved_top500"]))
        if len(ranking) < 500:
            repaired += 1
            candidates = list(dict.fromkeys(
                ranking + list(map(int, equal.at[request.request_idx, "retrieved_top500"]))))
            rows = np.asarray([lookup[note] for note in candidates], dtype=np.int64)
            scores = (np.asarray(emb[rows], dtype=np.float32) @ vectors[i].T).max(axis=1)
            order = np.argsort(-scores, kind="stable")
            ranking = [candidates[int(j)] for j in order[:500]]
        rankings.append(ranking)
    validation = validate_rankings(rankings, requests, set(map(int, ids)))
    if validation["shorter_than_topk"]:
        raise AssertionError(validation)
    metrics, per_request = evaluate_rankings(
        requests, rankings, data.train_exposed_items, data.train_users, "multi_interest",
        clicked_items=set(data.train_click_counts))
    per_request.to_parquet(practical_path, index=False, compression="zstd")
    result_path = OUT / "test_results/phase4_multi_interest_practical.json"
    payload = json.loads(result_path.read_text())
    payload.update({"validation": validation, "metrics": metrics,
                    "post_filter_buffer_repair": {
                        "requests_repaired": repaired,
                        "source": "equal-budget exact candidates",
                        "rescored_with": "max learned-interest cosine"}})
    json_save(result_path, payload)
    full = pd.read_csv(OUT / "full_test_results.csv")
    mask = (full.model == "multi_interest") & (full.budget == "practical")
    values = {f"Recall@{k}": metrics["overall"][f"Recall@{k}"]
              for k in (10, 50, 100, 200, 500)}
    values["MRR@100"] = metrics["overall"]["MRR@100"]
    for segment in ("train_clicked", "train_exposed_never_clicked", "warm_item", "cold_item",
                    "warm_user", "cold_user"):
        values[f"{segment}_Recall@500"] = metrics[segment]["Recall@500"]
    for column, value in values.items():
        full.loc[mask, column] = value
    full.to_csv(OUT / "full_test_results.csv", index=False)
    print(f"repaired={repaired} practical_R500={metrics['overall']['Recall@500']:.8f}", flush=True)


def phase4_load_titles(required: set[int]) -> dict[int, str]:
    found = {}
    for path in sorted((ROOT / "data/notes").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000,
                                                       columns=["note_idx", "note_title"]):
            note_ids, titles = batch.column(0).to_pylist(), batch.column(1).to_pylist()
            for note, title in zip(note_ids, titles):
                note = int(note)
                if note in required:
                    value = "" if title is None or pd.isna(title) else str(title)
                    found[note] = "[missing title]" if value.strip().lower() in {"", "nan", "none", "null"} else value
    return found


def phase4_diagnostics(config):
    data = QilinData(ROOT)
    requests = data.load_test_requests()
    request_by_id = {r.request_idx: r for r in requests}
    ids, emb, lookup, _ = resources(config.threads)
    diversity = pd.read_csv(PHASE3 / "diversity_analysis.csv").set_index("request_idx")
    candidates = {
        "kmeans_k4": PHASE3 / "per_request/full__kmeans_k4_equal_max.parquet",
        "single_attention": OUT / "per_request/phase4_test_single_attention_single.parquet",
        "learned_multi_equal": OUT / "per_request/phase4_test_multi_interest_equal.parquet",
    }
    frames = {name: pd.read_parquet(path).set_index("request_idx") for name, path in candidates.items()}
    bucket_rows = []
    for bucket in ("low", "medium", "high"):
        selected = [r for r in requests if diversity.loc[r.request_idx, "diversity_bucket"] == bucket]
        for name, frame in frames.items():
            rankings = [list(map(int, frame.loc[r.request_idx, "retrieved_top500"])) for r in selected]
            metrics, _ = evaluate_rankings(selected, rankings, data.train_exposed_items, data.train_users,
                                            name, clicked_items=set(data.train_click_counts))
            bucket_rows.append({"method": name, "diversity_bucket": bucket, "requests": len(selected),
                                "Recall@100": metrics["overall"]["Recall@100"],
                                "Recall@500": metrics["overall"]["Recall@500"]})
    pd.DataFrame(bucket_rows).to_csv(OUT / "diversity_analysis.csv", index=False)

    kmeans, learned = frames["kmeans_k4"], frames["learned_multi_equal"]
    eligible = [r for r in requests if pd.isna(kmeans.loc[r.request_idx, "first_hit_rank"])
                and pd.notna(learned.loc[r.request_idx, "first_hit_rank"])]
    rng = np.random.default_rng(SEED)
    chosen = ([eligible[int(i)] for i in np.sort(rng.choice(len(eligible),
               size=min(20, len(eligible)), replace=False))] if eligible else [])
    checkpoint = torch.load(OUT / "checkpoints/phase4_multi_interest_best.pt",
                            map_location="cpu", weights_only=False)
    device = resolve_device(config.device)
    model = phase4_make_model("multi_interest", emb.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    vectors, attention = phase4_encode_requests(model, chosen, emb, lookup, device,
                                                 return_attention=True)
    needed = {int(n) for r in chosen for n in (*r.history[-10:], *r.ground_truth)}
    titles = phase4_load_titles(needed)
    lines = ["# Phase 4: Learned Multi-interest Wins Over KMeans", "",
             f"Seed={SEED}; random sample of {len(chosen)} from eligible full-test wins. "
             "Labels are used only after checkpoint selection for diagnostics.", ""]
    case_rows = []
    for i, (request, user_vec, weights) in enumerate(zip(chosen, vectors, attention), 1):
        hist = [int(n) for n in request.history[-10:] if 0 <= int(n) < len(lookup) and lookup[int(n)] >= 0]
        retrieved = list(map(int, learned.loc[request.request_idx, "retrieved_top500"]))
        rank_lookup = {note: rank for rank, note in enumerate(retrieved, 1)}
        hit = min((n for n in request.ground_truth if n in rank_lookup), key=rank_lookup.get)
        target_vec = np.asarray(emb[int(lookup[hit])], dtype=np.float32)
        winner = int(np.argmax(user_vec @ target_vec))
        top_attended = {j: [titles.get(hist[t], "[missing title]") for t in np.argsort(-weights[j, :len(hist)])[:3]]
                        for j in range(user_vec.shape[0])}
        lines += [f"## Case {i}: request {request.request_idx}", "",
                  f"- History titles: {[titles.get(n, '[missing title]') for n in hist]}",
                  f"- Ground-truth titles: {[titles.get(n, '[missing title]') for n in sorted(request.ground_truth)]}",
                  f"- First hit title: {titles.get(hit, '[missing title]')}",
                  f"- KMeans first hit rank: miss; learned first hit rank: {rank_lookup[hit]}",
                  f"- Winning interest: {winner}",
                  f"- Attention weights (K x history): {np.round(weights[:, :len(hist)], 4).tolist()}",
                  f"- Top-attended titles: {json.dumps(top_attended, ensure_ascii=False)}", ""]
        case_rows.append({"request_idx": request.request_idx, "hit_target": hit,
                          "winner": winner, "learned_rank": rank_lookup[hit],
                          "attention": weights[:, :len(hist)].tolist()})
    (OUT / "cases.md").write_text("\n".join(lines), encoding="utf-8")
    json_save(OUT / "cases.json", case_rows)


def report(config):
    out = OUT
    split = json.loads((out / "configs/phase4_temporal_split.json").read_text())
    model_info = json.loads((out / "configs/phase4_models.json").read_text())
    full = pd.read_csv(out / "full_test_results.csv")
    validation = pd.read_csv(out / "validation_results.csv")
    usage = json.loads((out / "interest_usage.json").read_text())
    diversity = pd.read_csv(out / "diversity_analysis.csv")
    phase3 = pd.read_csv(PHASE3 / "full_results.csv")
    phase1 = pd.read_csv(ROOT / "results/phase_01/experiment_01_full_corpus_baselines/full/summary.csv")
    tfidf = phase1[(phase1.method == "tfidf_history20__with_history_filter") &
                   (phase1.segment == "overall")].iloc[0]
    results = [
        ("TF-IDF", float(tfidf["Recall@500"]), float(phase1[(phase1.method == "tfidf_history20__with_history_filter") & (phase1.segment == "cold_item")]["Recall@500"].iloc[0])),
        *[(name, float(phase3[phase3.method == method]["Recall@500"].iloc[0]),
           float(phase3[phase3.method == method]["Cold Recall@500"].iloc[0]))
          for name, method in (("BGE Mean", "single_mean"), ("Top3 History", "top3_history"),
                               ("KMeans K=4", "kmeans_k4_equal_max"))],
        *[(f"{row['model']} ({row['budget']})", float(row["Recall@500"]),
           float(row["cold_item_Recall@500"]))
          for _, row in full.iterrows()],
    ]
    lines = ["# Phase 4 — Learnable Multi-Interest User Tower", "",
             "## Protocol", "", "- Frozen BGE-base-zh basic-clean item embeddings; exact full-corpus IndexFlatIP.",
             "- History N=10; train-only positives; temporal train/validation split; test never used for model selection.",
             "- In-batch InfoNCE temperature 0.05; duplicate target, same-user, and current-history negatives masked.",
             "- User-tower optimization ran on CUDA; the installed exact Faiss IndexFlatIP is CPU-backed.",
             "- Multi-interest primary evaluation uses about 500 unique raw candidates total; practical 500/interest is also reported.", "",
             "## Temporal Split", "", f"- Train samples: {split['train_samples']:,}; validation samples: {split['valid_samples']:,}.",
             f"- Train users: {split['train_users']:,}; validation users: {split['valid_users']:,}; overlap: {split['overlap_users']:,}.",
             f"- Train timestamps: {split['train_timestamp_min']}–{split['train_timestamp_max']}; valid: {split['valid_timestamp_min']}–{split['valid_timestamp_max']}.",
             f"- Removed {split['history_target_overlap_removed']} history-target overlaps and {split['duplicate_target_removed']} duplicate positives.",
             f"- Caveat: {split['history_temporal_assumption']}. Candidate timestamps differ within {split['positive_timestamp_spread_requests']:,} positive requests; split uses minimum positive time per request.", "",
             "## Model Size and Best Validation", "",
             "| Model | Parameters | Best epoch | Valid R@500 | Checkpoint bytes |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in MODELS:
        d=model_info[name]
        lines.append(f"| {name} | {d['trainable_parameters']:,} | {d['best_epoch']} | {100*d['best_validation_recall500']:.4f}% | {d['checkpoint_bytes']:,} |")
    lines += ["", "## Full Test: Primary Comparison", "", "| Method | R@500 | Cold R@500 |", "| --- | ---: | ---: |"]
    for name, r, cold in results:
        lines.append(f"| {name} | {r*100:.4f}% | {cold*100:.4f}% |")
    lines += ["", "## Learned Tower Full-Test Detail", "",
              "| Model | Budget | R@100 | R@500 | MRR@100 | Train-clicked R@500 | Exposed-never-clicked R@500 | Cold R@500 | Warm-user R@500 | Cold-user R@500 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in full.iterrows():
        lines.append(
            f"| {row['model']} | {row['budget']} | {100*row['Recall@100']:.4f}% | "
            f"{100*row['Recall@500']:.4f}% | {100*row['MRR@100']:.4f}% | "
            f"{100*row['train_clicked_Recall@500']:.4f}% | "
            f"{100*row['train_exposed_never_clicked_Recall@500']:.4f}% | "
            f"{100*row['cold_item_Recall@500']:.4f}% | "
            f"{100*row['warm_user_Recall@500']:.4f}% | {100*row['cold_user_Recall@500']:.4f}% |")
    learned = full[(full.model == "multi_interest") & (full.budget == "equal")].iloc[0]
    mean = full[full.model == "mean_mlp"].iloc[0]
    single = full[full.model == "single_attention"].iloc[0]
    kmeans = float(phase3[phase3.method == "kmeans_k4_equal_max"]["Recall@500"].iloc[0])
    practical = full[(full.model == "multi_interest") & (full.budget == "practical")].iloc[0]
    lines += ["", "## Candidate-budget Check", "",
              f"- Equal-budget multi-interest R@500: {100*learned['Recall@500']:.4f}%.",
              f"- Practical 4 x 500 multi-interest R@500: {100*practical['Recall@500']:.4f}%.",
              "- The larger raw-candidate budget did not improve recall; the gain over KMeans is therefore not explained by the practical budget.",
              "", "## History-diversity Buckets", "",
              "| Method | Bucket | Requests | R@100 | R@500 |", "| --- | --- | ---: | ---: | ---: |"]
    for _, row in diversity.iterrows():
        lines.append(f"| {row['method']} | {row['diversity_bucket']} | {int(row['requests']):,} | "
                     f"{100*row['Recall@100']:.4f}% | {100*row['Recall@500']:.4f}% |")
    lines += ["", "Both learned towers improve every diversity bucket over KMeans. The absolute gain is largest "
              "in low-diversity histories, while the learned multi-interest model's relative gain remains large "
              "in high-diversity histories; however, the collapse diagnostic below prevents attributing that gain "
              "to four genuinely distinct learned interests.",
              "", "## Interest Collapse", "",
              f"- Mean/p50/p90 inter-interest cosine: {usage['mean_inter_interest_cosine']:.4f} / {usage['p50_inter_interest_cosine']:.4f} / {usage['p90_inter_interest_cosine']:.4f}.",
              f"- Positive winner rates: {', '.join(f'{100*x:.1f}%' for x in usage['positive_winner_rates'])}.",
              "- This is severe representational collapse: the K=4 outputs are almost the same vector, and one slot wins 64.6% of positives.",
              "- No diversity regularizer was silently added after observing this; the requested base architecture is reported as trained.", "",
              "## Findings", "",
              f"1. Mean+MLP test R@500={100*mean['Recall@500']:.4f}% versus frozen Mean=2.6622%.",
              f"2. Single Attention test R@500={100*single['Recall@500']:.4f}%.",
              f"3. Multi-interest equal-budget test R@500={100*learned['Recall@500']:.4f}%; KMeans=4.8467%, TF-IDF=7.2394%.",
              f"4. Multi-interest cold R@500={100*learned['cold_item_Recall@500']:.4f}%; KMeans=6.9938%, TF-IDF=9.9427%.",
              f"5. The strongest learned result is Single Attention at {100*single['Recall@500']:.4f}% overall and "
              f"{100*single['cold_item_Recall@500']:.4f}% cold; K=4 is not the winner.",
              "6. Mean pooling was a major bottleneck: even the light Mean+MLP more than doubled frozen-mean R@500.",
              "7. Supervised aggregation helps cold and warm items, but none of the learned towers reaches TF-IDF.",
              "8. The 20 fixed-seed learned-over-KMeans cases are in `cases.md`; they are diagnostic wins, not evidence "
              "of interpretable multi-interest separation because attention heads collapsed.",
              "", "## Decision", "",
              "Choose **D — optimize the user tower** before joint two-tower training. Supervision is valuable, but "
              "Single Attention beats K=4 and K=4 collapses. The next controlled user-tower experiment should address "
              "slot diversity/utilization (for example, the permitted small diversity loss) and retain the frozen item "
              "encoder. Do not unfreeze BGE or add hard negatives until a non-collapsed multi-interest tower beats the "
              "single-attention baseline. TF-IDF remains the strongest standalone retriever."]
    (out / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    config=args()
    OUT.mkdir(parents=True, exist_ok=True)
    phase4_seed(); torch.set_num_threads(config.threads)
    stages=[config.stage] if config.stage != "all" else ["prepare", "smoke", "train", "test", "diagnostics", "report"]
    for stage in stages:
        print(f"=== phase4 {stage} ===", flush=True)
        if stage == "prepare": load_split()
        elif stage == "smoke": smoke(config)
        elif stage == "train": train_models(config)
        elif stage == "test": test_models(config)
        elif stage == "repair-practical": repair_practical_result(config)
        elif stage == "diagnostics": phase4_diagnostics(config)
        elif stage == "report": report(config)


if __name__ == "__main__": main()
