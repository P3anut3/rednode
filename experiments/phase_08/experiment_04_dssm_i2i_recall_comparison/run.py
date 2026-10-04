#!/usr/bin/env python3
"""Validation-only author-style DSSM, DSSM-I2I and frozen BGE-I2I study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import grouped_requests
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (
    BGE_PATH, NOTE_IDS_PATH, PROTOCOL, TRAIN_PATH, VALID_PATH,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import (
    Phase7Collator, Phase7Dataset, load_training_frame,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.evaluation import (
    phase6_status_metrics,
)
from experiments.phase_08.experiment_02_image_hybrid_recall.data import image_paths
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.retrieval import (
    deterministic_proxy_candidates, filter_history, gpu_exact_search, validate_topk,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.run import (
    legacy_evaluator_sets, resources, save_csv_atomic, save_json,
    save_parquet_atomic, save_torch_atomic, select_requests,
)
from experiments.phase_07.experiment_03_author_concat_regularization_ablation.trainer import (
    encode_items, encode_queries, pack_item_features, parameter_report,
    train_epoch,
)
from experiments.phase_08.experiment_04_dssm_i2i_recall_comparison.i2i import (
    exact_knn_shards, load_knn, merge_request, normalized_corpus, sha256,
)
from experiments.phase_08.experiment_04_dssm_i2i_recall_comparison.models import (
    D_VARIANT, make_model,
)

OUT = ROOT / "results/phase_08/experiment_04_dssm_i2i_recall_comparison"
H2_RANKING = ROOT / "results/phase_07/experiment_04_h2_retrieval_dimension_ablation/validation/rankings/h2_256/seed42.parquet"
H2_RESULT = ROOT / "results/phase_07/experiment_04_h2_retrieval_dimension_ablation/validation/h2_256_seed42.json"
DEPTHS = (180, 500, 1200)
KNN_DEPTH = 1200
TOPK = 500
STAGES = ("plan", "audit", "smoke", "train", "validate-d", "knn-c", "knn-di", "evaluate-i2i", "fuse", "report")


def args_parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args()


def layout():
    for name in ("audit", "smoke", "configs", "checkpoints", "training",
                 "validation", "validation/per_request", "embeddings", "knn/c", "knn/di",
                 "metrics", "fusion", "markers"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verified_inputs():
    if not H2_RANKING.exists() or not H2_RESULT.exists():
        raise FileNotFoundError("Phase 7-04 H2 validation result and Top500 are required")
    catalog = np.load(NOTE_IDS_PATH, mmap_mode="r")
    if len(catalog) != PROTOCOL.corpus_items or len(np.unique(catalog)) != len(catalog):
        raise ValueError("canonical note mapping drift")
    return np.asarray(catalog)


def requests_and_store():
    train, valid, store = resources()
    requests = grouped_requests(valid)
    if len(requests) != 13_594:
        raise ValueError("temporal validation request count drift")
    return train, valid, store, requests


def status_sets(store):
    exposed, clicked, users = legacy_evaluator_sets()
    return (set(map(int, store.train_target_item_ids)),
            set(map(int, store.train_item_id_vocab)),
            set(map(int, store.train_user_ids)), exposed, clicked, users)


def evaluate(name, rankings, requests, store, *, strict=True):
    catalog = verified_inputs()
    if strict:
        validate_topk(rankings, catalog, [r.history for r in requests])
    else:
        allowed = set(map(int, catalog))
        for ranking, req in zip(rankings, requests):
            if len(ranking) > TOPK or len(ranking) != len(set(ranking)):
                raise ValueError("invalid I2I ranking")
            if not set(ranking).issubset(allowed) or set(ranking) & set(req.history):
                raise ValueError("I2I candidate out of corpus or history not filtered")
    targets, vocab, users, exposed, clicked, old_users = status_sets(store)
    metrics, temporal, per = phase6_status_metrics(
        requests, rankings, targets, vocab, users, name, exposed, clicked, old_users,
    )
    save_json(OUT / f"validation/{name}.json", {
        "method": name, "metrics": metrics, "temporal": temporal,
        "coverage_top500": float(np.mean([len(row) == TOPK for row in rankings])),
        "empty_history_fraction": float(np.mean([len(r.history) == 0 for r in requests])),
        "test_opened": False,
    })
    save_parquet_atomic(per, OUT / f"validation/per_request/{name}.parquet")
    return metrics, temporal, per


def audit():
    layout()
    catalog = verified_inputs()
    train, valid, store, requests = requests_and_store()
    if not np.array_equal(catalog, np.asarray(store.item_ids)):
        raise ValueError("FeatureStore mapping differs from canonical note IDs")
    if len(train) != 254_583 or len(valid) != 47_733:
        raise ValueError("temporal split drift")
    history = [int(note) for req in requests for note in req.history[-20:]]
    order = Counter(len(req.history[-20:]) for req in requests)
    profile = {
        "train_samples": len(train), "valid_samples": len(valid),
        "validation_requests": len(requests), "catalog_items": len(catalog),
        "history_references": len(history), "unique_history_items": len(set(history)),
        "empty_history_requests": order[0], "history_length_histogram": dict(sorted(order.items())),
        "train_path_sha256": sha256(TRAIN_PATH), "valid_path_sha256": sha256(VALID_PATH),
        "canonical_mapping_sha256": sha256(NOTE_IDS_PATH),
        "bge_embedding_sha256": sha256(BGE_PATH),
        "h2_ranking_sha256": sha256(H2_RANKING),
        "h2_result_sha256": sha256(H2_RESULT),
        "phase7_train_only_feature_audit_sha256": sha256(
            ROOT / "results/phase_07/experiment_01_feature_hybrid_two_tower/audit/feature_audit.json"
        ),
        "feature_schema": store.schema(),
        "forbidden_dense_features_used": False,
        "dynamic_user_fans_follows_used": False,
        "dynamic_user_counts_reason": "user_feat has no per-feature timestamp",
        "test_opened": False,
    }
    save_json(OUT / "audit/data_contract.json", profile)
    save_json(OUT / "markers/audit_complete.json", {"complete": True, "audit_sha256": sha256(OUT / "audit/data_contract.json")})
    print(json.dumps({k: v for k, v in profile.items() if k != "history_length_histogram"}, indent=2))


def require_audit():
    marker = json.loads((OUT / "markers/audit_complete.json").read_text())
    if marker["audit_sha256"] != sha256(OUT / "audit/data_contract.json"):
        raise RuntimeError("audit marker mismatch")


def smoke(args):
    require_audit()
    seed_all(args.seed)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("no CPU fallback for formal smoke")
    _, valid, store, requests = requests_and_store()
    frame = load_training_frame(True)
    dataset = Phase7Dataset(frame, store, True, limit=5000, seed=args.seed)
    model = make_model(store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    stats = train_epoch(model, dataset, optimizer, args.device, 1, 128, max_batches=8,
                        deadline=time.monotonic() + 600)
    sample = select_requests(requests, 100, 42)
    positives = set().union(*(r.ground_truth for r in sample))
    candidates = deterministic_proxy_candidates(np.asarray(store.item_ids), positives, 10_000, 42)
    items = encode_items(model, candidates, store, args.device,
                         packed=pack_item_features(candidates, store), deadline=time.monotonic() + 600)
    queries = encode_queries(model, sample, store, args.device, deadline=time.monotonic() + 600)
    if items.shape != (10_000, 128) or queries.shape != (100, 128):
        raise AssertionError("DSSM dimension mismatch")
    _, raw, _, _, _ = gpu_exact_search(items, queries, 550, args.device, query_batch=32,
                                       deadline=time.monotonic() + 600)
    rankings = filter_history(raw, candidates, [r.history for r in sample], 500)
    validate_topk(rankings, candidates, [r.history for r in sample])
    history = np.asarray(sorted({int(note) for r in sample for note in r.history[-20:]}), dtype=np.int64)
    mapping = {int(note): index for index, note in enumerate(candidates)}
    available = np.asarray([mapping[note] for note in history if int(note) in mapping], dtype=np.int64)
    if len(available):
        _, rows, _, _, _ = gpu_exact_search(items, items[available], 180, args.device,
                                            query_batch=32, deadline=time.monotonic() + 600)
        if (rows < 0).any() or (rows >= len(candidates)).any():
            raise AssertionError("KNN row mapping invalid")
    model.eval()
    batch = Phase7Collator(True)([dataset[index] for index in range(4)])
    batch = {key: value.to(args.device) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    with torch.inference_mode():
        original_query = model.query(batch)
        changed = dict(batch)
        changed["user_numeric"] = torch.randn_like(batch["user_numeric"]) * 1000
        changed_query = model.query(changed)
    numeric_invariance = float((original_query - changed_query).abs().max())
    if numeric_invariance > 1e-7:
        raise AssertionError("unverified fans/follows changed the D query")
    payload = {"passed": bool(np.isfinite(stats["loss"])), "loss": stats["loss"],
               "item_dim": items.shape[1], "history_order": "last_is_most_recent",
               "top500_unique": True, "bge_frozen": True,
               "gpu_peak_bytes": torch.cuda.max_memory_allocated(args.device),
               "unverified_user_numeric_invariance_max_abs": numeric_invariance,
               "test_opened": False}
    save_json(OUT / "smoke/seed42.json", payload)
    if not payload["passed"]:
        raise RuntimeError("D smoke failed")
    print(json.dumps(payload, indent=2))


def require_smoke():
    value = json.loads((OUT / "smoke/seed42.json").read_text())
    if not value.get("passed"):
        raise RuntimeError("successful smoke required")


def train(args):
    require_smoke()
    if args.seed != 42:
        raise RuntimeError("seed43/44 require a separate post-validation selection")
    completed = OUT / "markers/d_training_complete_seed42.json"
    if completed.exists():
        raise RuntimeError("completed training is immutable")
    seed_all(args.seed)
    deadline = time.monotonic() + 3600
    _, _, store, requests = requests_and_store()
    frame = load_training_frame(True)
    dataset = Phase7Dataset(frame, store, True, seed=args.seed)
    subset = select_requests(requests, 5000, 42)
    positives = set().union(*(r.ground_truth for r in subset))
    candidates = deterministic_proxy_candidates(np.asarray(store.item_ids), positives, 100_000, 42)
    packed = pack_item_features(candidates, store)
    model = make_model(store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    curves, best, stale = [], -1.0, 0
    checkpoint = OUT / "checkpoints/d_author_style_seed42_best.pt"
    for epoch in range(1, 7):
        epoch_stats = train_epoch(model, dataset, optimizer, args.device, epoch,
                                  args.batch_size, deadline=deadline)
        items = encode_items(model, candidates, store, args.device, packed=packed, deadline=deadline)
        queries = encode_queries(model, subset, store, args.device, deadline=deadline)
        _, raw, _, backend, timing = gpu_exact_search(items, queries, 600, args.device,
                                                       query_batch=128, deadline=deadline)
        ranking = filter_history(raw, candidates, [r.history for r in subset], 500, deadline)
        from experiments.common.metrics import evaluate_rankings
        proxy, _ = evaluate_rankings(subset, ranking, set(), set(), "d_proxy", deadline=deadline)
        score = float(proxy["overall"]["Recall@500"])
        row = {"epoch": epoch, "proxy_recall500": score, "search_backend": backend,
               **epoch_stats, **timing}
        curves.append(row)
        save_csv_atomic(pd.DataFrame(curves), OUT / "training/curve_seed42.csv")
        print(f"D epoch {epoch}: proxy R@500={score:.6f}, train={epoch_stats['seconds']:.1f}s", flush=True)
        if score > best:
            best, stale = score, 0
            save_torch_atomic({"state_dict": model.state_dict(), "epoch": epoch,
                               "seed": args.seed, "variant": D_VARIANT.to_dict()}, checkpoint)
        else:
            stale += 1
            if stale >= 2:
                break
    config = OUT / "configs/d_author_style_seed42.json"
    save_json(config, {"variant": D_VARIANT.to_dict(), "seed": 42, "history_n": 20,
                       "frozen_bge": True, "loss": "Phase7 TF-IDF hard-negative + in-batch",
                       "user_fans_follows_masked_to_zero": True,
                       "temperature": 0.05, "batch_size": args.batch_size,
                       "parameters": parameter_report(model), "proxy_recall500": best,
                       "test_opened": False})
    save_json(completed, {"complete": True, "checkpoint_sha256": sha256(checkpoint),
                          "config_sha256": sha256(config), "curve_sha256": sha256(OUT / "training/curve_seed42.csv")})


def load_d(store, device):
    done = json.loads((OUT / "markers/d_training_complete_seed42.json").read_text())
    checkpoint = OUT / "checkpoints/d_author_style_seed42_best.pt"
    if not done["complete"] or sha256(checkpoint) != done["checkpoint_sha256"]:
        raise RuntimeError("D checkpoint hash mismatch")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = make_model(store, device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    return model, done


def validate_d(args):
    marker = OUT / "markers/d_validation_complete.json"
    if marker.exists():
        raise RuntimeError("D validation is immutable")
    _, _, store, requests = requests_and_store()
    model, done = load_d(store, args.device)
    catalog = verified_inputs()
    deadline = time.monotonic() + 1800
    started = time.perf_counter()
    vectors = encode_items(model, catalog, store, args.device, deadline=deadline)
    embedding_path = OUT / "embeddings/d_item_vectors.f32.npy"
    temporary = Path(str(embedding_path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    temporary.replace(embedding_path)
    vector_hash = sha256(embedding_path)
    encode_seconds = time.perf_counter() - started
    queries = encode_queries(model, requests, store, args.device, deadline=deadline)
    _, raw, elapsed, backend, timing = gpu_exact_search(vectors, queries, 600, args.device,
                                                         query_batch=128, deadline=deadline)
    ranking = filter_history(raw, catalog, [r.history for r in requests], 500, deadline)
    metrics, temporal, _ = evaluate("d", ranking, requests, store)
    save_json(marker, {"complete": True, "checkpoint_sha256": done["checkpoint_sha256"],
                       "embedding_sha256": vector_hash, "embedding_shape": list(vectors.shape),
                       "mapping_sha256": sha256(NOTE_IDS_PATH),
                       "ranking_sha256": sha256(OUT / "validation/per_request/d.parquet"),
                       "encode_seconds": encode_seconds, "search_seconds": elapsed,
                       "backend": backend, **timing, "gpu_peak_bytes": torch.cuda.max_memory_allocated(args.device),
                       "test_opened": False})
    print(f"D full validation R@500={metrics['overall']['Recall@500']:.6f}", flush=True)


def history_rows(requests, catalog):
    note_to_row = {int(note): row for row, note in enumerate(catalog)}
    history_ids = sorted({int(note) for req in requests for note in req.history[-20:]})
    absent = [note for note in history_ids if note not in note_to_row]
    if absent:
        raise RuntimeError(f"history note outside corpus: {absent[:5]}")
    rows = np.asarray([note_to_row[note] for note in history_ids], dtype=np.int64)
    return rows, note_to_row


def knn(args, route):
    free = shutil.disk_usage(OUT).free
    expected = 114_234 * KNN_DEPTH * (4 + 4)
    if free < expected + 8 * 1024**3:
        raise RuntimeError(f"disk preflight failed: free={free}, new KNN bytes≈{expected}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("formal exact KNN requires CUDA")
    if route == "di":
        done = json.loads((OUT / "markers/d_validation_complete.json").read_text())
        path = OUT / "embeddings/d_item_vectors.f32.npy"
        if sha256(path) != done["embedding_sha256"]:
            raise RuntimeError("D embedding hash mismatch")
        dim, dtype = 128, "npy"
    else:
        path = BGE_PATH
        dim, dtype = 768, "float16_raw"
    _, _, _, requests = requests_and_store()
    catalog = verified_inputs()
    rows, _ = history_rows(requests, catalog)
    corpus = normalized_corpus(path, len(catalog), dim, dtype)
    if len(corpus) != len(catalog):
        raise ValueError("KNN corpus mapping mismatch")
    result = exact_knn_shards(corpus, rows, OUT / f"knn/{route}",
                              source_hash=sha256(path), depth=KNN_DEPTH,
                              device=args.device, deadline=time.monotonic() + 7200)
    save_json(OUT / f"markers/knn_{route}_complete.json", {
        "complete": True, **result, "mapping_sha256": sha256(NOTE_IDS_PATH),
        "test_opened": False,
    })


def evaluate_i2i():
    _, _, store, requests = requests_and_store()
    catalog = verified_inputs()
    rows, note_to_row = history_rows(requests, catalog)
    position = {int(row): index for index, row in enumerate(rows)}
    summary = {}
    for route in ("di", "c"):
        marker = json.loads((OUT / f"markers/knn_{route}_complete.json").read_text())
        if marker["mapping_sha256"] != sha256(NOTE_IDS_PATH):
            raise RuntimeError("KNN mapping mismatch")
        vectors_path = OUT / "embeddings/d_item_vectors.f32.npy" if route == "di" else BGE_PATH
        if marker["source_hash"] != sha256(vectors_path):
            raise RuntimeError("KNN vector source mismatch")
        knn_rows, knn_scores = load_knn(OUT / f"knn/{route}", rows, KNN_DEPTH, marker["source_hash"])
        route_summary = {}
        rankings_by_depth = {}
        eligible_history = np.asarray([len(req.history) > 0 for req in requests], dtype=bool)
        for depth in DEPTHS:
            started = time.perf_counter()
            rankings = [merge_request(req.history, note_to_row, catalog, position,
                                      knn_rows, knn_scores, depth) for req in requests]
            rankings_by_depth[depth] = rankings
            coverage = float(np.mean([len(r) == TOPK for r in rankings]))
            nonempty_coverage = float(np.mean([len(r) == TOPK for r, eligible in zip(rankings, eligible_history) if eligible]))
            route_summary[str(depth)] = {"coverage_top500": coverage,
                                         "coverage_nonempty_history": nonempty_coverage,
                                         "underfilled_requests": int(sum(len(r) < TOPK for r in rankings)),
                                         "aggregation_seconds": time.perf_counter() - started}
            if depth == 180:
                evaluate(f"{route}_author180", rankings, requests, store, strict=False)
            if nonempty_coverage >= 1.0 - 1e-12 and "selected_depth" not in route_summary:
                route_summary["selected_depth"] = depth
        selected = route_summary.get("selected_depth", KNN_DEPTH)
        core = rankings_by_depth[180]
        extended = rankings_by_depth[selected]
        rankings = []
        for first, extra in zip(core, extended):
            if len(first) == TOPK:
                rankings.append(first)
            else:
                seen = set(first)
                rankings.append(first + [note for note in extra if note not in seen][:TOPK - len(first)])
        # Empty-history requests cannot have an I2I ranking.  For fixed-budget
        # comparisons they receive deterministic H2 backfill, but the native
        # I2I coverage and metrics above/below remain unpadded.
        evaluate(route, rankings, requests, store, strict=False)
        route_summary["selected_depth"] = selected
        route_summary["formal_merge"] = "author180_order_then_overfetch_only_if_underfilled"
        summary[route] = route_summary
    save_json(OUT / "metrics/i2i_depth_coverage.json", summary)
    save_json(OUT / "markers/i2i_evaluation_complete.json", {"complete": True,
              "coverage_sha256": sha256(OUT / "metrics/i2i_depth_coverage.json"),
              "test_opened": False})


def read_rankings(name, requests):
    path = H2_RANKING if name == "h2" else OUT / f"validation/per_request/{name}.parquet"
    frame = pd.read_parquet(path, columns=["request_idx", "retrieved_top500", "ground_truth"])
    by_id = {int(row.request_idx): row for row in frame.itertuples(index=False)}
    if set(by_id) != {r.request_idx for r in requests}:
        raise RuntimeError(f"request alignment failed for {name}")
    for request in requests:
        if set(map(int, by_id[request.request_idx].ground_truth)) != set(request.ground_truth):
            raise RuntimeError(f"ground truth drift in {name}")
    return [list(map(int, by_id[r.request_idx].retrieved_top500)) for r in requests]


def combine(primary, secondary, quota_secondary, history):
    blocked = set(map(int, history))
    taken, output = set(), []
    for source, limit in ((primary, TOPK - quota_secondary), (secondary, quota_secondary),
                          (primary, TOPK), (secondary, TOPK)):
        picked = 0
        for note in source:
            if picked >= limit or len(output) >= TOPK:
                break
            note = int(note)
            if note not in blocked and note not in taken:
                output.append(note)
                taken.add(note)
                picked += 1
    return output


def fast_paired_bootstrap(requests, baseline_sets, candidate_rankings,
                          train_targets, train_vocab, *, method, replicates=1000):
    """Build each request's hit sets once, then bootstrap aligned recall deltas."""
    categories = ("overall", "train_target_seen", "train_history_only", "completely_unseen")
    history_only = train_vocab - train_targets
    values = {name: [] for name in categories}
    for request, base, ranking in zip(requests, baseline_sets, candidate_rankings):
        selected = set(ranking)
        truth = set(request.ground_truth)
        subsets = (truth, truth & train_targets, truth & history_only, truth - train_vocab)
        for name, positives in zip(categories, subsets):
            if positives:
                values[name].append(
                    (len(positives & selected) - len(positives & base)) / len(positives)
                )
    output = []
    for name in categories:
        array = np.asarray(values[name], dtype=np.float32)
        if not len(array):
            output.append({"method": method, "segment": name, "eligible_requests": 0,
                           "point_delta": float("nan"), "ci95_lower": float("nan"),
                           "ci95_upper": float("nan"), "replicates": 0})
            continue
        rng = np.random.default_rng(42)
        means = np.empty(replicates, dtype=np.float32)
        for start in range(0, replicates, 100):
            count = min(100, replicates - start)
            indices = rng.integers(0, len(array), size=(count, len(array)))
            means[start:start + count] = array[indices].mean(axis=1)
        output.append({"method": method, "segment": name,
                       "eligible_requests": len(array),
                       "point_delta": float(array.mean()),
                       "ci95_lower": float(np.quantile(means, 0.025)),
                       "ci95_upper": float(np.quantile(means, 0.975)),
                       "replicates": replicates})
    return output


def fusion():
    if not (OUT / "markers/i2i_evaluation_complete.json").exists():
        raise RuntimeError("I2I results required")
    _, _, store, requests = requests_and_store()
    routes = {name: read_rankings(name, requests) for name in ("h2", "d", "di", "c")}
    reference = json.loads(H2_RESULT.read_text())["metrics"]["overall"]["Recall@500"]
    definitions = (("d", "di"), ("d", "c"), ("h2", "c"))
    diagnostics, table, bootstrap_rows = {}, [], []
    for first, second in definitions:
        base, other = routes[first], routes[second]
        base_sets = [set(ranking) for ranking in base]
        a_hits = b_hits = both_hits = only_a = only_b = neither = 0
        overlap, oracle = [], []
        request_hit = Counter()
        unique_hits = {"primary": set(), "secondary": set(), "primary_only": set(), "secondary_only": set()}
        for req, aa, bb in zip(requests, base, other):
            truth = set(req.ground_truth)
            left, right = truth & set(aa), truth & set(bb)
            a_hits += len(left); b_hits += len(right); both_hits += len(left & right)
            only_a += len(left - right); only_b += len(right - left)
            neither += len(truth - (left | right))
            request_hit[(bool(left), bool(right))] += 1
            unique_hits["primary"].update(left)
            unique_hits["secondary"].update(right)
            unique_hits["primary_only"].update(left - right)
            unique_hits["secondary_only"].update(right - left)
            overlap.append(len(set(aa) & set(bb)) / max(1, len(set(aa) | set(bb))))
            oracle.append(len(left | right) / len(truth) if truth else 0.0)
        diagnostics[f"{first}+{second}"] = {
            "candidate_jaccard_mean": float(np.mean(overlap)),
            "primary_positive_hits": a_hits, "secondary_positive_hits": b_hits,
            "both_positive_hits": both_hits, "primary_only_positive_hits": only_a,
            "secondary_only_positive_hits": only_b, "neither_positive_hits": neither,
            "request_hit_contingency": {f"{int(a)}_{int(b)}": count for (a, b), count in request_hit.items()},
            "unique_positive_items_hit": {key: len(value) for key, value in unique_hits.items()},
            "union_oracle_request_macro_recall500": float(np.mean(oracle)),
            "union_oracle_is_not_fixed_budget": True,
        }
        for quota in (50, 100, 150):
            fused = [combine(aa, bb, quota, r.history)
                     for r, aa, bb in zip(requests, base, other)]
            if any(len(r) < TOPK for r in fused):
                raise RuntimeError("fixed-budget fusion underfilled; route has no valid fallback")
            name = f"{first}_{second}_q{quota}"
            metrics, temporal, per = evaluate(name, fused, requests, store)
            gained = lost = 0
            for req, aa, ff in zip(requests, base, fused):
                truth = set(req.ground_truth)
                gained += len((truth & set(ff)) - set(aa))
                lost += len((truth & set(aa)) - set(ff))
            baseline = json.loads((OUT / f"validation/{first}.json").read_text())["metrics"]["overall"]["Recall@500"] if first != "h2" else reference
            table.append({"method": name, "primary": first, "secondary": second,
                          "secondary_quota": quota, "recall500": metrics["overall"]["Recall@500"],
                          "baseline_recall500": baseline,
                          "delta_recall500": metrics["overall"]["Recall@500"] - baseline,
                          "added_positive_interactions": gained,
                          "displaced_positive_interactions": lost,
                          "net_positive_interactions": gained - lost,
                          "completely_unseen_recall500": temporal["completely_unseen"]["Recall@500"]})
            bootstrap_rows.extend(fast_paired_bootstrap(
                requests, base_sets, fused,
                set(map(int, store.train_target_item_ids)),
                set(map(int, store.train_item_id_vocab)),
                method=name, replicates=1000,
            ))
    for first, second in combinations(routes, 2):
        key = f"{first}+{second}"
        if key in diagnostics:
            continue
        aa, bb = routes[first], routes[second]
        jaccard, a_only, b_only, both, neither, oracle = [], 0, 0, 0, 0, []
        request_hits = Counter()
        for req, left_rank, right_rank in zip(requests, aa, bb):
            left, right = set(left_rank), set(right_rank)
            truth = set(req.ground_truth)
            ah, bh = truth & left, truth & right
            a_only += len(ah - bh); b_only += len(bh - ah)
            both += len(ah & bh); neither += len(truth - (ah | bh))
            request_hits[(bool(ah), bool(bh))] += 1
            jaccard.append(len(left & right) / max(1, len(left | right)))
            oracle.append(len(ah | bh) / len(truth) if truth else 0)
        diagnostics[key] = {"candidate_jaccard_mean": float(np.mean(jaccard)),
                            "primary_only_positive_hits": a_only,
                            "secondary_only_positive_hits": b_only,
                            "both_positive_hits": both, "neither_positive_hits": neither,
                            "request_hit_contingency": {f"{int(a)}_{int(b)}": count for (a, b), count in request_hits.items()},
                            "union_oracle_request_macro_recall500": float(np.mean(oracle)),
                            "union_oracle_is_not_fixed_budget": True}
    save_json(OUT / "metrics/route_overlap.json", diagnostics)
    save_csv_atomic(pd.DataFrame(table), OUT / "fusion/fixed_budget.csv")
    save_csv_atomic(pd.DataFrame(bootstrap_rows), OUT / "fusion/paired_bootstrap.csv")
    for first, second in definitions:
        options = [row for row in table if row["primary"] == first and row["secondary"] == second]
        best = max(options, key=lambda row: row["recall500"])
        routes[best["method"]] = read_rankings(best["method"], requests)
    segment_rows = segment_analysis(requests, routes, store)
    save_csv_atomic(pd.DataFrame(segment_rows), OUT / "metrics/route_segments.csv")
    save_json(OUT / "markers/fusion_complete.json", {"complete": True,
        "fusion_sha256": sha256(OUT / "fusion/fixed_budget.csv"),
        "test_opened": False})


def segment_analysis(requests, routes, store):
    """Same-request route slices, including image availability and BGE diversity."""
    catalog = verified_inputs()
    mask = np.load(image_paths("i1_first")[1], mmap_mode="r")
    if mask.shape != catalog.shape or mask.dtype != np.bool_:
        raise RuntimeError("Phase8 image availability mapping drift")
    bge = np.memmap(BGE_PATH, mode="r", dtype=np.float16,
                    shape=(len(catalog), 768))
    note_to_row = {int(note): index for index, note in enumerate(catalog)}
    diversity = np.full(len(requests), np.nan, dtype=np.float32)
    for index, request in enumerate(requests):
        rows = [note_to_row[int(note)] for note in request.history[-20:]
                if int(note) in note_to_row]
        if len(rows) < 2:
            continue
        vectors = np.asarray(bge[rows], dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarity = vectors @ vectors.T
        triangle = similarity[np.triu_indices(len(rows), k=1)]
        diversity[index] = 1.0 - float(np.mean(triangle))
    valid_diversity = diversity[np.isfinite(diversity)]
    cut1, cut2 = np.quantile(valid_diversity, [1 / 3, 2 / 3])
    target_seen = set(map(int, store.train_target_item_ids))
    train_vocab = set(map(int, store.train_item_id_vocab))
    history_only = train_vocab - target_seen
    rows = []
    for route, rankings in routes.items():
        for label in ("overall", "has_image", "no_image", "train_target_seen",
                      "train_history_only", "completely_unseen", "history_0_5",
                      "history_6_10", "history_11_20", "diversity_low",
                      "diversity_mid", "diversity_high"):
            values100, values500 = [], []
            for index, (req, ranking) in enumerate(zip(requests, rankings)):
                truth = set(req.ground_truth)
                if label == "has_image":
                    truth = {note for note in truth if mask[note_to_row[note]]}
                elif label == "no_image":
                    truth = {note for note in truth if not mask[note_to_row[note]]}
                elif label == "train_target_seen":
                    truth &= target_seen
                elif label == "train_history_only":
                    truth &= history_only
                elif label == "completely_unseen":
                    truth -= train_vocab
                elif label.startswith("history_"):
                    length = len(req.history[-20:])
                    lower, upper = {"history_0_5": (0, 5), "history_6_10": (6, 10),
                                    "history_11_20": (11, 20)}[label]
                    if not lower <= length <= upper:
                        continue
                elif label.startswith("diversity_"):
                    value = diversity[index]
                    if not np.isfinite(value):
                        continue
                    bucket = "diversity_low" if value <= cut1 else (
                        "diversity_mid" if value <= cut2 else "diversity_high")
                    if bucket != label:
                        continue
                if not truth:
                    continue
                values100.append(len(truth & set(ranking[:100])) / len(truth))
                values500.append(len(truth & set(ranking[:500])) / len(truth))
            rows.append({"route": route, "segment": label, "eligible_requests": len(values500),
                         "Recall@100": float(np.mean(values100)) if values100 else float("nan"),
                         "Recall@500": float(np.mean(values500)) if values500 else float("nan"),
                         "diversity_tertiles": [float(cut1), float(cut2)] if label.startswith("diversity_") else None})
    return rows


def report():
    if not (OUT / "markers/fusion_complete.json").exists():
        raise RuntimeError("fusion must complete before report")
    h2_seed_scores = [json.loads((H2_RESULT.parent / f"h2_256_seed{seed}.json").read_text())
                      ["metrics"]["overall"]["Recall@500"] for seed in (42, 43, 44)]
    def markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        rows = ["| " + " | ".join(map(str, columns)) + " |",
                "|" + "|".join("---" for _ in columns) + "|"]
        for values in frame.itertuples(index=False, name=None):
            cells = []
            for value in values:
                if isinstance(value, float):
                    cell = "" if np.isnan(value) else f"{value:.6g}"
                else:
                    cell = str(value)
                cells.append(cell.replace("|", "\\|"))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    lines = ["# Phase 8-04：作者式 DSSM 与逐历史 I2I 对照", "",
             "仅 temporal validation；未读取 recommendation test。D 是作者式、同协议受控实现，非原模型完整复现。", "",
             "特征边界：D 复用 Phase 7 train-only FeatureStore，但将无时间戳的 fans/follows 输入强制置零；不使用 `dense_feat*`、图片或物品累计行为字段。H2 基线沿用既有正式协议，H2+C 不新增画像字段。被替代的未屏蔽 D 预试运行只保留在 `debug/uncertain_profile_v1/`，不计入本表。", "",
             f"H2-256 validation 三 seed R@500 均值：{np.mean(h2_seed_scores):.4%}；表中配对主对照为同 seed42 的 {h2_seed_scores[0]:.4%}。", "",
             "## 单路", "",
             "| 路线 | R@100 | R@500 | MRR@100 | Top500 覆盖率 | Seen R@500 | History-only R@500 | Unseen R@500 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    h2 = json.loads(H2_RESULT.read_text())
    values = {"h2": h2}
    for name in ("d", "di", "c"):
        values[name] = json.loads((OUT / f"validation/{name}.json").read_text())
    for name, value in values.items():
        m = value["metrics"]
        temporal = value.get("temporal", value.get("phase6_item_status", {}))
        cols = [name, f"{m['overall']['Recall@100']:.4%}", f"{m['overall']['Recall@500']:.4%}",
                f"{m['overall']['MRR@100']:.4%}",
                f"{value.get('coverage_top500', 1.0):.2%}"]
        for key in ("train_target_seen", "train_history_only", "completely_unseen"):
            v = temporal.get(key, {}).get("Recall@500", float("nan"))
            cols.append(f"{v:.4%}")
        lines.append("| " + " | ".join(cols) + " |")
    lines += ["", "## 固定 Top500 配额", ""]
    fusion_table = pd.read_csv(OUT / "fusion/fixed_budget.csv")
    paired_table = pd.read_csv(OUT / "fusion/paired_bootstrap.csv")
    lines += [markdown(fusion_table), "",
              "### 同 request 配对 bootstrap", "",
              markdown(paired_table), "",
              "### 各路独有命中与候选重叠", ""]
    overlap = json.loads((OUT / "metrics/route_overlap.json").read_text())
    lines += [markdown(pd.DataFrame([{"pair": pair, **{key: value.get(key) for key in (
                "candidate_jaccard_mean", "primary_only_positive_hits",
                "secondary_only_positive_hits", "both_positive_hits",
                "union_oracle_request_macro_recall500")}}
                for pair, value in overlap.items()])), "",
              "### Warm / Cold、有图 / 无图、历史长度与多兴趣分桶", "",
              markdown(pd.read_csv(OUT / "metrics/route_segments.csv")), "",
              "## I2I 深度与覆盖", "",
              markdown(pd.DataFrame([{"route": route, "depth": depth, **numbers}
                            for route, values in json.loads((OUT / "metrics/i2i_depth_coverage.json").read_text()).items()
                            for depth, numbers in values.items() if isinstance(numbers, dict)])), "",
              "## 资源与可复现性", ""]
    d_marker = json.loads((OUT / "markers/d_validation_complete.json").read_text())
    d_config = json.loads((OUT / "configs/d_author_style_seed42.json").read_text())
    d_curve = pd.read_csv(OUT / "training/curve_seed42.csv")
    lines.append(f"- D 参数 {d_config['parameters']['total_parameters']:,}，其中 ID 参数 {d_config['parameters']['id_parameters']:,}；best epoch {int(d_curve.loc[d_curve.proxy_recall500.idxmax(), 'epoch'])}；训练累计 {d_curve.seconds.sum():.1f}s；checkpoint {(OUT / 'checkpoints/d_author_style_seed42_best.pt').stat().st_size/1024**2:.1f} MiB。")
    lines.append(f"- D 全库编码 {d_marker['encode_seconds']:.1f}s；精确 query 检索 {d_marker['search_seconds']:.1f}s；GPU 峰值 {d_marker['gpu_peak_bytes']/1024**3:.2f} GiB。")
    for route in ("c", "di"):
        value = json.loads((OUT / f"markers/knn_{route}_complete.json").read_text())
        cache_size = sum(path.stat().st_size for path in (OUT / f"knn/{route}").glob("*.npy"))
        lines.append(f"- {route.upper()} 去重历史 exact GPU KNN {value['unique_history_items']:,} 条，GPU索引构建+扫描+分块写入 {value['seconds']:.1f}s（不含源向量 CPU 归一化/预读），缓存 {cache_size/1024**3:.2f} GiB，GPU 峰值 {value['gpu_peak_bytes']/1024**3:.2f} GiB。")
    lines += [f"- D 128d float32 全库向量 / exact index 主体约 {1_983_938*128*4/1024**3:.2f} GiB；BGE 768d float32 GPU-resident exact index 主体约 {1_983_938*768*4/1024**3:.2f} GiB。",
              f"- DI ranking SHA-256: `{sha256(OUT / 'validation/per_request/di.parquet')}`",
              f"- C ranking SHA-256: `{sha256(OUT / 'validation/per_request/c.parquet')}`"]
    h2c = fusion_table[fusion_table.primary.eq("h2") & fusion_table.secondary.eq("c")]
    best = h2c.sort_values("recall500", ascending=False).iloc[0]
    ci = paired_table[(paired_table.method == best.method) & (paired_table.segment == "overall")].iloc[0]
    verdict = "validation GO 候选" if best.delta_recall500 > 0 and ci.ci95_lower > 0 else "validation No-Go"
    overlap_h2c = overlap["h2+c"]
    segments = pd.read_csv(OUT / "metrics/route_segments.csv")
    def segment_value(route, segment):
        return float(segments[(segments.route == route) & (segments.segment == segment)]["Recall@500"].iloc[0])
    lines += ["", "## 决策", "",
              f"1. D 单路 {values['d']['metrics']['overall']['Recall@500']:.4%}，未接近 H2 seed42 {h2_seed_scores[0]:.4%}。DI 使用同一 D item index，逐历史策略达到 {values['di']['metrics']['overall']['Recall@500']:.4%}，说明召回策略确有增量，但 D+DI 最佳固定预算仍仅 {fusion_table[(fusion_table.primary == 'd') & (fusion_table.secondary == 'di')].recall500.max():.4%}。",
              f"2. 冻结 BGE-I2I 与 H2 的候选 Jaccard 仅 {overlap_h2c['candidate_jaccard_mean']:.4%}，H2 未命中而 C 命中的正样本交互 {overlap_h2c['secondary_only_positive_hits']} 个；union oracle R@500 为 {overlap_h2c['union_oracle_request_macro_recall500']:.4%}，不能作为实际 Top500。",
              f"3. H2+C 最佳配额 `{best.method}`：相对同 seed42 H2 ΔR@500={best.delta_recall500:+.4%}，配对 95% CI [{ci.ci95_lower:+.4%}, {ci.ci95_upper:+.4%}]；{verdict}。",
              f"4. 该配额新增 {int(best.added_positive_interactions)} 个、挤掉 {int(best.displaced_positive_interactions)} 个 H2 正样本，净 {int(best.net_positive_interactions):+d}。高历史多样性分桶从 {segment_value('h2','diversity_high'):.4%} 到 {segment_value(best.method,'diversity_high'):.4%}，低多样性从 {segment_value('h2','diversity_low'):.4%} 到 {segment_value(best.method,'diversity_low'):.4%}。短历史 0–5 条仅 58 个请求，不足以得出稳定分层结论。",
              "5. DI/C 还需维护额外 KNN 缓存及 item index；在固定预算无稳健净增益时，不建议仅凭 union oracle 上线新路。", "",
              "这只是 validation 选型证据。即使 GO，也须先锁定配置并经用户确认，才能做一次 terminal test；本实验没有打开 test。", ""]
    lines += ["",
              "Union oracle 只衡量互补性，不是固定预算指标。不同向量空间的相似度没有直接相加。", "",
              "## 资产与边界", "",
              f"- Canonical mapping SHA-256: `{sha256(NOTE_IDS_PATH)}`",
              f"- D checkpoint SHA-256: `{sha256(OUT / 'checkpoints/d_author_style_seed42_best.pt')}`",
              f"- D vectors SHA-256: `{sha256(OUT / 'embeddings/d_item_vectors.f32.npy')}`",
              f"- BGE vectors SHA-256: `{sha256(BGE_PATH)}`",
              "- Test opened: `false`; H2 terminal test 9.0231% 不是本表 validation 口径。", ""]
    path = OUT / "summary.md"
    temporary = Path(str(path) + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    print(path)


def main():
    args = args_parse()
    if args.stage == "plan":
        print(json.dumps({"experiment": "phase_08/experiment_04_dssm_i2i_recall_comparison",
                          "stages": STAGES, "candidate_count": 1_983_938,
                          "test_opened": False}, indent=2))
        return
    if not args.confirm_run:
        raise SystemExit(f"{args.stage} requires --confirm-run")
    if args.seed != 42:
        raise SystemExit("seed43/44 are not unlocked by this validation-only first run")
    layout()
    if args.stage == "audit": audit()
    elif args.stage == "smoke": smoke(args)
    elif args.stage == "train": train(args)
    elif args.stage == "validate-d": validate_d(args)
    elif args.stage == "knn-c": knn(args, "c")
    elif args.stage == "knn-di": knn(args, "di")
    elif args.stage == "evaluate-i2i": evaluate_i2i()
    elif args.stage == "fuse": fusion()
    elif args.stage == "report": report()


if __name__ == "__main__":
    main()
