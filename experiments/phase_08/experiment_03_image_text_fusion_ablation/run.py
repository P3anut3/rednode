#!/usr/bin/env python3
"""Phase 8-03: matched-start image/text fusion ablation (test locked)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData  # noqa: E402
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import grouped_requests  # noqa: E402
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import PROTOCOL  # noqa: E402
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import Phase7Dataset, load_training_frame  # noqa: E402
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.evaluation import phase6_status_metrics  # noqa: E402
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.retrieval import (  # noqa: E402
    deterministic_proxy_candidates, filter_history, gpu_exact_search, validate_topk,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.run import (  # noqa: E402
    legacy_evaluator_sets, resources, save_csv_atomic, save_json,
    save_parquet_atomic, save_torch_atomic, select_requests,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.trainer import to_device  # noqa: E402
from experiments.phase_08.experiment_02_image_hybrid_recall import run as a0run  # noqa: E402
from experiments.phase_08.experiment_02_image_hybrid_recall.data import (  # noqa: E402
    IMAGE_OUT, ImageCollator, image_paths, verify_image_assets,
)
from experiments.phase_08.experiment_03_image_text_fusion_ablation.models import (  # noqa: E402
    A0Intervention, VARIANTS,
)
from experiments.phase_08.experiment_03_image_text_fusion_ablation.trainer import (  # noqa: E402
    IMAGE_STRATEGY, encode_items, encode_queries, make_model, pack_item_features,
    parameter_report, train_epoch,
)


OUT = ROOT / "results/phase_08/experiment_03_image_text_fusion_ablation"
SEEDS = (42, 43, 44)
STAGES = ("plan", "audit", "smoke", "train", "full-validation", "bootstrap-seed42",
          "select", "lock", "diagnose", "freeze-vector", "terminal-test", "report")
DIM = 256
MIN_GAIN = 0.001
MAX_DROP = 0.001
RESERVE_GIB = 16.0


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument("--model", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--min-host-available-gb", type=float, default=48)
    parser.add_argument("--max-process-tree-rss-gb", type=float, default=16)
    parser.add_argument("--max-gpu-allocated-gb", type=float, default=20)
    parser.add_argument("--min-disk-free-gb", type=float, default=16)
    parser.add_argument("--recover-stale-validation-lock", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args()


def marker(name):
    return OUT / "configs/markers" / name


def layout():
    for folder in ("audit", "smoke", "configs/markers", "configs", "checkpoints",
                   "training_curves", "validation/rankings", "validation", "diagnostics",
                   "embeddings", "metrics"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)


def sha(path):
    return a0run.sha(path)


def array_sha(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    view = memoryview(contiguous).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def resource_preflight(args, extra_gib=0.0):
    a0run.require_cuda(args.device)
    limits = a0run.resource_limits(args)
    # Five best/last checkpoint pairs (~2.1 GiB), rankings/markers (<0.5),
    # one locked vector (~1.9), one temporary publish copy (~1.9), slack (2).
    # Keep the complete 16 GiB reserve on top of remaining planned writes.
    planned_gib = 8.5 + extra_gib
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < RESERVE_GIB + planned_gib:
        raise RuntimeError(f"full-plan disk budget failed: free={free:.2f} GiB, required={RESERVE_GIB + planned_gib:.2f} GiB")
    if a0run.psutil.virtual_memory().available < limits["min_host_available_bytes"]:
        raise RuntimeError("host memory preflight failed")
    if torch.cuda.get_device_properties(args.device).total_memory < limits["max_gpu_allocated_bytes"]:
        raise RuntimeError("GPU memory preflight failed")
    return limits


def authorize(args):
    if args.stage != "plan" and not args.confirm_run:
        raise SystemExit("writing/evaluation stages require --confirm-run")
    if args.seed not in SEEDS or args.num_workers < 0 or args.prefetch_factor < 1:
        raise SystemExit("invalid seed or loader controls")
    if args.stage in {"smoke", "train", "full-validation"} and args.model is None:
        raise SystemExit("this stage needs --model")
    if args.stage == "train" and args.batch_size != 512:
        raise SystemExit("matched H2/A0 training requires batch-size=512")
    if args.recover_stale_validation_lock and args.stage != "full-validation":
        raise SystemExit("lock recovery is valid only for full-validation")
    if args.stage in {"train", "full-validation"} and args.seed != 42:
        path = marker("seed42_selection.json")
        if not path.exists() or json.loads(path.read_text()).get("best_model") != args.model:
            raise SystemExit("only selected structure may run seeds 43/44")


def plan():
    free = shutil.disk_usage(ROOT).free / 2**30
    print(json.dumps({"models": VARIANTS, "fixed_image": "mean_all", "train": 254583,
                      "validation": 47733, "items": PROTOCOL.corpus_items,
                      "disk_free_gib_now": round(free, 2),
                      "required_free_gib_before_run": RESERVE_GIB + 8.5,
                      "test_locked": True, "writes": False}, indent=2))


def a0_reference(seed):
    result, ranking = a0run.validation("i3_all", seed)
    path = a0run.OUT / f"validation/i3_all_seed{seed}.json"
    rank_path = a0run.OUT / f"validation/rankings/i3_all_seed{seed}.parquet"
    if result["seed"] != seed or result["candidate_universe"] != PROTOCOL.corpus_items:
        raise RuntimeError("A0 matched reference protocol mismatch")
    del ranking
    return {"result": str(path.relative_to(ROOT)), "result_sha256": sha(path),
            "ranking": str(rank_path.relative_to(ROOT)), "ranking_sha256": sha(rank_path),
            "metrics": result["metrics"]}


def audit(_):
    layout()
    if marker("audit_complete.json").exists():
        raise SystemExit("completed audit is immutable")
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < RESERVE_GIB + 8.5:
        raise RuntimeError(f"full-plan disk budget failed before audit: {free_gib:.2f} GiB free")
    prior = a0run.require_audit(IMAGE_STRATEGY)
    asset = verify_image_assets(IMAGE_STRATEGY, full_hash=True)
    _, valid, store = resources()
    image_ids = np.load(IMAGE_OUT / "mappings/note_ids.npy", mmap_mode="r")
    if not np.array_equal(store.item_ids, image_ids):
        raise RuntimeError("image and H2 canonical item mapping differ")
    coverage = prior["valid_coverage"]
    value = {"complete": True, "train_rows": prior["train_rows"],
             "valid_rows": len(valid), "valid_coverage": coverage,
             "image_asset": asset,
             "c0": {str(seed): a0run.control(seed) for seed in SEEDS},
             "a0": {str(seed): a0_reference(seed) for seed in SEEDS},
             "disk_free_gib_at_audit": free_gib,
             "disk_budget_gib": {"checkpoint_pairs": 2.1, "rankings_and_metadata": 0.5,
                                 "locked_vector": 1.9, "temporary_publish": 1.9,
                                 "other_slack": 2.1, "untouched_reserve": RESERVE_GIB},
             "test_opened": False}
    save_json(OUT / "audit/data_contract.json", value)
    save_json(marker("audit_complete.json"), {"complete": True,
              "data_sha256": sha(OUT / "audit/data_contract.json"), "test_opened": False})
    print(json.dumps({"train_rows": value["train_rows"], "valid_rows": value["valid_rows"],
                      "image_asset": asset["strategy"], "test_opened": False}, ensure_ascii=False))


def require_audit():
    done = json.loads(marker("audit_complete.json").read_text())
    path = OUT / "audit/data_contract.json"
    if not done.get("complete") or sha(path) != done["data_sha256"]:
        raise RuntimeError("audit contract/hash mismatch")
    value = json.loads(path.read_text())
    if verify_image_assets(IMAGE_STRATEGY, full_hash=True) != value["image_asset"]:
        raise RuntimeError("mean_all image asset changed")
    for seed in SEEDS:
        for label, current in (("c0", a0run.control(seed)), ("a0", a0_reference(seed))):
            old = value[label][str(seed)]
            if old["result_sha256"] != current["result_sha256"] or old["ranking_sha256"] != current["ranking_sha256"]:
                raise RuntimeError(f"{label} reference changed after audit")
    return value


def model_batch(dataset, indices):
    return ImageCollator(IMAGE_STRATEGY)([dataset[int(index)] for index in indices])


def smoke(args):
    layout(); limits = resource_preflight(args); require_audit(); a0run.seed_all(42)
    deadline = time.monotonic() + 600
    _, valid, store = resources()
    dataset = Phase7Dataset(load_training_frame(True), store, True, limit=10_000, seed=42)
    model = make_model(store, args.device, args.model)
    # The actual copied A0 tensors must reproduce both query and item before
    # any optimizer step, not merely share a nominal random seed.
    schema = store.schema()
    a0 = A0Intervention(user_vocab=schema["user_count"], item_vocab=schema["item_id_count"],
                        user_category_sizes=schema["user_category_sizes"],
                        item_category_sizes=schema["item_category_sizes"],
                        retrieval_dim=256, id_dim=128).to(args.device)
    shared = a0.state_dict()
    current = model.state_dict()
    for key in shared:
        shared[key] = current[key].detach().clone()
    a0.load_state_dict(shared, strict=True)
    mask = np.load(image_paths(IMAGE_STRATEGY)[1], mmap_mode="r")
    available = np.flatnonzero(mask[dataset.target_rows])
    missing = np.flatnonzero(~mask[dataset.target_rows])
    if not len(available) or not len(missing):
        raise RuntimeError("smoke requires one image and one no-image target")
    chosen = [int(available[0]), int(missing[0])]
    batch = to_device(model_batch(dataset, chosen), args.device)
    model.eval(); a0.eval()
    with torch.inference_mode():
        item_diff = float((model.encode_item(batch, "target") - a0.encode_item(batch, "target")).abs().max())
        query_diff = float((model.query(batch) - a0.query(batch)).abs().max())
        unavailable = (~batch["target_image_available"]).nonzero(as_tuple=False)
        no_image_diff = None
        if len(unavailable):
            index = unavailable[0, 0]
            changed = dict(batch); changed["target_image"] = batch["target_image"].clone()
            changed["target_image"][index] = torch.randn_like(changed["target_image"][index])
            no_image_diff = float((model.encode_item(changed, "target")[index] - model.encode_item(batch, "target")[index]).abs().max())
    if max(item_diff, query_diff) > 1e-6 or (no_image_diff is not None and no_image_diff > 1e-7):
        raise RuntimeError("A0 matched-start/no-image invariant failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    stats = train_epoch(model, dataset, optimizer, args.device, 256, 1, IMAGE_STRATEGY,
                        max_batches=40, deadline=deadline, num_workers=args.num_workers,
                        prefetch_factor=args.prefetch_factor, limits=limits)
    gradients = {prefix: any(name.startswith(prefix) and p.grad is not None
                             and torch.count_nonzero(p.grad).item() > 0
                             for name, p in model.named_parameters())
                 for prefix in ("content", "item_meta", "item_id", "attention",
                                "history_projection", "user_profile", "user_id",
                                "image_projection", "image_gate")}
    if model.interaction is not None:
        gradients["interaction"] = any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                                        for p in model.interaction.parameters())
    requests = select_requests(grouped_requests(valid), 1_000, 42)
    ids = deterministic_proxy_candidates(np.asarray(store.item_ids),
                                         set().union(*(r.ground_truth for r in requests)), 100_000, 42)
    proxy = a0run.proxy(model, requests, store, IMAGE_STRATEGY, args.device, ids,
                        pack_item_features(ids, store, True), deadline,
                        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, limits=limits)
    passed = bool(np.isfinite(stats["loss"]) and item_diff <= 1e-6 and query_diff <= 1e-6
                  and no_image_diff is not None and no_image_diff <= 1e-7
                  and all(gradients.values()) and not batch["target_content"].requires_grad
                  and not batch["target_image"].requires_grad)
    payload = {"passed": passed, "model": args.model, "matched_item_max_abs": item_diff,
               "matched_query_max_abs": query_diff, "no_image_max_abs": no_image_diff,
               "gradients": gradients, "train": stats, "proxy": proxy, "test_opened": False}
    save_json(OUT / f"smoke/{args.model}_seed42.json", payload)
    if not passed:
        raise RuntimeError("fusion smoke failed")
    print(json.dumps(payload, ensure_ascii=False))


def train_done(name, seed):
    return marker(f"training_complete_{name}_seed{seed}.json")


def train(args):
    layout(); limits = resource_preflight(args); audit_value = require_audit()
    smoke_path = OUT / f"smoke/{args.model}_seed42.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get("passed"):
        raise RuntimeError("successful smoke required")
    if train_done(args.model, args.seed).exists() or marker(f"full_validation_{args.model}_seed{args.seed}.json").exists():
        raise RuntimeError("completed training/validation cannot be overwritten")
    a0run.seed_all(args.seed); deadline = time.monotonic() + 3600
    _, valid, store = resources()
    dataset = Phase7Dataset(load_training_frame(True), store, True, seed=args.seed)
    requests = select_requests(grouped_requests(valid), 5_000, 42)
    ids = deterministic_proxy_candidates(np.asarray(store.item_ids),
                                         set().union(*(r.ground_truth for r in requests)), 100_000, 42)
    packed = pack_item_features(ids, store, True)
    model = make_model(store, args.device, args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best, stale, curve = -1.0, 0, []
    best_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_best.pt"
    last_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_last.pt"
    for epoch in range(1, PROTOCOL.epochs + 1):
        stats = train_epoch(model, dataset, optimizer, args.device, 512, epoch,
                            IMAGE_STRATEGY, deadline=deadline, num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor, limits=limits)
        proxy = a0run.proxy(model, requests, store, IMAGE_STRATEGY, args.device,
                            ids, packed, deadline, num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor, limits=limits)
        curve.append({"epoch": epoch, "seed": args.seed, "model": args.model, **stats,
                      **{f"proxy_{key}": value for key, value in proxy.items()}})
        state = {"state_dict": model.state_dict(), "variant": args.model, "seed": args.seed,
                 "epoch": epoch, "proxy_recall500": proxy["Recall@500"],
                 "image_sha256": audit_value["image_asset"]["image_sha256"]}
        save_torch_atomic(state, last_path)
        if proxy["Recall@500"] > best:
            best, stale = proxy["Recall@500"], 0
            save_torch_atomic(state, best_path)
        else:
            stale += 1
            if stale >= PROTOCOL.patience:
                break
    curve_path = OUT / f"training_curves/{args.model}_seed{args.seed}.csv"
    config_path = OUT / f"configs/{args.model}_seed{args.seed}.json"
    save_csv_atomic(pd.DataFrame(curve), curve_path)
    save_json(config_path, {"variant": args.model, "seed": args.seed, "batch_size": 512,
                           "temperature": 0.05, "loss": "H2 in-batch + 0.5 TF-IDF pairwise",
                           "image_asset": audit_value["image_asset"],
                           "parameters": parameter_report(model), "test_opened": False})
    save_json(train_done(args.model, args.seed), {"complete": True, "variant": args.model,
              "seed": args.seed, "epochs_completed": len(curve),
              "checkpoint": str(best_path.relative_to(ROOT)), "checkpoint_sha256": sha(best_path),
              "curve": str(curve_path.relative_to(ROOT)), "curve_sha256": sha(curve_path),
              "config": str(config_path.relative_to(ROOT)), "config_sha256": sha(config_path),
              "test_opened": False})


def load_best(name, seed, store, device):
    done = json.loads(train_done(name, seed).read_text())
    checkpoint_path = ROOT / done["checkpoint"]
    if (not done.get("complete") or sha(checkpoint_path) != done["checkpoint_sha256"]
            or sha(ROOT / done["curve"]) != done["curve_sha256"]
            or sha(ROOT / done["config"]) != done["config_sha256"]):
        raise RuntimeError("training completion/hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(store, device, name)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint, done


def cross_metrics(requests, rankings, store, image_mask, train_vocab):
    rows = {key: [] for key in ("has_image_and_unseen", "no_image_and_unseen")}
    counts = {key: 0 for key in rows}
    for request, ranking in zip(requests, rankings):
        selected = set(map(int, request.ground_truth)) - train_vocab
        has = {note for note in selected if image_mask[store.item_lookup[note]]}
        for key, truth in (("has_image_and_unseen", has), ("no_image_and_unseen", selected - has)):
            counts[key] += len(truth)
            if truth:
                rows[key].append(len(truth.intersection(map(int, ranking[:500]))) / len(truth))
    return {key: {"eligible_requests": len(values), "positive_items": counts[key],
                  "Recall@500": float(np.mean(values)) if values else None}
            for key, values in rows.items()}


def validation_done(name, seed):
    return marker(f"full_validation_{name}_seed{seed}.json")


def validation(name, seed):
    done = json.loads(validation_done(name, seed).read_text())
    result_path, ranking_path = ROOT / done["result"], ROOT / done["ranking"]
    if (not done.get("complete") or sha(result_path) != done["result_sha256"]
            or sha(ranking_path) != done["ranking_sha256"]):
        raise RuntimeError("full validation marker/hash mismatch")
    result = json.loads(result_path.read_text())
    if result["checkpoint_sha256"] != done["checkpoint_sha256"]:
        raise RuntimeError("validation/checkpoint mismatch")
    return result, pd.read_parquet(ranking_path)


def validation_lock(args):
    lock = marker("full_validation_active.lock")
    if lock.exists() and args.recover_stale_validation_lock:
        owner = json.loads(lock.read_text())
        if owner.get("hostname") != socket.gethostname() or not isinstance(owner.get("pid"), int):
            raise RuntimeError("unverifiable validation lock")
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError("validation lock owner is alive")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"hostname": socket.gethostname(), "pid": os.getpid(),
                   "created_at": time.time()}, stream)
    return lock


def full_validation(args):
    layout(); limits = resource_preflight(args); audit_value = require_audit()
    if validation_done(args.model, args.seed).exists():
        raise RuntimeError("completed full validation is immutable")
    lock = validation_lock(args)
    try:
        started, deadline = time.perf_counter(), time.monotonic() + 1200
        _, valid, store = resources()
        requests = grouped_requests(valid)
        model, checkpoint, trained = load_best(args.model, args.seed, store, args.device)
        ids = np.asarray(store.item_ids, dtype=np.int64)
        item_started = time.perf_counter()
        items = encode_items(model, ids, store, args.device, IMAGE_STRATEGY,
                             deadline=deadline, limits=limits)
        item_seconds = time.perf_counter() - item_started
        vector_hash = array_sha(items)
        query_started = time.perf_counter()
        queries = encode_queries(model, requests, store, args.device, IMAGE_STRATEGY,
                                 deadline=deadline, num_workers=args.num_workers,
                                 prefetch_factor=args.prefetch_factor, limits=limits)
        query_seconds = time.perf_counter() - query_started
        _, raw, _, backend, timing = gpu_exact_search(items, queries, PROTOCOL.overfetch,
                                                       args.device, query_batch=256, deadline=deadline)
        rankings = filter_history(raw, ids, [r.history for r in requests], 500, deadline)
        validate_topk(rankings, ids, [r.history for r in requests], deadline)
        targets = set(map(int, store.train_target_item_ids))
        vocab = set(map(int, store.train_item_id_vocab))
        users = set(map(int, store.train_user_ids))
        old_exposed, old_clicked, old_users = legacy_evaluator_sets()
        metrics, status, per = phase6_status_metrics(requests, rankings, targets, vocab,
                                                     users, args.model, old_exposed,
                                                     old_clicked, old_users, deadline)
        image_mask = np.load(image_paths(IMAGE_STRATEGY)[1], mmap_mode="r")
        image_status = a0run.image_segment_metrics(requests, rankings, store, image_mask)
        cross = cross_metrics(requests, rankings, store, image_mask, vocab)
        ranking_path = OUT / f"validation/rankings/{args.model}_seed{args.seed}.parquet"
        result_path = OUT / f"validation/{args.model}_seed{args.seed}.json"
        save_parquet_atomic(per, ranking_path)
        save_json(result_path, {"variant": args.model, "seed": args.seed,
                  "checkpoint_epoch": checkpoint["epoch"],
                  "checkpoint_sha256": trained["checkpoint_sha256"],
                  "vector_sha256": vector_hash, "vector_shape": list(items.shape),
                  "vector_dtype": str(items.dtype), "mapping_sha256": audit_value["image_asset"]["mapping_sha256"],
                  "metrics": metrics, "phase6_item_status": status,
                  "image_status": image_status, "cross_status": cross,
                  "candidate_universe": len(ids), "item_encode_seconds": item_seconds,
                  "query_encode_seconds": query_seconds, "search_backend": backend,
                  **timing, "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                  "elapsed_seconds": time.perf_counter() - started, "test_opened": False})
        save_json(validation_done(args.model, args.seed), {"complete": True,
                  "variant": args.model, "seed": args.seed,
                  "checkpoint_sha256": trained["checkpoint_sha256"],
                  "result": str(result_path.relative_to(ROOT)), "result_sha256": sha(result_path),
                  "ranking": str(ranking_path.relative_to(ROOT)), "ranking_sha256": sha(ranking_path),
                  "test_opened": False})
    finally:
        lock.unlink(missing_ok=True)


def ranking_dict(frame):
    value = {int(row.request_idx): set(map(int, row.retrieved_top500)) for row in frame.itertuples(index=False)}
    if len(value) != len(frame):
        raise RuntimeError("duplicate request IDs in ranking")
    return value


def paired_deltas(requests, base_frame, candidate_frame, store, mask, targets, vocab):
    base, candidate = ranking_dict(base_frame), ranking_dict(candidate_frame)
    if set(base) != set(candidate) or len(base) != len(requests):
        raise RuntimeError("paired request set mismatch")
    segments = ("overall", "has_image", "no_image", "train_target_seen",
                "completely_unseen", "has_image_and_unseen", "no_image_and_unseen")
    output = {key: [] for key in segments}
    for request in requests:
        truth = set(map(int, request.ground_truth))
        has = {note for note in truth if mask[store.item_lookup[note]]}
        unseen = truth - vocab
        parts = {"overall": truth, "has_image": has, "no_image": truth - has,
                 "train_target_seen": truth & targets, "completely_unseen": unseen,
                 "has_image_and_unseen": has & unseen,
                 "no_image_and_unseen": unseen - has}
        for label, selected in parts.items():
            if selected:
                output[label].append(len(selected & candidate[request.request_idx]) / len(selected)
                                     - len(selected & base[request.request_idx]) / len(selected))
    return {key: np.asarray(value, np.float64) for key, value in output.items()}


def reference_frame(label, seed):
    if label == "c0":
        path = ROOT / a0run.control(seed)["ranking"]
        return pd.read_parquet(path)
    return a0run.validation("i3_all", seed)[1]


def paired_for(name, seed, reference, requests, store, mask, targets, vocab):
    return paired_deltas(requests, reference_frame(reference, seed), validation(name, seed)[1],
                         store, mask, targets, vocab)


def bootstrap_seed42(_):
    layout(); require_audit()
    _, valid, store = resources(); requests = grouped_requests(valid)
    mask = np.load(image_paths(IMAGE_STRATEGY)[1], mmap_mode="r")
    targets, vocab = set(map(int, store.train_target_item_ids)), set(map(int, store.train_item_id_vocab))
    output = {}
    for name in VARIANTS:
        output[name] = {}
        for reference in ("c0", "a0"):
            arrays = paired_for(name, 42, reference, requests, store, mask, targets, vocab)
            output[name][reference] = {label: a0run.bootstrap_array(values, seed=42 + index)
                                       for index, (label, values) in enumerate(arrays.items())}
    save_json(OUT / "validation/seed42_paired_bootstrap.json", output)


def select(_):
    path = marker("seed42_selection.json")
    if path.exists():
        raise RuntimeError("seed42 selection is immutable")
    paired_path = OUT / "validation/seed42_paired_bootstrap.json"
    values = json.loads(paired_path.read_text())
    eligible, scores = [], {}
    for name in VARIANTS:
        result, _ = validation(name, 42)
        score = float(result["metrics"]["overall"]["Recall@500"])
        scores[name] = score
        if (values[name]["c0"]["no_image"]["point_delta"] >= -MAX_DROP
                and values[name]["c0"]["completely_unseen"]["point_delta"] >= -MAX_DROP):
            eligible.append((score, float(result["metrics"]["overall"]["Recall@100"]), name))
    best = max(eligible)[2] if eligible else None
    diagnostic_model = max((score, name) for name, score in scores.items())[1]
    save_json(path, {"best_model": best, "eligible": [x[2] for x in eligible],
                     "diagnostic_model": diagnostic_model,
                     "seed42_scores": scores, "selection_rule": "screen no-image/unseen point loss <=0.1pp; then max overall R@500, tie R@100/name",
                     "bootstrap_sha256": sha(paired_path), "test_opened": False})


def lock(_):
    path = marker("final_decision.json")
    if path.exists():
        raise RuntimeError("final decision is immutable")
    selection = json.loads(marker("seed42_selection.json").read_text())
    name = selection["best_model"]
    if name is None:
        save_json(path, {"best_model": None, "terminal_allowed": False,
                         "reason": "no seed42 structure passed no-image/unseen screen",
                         "test_opened": False})
        return
    require_audit()
    _, valid, store = resources(); requests = grouped_requests(valid)
    mask = np.load(image_paths(IMAGE_STRATEGY)[1], mmap_mode="r")
    targets, vocab = set(map(int, store.train_target_item_ids)), set(map(int, store.train_item_id_vocab))
    scores, controls = [], {"c0": [], "a0": []}
    paired = {label: [] for label in controls}
    for seed in SEEDS:
        value, _ = validation(name, seed)
        scores.append(float(value["metrics"]["overall"]["Recall@500"]))
        controls["c0"].append(float(a0run.control(seed)["metrics"]["overall"]["Recall@500"]))
        controls["a0"].append(float(a0run.validation("i3_all", seed)[0]["metrics"]["overall"]["Recall@500"]))
        for reference in controls:
            paired[reference].append(paired_for(name, seed, reference, requests, store, mask, targets, vocab))
    boot = {}
    per_seed = {}
    for reference in controls:
        boot[reference], per_seed[reference] = {}, {}
        for index, segment in enumerate(paired[reference][0]):
            arrays = [row[segment] for row in paired[reference]]
            if any(len(array) != len(arrays[0]) for array in arrays):
                raise RuntimeError("seed paired request segment mismatch")
            averaged = np.mean(np.stack(arrays), axis=0) if len(arrays[0]) else np.empty(0)
            boot[reference][segment] = a0run.bootstrap_array(averaged, seed=100 + index)
            per_seed[reference][segment] = [float(array.mean()) if len(array) else None for array in arrays]
    c0 = boot["c0"]
    if any(c0[key]["eligible_requests"] == 0 for key in ("overall", "has_image", "no_image", "completely_unseen")):
        raise RuntimeError("required validation slice is empty")
    deltas = [score - control for score, control in zip(scores, controls["c0"])]
    gates = {
        "practical_gain_vs_c0": float(np.mean(deltas)) >= MIN_GAIN,
        "overall_ci_positive": c0["overall"]["ci95_lower"] > 0,
        "not_worse_than_a0": boot["a0"]["overall"]["point_delta"] >= 0,
        "has_image_positive": c0["has_image"]["point_delta"] > 0,
        "no_image_noninferior": c0["no_image"]["ci95_lower"] >= -MAX_DROP,
        "unseen_noninferior": c0["completely_unseen"]["ci95_lower"] >= -MAX_DROP,
        "all_seed_directions_positive": all(delta > 0 for delta in deltas),
    }
    middle = sorted(zip(scores, SEEDS))[1][1]
    chosen, _ = validation(name, middle)
    trained = json.loads(train_done(name, middle).read_text())
    save_json(path, {"best_model": name, "scores": scores, "controls": controls,
              "seed_deltas_vs_c0": deltas, "seed_delta_std_vs_c0": float(np.std(deltas)),
              "mean": float(np.mean(scores)), "paired_bootstrap": boot,
              "per_seed_paired_deltas": per_seed, "gates": gates,
              "terminal_allowed": all(gates.values()), "seed": middle,
              "checkpoint": trained["checkpoint"], "checkpoint_sha256": trained["checkpoint_sha256"],
              "validation_vector_sha256": chosen["vector_sha256"],
              "test_opened": False})


def intervention_summary(requests, rankings, normal, store, image_mask):
    metrics, _ = evaluate_rankings(requests, rankings, set(), set(), "diagnostic")
    image = a0run.image_segment_metrics(requests, rankings, store, image_mask)
    overlap, newly_hit, lost_hit, candidate_has_image = [], [], [], []
    per_request = []
    for request, rank, baseline in zip(requests, rankings, normal):
        current, original = set(map(int, rank[:500])), set(map(int, baseline[:500]))
        truth = set(map(int, request.ground_truth))
        new_hits, lost_hits = len((current - original) & truth), len((original - current) & truth)
        shared = len(current & original)
        image_count = sum(bool(image_mask[store.item_lookup[int(note)]]) for note in rank[:500])
        overlap.append(shared / 500)
        newly_hit.append(new_hits); lost_hit.append(lost_hits)
        candidate_has_image.append(image_count / 500)
        per_request.append({"request_idx": request.request_idx, "overlap_top500": shared,
                            "new_positive_hits": new_hits, "lost_positive_hits": lost_hits,
                            "has_image_candidates": image_count})
    return {"overall_recall500": metrics["overall"]["Recall@500"],
            "has_image_recall500": image["has_image"]["Recall@500"],
            "no_image_recall500": image["no_image"]["Recall@500"],
            "mean_top500_overlap": float(np.mean(overlap)),
            "new_positive_hits": int(sum(newly_hit)), "lost_positive_hits": int(sum(lost_hit)),
            "mean_has_image_candidate_fraction": float(np.mean(candidate_has_image))}, per_request


def diagnose(args):
    """Read-only validation interventions; never used in selection or GO gates."""
    layout(); limits = resource_preflight(args); require_audit()
    decision_path = marker("final_decision.json")
    if not decision_path.exists():
        raise RuntimeError("lock validation decision before mechanism diagnostics")
    decision = json.loads(decision_path.read_text())
    selection = json.loads(marker("seed42_selection.json").read_text())
    done_path = marker("diagnostics_complete.json")
    if done_path.exists():
        raise RuntimeError("completed mechanism diagnostics are immutable")
    # If all structures fail the prespecified screening rule, diagnose only
    # the seed42 highest-overall model. This cannot revive its No-Go decision.
    seed = decision.get("seed", 42)
    name = decision.get("best_model") or selection["diagnostic_model"]
    _, valid, store = resources(); requests = grouped_requests(valid)
    ids = np.asarray(store.item_ids, np.int64)
    image_mask = np.load(image_paths(IMAGE_STRATEGY)[1], mmap_mode="r")
    deadline = time.monotonic() + 1200
    schema = store.schema()
    a0_model = A0Intervention(user_vocab=schema["user_count"], item_vocab=schema["item_id_count"],
                              user_category_sizes=schema["user_category_sizes"],
                              item_category_sizes=schema["item_category_sizes"],
                              retrieval_dim=256, id_dim=128).to(args.device)
    a0_done = json.loads(a0run.completion("i3_all", seed).read_text())
    if sha(ROOT / a0_done["checkpoint"]) != a0_done["checkpoint_sha256"]:
        raise RuntimeError("A0 checkpoint changed")
    a0_checkpoint = torch.load(ROOT / a0_done["checkpoint"], map_location=args.device, weights_only=False)
    a0_model.load_state_dict(a0_checkpoint["state_dict"], strict=True)
    a0_model.eval()
    fusion_model, _, _ = load_best(name, seed, store, args.device)
    outcomes = {"seed": seed, "fusion": name, "test_opened": False,
                "warning": "inference interventions are out-of-distribution mechanism clues, not formal model scores"}
    per_request_rows = []
    for label, model in (("a0", a0_model), (name, fusion_model)):
        model.eval()
        model.disable_history_image = False; model.disable_candidate_image = False
        query_on = encode_queries(model, requests, store, args.device, IMAGE_STRATEGY,
                                  deadline=deadline, num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch_factor, limits=limits)
        model.disable_history_image = True
        query_off = encode_queries(model, requests, store, args.device, IMAGE_STRATEGY,
                                   deadline=deadline, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch_factor, limits=limits)
        model.disable_history_image = False
        item_on = encode_items(model, ids, store, args.device, IMAGE_STRATEGY,
                               deadline=deadline, limits=limits)
        normal = None
        variants = {}
        for switch, queries in (("on_on", query_on), ("history_off", query_off)):
            _, raw, _, _, _ = gpu_exact_search(item_on, queries, PROTOCOL.overfetch,
                                               args.device, query_batch=256, deadline=deadline)
            ranked = filter_history(raw, ids, [r.history for r in requests], 500, deadline)
            validate_topk(ranked, ids, [r.history for r in requests], deadline)
            if normal is None:
                normal = ranked
            variants[switch] = ranked
        del item_on
        model.disable_candidate_image = True
        item_off = encode_items(model, ids, store, args.device, IMAGE_STRATEGY,
                                deadline=deadline, limits=limits)
        for switch, queries in (("candidate_off", query_on), ("both_off", query_off)):
            _, raw, _, _, _ = gpu_exact_search(item_off, queries, PROTOCOL.overfetch,
                                               args.device, query_batch=256, deadline=deadline)
            ranked = filter_history(raw, ids, [r.history for r in requests], 500, deadline)
            validate_topk(ranked, ids, [r.history for r in requests], deadline)
            variants[switch] = ranked
        del item_off
        model.disable_candidate_image = False
        if label != "a0":
            model.zero_image_values = True  # availability flags stay unchanged
            q_zero = encode_queries(model, requests, store, args.device, IMAGE_STRATEGY,
                                    deadline=deadline, num_workers=args.num_workers,
                                    prefetch_factor=args.prefetch_factor, limits=limits)
            i_zero = encode_items(model, ids, store, args.device, IMAGE_STRATEGY,
                                  deadline=deadline, limits=limits)
            _, raw, _, _, _ = gpu_exact_search(i_zero, q_zero, PROTOCOL.overfetch,
                                               args.device, query_batch=256, deadline=deadline)
            ranked = filter_history(raw, ids, [r.history for r in requests], 500, deadline)
            validate_topk(ranked, ids, [r.history for r in requests], deadline)
            variants["semantic_zero_keep_available"] = ranked
            model.zero_image_values = False
            del i_zero
        outcomes[label] = {}
        for switch, ranked in variants.items():
            summary, rows = intervention_summary(requests, ranked, normal, store, image_mask)
            outcomes[label][switch] = summary
            for row in rows:
                per_request_rows.append({"model": label, "intervention": switch, **row})
    result_path = OUT / "diagnostics/mechanism.json"
    per_path = OUT / "diagnostics/mechanism_per_request.parquet"
    save_json(result_path, outcomes)
    save_parquet_atomic(pd.DataFrame(per_request_rows), per_path)
    save_json(done_path, {"complete": True, "result_sha256": sha(result_path),
                          "per_request_sha256": sha(per_path), "test_opened": False})


def freeze_vector(args):
    """Regenerate only the locked GO vector and bind it to validation hash."""
    layout(); limits = resource_preflight(args); require_audit()
    decision = json.loads(marker("final_decision.json").read_text())
    if not decision.get("terminal_allowed"):
        raise SystemExit("validation No-Go; no terminal vector")
    done_path = marker("frozen_vector_complete.json")
    if done_path.exists():
        raise RuntimeError("frozen terminal vector is immutable")
    _, _, store = resources()
    model, _, trained = load_best(decision["best_model"], decision["seed"], store, args.device)
    deadline = time.monotonic() + 1200
    items = encode_items(model, np.asarray(store.item_ids, np.int64), store, args.device,
                         IMAGE_STRATEGY, deadline=deadline, limits=limits)
    digest = array_sha(items)
    if digest != decision["validation_vector_sha256"]:
        raise RuntimeError("regenerated item vectors differ from selected validation checkpoint")
    path = OUT / "embeddings/locked_terminal_items.f32.npy"
    a0run.save_npy(path, items)
    mapping = IMAGE_OUT / "mappings/note_ids.npy"
    metadata = {"shape": list(items.shape), "dtype": str(items.dtype),
                "array_sha256": digest, "file_sha256": sha(path),
                "mapping": str(mapping.relative_to(ROOT)), "mapping_sha256": sha(mapping),
                "checkpoint_sha256": trained["checkpoint_sha256"], "test_opened": False}
    metadata_path = OUT / "embeddings/locked_terminal_metadata.json"
    save_json(metadata_path, metadata)
    save_json(done_path, {"complete": True, "embedding": str(path.relative_to(ROOT)),
                          "embedding_sha256": sha(path), "metadata_sha256": sha(metadata_path),
                          "test_opened": False})


def terminal_test(args):
    decision_path = marker("final_decision.json")
    decision = json.loads(decision_path.read_text())
    if not decision.get("terminal_allowed"):
        raise SystemExit("validation No-Go; test remains unopened")
    manifest = marker("terminal_test_manifest.json")
    if manifest.exists():
        raise RuntimeError("terminal test is one-shot")
    limits = resource_preflight(args); require_audit()
    frozen = json.loads(marker("frozen_vector_complete.json").read_text())
    path = ROOT / frozen["embedding"]
    metadata_path = OUT / "embeddings/locked_terminal_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (not frozen.get("complete") or sha(path) != frozen["embedding_sha256"]
            or sha(metadata_path) != frozen["metadata_sha256"]
            or metadata["checkpoint_sha256"] != decision["checkpoint_sha256"]
            or metadata["array_sha256"] != decision["validation_vector_sha256"]):
        raise RuntimeError("locked terminal vector contract failed")
    mapping = ROOT / metadata["mapping"]
    if sha(mapping) != metadata["mapping_sha256"]:
        raise RuntimeError("canonical mapping changed")
    items, ids = np.load(path, mmap_mode="r"), np.load(mapping, mmap_mode="r")
    if items.shape != (PROTOCOL.corpus_items, DIM) or items.dtype != np.float32 or ids.shape != (PROTOCOL.corpus_items,):
        raise RuntimeError("terminal vector/mapping shape or dtype mismatch")
    _, _, store = resources()
    model, checkpoint, trained = load_best(decision["best_model"], decision["seed"], store, args.device)
    if trained["checkpoint_sha256"] != decision["checkpoint_sha256"]:
        raise RuntimeError("terminal checkpoint changed")
    save_json(manifest, {"status": "running", "decision_sha256": sha(decision_path), "test_reads": 1})
    deadline = time.monotonic() + 1200
    requests = QilinData(ROOT).load_test_requests()  # first and only test read
    queries = encode_queries(model, requests, store, args.device, IMAGE_STRATEGY,
                             deadline=deadline, num_workers=args.num_workers,
                             prefetch_factor=args.prefetch_factor, limits=limits)
    _, raw, _, backend, timing = gpu_exact_search(items, queries, PROTOCOL.overfetch,
                                                   args.device, query_batch=256, deadline=deadline)
    rankings = filter_history(raw, ids, [r.history for r in requests], 500, deadline)
    validate_topk(rankings, ids, [r.history for r in requests], deadline)
    targets, vocab, users = (set(map(int, store.train_target_item_ids)),
                             set(map(int, store.train_item_id_vocab)),
                             set(map(int, store.train_user_ids)))
    exposed, clicked, warm_users = legacy_evaluator_sets()
    metrics, status, per = phase6_status_metrics(requests, rankings, targets, vocab,
                                                 users, decision["best_model"], exposed,
                                                 clicked, warm_users, deadline)
    mask = np.load(image_paths(IMAGE_STRATEGY)[1], mmap_mode="r")
    result_path = OUT / "metrics/terminal_test.json"
    save_json(result_path, {"model": decision["best_model"], "seed": decision["seed"],
              "checkpoint_epoch": checkpoint["epoch"], "metrics": metrics,
              "phase6_item_status": status,
              "image_status": a0run.image_segment_metrics(requests, rankings, store, mask),
              "cross_status": cross_metrics(requests, rankings, store, mask, vocab),
              "candidate_universe": len(ids), "search_backend": backend, **timing,
              "test_reads": 1})
    save_parquet_atomic(per, OUT / "metrics/terminal_test_rankings.parquet")
    save_json(manifest, {"status": "complete", "decision_sha256": sha(decision_path),
                         "result_sha256": sha(result_path), "test_reads": 1})


def report(_):
    lines = ["# Phase 8-03：图文融合与受控残差", "", "本实验只使用 validation 选型；test 仅在全部预注册门禁通过后打开。", ""]
    selection_path = marker("seed42_selection.json")
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())
        _, valid, store = resources()
        requests = grouped_requests(valid)
        image_mask = np.load(image_paths(IMAGE_STRATEGY)[1], mmap_mode="r")
        train_vocab = set(map(int, store.train_item_id_vocab))
        lines += ["| Model | Seed | R@100 | R@500 | MRR@100 | 有图 R@500 | 无图 R@500 | Completely-unseen R@500 | 有图×Unseen R@500 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for seed in SEEDS:
            for label, result in (("C0", json.loads((ROOT / a0run.control(seed)["result"]).read_text())),
                                  ("A0", a0run.validation("i3_all", seed)[0])):
                reference_rank = reference_frame(label.lower(), seed)
                lookup = {int(row.request_idx): row.retrieved_top500
                          for row in reference_rank.itertuples(index=False)}
                if len(lookup) != len(requests):
                    raise RuntimeError("reference report ranking/request mismatch")
                ordered = [lookup[request.request_idx] for request in requests]
                image = a0run.image_segment_metrics(requests, ordered, store, image_mask)
                cross = cross_metrics(requests, ordered, store, image_mask, train_vocab)
                overall = result["metrics"]["overall"]
                unseen = result["phase6_item_status"]["completely_unseen"]
                lines.append(f"| {label} | {seed} | {overall['Recall@100']:.4%} | {overall['Recall@500']:.4%} | {overall['MRR@100']:.4%} | "
                             f"{image['has_image']['Recall@500']:.4%} | {image['no_image']['Recall@500']:.4%} | "
                             f"{unseen['Recall@500']:.4%} | {cross['has_image_and_unseen']['Recall@500']:.4%} |")
            for name in VARIANTS:
                if not validation_done(name, seed).exists():
                    continue
                result, _ = validation(name, seed)
                m, image, unseen, cross = (result["metrics"]["overall"], result["image_status"],
                                            result["phase6_item_status"]["completely_unseen"], result["cross_status"])
                lines.append(f"| {name} | {seed} | {m['Recall@100']:.4%} | {m['Recall@500']:.4%} | {m['MRR@100']:.4%} | "
                             f"{image['has_image']['Recall@500']:.4%} | {image['no_image']['Recall@500']:.4%} | "
                             f"{unseen['Recall@500']:.4%} | {cross['has_image_and_unseen']['Recall@500']:.4%} |")
        lines += ["", f"Seed42 正式选型：`{selected['best_model']}`；eligible：`{selected['eligible']}`。最高总体分模型 `{selected['diagnostic_model']}` 仅用于只读机制诊断，不能绕过初筛。"]
        boot_path = OUT / "validation/seed42_paired_bootstrap.json"
        if boot_path.exists():
            boot = json.loads(boot_path.read_text())
            lines += ["", "## Seed42 同 request 配对差值", "",
                      "| Model | 对照 | ΔOverall R@500 | Overall 95% CI | Δ有图 | Δ无图 | ΔCompletely-unseen | Δ有图×Unseen |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
            fmt = lambda number: f"{number:+.4%}" if number is not None else "N/A"
            for name in VARIANTS:
                for reference in ("c0", "a0"):
                    row = boot[name][reference]
                    overall = row["overall"]
                    lines.append(f"| {name} | {reference} | {fmt(overall['point_delta'])} | "
                                 f"[{fmt(overall['ci95_lower'])}, {fmt(overall['ci95_upper'])}] | "
                                 f"{fmt(row['has_image']['point_delta'])} | {fmt(row['no_image']['point_delta'])} | "
                                 f"{fmt(row['completely_unseen']['point_delta'])} | "
                                 f"{fmt(row['has_image_and_unseen']['point_delta'])} |")
        lines += ["", "## 模型与工程成本", "",
                  "| Model | 参数量 | 融合分支参数 | Best epoch | 训练秒数 | 全库编码秒数 | Exact 检索秒数 | GPU 峰值 GiB |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in VARIANTS:
            result, _ = validation(name, 42)
            config = json.loads((OUT / f"configs/{name}_seed42.json").read_text())
            curve = pd.read_csv(OUT / f"training_curves/{name}_seed42.csv")
            parameters = config["parameters"]
            lines.append(f"| {name} | {parameters['total_parameters']:,} | {parameters['fusion_parameters']:,} | "
                         f"{result['checkpoint_epoch']} | {float(curve.seconds.sum()):.1f} | "
                         f"{result['item_encode_seconds']:.1f} | {result['query_search_seconds']:.2f} | "
                         f"{result['gpu_peak_memory_bytes']/2**30:.2f} |")
    decision_path = marker("final_decision.json")
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        lines += ["", "## 最终 Validation 决策", ""]
        if decision["best_model"] is None:
            lines.append(f"**No-Go**：{decision['reason']}。test 未读取。")
        else:
            lines += [f"- 最佳结构：`{decision['best_model']}`；三 seed mean R@500：{decision['mean']:.4%}。",
                      f"- 各 seed 相对 C0 的配对差值：`{decision['seed_deltas_vs_c0']}`；std：{decision['seed_delta_std_vs_c0']:.4%}。",
                      f"- GO：`{decision['terminal_allowed']}`；门禁：`{json.dumps(decision['gates'], ensure_ascii=False)}`。",
                      "", "| Reference | Segment | ΔR@500 | 95% CI | Eligible requests |",
                      "|---|---|---:|---:|---:|"]
            for reference, segments in decision["paired_bootstrap"].items():
                for segment, value in segments.items():
                    fmt = lambda x: f"{x:+.4%}" if x is not None else "N/A"
                    lines.append(f"| {reference} | {segment} | {fmt(value['point_delta'])} | "
                                 f"[{fmt(value['ci95_lower'])}, {fmt(value['ci95_upper'])}] | {value['eligible_requests']:,} |")
            if not decision["terminal_allowed"]:
                lines += ["", "**No-Go：不打开 terminal test，不替代 H2-256。**"]
            else:
                lines += ["", "Validation GO；仅在向量冻结与 hash 校验后允许一次 terminal test。"]
    diag = OUT / "diagnostics/mechanism.json"
    if diag.exists():
        diagnostic = json.loads(diag.read_text())
        lines += ["", "## 推理干预诊断（不参与选型）", "",
                  f"同 seed42 的 A0 与 `{diagnostic['fusion']}`，详见 `{diag.relative_to(ROOT)}`。",
                  "| Model | Intervention | Overall R@500 | 有图 R@500 | 无图 R@500 | Top500 overlap | 新增 positive | 丢失 positive | 有图候选占比 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for model_name in ("a0", diagnostic["fusion"]):
            for switch, value in diagnostic[model_name].items():
                lines.append(f"| {model_name} | {switch} | {value['overall_recall500']:.4%} | "
                             f"{value['has_image_recall500']:.4%} | {value['no_image_recall500']:.4%} | "
                             f"{value['mean_top500_overlap']:.4%} | {value['new_positive_hits']} | "
                             f"{value['lost_positive_hits']} | {value['mean_has_image_candidate_fraction']:.4%} |")
        lines += ["", "关闭历史图片、关闭候选图片及图片数值置零均为**分布外推理干预**，只能说明机制线索；尤其保留有图标记而置零图片数值会强烈改变候选组成，不能直接量化纯图片语义贡献。"]
    terminal = OUT / "metrics/terminal_test.json"
    if terminal.exists():
        value = json.loads(terminal.read_text())
        lines += ["", "## Terminal test", "", f"最终 R@500：{value['metrics']['overall']['Recall@500']:.4%}；仅作锁定配置的一次终局观察。"]
    test_opened = marker("terminal_test_manifest.json").exists()
    lines += ["", "Bootstrap CI 主要反映 request 波动；单 seed 结构消融不能充分刻画初始化不确定性。同一 validation 经多结构选优，结论仅为 validation 选型证据。",
              "", f"recommendation test opened：`{test_opened}`；未用于结构、阈值、seed 或 checkpoint 选择。",
              f"当前文件系统剩余空间：{shutil.disk_usage(ROOT).free/2**30:.2f} GiB；未清理或覆盖 Phase 8-01/02 资产。", ""]
    path = OUT / "summary.md"; path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp"); temporary.write_text("\n".join(lines), encoding="utf-8"); temporary.replace(path)


def main():
    args = args_parser(); authorize(args)
    if args.stage == "plan":
        plan(); return
    actions = {"audit": audit, "smoke": smoke, "train": train,
               "full-validation": full_validation, "bootstrap-seed42": bootstrap_seed42,
               "select": select, "lock": lock, "diagnose": diagnose,
               "freeze-vector": freeze_vector, "terminal-test": terminal_test,
               "report": report}
    actions[args.stage](args)


if __name__ == "__main__":
    main()
