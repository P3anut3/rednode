#!/usr/bin/env python3
"""Phase 8-02: frozen pooled-image increment over the matched H2-256 tower."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
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
from experiments.phase_08.experiment_02_image_hybrid_recall.data import (  # noqa: E402
    IMAGE_OUT, STRATEGIES, ImageCollator, image_paths, verify_image_assets,
)
from experiments.phase_08.experiment_02_image_hybrid_recall.trainer import (  # noqa: E402
    DIM, ID_DIM, encode_items, encode_queries, make_model,
    pack_item_features, parameter_report, train_epoch,
)
from experiments.phase_07.experiment_04_h2_retrieval_dimension_ablation.models import HybridTower256  # noqa: E402


OUT = ROOT / "results/phase_08/experiment_02_image_hybrid_recall"
H2 = ROOT / "results/phase_07/experiment_04_h2_retrieval_dimension_ablation"
SEEDS = (42, 43, 44)
STAGES = ("plan", "audit", "smoke", "train", "full-validation", "bootstrap-seed42", "select", "lock", "terminal-test", "report")
WRITE_STAGES = set(STAGES) - {"plan"}
MIN_PRACTICAL_GAIN = 0.001  # +0.1 percentage point in request-macro Recall@500
MAX_SEGMENT_DROP = 0.001  # -0.1 percentage point non-inferiority guard


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument("--model", choices=tuple(STRATEGIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--min-host-available-gb", type=float, default=48.0)
    parser.add_argument("--max-process-tree-rss-gb", type=float, default=16.0)
    parser.add_argument("--max-gpu-allocated-gb", type=float, default=20.0)
    parser.add_argument("--min-disk-free-gb", type=float, default=16.0)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--recover-stale-validation-lock", action="store_true",
                        help="Explicitly clear only a dead same-host full-validation lock")
    return parser.parse_args()


def marker(name: str) -> Path:
    return OUT / "configs/markers" / name


def layout():
    for name in ("audit", "smoke", "configs/markers", "checkpoints", "training_curves", "proxy", "validation/rankings", "embeddings", "diagnostics", "metrics"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_npy(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, data, allow_pickle=False)
    temporary.replace(path)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda(device: str):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("CUDA required; no CPU fallback")


def resource_limits(args) -> dict:
    return {
        "min_host_available_bytes": int(args.min_host_available_gb * 2**30),
        "max_process_tree_rss_bytes": int(args.max_process_tree_rss_gb * 2**30),
        "max_gpu_allocated_bytes": int(args.max_gpu_allocated_gb * 2**30),
        "min_disk_free_bytes": int(args.min_disk_free_gb * 2**30),
        "disk_path": ROOT,
    }


def resource_preflight(args, formal_vectors: bool = False):
    limits = resource_limits(args)
    free = shutil.disk_usage(ROOT).free
    required = limits["min_disk_free_bytes"] + (PROTOCOL.corpus_items * DIM * 4 if formal_vectors else 0)
    if psutil.virtual_memory().available < limits["min_host_available_bytes"] or free < required:
        raise SystemExit("host memory/disk safety preflight failed")
    if args.device.startswith("cuda"):
        gpu = torch.cuda.get_device_properties(args.device)
        if gpu.total_memory < limits["max_gpu_allocated_bytes"]:
            raise SystemExit("GPU memory is below configured training safety cap")
    return limits


def authorize(args):
    if args.stage in WRITE_STAGES and not args.confirm_run:
        raise SystemExit("this stage needs --confirm-run")
    if args.stage in {"smoke", "train", "full-validation"} and args.model is None:
        raise SystemExit("this stage needs --model")
    if args.recover_stale_validation_lock and args.stage != "full-validation":
        raise SystemExit("stale-lock recovery is only valid for full-validation")
    if args.seed not in SEEDS:
        raise SystemExit("formal seeds are 42/43/44")
    if args.stage == "train" and args.batch_size != 512:
        raise SystemExit("matched H2 protocol requires batch-size=512; changed batch needs a new matched C0 control")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise SystemExit("num-workers must be >=0 and prefetch-factor >=1")
    if min(args.min_host_available_gb, args.max_process_tree_rss_gb,
           args.max_gpu_allocated_gb, args.min_disk_free_gb) <= 0:
        raise SystemExit("resource limits must be positive")
    if args.stage == "smoke" and args.seed != 42:
        raise SystemExit("smoke is fixed to seed42 for all three image strategies")
    if args.stage in {"train", "full-validation"} and args.seed != 42:
        selection = marker("seed42_selection.json")
        if not selection.exists() or json.loads(selection.read_text()).get("best_strategy") != args.model:
            raise SystemExit("only the validation-selected strategy may use seeds 43/44")


def plan():
    print(json.dumps({"control": "Phase 7-04 H2-256, same seed", "models": STRATEGIES,
                      "history_n": 20, "retrieval_dim": DIM, "id_dim": ID_DIM,
                      "train_samples": PROTOCOL.train_samples, "valid_samples": PROTOCOL.valid_samples,
                      "test_locked_until_validation": True, "default_action": "read-only plan"}, indent=2))


def control(seed: int) -> dict:
    result = H2 / f"validation/h2_256_seed{seed}.json"
    ranking = H2 / f"validation/rankings/h2_256/seed{seed}.parquet"
    completion = H2 / f"configs/markers/full_validation_h2_256_seed{seed}.json"
    done = json.loads(completion.read_text())
    if (not done.get("complete") or ROOT / done["result"] != result
            or ROOT / done["ranking"] != ranking):
        raise RuntimeError("H2-256 control is incomplete")
    value = json.loads(result.read_text())
    if value["candidate_universe"] != PROTOCOL.corpus_items or value["seed"] != seed:
        raise RuntimeError("H2-256 control protocol mismatch")
    return {"result": str(result.relative_to(ROOT)), "result_sha256": sha(result),
            "ranking": str(ranking.relative_to(ROOT)), "ranking_sha256": sha(ranking),
            "metrics": value["metrics"], "status": value["phase6_item_status"]}


def audit(args):
    layout()
    if marker("audit_complete.json").exists():
        raise SystemExit("completed audit is immutable")
    train, valid, store = resources()
    references = {str(seed): control(seed) for seed in SEEDS}
    assets = {name: verify_image_assets(name, full_hash=True) for name in STRATEGIES}
    image_ids = np.load(IMAGE_OUT / "mappings/note_ids.npy", mmap_mode="r")
    if store.item_ids.shape != image_ids.shape or not np.array_equal(store.item_ids, image_ids):
        raise RuntimeError("Phase-7 feature store and image corpus row order differ")
    masks = [np.load(image_paths(name)[1], mmap_mode="r") for name in STRATEGIES]
    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise RuntimeError("pooling strategies have different availability masks")
    mask = masks[0]
    train_targets = set(map(int, store.train_target_item_ids))
    train_vocab = set(map(int, store.train_item_id_vocab))

    def positive_coverage(frame):
        ids = frame.positive_item_id.to_numpy(np.int64)
        rows = store.item_lookup[ids]
        if np.any(rows < 0):
            raise RuntimeError("positive outside corpus")
        available = np.asarray(mask[rows], dtype=np.bool_)
        categories = {
            "train_target_seen": np.isin(ids, list(train_targets)),
            "train_history_only": np.isin(ids, list(train_vocab - train_targets)),
            "completely_unseen": ~np.isin(ids, list(train_vocab)),
        }
        return {key: {"positive_samples": int(selected.sum()),
                      "image_positive_samples": int((available & selected).sum()),
                      "image_rate": float(available[selected].mean()) if selected.any() else None}
                for key, selected in {"overall": np.ones(len(ids), bool), **categories}.items()}

    payload = {"complete": True, "train_rows": len(train), "valid_rows": len(valid),
               "train_coverage": positive_coverage(train), "valid_coverage": positive_coverage(valid),
               "image_assets": assets, "controls": references, "test_opened": False}
    save_json(OUT / "audit/feature_audit.json", payload)
    save_json(marker("audit_complete.json"), {"complete": True, "audit_sha256": sha(OUT / "audit/feature_audit.json"), "test_opened": False})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def require_audit(strategy: str) -> dict:
    done = json.loads(marker("audit_complete.json").read_text())
    path = OUT / "audit/feature_audit.json"
    if not done.get("complete") or sha(path) != done["audit_sha256"]:
        raise RuntimeError("audit marker/hash mismatch")
    audit_value = json.loads(path.read_text())
    asset = verify_image_assets(strategy, full_hash=True)
    if asset != audit_value["image_assets"][strategy]:
        raise RuntimeError("pooled image asset changed after audit")
    for seed in SEEDS:
        current = control(seed)
        frozen = audit_value["controls"][str(seed)]
        if (current["result_sha256"] != frozen["result_sha256"]
                or current["ranking_sha256"] != frozen["ranking_sha256"]):
            raise RuntimeError("matched H2-256 control changed after audit")
    return audit_value


def proxy(model, requests, store, strategy, device, ids, packed, deadline,
          num_workers=2, prefetch_factor=1, limits=None):
    items = encode_items(model, ids, store, device, strategy, packed=packed,
                         deadline=deadline, limits=limits)
    queries = encode_queries(model, requests, store, device, strategy, deadline=deadline,
                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                             limits=limits)
    _, raw, _, backend, timing = gpu_exact_search(items, queries, min(600, len(ids)), device, deadline=deadline)
    rankings = filter_history(raw, ids, [r.history for r in requests], 500, deadline)
    validate_topk(rankings, ids, [r.history for r in requests], deadline)
    metrics, _ = evaluate_rankings(requests, rankings, set(), set(), "proxy", deadline=deadline)
    return {"Recall@100": metrics["overall"]["Recall@100"],
            "Recall@500": metrics["overall"]["Recall@500"],
            "MRR@100": metrics["overall"]["MRR@100"],
            "backend": backend, **timing}


def smoke(args):
    layout(); require_cuda(args.device); limits = resource_preflight(args)
    require_audit(args.model); seed_all(args.seed)
    deadline = time.monotonic() + 600
    _, valid, store = resources()
    dataset = Phase7Dataset(load_training_frame(True), store, True, limit=10_000, seed=args.seed)
    model = make_model(store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    stats = train_epoch(model, dataset, optimizer, args.device, min(args.batch_size, 256), 1,
                        args.model, max_batches=40, deadline=deadline,
                        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                        limits=limits)
    gradients = {prefix: any(name.startswith(prefix) and p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                             for name, p in model.named_parameters())
                 for prefix in ("content", "item_meta", "item_id", "attention", "history_projection", "user_profile", "user_id", "image_projection", "alpha_image")}
    mask = np.load(image_paths(args.model)[1], mmap_mode="r")
    has_history = (dataset.history_rows >= 0).any(axis=1)
    available_indices = np.flatnonzero(mask[dataset.target_rows] & has_history)
    missing_indices = np.flatnonzero(~mask[dataset.target_rows] & has_history)
    if not len(available_indices) or not len(missing_indices):
        raise RuntimeError("smoke requires both image and no-image samples")
    chosen = [int(available_indices[0]), int(missing_indices[0])]
    batch = to_device(ImageCollator(args.model)([dataset[i] for i in chosen]), args.device)
    model.eval()
    with torch.inference_mode():
        query, target = model(batch)
        direct = encode_items(model, dataset.targets[chosen], store, args.device, args.model,
                              limits=limits)
        missing = 1
        plain = HybridTower256.encode_item_values(
            model,
            batch["target_content"][missing:missing + 1],
            batch["target_item_id_row"][missing:missing + 1],
            batch["target_categorical"][missing:missing + 1],
            batch["target_numeric"][missing:missing + 1],
        )
        history_index = int(torch.nonzero(batch["history_mask"][0], as_tuple=False)[0])
        history_id = np.asarray([int(batch["history_note_ids"][0, history_index])], dtype=np.int64)
        history_direct = encode_items(model, history_id, store, args.device, args.model,
                                      limits=limits)
        history_batch = model.encode_item(batch, "history")[0, history_index]
        changed = dict(batch)
        changed["target_content"] = torch.randn_like(batch["target_content"])
        changed["target_image"] = torch.randn_like(batch["target_image"])
        changed["target_item_id_row"] = torch.zeros_like(batch["target_item_id_row"])
        query_after_target_change = model.query(changed)
    requests = select_requests(grouped_requests(valid), 1_000, 42)
    ids = deterministic_proxy_candidates(np.asarray(store.item_ids),
                                         set().union(*(r.ground_truth for r in requests)), 100_000, 42)
    proxy_result = proxy(model, requests, store, args.model, args.device, ids,
                         pack_item_features(ids, store, True), deadline,
                         num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                         limits=limits)
    passed = bool(np.isfinite(stats["loss"]) and all(gradients.values())
                  and not batch["target_image"].requires_grad
                  and target.shape == (2, DIM) and query.shape == (2, DIM)
                  and torch.allclose(target.norm(dim=-1), torch.ones(2, device=args.device), atol=1e-4)
                  and np.allclose(target.cpu().numpy(), direct, atol=1e-5)
                  and np.allclose(history_batch.cpu().numpy(), history_direct[0], atol=1e-5)
                  and torch.allclose(query, query_after_target_change, atol=1e-7)
                  and torch.allclose(target[missing:missing + 1], plain, atol=1e-6)
                  and not batch["target_content"].requires_grad
                  and bool(batch["history_image_available"].logical_not().any()))
    payload = {"passed": passed, "model": args.model, "seed": args.seed,
               "train": stats, "proxy": proxy_result, "gradients": gradients,
               "image_available_count": int(batch["target_image_available"].sum()),
               "no_image_max_abs_vs_h2": float((target[missing:missing + 1] - plain).abs().max()),
               "parameters": parameter_report(model), "test_opened": False}
    payload["resource_limits"] = {key: value for key, value in limits.items() if key != "disk_path"}
    payload["data_loader"] = {"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor}
    save_json(OUT / f"smoke/{args.model}_seed{args.seed}.json", payload)
    if not passed:
        raise RuntimeError("image-hybrid smoke failed")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def completion(name, seed):
    return marker(f"training_complete_{name}_seed{seed}.json")


def train(args):
    layout(); require_cuda(args.device); limits = resource_preflight(args)
    asset = require_audit(args.model)
    smoke_path = OUT / f"smoke/{args.model}_seed42.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get("passed"):
        raise SystemExit("successful seed42 smoke required")
    if completion(args.model, args.seed).exists() or marker(f"full_validation_{args.model}_seed{args.seed}.json").exists():
        raise SystemExit("completed training/validation cannot be overwritten")
    seed_all(args.seed); deadline = time.monotonic() + 3600
    _, valid, store = resources()
    dataset = Phase7Dataset(load_training_frame(True), store, True, seed=args.seed)
    requests = select_requests(grouped_requests(valid), 5_000, 42)
    ids = deterministic_proxy_candidates(np.asarray(store.item_ids),
                                         set().union(*(r.ground_truth for r in requests)), 100_000, 42)
    packed = pack_item_features(ids, store, True)
    model = make_model(store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best, stale, curve = -1.0, 0, []
    best_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_best.pt"
    last_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_last.pt"
    for epoch in range(1, PROTOCOL.epochs + 1):
        stats = train_epoch(model, dataset, optimizer, args.device, args.batch_size,
                            epoch, args.model, deadline=deadline,
                            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                            limits=limits)
        metrics = proxy(model, requests, store, args.model, args.device, ids, packed, deadline,
                        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                        limits=limits)
        curve.append({"epoch": epoch, "seed": args.seed, "model": args.model, **stats,
                      **{f"proxy_{k}": v for k, v in metrics.items()}})
        state = {"state_dict": model.state_dict(), "model": args.model, "seed": args.seed,
                 "epoch": epoch, "proxy_recall500": metrics["Recall@500"],
                 "image_asset_sha256": asset["image_assets"][args.model]["image_sha256"]}
        save_torch_atomic(state, last_path)
        if metrics["Recall@500"] > best:
            best, stale = metrics["Recall@500"], 0
            save_torch_atomic(state, best_path)
        else:
            stale += 1
            if stale >= PROTOCOL.patience:
                break
    curve_path = OUT / f"training_curves/{args.model}_seed{args.seed}.csv"
    config_path = OUT / f"configs/{args.model}_seed{args.seed}.json"
    save_csv_atomic(pd.DataFrame(curve), curve_path)
    save_json(config_path, {"model": args.model, "seed": args.seed, "batch_size": 512,
                            "loss": "Phase7 TF-IDF HN + in-batch InfoNCE", "temperature": 0.05,
                            "best_proxy_recall500": best, "image_asset": asset["image_assets"][args.model],
                            "data_loader": {"num_workers": args.num_workers,
                                            "prefetch_factor": args.prefetch_factor},
                            "resource_limits": {key: value for key, value in limits.items() if key != "disk_path"},
                            "parameters": parameter_report(model), "test_opened": False})
    save_json(completion(args.model, args.seed), {"complete": True, "model": args.model,
              "seed": args.seed, "epochs_completed": len(curve),
              "checkpoint": str(best_path.relative_to(ROOT)), "checkpoint_sha256": sha(best_path),
              "curve": str(curve_path.relative_to(ROOT)), "config": str(config_path.relative_to(ROOT)),
              "test_opened": False})


def load_best(name, seed, store, device):
    done = json.loads(completion(name, seed).read_text())
    checkpoint_path = ROOT / done["checkpoint"]
    if not done.get("complete") or sha(checkpoint_path) != done["checkpoint_sha256"]:
        raise RuntimeError("training checkpoint/hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(store, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint, done


def image_segment_metrics(requests, rankings, store, mask) -> dict:
    if len(requests) != len(rankings):
        raise RuntimeError("request/ranking count mismatch")
    values = {"has_image": [], "no_image": []}
    for request, ranking in zip(requests, rankings):
        truth = set(map(int, request.ground_truth))
        available = {note for note in truth if mask[store.item_lookup[note]]}
        for label, selected in (("has_image", available), ("no_image", truth - available)):
            if selected:
                hits = {int(note): rank for rank, note in enumerate(ranking[:500], 1) if int(note) in selected}
                first = min(hits.values(), default=0)
                values[label].append((sum(rank <= 100 for rank in hits.values()) / len(selected),
                                      len(hits) / len(selected), 1 / first if 0 < first <= 100 else 0.0))
    return {label: {"eligible_requests": len(rows), "Recall@100": float(np.mean([x[0] for x in rows])),
                    "Recall@500": float(np.mean([x[1] for x in rows])),
                    "MRR@100": float(np.mean([x[2] for x in rows]))}
            for label, rows in values.items()}


def full_validation(args):
    layout(); require_cuda(args.device); limits = resource_preflight(args, formal_vectors=True)
    audit_value = require_audit(args.model)
    done_path = marker(f"full_validation_{args.model}_seed{args.seed}.json")
    if done_path.exists():
        raise SystemExit("completed full validation is immutable")
    lock = marker("full_validation_active.lock")
    if lock.exists() and args.recover_stale_validation_lock:
        try:
            owner = json.loads(lock.read_text())
        except (ValueError, OSError) as exc:
            raise SystemExit("unreadable lock requires manual inspection; refusing recovery") from exc
        if owner.get("hostname") != socket.gethostname() or not isinstance(owner.get("pid"), int):
            raise SystemExit("lock is not a verifiable same-host process")
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            lock.unlink()
        except PermissionError as exc:
            raise SystemExit("lock owner may still be alive; refusing recovery") from exc
        else:
            raise SystemExit("full-validation lock owner is still alive")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit("another full validation is active") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"hostname": socket.gethostname(), "pid": os.getpid(), "created_at": time.time()}, stream)
        started, deadline = time.perf_counter(), time.monotonic() + 1200
        torch.cuda.reset_peak_memory_stats(args.device)
        train_frame, valid, store = resources()
        requests = grouped_requests(valid)
        model, checkpoint, trained = load_best(args.model, args.seed, store, args.device)
        candidates = np.asarray(store.item_ids)
        item_start = time.perf_counter()
        items = encode_items(model, candidates, store, args.device, args.model,
                             deadline=deadline, limits=limits)
        item_seconds = time.perf_counter() - item_start
        embedding = OUT / f"embeddings/{args.model}/seed{args.seed}_item_vectors.f32.npy"
        save_npy(embedding, items)
        embedding_hash = sha(embedding)
        query_start = time.perf_counter()
        queries = encode_queries(model, requests, store, args.device, args.model, deadline=deadline,
                                 num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                 limits=limits)
        query_seconds = time.perf_counter() - query_start
        _, raw, _, backend, timing = gpu_exact_search(items, queries, PROTOCOL.overfetch,
                                                       args.device, query_batch=256, deadline=deadline)
        rankings = filter_history(raw, candidates, [r.history for r in requests], 500, deadline)
        validate_topk(rankings, candidates, [r.history for r in requests], deadline)
        targets = set(map(int, store.train_target_item_ids))
        vocab = set(map(int, store.train_item_id_vocab))
        users = set(map(int, store.train_user_ids))
        old_exposed, old_clicked, old_users = legacy_evaluator_sets()
        metrics, status, per = phase6_status_metrics(requests, rankings, targets, vocab,
                                                     users, args.model, old_exposed,
                                                     old_clicked, old_users, deadline)
        mask = np.load(image_paths(args.model)[1], mmap_mode="r")
        image_segments = image_segment_metrics(requests, rankings, store, mask)
        ranking_path = OUT / f"validation/rankings/{args.model}_seed{args.seed}.parquet"
        save_parquet_atomic(per, ranking_path)
        result_path = OUT / f"validation/{args.model}_seed{args.seed}.json"
        result = {"model": args.model, "seed": args.seed, "checkpoint_epoch": checkpoint["epoch"],
                  "checkpoint_sha256": trained["checkpoint_sha256"],
                  "metrics": metrics, "phase6_item_status": status, "image_status": image_segments,
                  "candidate_universe": len(candidates), "parameters": parameter_report(model),
                  "alpha": model.alpha_values(), "item_vector_bytes": embedding.stat().st_size,
                  "item_encode_seconds": item_seconds, "query_encode_seconds": query_seconds,
                  "search_backend": backend, **timing,
                  "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                  "elapsed_seconds": time.perf_counter() - started, "test_opened": False}
        save_json(result_path, result)
        mapping = IMAGE_OUT / "mappings/note_ids.npy"
        metadata_path = OUT / f"embeddings/{args.model}/seed{args.seed}_metadata.json"
        save_json(metadata_path, {"shape": list(items.shape), "dtype": str(items.dtype),
                  "embedding_sha256": embedding_hash, "checkpoint_sha256": trained["checkpoint_sha256"],
                  "mapping": str(mapping.relative_to(ROOT)), "mapping_sha256": sha(mapping),
                  "source_image_sha256": audit_value["image_assets"][args.model]["image_sha256"]})
        save_json(done_path, {"complete": True, "model": args.model, "seed": args.seed,
                  "checkpoint_sha256": trained["checkpoint_sha256"],
                  "result": str(result_path.relative_to(ROOT)), "result_sha256": sha(result_path),
                  "ranking": str(ranking_path.relative_to(ROOT)), "ranking_sha256": sha(ranking_path),
                  "embedding": str(embedding.relative_to(ROOT)), "embedding_sha256": embedding_hash,
                  "metadata": str(metadata_path.relative_to(ROOT)), "test_opened": False})
    finally:
        lock.unlink(missing_ok=True)


def validation(name, seed):
    done = json.loads(marker(f"full_validation_{name}_seed{seed}.json").read_text())
    result_path, ranking_path = ROOT / done["result"], ROOT / done["ranking"]
    if not done.get("complete") or sha(result_path) != done["result_sha256"] or sha(ranking_path) != done["ranking_sha256"]:
        raise RuntimeError("validation marker/artifact mismatch")
    result = json.loads(result_path.read_text())
    if result["checkpoint_sha256"] != done["checkpoint_sha256"]:
        raise RuntimeError("validation checkpoint mismatch")
    return result, pd.read_parquet(ranking_path)


def paired_deltas(requests, base_frame, new_frame, store, image_mask, train_targets, train_vocab, warm_items):
    base = {int(r.request_idx): set(r.retrieved_top500) for r in base_frame.itertuples(index=False)}
    new = {int(r.request_idx): set(r.retrieved_top500) for r in new_frame.itertuples(index=False)}
    if set(base) != set(new) or len(base) != len(requests):
        raise RuntimeError("paired request IDs differ")
    deltas = {label: [] for label in ("overall", "has_image", "no_image", "warm_item", "cold_item", "completely_unseen", "train_target_seen")}
    for request in requests:
        truth = set(map(int, request.ground_truth))
        has = {note for note in truth if image_mask[store.item_lookup[note]]}
        segments = {"overall": truth, "has_image": has, "no_image": truth - has,
                    "warm_item": truth & warm_items, "cold_item": truth - warm_items,
                    "completely_unseen": truth - train_vocab,
                    "train_target_seen": truth & train_targets}
        for label, selected in segments.items():
            if selected:
                deltas[label].append(len(selected & new[request.request_idx]) / len(selected)
                                     - len(selected & base[request.request_idx]) / len(selected))
    return {key: np.asarray(value, dtype=np.float64) for key, value in deltas.items()}


def bootstrap_array(deltas: np.ndarray, seed: int = 42, replicates: int = 2000) -> dict:
    if not len(deltas):
        return {"eligible_requests": 0, "status": "N/A", "point_delta": None,
                "ci95_lower": None, "ci95_upper": None, "replicates": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        sample = rng.integers(0, len(deltas), size=(count, len(deltas)))
        means[start:start + count] = deltas[sample].mean(axis=1)
    return {"eligible_requests": len(deltas), "point_delta": float(deltas.mean()),
            "ci95_lower": float(np.quantile(means, .025)),
            "ci95_upper": float(np.quantile(means, .975)), "replicates": replicates}


def deltas_for(name, seed, requests, store, mask, targets, vocab, warm):
    reference = control(seed)
    base = pd.read_parquet(ROOT / reference["ranking"])
    _, candidate = validation(name, seed)
    return paired_deltas(requests, base, candidate, store, mask, targets, vocab, warm)


def bootstrap_seed42():
    layout(); _, valid, store = resources()
    requests = grouped_requests(valid)
    mask = np.load(image_paths("i1_first")[1], mmap_mode="r")
    targets, vocab = set(map(int, store.train_target_item_ids)), set(map(int, store.train_item_id_vocab))
    warm, _, _ = legacy_evaluator_sets()
    results = {}
    for name in STRATEGIES:
        require_audit(name)
        deltas = deltas_for(name, 42, requests, store, mask, targets, vocab, warm)
        results[name] = {segment: bootstrap_array(values, seed=42 + index)
                         for index, (segment, values) in enumerate(deltas.items())}
    save_json(OUT / "validation/seed42_paired_bootstrap.json", results)


def select():
    if marker("seed42_selection.json").exists():
        raise SystemExit("seed42 strategy selection is immutable")
    path = OUT / "validation/seed42_paired_bootstrap.json"
    if not path.exists():
        raise SystemExit("seed42 bootstrap required")
    candidates = []
    for name in STRATEGIES:
        result, _ = validation(name, 42)
        score = result["metrics"]["overall"]
        candidates.append((float(score["Recall@500"]), float(score["Recall@100"]), name))
    best = max(candidates)
    save_json(marker("seed42_selection.json"), {"best_strategy": best[2],
              "seed42_scores": {name: score for score, _, name in candidates},
              "selection_rule": "max validation Recall@500; tie Recall@100, then name",
              "paired_bootstrap_sha256": sha(path), "test_opened": False})


def lock():
    if marker("final_decision.json").exists():
        raise SystemExit("final validation decision is immutable")
    selection = json.loads(marker("seed42_selection.json").read_text())
    name = selection["best_strategy"]
    audit_value = require_audit(name)
    _, valid, store = resources()
    requests = grouped_requests(valid)
    mask = np.load(image_paths(name)[1], mmap_mode="r")
    targets, vocab = set(map(int, store.train_target_item_ids)), set(map(int, store.train_item_id_vocab))
    warm, _, _ = legacy_evaluator_sets()
    scores, base_scores, paired = [], [], []
    for seed in SEEDS:
        result, _ = validation(name, seed)
        scores.append(float(result["metrics"]["overall"]["Recall@500"]))
        base_scores.append(float(control(seed)["metrics"]["overall"]["Recall@500"]))
        paired.append(deltas_for(name, seed, requests, store, mask, targets, vocab, warm))
    # Average the three same-request, same-seed paired deltas before request bootstrap.
    aggregated = {}
    for index, label in enumerate(paired[0]):
        arrays = [row[label] for row in paired]
        pooled = np.mean(np.stack(arrays), axis=0) if len(arrays[0]) else np.empty(0, np.float64)
        aggregated[label] = bootstrap_array(pooled, seed=100 + index)
    save_json(OUT / "validation/three_seed_paired_bootstrap.json", aggregated)
    for required in ("overall", "no_image", "completely_unseen"):
        if aggregated[required]["eligible_requests"] == 0:
            raise RuntimeError(f"required validation segment is empty: {required}")
    gates = {
        "practical_overall_gain": float(np.mean(scores) - np.mean(base_scores)) >= MIN_PRACTICAL_GAIN,
        "overall_ci_lower_positive": aggregated["overall"]["ci95_lower"] > 0,
        "no_image_noninferior": aggregated["no_image"]["ci95_lower"] >= -MAX_SEGMENT_DROP,
        "completely_unseen_noninferior": aggregated["completely_unseen"]["ci95_lower"] >= -MAX_SEGMENT_DROP,
    }
    middle = sorted(zip(scores, SEEDS))[1][1]
    done = json.loads(marker(f"full_validation_{name}_seed{middle}.json").read_text())
    trained = json.loads(completion(name, middle).read_text())
    decision = {"best_strategy": name, "scores": scores, "control_scores": base_scores,
                "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                "paired_bootstrap": aggregated, "gates": gates,
                "legacy_cold_item_note": "N/A: zero eligible validation requests; not a GO gate",
                "terminal_allowed": all(gates.values()), "seed": middle,
                "checkpoint": trained["checkpoint"], "checkpoint_sha256": trained["checkpoint_sha256"],
                "embedding": done["embedding"], "embedding_sha256": done["embedding_sha256"],
                "metadata": done["metadata"],
                "image_asset": audit_value["image_assets"][name], "test_opened": False}
    save_json(marker("final_decision.json"), decision)


def terminal_test(args):
    decision_path = marker("final_decision.json")
    decision = json.loads(decision_path.read_text())
    if not decision["terminal_allowed"]:
        raise SystemExit("validation No-Go; test remains unopened")
    manifest = marker("terminal_test_manifest.json")
    if manifest.exists():
        raise SystemExit("terminal test is one-shot and already opened")
    require_cuda(args.device)
    limits = resource_preflight(args)
    strategy, seed = decision["best_strategy"], decision["seed"]
    current_asset = verify_image_assets(strategy, full_hash=True)
    if current_asset != decision["image_asset"]:
        raise RuntimeError("image asset changed after decision")
    checkpoint, embedding = ROOT / decision["checkpoint"], ROOT / decision["embedding"]
    metadata = json.loads((ROOT / decision["metadata"]).read_text())
    if sha(checkpoint) != decision["checkpoint_sha256"] or sha(embedding) != decision["embedding_sha256"]:
        raise RuntimeError("terminal checkpoint or item vector hash mismatch")
    mapping = ROOT / metadata["mapping"]
    if (sha(mapping) != metadata["mapping_sha256"] or metadata["checkpoint_sha256"] != decision["checkpoint_sha256"]
            or metadata["embedding_sha256"] != decision["embedding_sha256"]
            or metadata["source_image_sha256"] != current_asset["image_sha256"]):
        raise RuntimeError("terminal mapping/vector/model contract mismatch")
    items = np.load(embedding, mmap_mode="r")
    candidate_ids = np.load(mapping, mmap_mode="r")
    if (items.shape != (PROTOCOL.corpus_items, DIM) or items.dtype != np.float32
            or candidate_ids.shape != (PROTOCOL.corpus_items,) or not np.array_equal(candidate_ids, np.load(IMAGE_OUT / "mappings/note_ids.npy", mmap_mode="r"))):
        raise RuntimeError("terminal candidate shape/dtype/mapping mismatch")
    save_json(manifest, {"status": "running", "test_reads": 1, "decision_sha256": sha(decision_path)})
    deadline = time.monotonic() + 1200
    requests = QilinData(ROOT).load_test_requests()
    _, _, store = resources()
    model, checkpoint_value, _ = load_best(strategy, seed, store, args.device)
    query_started = time.perf_counter()
    queries = encode_queries(model, requests, store, args.device, strategy, deadline=deadline,
                             num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                             limits=limits)
    query_seconds = time.perf_counter() - query_started
    _, raw, _, backend, timing = gpu_exact_search(items, queries, 600, args.device,
                                                   query_batch=256, deadline=deadline)
    rankings = filter_history(raw, candidate_ids, [r.history for r in requests], 500, deadline)
    validate_topk(rankings, candidate_ids, [r.history for r in requests], deadline)
    targets, vocab, users = set(map(int, store.train_target_item_ids)), set(map(int, store.train_item_id_vocab)), set(map(int, store.train_user_ids))
    exposed, clicked, warm_users = legacy_evaluator_sets()
    metrics, status, per = phase6_status_metrics(requests, rankings, targets, vocab,
                                                 users, strategy, exposed, clicked, warm_users, deadline)
    mask = np.load(image_paths(strategy)[1], mmap_mode="r")
    image_status = image_segment_metrics(requests, rankings, store, mask)
    # Reference test is read only after this experiment's model/strategy/seed
    # are locked and its own terminal ranking has already been evaluated.
    h2_terminal_path = H2 / "metrics/terminal_test.json"
    h2_terminal = json.loads(h2_terminal_path.read_text())
    h2_recall500 = float(h2_terminal["metrics"]["overall"]["Recall@500"])
    image_recall500 = float(metrics["overall"]["Recall@500"])
    result_path = OUT / "metrics/terminal_test.json"
    save_json(result_path, {"strategy": strategy, "seed": seed, "checkpoint_epoch": checkpoint_value["epoch"],
              "metrics": metrics, "phase6_item_status": status, "image_status": image_status,
              "h2_256_terminal_reference": {"path": str(h2_terminal_path.relative_to(ROOT)),
                                             "sha256": sha(h2_terminal_path),
                                             "seed": h2_terminal["seed"],
                                             "Recall@500": h2_recall500},
              "delta_recall500_vs_h2_terminal": image_recall500 - h2_recall500,
              "candidate_universe": len(candidate_ids), "query_encode_seconds": query_seconds,
              "search_backend": backend, **timing, "test_reads": 1})
    save_parquet_atomic(per, OUT / "metrics/terminal_test_rankings.parquet")
    save_json(manifest, {"status": "complete", "test_reads": 1,
              "result": str(result_path.relative_to(ROOT)), "decision_sha256": sha(decision_path)})


def report():
    def percent(value):
        return f"{value:.4%}" if value is not None and np.isfinite(value) else "N/A"

    def signed_percent(value):
        return f"{value:+.4%}" if value is not None and np.isfinite(value) else "N/A"

    lines = ["# Phase 8-02：冻结图片特征对 H2-256 全库召回的增益", "",
             "主对照为相同 seed 的 Phase 7-04 H2-256；未重新训练 C0。", "",
             "| Model | Seed | R@100 | R@500 | MRR@100 | Has-image R@500 | No-image R@500 | Warm R@500 | Cold R@500 | Unseen R@500 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    engineering = []
    for seed in SEEDS:
        reference = control(seed)
        value = json.loads((ROOT / reference["result"]).read_text())
        # Image/no-image control strata are computed from the existing C0 ranking,
        # never from a re-run of C0 or from the test set.
        if seed == 42 or marker("final_decision.json").exists():
            _, valid, store = resources()
            requests = grouped_requests(valid)
            mask = np.load(image_paths("i1_first")[1], mmap_mode="r")
            base = pd.read_parquet(ROOT / reference["ranking"])
            rankings_by_request = {int(row.request_idx): row.retrieved_top500 for row in base.itertuples(index=False)}
            if len(rankings_by_request) != len(requests):
                raise RuntimeError("C0 per-request ranking has missing or duplicate request IDs")
            image = image_segment_metrics(
                requests, [rankings_by_request[request.request_idx] for request in requests], store, mask
            )
            m, w, s = (value["metrics"]["overall"], value["metrics"]["warm_item"],
                       value["metrics"]["cold_item"])
            u = value["phase6_item_status"]["completely_unseen"]
            lines.append(f"| C0 H2-256 | {seed} | {percent(m['Recall@100'])} | {percent(m['Recall@500'])} | {percent(m['MRR@100'])} | {percent(image['has_image']['Recall@500'])} | {percent(image['no_image']['Recall@500'])} | {percent(w['Recall@500'])} | {percent(s['Recall@500'])} | {percent(u['Recall@500'])} |")
        for name in STRATEGIES:
            path = marker(f"full_validation_{name}_seed{seed}.json")
            if not path.exists():
                continue
            candidate, _ = validation(name, seed)
            m, w, s, u, image = (candidate["metrics"]["overall"], candidate["metrics"]["warm_item"], candidate["metrics"]["cold_item"],
                              candidate["phase6_item_status"]["completely_unseen"], candidate["image_status"])
            lines.append(f"| {name} | {seed} | {percent(m['Recall@100'])} | {percent(m['Recall@500'])} | {percent(m['MRR@100'])} | {percent(image['has_image']['Recall@500'])} | {percent(image['no_image']['Recall@500'])} | {percent(w['Recall@500'])} | {percent(s['Recall@500'])} | {percent(u['Recall@500'])} |")
            engineering.append((name, seed, candidate))
    if engineering:
        lines += ["", "## 资源与参数", "",
                  "| Model | Seed | Image alpha | Params | Trainable | Best epoch | Train (s) | ΔTrain vs C0 (s) | Checkpoint (MiB) | Item encode (s) | Index build (s) | Search (s) | GPU peak (GiB) | Index/vector (GiB) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, seed, candidate in engineering:
            p = candidate["parameters"]
            own_curve = pd.read_csv(OUT / f"training_curves/{name}_seed{seed}.csv")
            base_curve = pd.read_csv(H2 / f"metrics/training_curves/h2_256_seed{seed}.csv")
            own_seconds, base_seconds = float(own_curve.seconds.sum()), float(base_curve.seconds.sum())
            checkpoint_size = (OUT / f"checkpoints/{name}_seed{seed}_best.pt").stat().st_size / 2**20
            lines.append(f"| {name} | {seed} | {candidate['alpha']['image']:.5f} | {p['total_parameters']:,} | {p['trainable_parameters']:,} | {candidate['checkpoint_epoch']} | {own_seconds:.1f} | {own_seconds-base_seconds:+.1f} | {checkpoint_size:.1f} | {candidate['item_encode_seconds']:.1f} | {candidate['index_build_seconds']:.2f} | {candidate['query_search_seconds']:.2f} | {candidate['gpu_peak_memory_bytes']/2**30:.2f} | {candidate['item_vector_bytes']/2**30:.2f} |")
        lines += ["", "训练总耗时包含实际执行 epoch，若 early stopping epoch 数不同，ΔTrain 同时含 epoch 数差异；逐 epoch 吞吐与 data_wait_fraction 见 `training_curves/`。"]
    seed42_bootstrap = OUT / "validation/seed42_paired_bootstrap.json"
    if seed42_bootstrap.exists():
        values = json.loads(seed42_bootstrap.read_text())
        lines += ["", "## Seed42 同 request 配对 bootstrap", "",
                  "| Strategy | ΔOverall R@500 | 95% CI | ΔNo-image | ΔCold |",
                  "|---|---:|---:|---:|---:|"]
        for name in STRATEGIES:
            row = values[name]
            lines.append(f"| {name} | {signed_percent(row['overall']['point_delta'])} | [{signed_percent(row['overall']['ci95_lower'])}, {signed_percent(row['overall']['ci95_upper'])}] | {signed_percent(row['no_image']['point_delta'])} | {signed_percent(row['cold_item']['point_delta'])} |")
    decision_path = marker("final_decision.json")
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        control_mean = float(np.mean(decision["control_scores"]))
        lines += ["", "## 预注册门禁", "",
                  f"- 最佳策略：`{decision['best_strategy']}`；三 seed mean R@500：{decision['mean']:.4%}，同 seed C0 均值：{control_mean:.4%}，差值：{decision['mean']-control_mean:+.4%}；图片模型 std：{decision['std']:.4%}。",
                  f"- GO：`{decision['terminal_allowed']}`；门禁：`{json.dumps(decision['gates'], ensure_ascii=False)}`。",
                  "", "| 配对分层 | Eligible requests | ΔR@500 | 95% CI |", "|---|---:|---:|---:|"]
        for segment in ("overall", "has_image", "no_image", "train_target_seen", "completely_unseen", "cold_item"):
            row = decision["paired_bootstrap"][segment]
            lines.append(f"| {segment} | {row['eligible_requests']:,} | {signed_percent(row['point_delta'])} | [{signed_percent(row['ci95_lower'])}, {signed_percent(row['ci95_upper'])}] |")
        if not decision["terminal_allowed"]:
            lines += ["", "**结论：No-Go。** 图片对有图正样本有正增益，但总体增益未达到预设 +0.1pp，且总体 CI 跨零；无图正样本显著退化，completely-unseen 未通过预设非劣效界。保留 Phase 8-01 图片资产，不将图片分支并入当前 H2-256；按协议不读取 terminal test。"]
    else:
        lines += ["", "尚未完成三 seed validation 锁定；不得读取 test。"]
    terminal = OUT / "metrics/terminal_test.json"
    if terminal.exists():
        value = json.loads(terminal.read_text())
        reference = value["h2_256_terminal_reference"]
        lines += ["", "## Terminal test（仅门禁通过后）", "",
                  f"- 图片模型 seed{value['seed']} R@100/R@500：{value['metrics']['overall']['Recall@100']:.4%}/{value['metrics']['overall']['Recall@500']:.4%}。",
                  f"- 已锁定 H2-256 seed{reference['seed']} terminal R@500：{reference['Recall@500']:.4%}；图片模型差值：{value['delta_recall500_vs_h2_terminal']:+.4%}（仅终局观察，不用于选择）。"]
    lines += ["", "## 产物", "", f"- 结果：`{OUT.relative_to(ROOT)}`。",
              "- Phase 8-01 pooled image 资产只读；本实验未重新运行 SigLIP。",
              "- 本实验不开展独立图片召回、搜索多任务或粗排。", ""]
    path = OUT / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    authorize(args)
    if args.stage == "plan":
        plan()
        return
    if args.stage in {"smoke", "train", "full-validation", "terminal-test"}:
        require_cuda(args.device)
    actions = {"audit": audit, "smoke": smoke, "train": train,
               "full-validation": full_validation,
               "bootstrap-seed42": lambda _: bootstrap_seed42(),
               "select": lambda _: select(), "lock": lambda _: lock(),
               "terminal-test": terminal_test, "report": lambda _: report()}
    actions[args.stage](args)


if __name__ == "__main__":
    main()
