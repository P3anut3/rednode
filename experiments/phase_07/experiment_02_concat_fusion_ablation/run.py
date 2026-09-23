#!/usr/bin/env python3
"""Phase 7 Experiment 02: ID64 concat-fusion ablation.

The default stage is a read-only plan.  No data scan, training, validation, or
test is allowed without an explicit ``--confirm-run`` after human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import (  # noqa: E402
    Phase7Collator,
    Phase7Dataset,
    load_training_frame,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.evaluation import (  # noqa: E402
    feature_slices,
    paired_bootstrap_segments,
    phase6_status_metrics,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.features import FeatureStore  # noqa: E402
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.retrieval import (  # noqa: E402
    deterministic_proxy_candidates,
    filter_history,
    gpu_exact_search,
    validate_topk,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.run import (  # noqa: E402
    json_default,
    legacy_evaluator_sets,
    resources,
    save_csv_atomic,
    save_json,
    save_parquet_atomic,
    save_torch_atomic,
    select_requests,
)

from experiments.phase_07.experiment_02_concat_fusion_ablation.config import (  # noqa: E402
    CONDITIONAL_MODEL,
    EXP01_OUT,
    MODEL_NAMES,
    OUT,
    PROTOCOL,
    DeadlineExceeded,
    check_deadline,
    ensure_output_layout,
)
from experiments.phase_07.experiment_02_concat_fusion_ablation.models import ConcatFusionTwoTower  # noqa: E402
from experiments.phase_07.experiment_02_concat_fusion_ablation.trainer import (  # noqa: E402
    encode_items,
    encode_queries,
    make_model,
    pack_item_features,
    parameter_report,
    to_device,
    train_epoch,
    validation_residual_diagnostics,
)


MUTATING_STAGES = {
    "smoke", "train", "full-validation", "select", "decision", "lock",
    "terminal-test", "report"
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("plan", "smoke", "train", "full-validation", "select", "decision", "lock", "terminal-test", "report"),
        default="plan",
    )
    parser.add_argument("--model", choices=(*MODEL_NAMES, CONDITIONAL_MODEL))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=PROTOCOL.batch_size)
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args()


def authorize(args) -> None:
    if args.stage in MUTATING_STAGES and not args.confirm_run:
        raise SystemExit(f"REFUSED: {args.stage} requires --confirm-run after review")
    if args.stage in {"smoke", "train", "full-validation"} and not args.model:
        raise SystemExit(f"--stage {args.stage} requires --model")
    if args.seed not in {42, 43, 44}:
        raise SystemExit("formal seeds are fixed to 42/43/44")
    if args.stage in {"train", "full-validation"} and args.seed in {43, 44}:
        selection = OUT / "locks/selected_structures.json"
        if not selection.exists() or args.model not in json.loads(selection.read_text())["multi_seed_models"]:
            raise SystemExit("seed43/44 requires validation selection lock")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def plan() -> None:
    print(json.dumps({
        "status": "implementation_only_waiting_for_review",
        "models": list(MODEL_NAMES),
        "protocol": PROTOCOL.to_dict(),
        "reuse": "Experiment 01 train-only cache, dataset, HN loss, evaluator and retrieval",
        "execution": ["smoke M1/M2", "train seed42", "serial full-validation", "select", "winner seed43/44", "lock", "one-shot terminal-test"],
        "test_opened": False,
    }, ensure_ascii=False, indent=2))


def _parameter_grad(model, prefix: str) -> bool:
    return any(
        name.startswith(prefix) and p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad).item() > 0
        for name, p in model.named_parameters()
    )


def structural_audit(model: ConcatFusionTwoTower, dataset, device: str) -> dict:
    cpu = Phase7Collator(True)([dataset[i] for i in range(8)])
    batch = to_device(cpu, device)
    model.eval()
    with torch.inference_mode():
        query = model.query(batch)
        item = model.encode_item(batch, "target")
        initial = {}
        if model.residual:
            item_parts = model.item_components(batch["target_content"], batch["target_item_id_row"], batch["target_categorical"], batch["target_numeric"])
            user_parts = model.user_components(batch)
            initial = {
                "initial_item_vs_normalized_content_max_abs": float((item_parts["final"] - torch.nn.functional.normalize(item_parts["base"], dim=-1)).abs().max()),
                "initial_user_vs_normalized_history_max_abs": float((user_parts["final"] - torch.nn.functional.normalize(user_parts["base"], dim=-1)).abs().max()),
            }
        cold_item_batch = dict(batch)
        cold_item_batch["target_item_id_row"] = torch.zeros_like(batch["target_item_id_row"])
        item_before = model.encode_item(cold_item_batch, "target")
        item_saved = model.item_id.weight[1:].clone()
        model.item_id.weight[1:].normal_()
        item_after = model.encode_item(cold_item_batch, "target")
        model.item_id.weight[1:].copy_(item_saved)

        # Keep history Item IDs untouched: this check isolates only the cold
        # user's OOV User-ID branch.
        cold_user_batch = dict(batch)
        cold_user_batch["user_id_row"] = torch.zeros_like(batch["user_id_row"])
        user_before = model.query(cold_user_batch)
        user_saved = model.user_id.weight[1:].clone()
        model.user_id.weight[1:].normal_()
        user_after = model.query(cold_user_batch)
        model.user_id.weight[1:].copy_(user_saved)
    required = (
        (
            "content", "item_meta", "user_profile", "item_id", "user_id",
            "attention", "history_mlp", "item_id_projection",
            "user_id_projection", "alpha_meta_raw", "alpha_profile_raw",
            "alpha_item_id_raw", "alpha_user_id_raw",
        )
        if getattr(model, "additive", False)
        else (
            "content", "item_meta", "user_profile", "item_id", "user_id",
            "attention", "history_mlp", "item_fusion", "user_fusion",
        )
    )
    gradients = {name: _parameter_grad(model, name) for name in required}
    if model.residual:
        gradients.update(beta_item=_parameter_grad(model, "beta_item_raw"), beta_user=_parameter_grad(model, "beta_user_raw"))
    padding_gradient_zero = all(
        module.weight.grad is None or torch.count_nonzero(module.weight.grad[0]).item() == 0
        for module in model.modules() if isinstance(module, torch.nn.Embedding) and module.padding_idx == 0
    )
    negative_ids = batch["negative_note_ids"]
    target_ids = batch["target_note_id"][:, None]
    negative_valid = batch["negative_mask"]
    negative_hits_target = negative_valid & negative_ids.eq(target_ids)
    negative_hits_history = negative_valid & (
        negative_ids[:, :, None].eq(batch["history_note_ids"][:, None, :])
    ).any(-1)
    negative_hits_same_request = torch.zeros_like(negative_valid)
    for row, positives in enumerate(batch["same_request_positive_ids"]):
        if positives:
            positive_tensor = torch.as_tensor(positives, device=device)
            negative_hits_same_request[row] = negative_valid[row] & (
                negative_ids[row, :, None].eq(positive_tensor[None, :])
            ).any(-1)
    return {
        "id_dim_64": model.item_id.embedding_dim == model.user_id.embedding_dim == 64,
        "item_id_rows": model.item_id.num_embeddings,
        "user_id_rows": model.user_id.num_embeddings,
        "id_vocab_shape_matches_contract": (
            model.item_id.num_embeddings == PROTOCOL.item_id_rows
            and model.user_id.num_embeddings == PROTOCOL.user_id_rows
        ),
        "query_shape": list(query.shape),
        "item_shape": list(item.shape),
        "query_norm_max_error": float((query.norm(dim=-1) - 1).abs().max()),
        "item_norm_max_error": float((item.norm(dim=-1) - 1).abs().max()),
        "bge_external_frozen_memmap": not any("bge" in name.lower() for name, _ in model.named_parameters()),
        "shared_history_candidate_item_tower": True,
        "branch_gradients": gradients,
        "all_required_gradients": all(gradients.values()),
        "padding_gradient_zero": padding_gradient_zero,
        "cold_item_id_invariance_max_abs": float((item_before - item_after).abs().max()),
        "cold_user_id_invariance_max_abs": float((user_before - user_after).abs().max()),
        "hard_negative_valid_count": int(batch["negative_mask"].sum()),
        "hard_negative_target_overlap": int(negative_hits_target.sum()),
        "hard_negative_history_overlap": int(negative_hits_history.sum()),
        "hard_negative_same_request_positive_overlap": int(negative_hits_same_request.sum()),
        **initial,
    }


def proxy_evaluate(model, requests, store, device, proxy_ids=None, packed=None, deadline=None):
    catalog = np.asarray(store.item_ids)
    if proxy_ids is None:
        positives = set().union(*(set(r.ground_truth) for r in requests)) & set(map(int, catalog))
        proxy_ids = deterministic_proxy_candidates(catalog, positives, PROTOCOL.proxy_candidates, PROTOCOL.seed)
    items = encode_items(model, proxy_ids, store, device, deadline=deadline, packed=packed)
    queries = encode_queries(model, requests, store, device, deadline=deadline)
    _, rows, seconds, backend, timing = gpu_exact_search(items, queries, min(PROTOCOL.overfetch, len(proxy_ids)), device, query_batch=512, deadline=deadline)
    rankings = filter_history(rows, proxy_ids, [r.history for r in requests], PROTOCOL.topk, deadline)
    validate_topk(rankings, proxy_ids, [r.history for r in requests], deadline=deadline)
    metrics, _ = evaluate_rankings(requests, rankings, set(), set(), "proxy", deadline=deadline)
    return {
        "proxy_Recall@100": metrics["overall"]["Recall@100"],
        "proxy_Recall@500": metrics["overall"]["Recall@500"],
        "proxy_MRR@100": metrics["overall"]["MRR@100"],
        "proxy_candidates": len(proxy_ids),
        "search_seconds": seconds,
        "search_backend": backend,
        **timing,
    }


def smoke(args) -> None:
    seed_all(args.seed)
    started, deadline = time.perf_counter(), time.monotonic() + 600
    train_frame, valid, store = resources()
    frame = load_training_frame(True)
    dataset = Phase7Dataset(frame, store, True, PROTOCOL.history_n, args.seed, PROTOCOL.smoke_train_samples)
    model = make_model(args.model, store, args.device)
    initial_audit = structural_audit(model, dataset, args.device) if model.residual else {}
    # M2 has an exactly zero final fusion layer. Two or more updates are required
    # before beta receives a mathematically non-zero gradient.
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL.learning_rate, weight_decay=PROTOCOL.weight_decay)
    stats = train_epoch(model, dataset, optimizer, args.device, 1, min(args.batch_size, 256), 8, deadline)
    audit = structural_audit(model, dataset, args.device)
    if model.residual:
        audit["initial_item_vs_normalized_content_max_abs"] = initial_audit["initial_item_vs_normalized_content_max_abs"]
        audit["initial_user_vs_normalized_history_max_abs"] = initial_audit["initial_user_vs_normalized_history_max_abs"]
    requests = select_requests(grouped_requests(valid), PROTOCOL.smoke_valid_requests, PROTOCOL.seed)
    proxy = proxy_evaluate(model, requests, store, args.device, deadline=deadline)
    passed = bool(
        np.isfinite(stats["loss"])
        and audit["id_dim_64"]
        and audit["id_vocab_shape_matches_contract"]
        and audit["query_shape"][1] == audit["item_shape"][1] == 128
        and audit["all_required_gradients"]
        and audit["padding_gradient_zero"]
        and audit["cold_item_id_invariance_max_abs"] <= 1e-6
        and audit["cold_user_id_invariance_max_abs"] <= 1e-6
        and audit["query_norm_max_error"] <= 1e-4
        and audit["item_norm_max_error"] <= 1e-4
        and audit["hard_negative_valid_count"] > 0
        and audit["hard_negative_target_overlap"] == 0
        and audit["hard_negative_history_overlap"] == 0
        and audit["hard_negative_same_request_positive_overlap"] == 0
        and (not model.residual or audit["initial_item_vs_normalized_content_max_abs"] <= 1e-6)
        and (not model.residual or audit["initial_user_vs_normalized_history_max_abs"] <= 1e-6)
    )
    payload = {
        "model": args.model, "seed": args.seed, "train": stats, "proxy": proxy,
        "audit": audit, "parameters": parameter_report(model), "smoke_passed": passed,
        "elapsed_seconds": time.perf_counter() - started, "test_opened": False,
    }
    save_json(OUT / f"smoke/{args.model}_seed{args.seed}.json", payload)
    if not passed:
        raise RuntimeError("smoke failed; training remains blocked")
    print(json.dumps(payload, indent=2, default=json_default))


def _train_impl(args) -> None:
    completion = OUT / f"locks/training_complete_{args.model}_seed{args.seed}.json"
    validation_marker = OUT / f"locks/full_validation_{args.model}_seed{args.seed}.json"
    if completion.exists() or validation_marker.exists():
        raise SystemExit("training artifact is immutable after completion/validation")
    smoke_path = OUT / f"smoke/{args.model}_seed{args.seed}.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text())["smoke_passed"]:
        raise SystemExit("successful smoke required")
    seed_all(args.seed)
    started, deadline = time.perf_counter(), time.monotonic() + 3600
    _, valid, store = resources()
    frame = load_training_frame(True)
    dataset = Phase7Dataset(frame, store, True, PROTOCOL.history_n, args.seed)
    requests = select_requests(grouped_requests(valid), PROTOCOL.proxy_requests, PROTOCOL.seed)
    catalog = np.asarray(store.item_ids)
    def build_proxy(selected):
        positives = set().union(*(set(r.ground_truth) for r in selected)) & set(map(int, catalog))
        ids = deterministic_proxy_candidates(catalog, positives, PROTOCOL.proxy_candidates, PROTOCOL.seed)
        return ids, pack_item_features(ids, store)
    proxy_ids, proxy_pack = build_proxy(requests)
    model = make_model(args.model, store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL.learning_rate, weight_decay=PROTOCOL.weight_decay)
    best, stale, curves = -1.0, 0, []
    best_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_best.pt"
    last_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_last.pt"
    for epoch in range(1, PROTOCOL.epochs + 1):
        epoch_started = time.perf_counter()
        try:
            stats = train_epoch(model, dataset, optimizer, args.device, epoch, args.batch_size, deadline=deadline)
            proxy = proxy_evaluate(
                model, requests, store, args.device, proxy_ids, proxy_pack, deadline
            )
        except DeadlineExceeded:
            save_torch_atomic({
                "state_dict": model.state_dict(), "model": args.model,
                "seed": args.seed, "epoch": epoch, "aborted_by_deadline": True,
                "protocol": PROTOCOL.to_dict(),
            }, last_path)
            save_json(OUT / f"timing/{args.model}_seed{args.seed}_deadline.json", {
                "stage": "train", "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - started,
            })
            raise
        row = {"model": args.model, "seed": args.seed, "epoch": epoch, **stats, **proxy, "epoch_seconds": time.perf_counter() - epoch_started}
        curves.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {"state_dict": model.state_dict(), "model": args.model, "seed": args.seed, "epoch": epoch, "proxy_recall500": proxy["proxy_Recall@500"], "protocol": PROTOCOL.to_dict(), "parameters": parameter_report(model)}
        save_torch_atomic(checkpoint, last_path)
        if proxy["proxy_Recall@500"] > best:
            best, stale = proxy["proxy_Recall@500"], 0
            save_torch_atomic(checkpoint, best_path)
        else:
            stale += 1
            if stale >= PROTOCOL.patience:
                break
    save_csv_atomic(pd.DataFrame(curves), OUT / f"training_curves/{args.model}_seed{args.seed}.csv")
    save_json(OUT / f"configs/{args.model}_seed{args.seed}.json", {
        "model": args.model, "seed": args.seed, "best_proxy_recall500": best,
        "best_checkpoint": str(best_path.relative_to(ROOT)), "loss": "Phase5 TF-IDF hard-negative protocol",
        "parameters": parameter_report(model), "elapsed_seconds": time.perf_counter() - started,
        "test_opened": False,
    })
    if not best_path.exists() or not last_path.exists():
        raise RuntimeError("normal training ended without best/last checkpoint")
    best_sha256, last_sha256 = sha256_file(best_path), sha256_file(last_path)
    save_json(completion, {
        "complete": True,
        "deadline_aborted": False,
        "model": args.model,
        "seed": args.seed,
        "epochs_completed": len(curves),
        "training_curve": str((OUT / f"training_curves/{args.model}_seed{args.seed}.csv").relative_to(ROOT)),
        "config": str((OUT / f"configs/{args.model}_seed{args.seed}.json").relative_to(ROOT)),
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": best_sha256,
        "last_checkpoint_sha256": last_sha256,
        "fixed_proxy_requests": len(requests),
        "fixed_proxy_candidates": len(proxy_ids),
        "test_opened": False,
    })


def train(args) -> None:
    active = OUT / f"locks/training_active_{args.model}_seed{args.seed}.lock"
    try:
        descriptor = os.open(active, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit("the same model/seed training job is already active") from exc
    try:
        os.write(descriptor, f"{args.model} seed={args.seed}\n".encode())
        os.close(descriptor)
        _train_impl(args)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        active.unlink(missing_ok=True)


def load_best(name, seed, store, device):
    checkpoint = torch.load(OUT / f"checkpoints/{name}_seed{seed}_best.pt", map_location=device, weights_only=False)
    model = make_model(name, store, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def verified_training(name: str, seed: int) -> tuple[Path, dict, str]:
    marker_path = OUT / f"locks/training_complete_{name}_seed{seed}.json"
    if not marker_path.exists():
        raise SystemExit("full validation requires a training completion marker")
    marker = json.loads(marker_path.read_text())
    if not marker.get("complete") or marker.get("deadline_aborted"):
        raise SystemExit("training marker is incomplete or deadline-aborted")
    curve = ROOT / marker["training_curve"]
    config = ROOT / marker["config"]
    checkpoint = ROOT / marker["best_checkpoint"]
    if not curve.exists() or not config.exists() or not checkpoint.exists():
        raise SystemExit("training marker references a missing artifact")
    curve_frame = pd.read_csv(curve)
    config_value = json.loads(config.read_text())
    if len(curve_frame) != int(marker["epochs_completed"]) or len(curve_frame) == 0:
        raise SystemExit("training curve is incomplete relative to completion marker")
    if (
        config_value.get("model") != name
        or int(config_value.get("seed", -1)) != seed
        or config_value.get("best_checkpoint") != marker["best_checkpoint"]
    ):
        raise SystemExit("training config does not match completion marker")
    actual = sha256_file(checkpoint)
    if actual != marker["best_checkpoint_sha256"]:
        raise SystemExit("best checkpoint SHA-256 does not match training marker")
    return checkpoint, marker, actual


def acquire_full_validation_lock() -> int:
    path = OUT / "locks/full_validation_active.lock"
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit("another full validation is active; runs must be serial") from exc


def full_validation(args) -> None:
    marker = OUT / f"locks/full_validation_{args.model}_seed{args.seed}.json"
    if marker.exists():
        raise SystemExit("completed full validation is immutable")
    _, training_marker, checkpoint_sha256 = verified_training(args.model, args.seed)
    descriptor = acquire_full_validation_lock()
    lock_path = OUT / "locks/full_validation_active.lock"
    try:
        os.write(descriptor, f"{args.model} seed={args.seed}\n".encode())
        os.close(descriptor)
        started, deadline = time.perf_counter(), time.monotonic() + 1200
        train_frame, valid, store = resources()
        requests = grouped_requests(valid)
        model, checkpoint = load_best(args.model, args.seed, store, args.device)
        item_started = time.perf_counter()
        candidate_ids = np.asarray(store.item_ids)
        items = encode_items(model, candidate_ids, store, args.device, deadline=deadline)
        item_seconds = time.perf_counter() - item_started
        embedding_path = OUT / f"embeddings/{args.model}/seed{args.seed}_item_vectors.f32.npy"
        save_npy_atomic(embedding_path, items)
        save_json(OUT / f"embeddings/{args.model}/seed{args.seed}_metadata.json", {
            "model": args.model, "seed": args.seed, "checkpoint_epoch": checkpoint["epoch"],
            "shape": list(items.shape), "dtype": str(items.dtype), "normalized": True,
            "candidate_mapping": str((EXP01_OUT / "cache/item_ids.npy").relative_to(ROOT)),
            "candidate_mapping_sha256": sha256_file(EXP01_OUT / "cache/item_ids.npy"),
            "checkpoint": training_marker["best_checkpoint"],
            "checkpoint_sha256": checkpoint_sha256,
            "test_opened": False,
        })
        query_started = time.perf_counter()
        queries = encode_queries(model, requests, store, args.device, deadline=deadline)
        query_seconds = time.perf_counter() - query_started
        _, raw, search_seconds, backend, search_timing = gpu_exact_search(items, queries, PROTOCOL.overfetch, args.device, deadline=deadline)
        rankings = filter_history(raw, candidate_ids, [r.history for r in requests], PROTOCOL.topk, deadline)
        validate_topk(rankings, candidate_ids, [r.history for r in requests], deadline=deadline)
        targets = set(map(int, store.train_target_item_ids))
        vocab = set(map(int, store.train_item_id_vocab))
        users = set(map(int, store.train_user_ids))
        old_exposed, old_clicked, old_users = legacy_evaluator_sets()
        metrics, status, per = phase6_status_metrics(requests, rankings, targets, vocab, users, args.model, old_exposed, old_clicked, old_users, deadline)
        save_parquet_atomic(per, OUT / f"validation_rankings/{args.model}/seed{args.seed}.parquet")
        counts = train_frame.positive_item_id.value_counts().to_dict()
        save_csv_atomic(feature_slices(requests, rankings, store, counts), OUT / f"validation/{args.model}_seed{args.seed}_feature_slices.csv")
        residual = validation_residual_diagnostics(
            model, requests, store, targets, vocab, users, args.device,
            seed=args.seed, deadline=deadline
        )
        save_json(OUT / f"diagnostics/{args.model}_seed{args.seed}_residual.json", residual)
        payload = {
            "model": args.model, "seed": args.seed, "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": checkpoint_sha256,
            "metrics": metrics, "temporal_item_status": status,
            "parameters": parameter_report(model), "residual_diagnostics": residual,
            "checkpoint_size_bytes": (OUT / f"checkpoints/{args.model}_seed{args.seed}_best.pt").stat().st_size,
            "candidate_universe": len(candidate_ids), "item_encode_seconds": item_seconds,
            "query_encode_seconds": query_seconds, "search_seconds": search_seconds,
            "mean_query_search_ms": search_timing["query_search_seconds"] * 1000 / len(requests),
            "search_backend": backend, **search_timing,
            "elapsed_seconds": time.perf_counter() - started, "test_opened": False,
        }
        save_json(OUT / f"validation/{args.model}_seed{args.seed}.json", payload)
        save_json(OUT / f"timing/{args.model}_seed{args.seed}_full_validation.json", {
            key: payload[key] for key in (
                "item_encode_seconds", "query_encode_seconds", "index_build_seconds",
                "query_search_seconds", "mean_query_search_ms", "elapsed_seconds",
            )
        })
        if args.seed == 42 and args.model in MODEL_NAMES:
            other = MODEL_NAMES[1] if args.model == MODEL_NAMES[0] else MODEL_NAMES[0]
            other_path = OUT / f"validation_rankings/{other}/seed42.parquet"
            other_marker_path = OUT / f"locks/full_validation_{other}_seed42.json"
            other_complete = (
                json.loads(other_marker_path.read_text())
                if other_marker_path.exists()
                else {}
            )
            if (
                other_path.exists()
                and other_complete.get("complete")
                and other_complete.get("all_artifacts_complete")
            ):
                base_name, candidate_name = MODEL_NAMES[0], MODEL_NAMES[1]
                base = pd.read_parquet(OUT / f"validation_rankings/{base_name}/seed42.parquet")
                candidate = pd.read_parquet(OUT / f"validation_rankings/{candidate_name}/seed42.parquet")
                bootstrap = paired_bootstrap_segments(base, candidate, requests, ("overall", "warm", "cold", "train_target_seen", "train_history_only", "completely_unseen"), old_exposed, targets, vocab, deadline=deadline)
                save_json(OUT / "validation/m2_vs_m1_bootstrap.json", bootstrap)
        save_json(marker, {
            "complete": True,
            "all_artifacts_complete": True,
            "checkpoint_sha256": checkpoint_sha256,
            "embedding": str(embedding_path.relative_to(ROOT)),
            "test_opened": False,
        })
    except Exception:
        raise
    finally:
        if descriptor:
            try:
                os.close(descriptor)
            except OSError:
                pass
        lock_path.unlink(missing_ok=True)


def select() -> None:
    values, segments = {}, {}
    for model in MODEL_NAMES:
        path = OUT / f"validation/{model}_seed42.json"
        marker = OUT / f"locks/full_validation_{model}_seed42.json"
        if not path.exists() or not marker.exists():
            raise SystemExit("both seed42 full validations are required")
        marker_value = json.loads(marker.read_text())
        if not marker_value.get("complete") or not marker_value.get(
            "all_artifacts_complete"
        ):
            raise SystemExit(f"incomplete full validation marker: {model}")
        result = json.loads(path.read_text())
        if result.get("checkpoint_sha256") != marker_value.get("checkpoint_sha256"):
            raise SystemExit(f"validation result/checkpoint hash mismatch: {model}")
        values[model] = float(result["metrics"]["overall"]["Recall@500"])
        segments[model] = {
            "warm": float(result["temporal_item_status"]["train_target_seen"]["Recall@500"]),
            "cold": float(result["temporal_item_status"]["completely_unseen"]["Recall@500"]),
        }
    m1, m2 = MODEL_NAMES
    if (
        values[m2] >= values[m1]
        and segments[m2]["warm"] >= segments[m1]["warm"] - 0.001
        and segments[m2]["cold"] > segments[m1]["cold"]
    ):
        winner, rationale = m2, "M2 overall>=M1, temporal-warm non-inferior, temporal-cold better"
    elif (
        values[m1] > values[m2]
        and segments[m1]["warm"] > segments[m2]["warm"]
        and segments[m1]["cold"] >= segments[m2]["cold"] - 0.001
    ):
        winner, rationale = m1, "M1 overall/warm better, temporal-cold within 0.1pp"
    else:
        winner = max(values, key=values.get)
        rationale = "predefined dominance rules inconclusive; use overall R@500 and retain close-model multi-seed audit"
    close = abs(values[MODEL_NAMES[0]] - values[MODEL_NAMES[1]]) <= 0.001
    multi = list(MODEL_NAMES) if close else [winner]
    h2_validation = float(json.loads((EXP01_OUT / "validation/h2_side_features_id_seed42.json").read_text())["metrics"]["overall"]["Recall@500"])
    save_json(OUT / "locks/selected_structures.json", {
        "winner_seed42": winner, "selection_rationale": rationale,
        "seed42_recall500": values, "seed42_temporal_segments": segments,
        "close_within_0.1pp": close,
        "multi_seed_models": multi, "existing_h2_recall500": h2_validation,
        "add_id64_eligible": all(score < h2_validation - 0.001 for score in values.values()),
        "add_id64_control": CONDITIONAL_MODEL,
        "add_id64_seed_policy": "seed42_only_conditional_control",
        "test_opened": False,
    })


def lock_final() -> None:
    selection = json.loads((OUT / "locks/selected_structures.json").read_text())
    candidates = {}
    for model in selection["multi_seed_models"]:
        rows = []
        for seed in (42, 43, 44):
            path = OUT / f"validation/{model}_seed{seed}.json"
            marker = OUT / f"locks/full_validation_{model}_seed{seed}.json"
            if not path.exists() or not marker.exists():
                raise SystemExit(f"missing full validation: {model} seed{seed}")
            result = json.loads(path.read_text())
            marker_value = json.loads(marker.read_text())
            if (
                not marker_value.get("complete")
                or not marker_value.get("all_artifacts_complete")
                or result.get("checkpoint_sha256")
                != marker_value.get("checkpoint_sha256")
            ):
                raise SystemExit(
                    f"incomplete or hash-mismatched validation: {model} seed{seed}"
                )
            embedding = ROOT / marker_value["embedding"]
            if not embedding.exists():
                raise SystemExit(f"missing locked embedding: {model} seed{seed}")
            rows.append(
                (
                    seed,
                    float(result["metrics"]["overall"]["Recall@500"]),
                    marker_value,
                )
            )
        candidates[model] = rows
    winner = max(
        candidates,
        key=lambda name: np.mean([score for _, score, _ in candidates[name]]),
    )
    rows = candidates[winner]
    median_seed, median_score, validation_marker = sorted(
        rows, key=lambda value: value[1]
    )[1]
    save_json(OUT / "locks/final_model.json", {
        "model": winner, "seed": median_seed, "validation_recall500": median_score,
        "three_seed_mean_recall500": float(np.mean([x[1] for x in rows])),
        "three_seed_std_recall500": float(np.std([x[1] for x in rows])),
        "three_seed_results": [
            {"seed": seed, "Recall@500": score}
            for seed, score, _ in rows
        ],
        "checkpoint": f"checkpoints/{winner}_seed{median_seed}_best.pt",
        "checkpoint_sha256": validation_marker["checkpoint_sha256"],
        "embedding": validation_marker["embedding"],
        "selection_rule": "best three-seed mean; terminal uses median seed", "test_opened": False,
    })


def validation_decision() -> None:
    """Freeze the no-test decision after the conditional ID64 control."""
    paths = {
        "m1": OUT / "validation/m1_direct_concat_id64_seed42.json",
        "m2_42": OUT / "validation/m2_concat_residual_id64_seed42.json",
        "m2_43": OUT / "validation/m2_concat_residual_id64_seed43.json",
        "m2_44": OUT / "validation/m2_concat_residual_id64_seed44.json",
        "add64": OUT / "validation/add_id64_control_seed42.json",
        "h2": EXP01_OUT / "validation/h2_side_features_id_seed42.json",
    }
    if any(not path.exists() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.exists()]
        raise SystemExit(f"validation decision missing artifacts: {missing}")
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    recall = {
        name: float(value["metrics"]["overall"]["Recall@500"])
        for name, value in values.items()
    }
    m2_seeds = [recall[f"m2_{seed}"] for seed in (42, 43, 44)]
    add_status = values["add64"]["temporal_item_status"]
    h2_status = values["h2"]["phase6_item_status"]
    payload = {
        "decision": "keep_existing_h2",
        "terminal_test_required": False,
        "terminal_test_skipped_to_preserve_one_shot_test": True,
        "reason": (
            "M1 and M2 are materially below existing H2; conditional Add-ID64 "
            "also remains below H2 on validation, so no new candidate qualifies."
        ),
        "validation_recall500": recall,
        "m2_three_seed_mean_recall500": float(np.mean(m2_seeds)),
        "m2_three_seed_std_recall500": float(np.std(m2_seeds)),
        "add64_minus_h2_overall": recall["add64"] - recall["h2"],
        "add64_minus_h2_train_target_seen": (
            add_status["train_target_seen"]["Recall@500"]
            - h2_status["train_target_seen"]["Recall@500"]
        ),
        "add64_minus_h2_completely_unseen": (
            add_status["completely_unseen"]["Recall@500"]
            - h2_status["completely_unseen"]["Recall@500"]
        ),
        "test_opened": False,
    }
    save_json(OUT / "locks/validation_decision.json", payload)


def terminal_preflight(args, lock: dict):
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("terminal preflight failed: CUDA is unavailable")
    try:
        probe = torch.empty(1, device=args.device)
        torch.cuda.synchronize(args.device)
        del probe
    except Exception as exc:
        raise SystemExit(f"terminal preflight failed: GPU allocation error: {exc}") from exc

    _, _, checkpoint_sha256 = verified_training(
        lock["model"], lock["seed"]
    )
    if checkpoint_sha256 != lock["checkpoint_sha256"]:
        raise SystemExit("terminal preflight failed: final lock checkpoint hash drift")
    validation_marker_path = (
        OUT / f"locks/full_validation_{lock['model']}_seed{lock['seed']}.json"
    )
    if not validation_marker_path.exists():
        raise SystemExit("terminal preflight failed: full-validation marker missing")
    validation_marker = json.loads(validation_marker_path.read_text())
    if (
        not validation_marker.get("complete")
        or validation_marker.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise SystemExit("terminal preflight failed: validation marker/hash mismatch")

    embedding_path = ROOT / lock["embedding"]
    metadata_path = (
        OUT / f"embeddings/{lock['model']}/seed{lock['seed']}_metadata.json"
    )
    if not embedding_path.exists() or not metadata_path.exists():
        raise SystemExit("terminal preflight failed: embedding or metadata missing")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("model") != lock["model"]
        or int(metadata.get("seed", -1)) != int(lock["seed"])
        or metadata.get("checkpoint_sha256") != checkpoint_sha256
        or metadata.get("shape") != [PROTOCOL.corpus_items, PROTOCOL.output_dim]
        or metadata.get("dtype") != "float32"
        or metadata.get("normalized") is not True
    ):
        raise SystemExit("terminal preflight failed: embedding/checkpoint hash mismatch")
    items = np.load(embedding_path, mmap_mode="r")
    if items.shape != (PROTOCOL.corpus_items, PROTOCOL.output_dim):
        raise SystemExit(f"terminal preflight failed: embedding shape {items.shape}")
    if items.dtype != np.float32:
        raise SystemExit(f"terminal preflight failed: embedding dtype {items.dtype}")
    probe_rows = np.linspace(0, len(items) - 1, 4096, dtype=np.int64)
    probe_vectors = np.asarray(items[probe_rows], dtype=np.float32)
    if not np.isfinite(probe_vectors).all() or not np.allclose(
        np.linalg.norm(probe_vectors, axis=1), 1.0, atol=1e-3
    ):
        raise SystemExit("terminal preflight failed: embedding normalization/finite check")
    mapping_path = ROOT / metadata["candidate_mapping"]
    if not mapping_path.exists():
        raise SystemExit("terminal preflight failed: candidate mapping missing")
    if sha256_file(mapping_path) != metadata.get("candidate_mapping_sha256"):
        raise SystemExit("terminal preflight failed: candidate mapping hash mismatch")
    mapping = np.load(mapping_path, mmap_mode="r")
    if mapping.shape != (PROTOCOL.corpus_items,):
        raise SystemExit(f"terminal preflight failed: mapping shape {mapping.shape}")
    train_frame, _, store = resources()
    candidate_ids = np.asarray(store.item_ids)
    if not np.array_equal(candidate_ids, mapping):
        raise SystemExit("terminal preflight failed: FeatureStore mapping drift")
    model, checkpoint = load_best(lock["model"], lock["seed"], store, args.device)
    return train_frame, store, model, checkpoint, items, candidate_ids


def terminal_test(args) -> None:
    manifest = OUT / "terminal_test/manifest.json"
    if manifest.exists():
        raise SystemExit("terminal test is one-shot and already opened")
    lock = json.loads((OUT / "locks/final_model.json").read_text())
    train_frame, store, model, checkpoint, items, candidate_ids = (
        terminal_preflight(args, lock)
    )
    # From this line onward the terminal test is considered opened exactly once.
    save_json(manifest, {"status": "running", "test_reads": 1, "locked_model": lock})
    deadline = time.monotonic() + 1200
    requests = QilinData(ROOT).load_test_requests()
    query_started = time.perf_counter(); queries = encode_queries(model, requests, store, args.device, deadline=deadline); query_seconds = time.perf_counter() - query_started
    _, raw, search_seconds, backend, timing = gpu_exact_search(items, queries, PROTOCOL.overfetch, args.device, deadline=deadline)
    rankings = filter_history(raw, candidate_ids, [r.history for r in requests], PROTOCOL.topk, deadline)
    validate_topk(rankings, candidate_ids, [r.history for r in requests], deadline=deadline)
    targets, vocab, users = set(map(int, store.train_target_item_ids)), set(map(int, store.train_item_id_vocab)), set(map(int, store.train_user_ids))
    old_exposed, old_clicked, old_users = legacy_evaluator_sets()
    metrics, status, per = phase6_status_metrics(requests, rankings, targets, vocab, users, lock["model"], old_exposed, old_clicked, old_users, deadline)
    save_parquet_atomic(per, OUT / "terminal_test/per_request.parquet")
    save_csv_atomic(
        feature_slices(requests, rankings, store, train_frame.positive_item_id.value_counts().to_dict()),
        OUT / "terminal_test/feature_slices.csv",
    )
    save_json(OUT / "terminal_test/result.json", {
        "model": lock["model"], "seed": lock["seed"], "checkpoint_epoch": checkpoint["epoch"],
        "metrics": metrics, "temporal_item_status": status, "candidate_universe": len(candidate_ids),
        "item_encode_seconds": 0.0, "reused_locked_item_embedding": True,
        "query_encode_seconds": query_seconds,
        "search_seconds": search_seconds, "search_backend": backend, **timing,
        "residual_diagnostics": validation_residual_diagnostics(
            model, requests, store, targets, vocab, users, args.device,
            seed=lock["seed"], deadline=deadline
        ),
        "test_reads": 1,
    })
    save_json(manifest, {"status": "complete", "test_reads": 1, "locked_model": lock})


def report() -> None:
    def fmt_pct(value: float) -> str:
        return "N/A" if not np.isfinite(value) else f"{value:.4%}"

    rows = []
    for path in sorted((OUT / "validation").glob("*_seed*.json")):
        value = json.loads(path.read_text())
        if "metrics" not in value:
            continue
        rows.append({
            "model": value["model"], "seed": value["seed"],
            **{f"r{k}": value["metrics"]["overall"][f"Recall@{k}"] for k in (10, 50, 100, 200, 500)},
            "mrr100": value["metrics"]["overall"]["MRR@100"],
            "warm": value["metrics"]["warm_item"]["Recall@500"],
            "cold": value["metrics"]["cold_item"]["Recall@500"],
            "warm_user": value["metrics"]["warm_user"]["Recall@500"],
            "cold_user": value["metrics"]["cold_user"]["Recall@500"],
            "seen": value["temporal_item_status"]["train_target_seen"]["Recall@500"],
            "unseen": value["temporal_item_status"]["completely_unseen"]["Recall@500"],
            "parameters": value["parameters"], "diagnostics": value["residual_diagnostics"],
        })
    lines = ["# Phase 7 Experiment 02：Concat Fusion 消融", "", "| Model | Seed | R@10 | R@50 | R@100 | R@200 | R@500 | MRR@100 | Target-seen R@500 | Completely-unseen R@500 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['seed']} | {row['r10']:.4%} | {row['r50']:.4%} | {row['r100']:.4%} | {row['r200']:.4%} | {row['r500']:.4%} | {row['mrr100']:.6f} | {row['seen']:.4%} | {row['unseen']:.4%} |")
    lines += ["", "## Legacy Warm / Cold User-Item", "", "| Model | Seed | Warm-item R@500 | Cold-item R@500 | Warm-user R@500 | Cold-user R@500 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['seed']} | {fmt_pct(row['warm'])} | "
            f"{fmt_pct(row['cold'])} | {fmt_pct(row['warm_user'])} | "
            f"{fmt_pct(row['cold_user'])} |"
        )
    lines += [
        "",
        "> Temporal-validation requests 在旧 Phase 1 train/test 定义下全部属于 legacy warm，"
        "因此 legacy cold item/user 无合格样本并记为 N/A；模型选择使用下方 temporal "
        "target-seen/history-only/completely-unseen 分桶。",
    ]
    lines += ["", "## 固定参考", "", "- Phase 5 Content terminal R@500：8.1282%。", "- Phase 7 H2 terminal R@500：8.7028%；validation 三 seed均值：9.5441%。", "- 本实验不运行融合。", "", "## 参数与 Residual 诊断", ""]
    for row in rows:
        lines.append(f"- `{row['model']}` seed {row['seed']}：parameters=`{json.dumps(row['parameters'])}`；diagnostics=`{json.dumps(row['diagnostics'])}`。")
    lines += ["", "## Frequency Buckets", ""]
    for row in rows:
        path = OUT / f"validation/{row['model']}_seed{row['seed']}_feature_slices.csv"
        if not path.exists():
            continue
        frequency = pd.read_csv(path)
        frequency = frequency[frequency["slice"] == "train_target_frequency"]
        lines.append(f"### {row['model']} seed {row['seed']}")
        lines.append("")
        lines.append("| Bucket | Positives | R@500 |")
        lines.append("| --- | ---: | ---: |")
        for _, value in frequency.iterrows():
            lines.append(f"| {value['value']} | {int(value['positive_count'])} | {value['positive_recall@500']:.4%} |")
        lines.append("")
    bootstrap_path = OUT / "validation/m2_vs_m1_bootstrap.json"
    if bootstrap_path.exists():
        bootstrap = json.loads(bootstrap_path.read_text())
        lines += ["", "## M2 - M1 Paired Bootstrap", "", "| Segment | Delta R@500 | CI95 |", "| --- | ---: | --- |"]
        for segment in ("overall", "train_target_seen", "train_history_only", "completely_unseen"):
            value = bootstrap[segment]
            lines.append(f"| {segment} | {value['point_delta']:.4%} | [{value['ci95_lower']:.4%}, {value['ci95_upper']:.4%}] |")
    selection_path = OUT / "locks/selected_structures.json"
    if selection_path.exists():
        lines += ["", "## Validation Selection", "", f"```json\n{selection_path.read_text().strip()}\n```"]
    decision_path = OUT / "locks/validation_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        lines += [
            "", "## 最终 Validation 决策", "",
            f"- 决策：`{decision['decision']}`。",
            f"- M2 三 seed R@500：均值 {decision['m2_three_seed_mean_recall500']:.4%}，std {decision['m2_three_seed_std_recall500']:.4%}。",
            f"- Add-ID64 相对现有 H2 Overall：{decision['add64_minus_h2_overall']:+.4%}。",
            f"- Add-ID64 相对 H2 target-seen：{decision['add64_minus_h2_train_target_seen']:+.4%}。",
            f"- Add-ID64 相对 H2 completely-unseen：{decision['add64_minus_h2_completely_unseen']:+.4%}。",
            "- 新结构未达到替换门槛，因此不读取 terminal test。",
        ]
    terminal = OUT / "terminal_test/result.json"
    if terminal.exists():
        value = json.loads(terminal.read_text())
        overall = value["metrics"]["overall"]["Recall@500"]
        warm = value["temporal_item_status"]["train_target_seen"]["Recall@500"]
        cold = value["temporal_item_status"]["completely_unseen"]["Recall@500"]
        h2_overall, h2_warm, h2_cold = 0.08702838361836934, 0.022355799927943308, 0.10480170688334324
        final_lock = json.loads((OUT / "locks/final_model.json").read_text())
        stable = final_lock["three_seed_std_recall500"] <= 0.002
        unified = bool(overall > 0.081282 and warm > h2_warm and cold >= h2_cold - 0.001 and stable)
        lines += [
            "", "## Terminal Test（唯一一次）", "",
            f"- Model：`{value['model']}`，seed={value['seed']}。",
            f"- Recall@100：{value['metrics']['overall']['Recall@100']:.4%}。",
            f"- Recall@500：{overall:.4%}；相对 Phase 5 Content {overall - 0.081282:+.4%}，相对 H2 {overall - h2_overall:+.4%}。",
            f"- Temporal warm：{warm:.4%}（相对 H2 {warm - h2_warm:+.4%}）。",
            f"- Completely-unseen：{cold:.4%}（相对 H2 {cold - h2_cold:+.4%}）。",
            f"- 三 seed std：{final_lock['three_seed_std_recall500']:.4%}。",
            f"- 统一召回候选门槛：{'PASS' if unified else 'FAIL'}。",
            "- Test 结果未用于调结构。",
        ]
    temporary = OUT / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(OUT / "summary.md")


def main() -> None:
    args = parse_args()
    authorize(args)
    if args.stage == "plan":
        return plan()
    ensure_output_layout()
    if args.stage == "smoke": smoke(args)
    elif args.stage == "train": train(args)
    elif args.stage == "full-validation": full_validation(args)
    elif args.stage == "select": select()
    elif args.stage == "decision": validation_decision()
    elif args.stage == "lock": lock_final()
    elif args.stage == "terminal-test": terminal_test(args)
    elif args.stage == "report": report()


if __name__ == "__main__":
    main()
