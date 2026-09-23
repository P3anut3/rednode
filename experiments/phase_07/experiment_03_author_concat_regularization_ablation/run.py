#!/usr/bin/env python3
"""Phase 7 Experiment 03: author concat and structured-ID regularization.

The default action is a read-only plan.  Every stage that reads Qilin data or
writes an artifact requires ``--confirm-run`` after human review.
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
from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import (  # noqa: E402
    grouped_requests,
)
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
from experiments.phase_07.experiment_02_concat_fusion_ablation.trainer import (  # noqa: E402
    to_device,
)
from experiments.phase_07.experiment_03_author_concat_regularization_ablation.models import (  # noqa: E402
    E0,
    E0_VARIANT,
    FULL_VARIANTS,
    VARIANTS,
    AuthorConcatTower,
    make_e0_compatible_model,
)
from experiments.phase_07.experiment_03_author_concat_regularization_ablation.trainer import (  # noqa: E402
    BATCH_SIZE,
    CORPUS_ITEMS,
    MAX_EPOCHS,
    OUT,
    OVERFETCH,
    PATIENCE,
    PROXY_CANDIDATES,
    PROXY_REQUESTS,
    TOPK,
    DeadlineExceeded,
    encode_items,
    encode_queries,
    make_model,
    pack_item_features,
    parameter_report,
    representation_stability,
    train_epoch,
)


EXP01_OUT = ROOT / "results/phase_07/experiment_01_feature_hybrid_two_tower"
EXP02_OUT = ROOT / "results/phase_07/experiment_02_concat_fusion_ablation"
MODEL_NAMES = tuple(VARIANTS)
SEEDS = (42, 43, 44)
TRAIN_BUDGET_SECONDS = 3600
VALIDATION_BUDGET_SECONDS = 1200
TEST_BUDGET_SECONDS = 1200
MUTATING_STAGES = {
    "static-audit",
    "smoke",
    "train",
    "e0-diagnostics",
    "full-validation",
    "select",
    "lock",
    "terminal-test",
    "report",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "plan",
            "static-audit",
            "smoke",
            "train",
            "e0-diagnostics",
            "full-validation",
            "select",
            "lock",
            "terminal-test",
            "report",
        ),
        default="plan",
    )
    parser.add_argument("--model", choices=(E0, *MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args()


def marker(name: str) -> Path:
    return OUT / "configs/markers" / name


def authorize(args) -> None:
    if args.stage in MUTATING_STAGES and not args.confirm_run:
        raise SystemExit(f"REFUSED: {args.stage} requires --confirm-run after review")
    if args.stage in {"smoke", "train", "full-validation"} and args.model not in MODEL_NAMES:
        raise SystemExit(f"{args.stage} requires one of E1-E7; E0 is never retrained")
    if args.stage == "e0-diagnostics" and args.model not in {None, E0}:
        raise SystemExit("e0-diagnostics only accepts E0")
    if args.seed not in SEEDS:
        raise SystemExit("formal seeds are fixed to 42/43/44")
    if args.stage in {"smoke", "train", "full-validation"} and args.seed in {43, 44}:
        path = marker("selected_structures.json")
        if not path.exists() or args.model not in json.loads(path.read_text())["multi_seed_models"]:
            raise SystemExit("seed43/44 is allowed only after seed42 selection")


def ensure_layout() -> None:
    for name in (
        "configs", "configs/markers", "checkpoints", "metrics",
        "metrics/training_curves", "metrics/timing", "validation",
        "validation/rankings", "embeddings", "diagnostics", "smoke",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_baseline_references() -> dict:
    """Load exact immutable baselines and bind every value to its source hash."""
    e0_path = EXP02_OUT / "validation/m1_direct_concat_id64_seed42.json"
    h2_paths = {
        seed: EXP01_OUT / f"validation/h2_side_features_id_seed{seed}.json"
        for seed in SEEDS
    }
    e0 = json.loads(e0_path.read_text())
    h2 = {seed: json.loads(path.read_text()) for seed, path in h2_paths.items()}
    h2_scores = {
        seed: float(value["metrics"]["overall"]["Recall@500"])
        for seed, value in h2.items()
    }
    return {
        "e0_seed42_recall500": float(e0["metrics"]["overall"]["Recall@500"]),
        "h2_seed42_recall500": h2_scores[42],
        "h2_three_seed_mean_recall500": float(np.mean(list(h2_scores.values()))),
        "h2_seed_recall500": h2_scores,
        "sources": {
            "e0": {
                "path": str(e0_path.relative_to(ROOT)),
                "sha256": sha256_file(e0_path),
            },
            **{
                f"h2_seed{seed}": {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for seed, path in h2_paths.items()
            },
        },
    }


def _verify_baseline_references(references: dict) -> None:
    for value in references["sources"].values():
        path = ROOT / value["path"]
        if not path.exists() or sha256_file(path) != value["sha256"]:
            raise SystemExit(f"formal baseline source changed: {value['path']}")


def save_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def plan() -> None:
    print(json.dumps({
        "status": "implementation_only_waiting_for_review",
        "e0": {"source": "Experiment 02 formal validation JSON; exact value loaded at selection"},
        "variants": {name: value.to_dict() for name, value in VARIANTS.items()},
        "controls": {
            "train_samples": 254_583, "validation_samples": 47_733,
            "history_n": 20, "output_dim": 128, "candidate_universe": CORPUS_ITEMS,
            "temperature": 0.05, "batch_size": BATCH_SIZE,
            "epochs": MAX_EPOCHS, "patience": PATIENCE,
        },
        "execution_requires_confirm_run": True,
        "test_opened": False,
    }, ensure_ascii=False, indent=2))


def _old_checkpoint() -> Path:
    return EXP02_OUT / "checkpoints/m1_direct_concat_id64_seed42_best.pt"


def static_audit(args) -> None:
    """Data-backed audit; deliberately separate from the synthetic self-check."""
    ensure_layout()
    _, _, store = resources()
    schema = store.schema()
    old = make_e0_compatible_model(
        schema["user_count"], schema["item_id_count"],
        schema["user_category_sizes"], schema["item_category_sizes"],
    )
    checkpoint = torch.load(_old_checkpoint(), map_location="cpu", weights_only=False)
    old.load_state_dict(checkpoint["state_dict"], strict=True)
    rows = []
    for name in MODEL_NAMES:
        seed_all(42)
        model = make_model(name, store, "cpu")
        report = parameter_report(model)
        variant = VARIANTS[name]
        rows.append({
            "model": name, **report,
            "item_fusion_contract": report["item_fusion_input_dim"] == variant.item_fusion_input_dim,
            "user_fusion_contract": report["user_fusion_input_dim"] == variant.user_fusion_input_dim,
            "item_vocab_equal": model.item_id.num_embeddings == old.item_id.num_embeddings,
            "user_vocab_equal": model.user_id.num_embeddings == old.user_id.num_embeddings,
            "padding_rows_zero": bool(
                torch.count_nonzero(model.item_id.weight[0]) == 0
                and torch.count_nonzero(model.user_id.weight[0]) == 0
            ),
        })
    payload = {
        "e0_checkpoint_strict_load": True,
        "e0_checkpoint": str(_old_checkpoint().relative_to(ROOT)),
        "e0_checkpoint_sha256": sha256_file(_old_checkpoint()),
        "variants": rows,
        "test_opened": False,
    }
    save_json(OUT / "diagnostics/static_audit.json", payload)
    print(json.dumps(payload, indent=2, default=json_default))


def _small_batch(dataset, device: str, size: int = 16) -> dict:
    return to_device(Phase7Collator(True)([dataset[i] for i in range(size)]), device)


def _has_gradient(model, prefix: str) -> bool:
    return any(
        name.startswith(prefix) and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in model.named_parameters()
    )


def smoke(args) -> None:
    ensure_layout()
    seed_all(args.seed)
    frame = load_training_frame(True).iloc[:10_000].reset_index(drop=True)
    _, valid, store = resources()
    dataset = Phase7Dataset(frame, store, True, 20, args.seed)
    model = make_model(args.model, store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    stats = train_epoch(
        model, dataset, optimizer, args.device, 1,
        batch_size=min(args.batch_size, 256), max_batches=8,
        deadline=time.monotonic() + 600,
    )
    batch = _small_batch(dataset, args.device)
    model.eval()
    with torch.inference_mode():
        query = model.query(batch)
        item = model.encode_item(batch, "target")
        item_off = model.encode_item(batch, "target", item_id_enabled=False)
        user_off = model.query(batch, user_id_enabled=False)
    gradient_branches = [
        "item_meta", "user_profile", "item_id", "user_id", "attention",
        "history_mlp", "item_fusion", "user_fusion",
    ]
    if VARIANTS[args.model].content_input_dim == 128:
        gradient_branches.append("content")
    gradients = {name: _has_gradient(model, name) for name in gradient_branches}
    dropout = stats["structured_id_dropout"]
    p = VARIANTS[args.model].id_dropout
    ratio_ok = all(
        abs(dropout[role]["seen_drop_ratio"] - p) <= (0.04 if p else 1e-12)
        for role in ("target", "history", "user")
    )
    requests = select_requests(grouped_requests(valid), 100, 42)
    positives = set().union(*(request.ground_truth for request in requests))
    candidates = deterministic_proxy_candidates(
        np.asarray(store.item_ids), positives, 100_000, 42
    )
    packed = pack_item_features(candidates, store)
    vectors = encode_items(model, candidates, store, args.device, packed=packed)
    queries = encode_queries(model, requests, store, args.device)
    _, raw, _, backend, _ = gpu_exact_search(
        vectors, queries, min(OVERFETCH, len(candidates)), args.device,
        deadline=time.monotonic() + 600,
    )
    rankings = filter_history(raw, candidates, [r.history for r in requests], TOPK)
    validate_topk(rankings, candidates, [r.history for r in requests])
    negative_ids = batch["negative_note_ids"]
    negative_valid = batch["negative_mask"]
    negative_target_overlap = negative_valid & negative_ids.eq(
        batch["target_note_id"][:, None]
    )
    negative_history_overlap = negative_valid & (
        negative_ids[:, :, None].eq(batch["history_note_ids"][:, None, :])
    ).any(-1)
    negative_same_request_overlap = torch.zeros_like(negative_valid)
    for row, positives_in_request in enumerate(batch["same_request_positive_ids"]):
        if positives_in_request:
            positive_tensor = torch.as_tensor(
                positives_in_request, device=args.device, dtype=negative_ids.dtype
            )
            negative_same_request_overlap[row] = negative_valid[row] & (
                negative_ids[row, :, None].eq(positive_tensor[None, :])
            ).any(-1)
    passed = bool(
        np.isfinite(stats["loss"])
        and max(abs(query.norm(dim=-1).cpu().numpy() - 1)) < 1e-4
        and max(abs(item.norm(dim=-1).cpu().numpy() - 1)) < 1e-4
        and all(gradients.values())
        and ratio_ok
        and torch.isfinite(item_off).all()
        and torch.isfinite(user_off).all()
        and torch.count_nonzero(model.item_id.weight[0]) == 0
        and torch.count_nonzero(model.user_id.weight[0]) == 0
        and not negative_target_overlap.any()
        and not negative_history_overlap.any()
        and not negative_same_request_overlap.any()
    )
    payload = {
        "model": args.model, "seed": args.seed, "passed": passed,
        "train": stats, "gradients": gradients, "dropout_ratio_ok": ratio_ok,
        "query_shape": list(query.shape), "item_shape": list(item.shape),
        "search_backend": backend,
        "retrieved_shape": [len(rankings), len(rankings[0]) if rankings else 0],
        "raw_bge_is_external_input": VARIANTS[args.model].content_input_dim == 768,
        "negative_target_overlap": int(negative_target_overlap.sum()),
        "negative_history_overlap": int(negative_history_overlap.sum()),
        "negative_same_request_positive_overlap": int(negative_same_request_overlap.sum()),
        "item_id_off_mean_cosine": float((item * item_off).sum(-1).mean()),
        "user_id_off_mean_cosine": float((query * user_off).sum(-1).mean()),
        "test_opened": False,
    }
    save_json(OUT / f"smoke/{args.model}_seed{args.seed}.json", payload)
    if not passed:
        raise RuntimeError("smoke failed; formal training remains blocked")
    print(json.dumps(payload, indent=2, default=json_default))


def _proxy_evaluate(model, requests, store, candidates, packed, device, deadline):
    vectors = encode_items(model, candidates, store, device, packed=packed, deadline=deadline)
    queries = encode_queries(model, requests, store, device, deadline=deadline)
    _, raw, seconds, backend, timing = gpu_exact_search(
        vectors, queries, min(OVERFETCH, len(candidates)), device,
        query_batch=512, deadline=deadline,
    )
    rankings = filter_history(raw, candidates, [r.history for r in requests], TOPK, deadline)
    validate_topk(rankings, candidates, [r.history for r in requests], deadline=deadline)
    metrics, _ = evaluate_rankings(requests, rankings, set(), set(), "proxy", deadline=deadline)
    return {
        "proxy_Recall@100": metrics["overall"]["Recall@100"],
        "proxy_Recall@500": metrics["overall"]["Recall@500"],
        "proxy_MRR@100": metrics["overall"]["MRR@100"],
        "search_seconds": seconds, "search_backend": backend, **timing,
    }


def _completion(name: str, seed: int) -> Path:
    return marker(f"training_complete_{name}_seed{seed}.json")


def train(args) -> None:
    ensure_layout()
    smoke_path = OUT / f"smoke/{args.model}_seed{args.seed}.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get("passed"):
        raise SystemExit("successful matching model/seed smoke is required")
    completion = _completion(args.model, args.seed)
    validation_done = marker(f"full_validation_{args.model}_seed{args.seed}.json")
    if completion.exists() or validation_done.exists():
        raise SystemExit("completed training/validation artifacts are immutable")
    active = marker(f"training_active_{args.model}_seed{args.seed}.lock")
    try:
        descriptor = os.open(active, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit("matching training job is already active") from exc
    try:
        os.write(descriptor, b"active\n")
        os.close(descriptor)
        seed_all(args.seed)
        started, deadline = time.perf_counter(), time.monotonic() + TRAIN_BUDGET_SECONDS
        _, valid, store = resources()
        frame = load_training_frame(True)
        dataset = Phase7Dataset(frame, store, True, 20, args.seed)
        requests = select_requests(grouped_requests(valid), PROXY_REQUESTS, 42)
        positives = set().union(*(request.ground_truth for request in requests))
        candidates = deterministic_proxy_candidates(
            np.asarray(store.item_ids), positives, PROXY_CANDIDATES, 42
        )
        packed = pack_item_features(candidates, store)
        model = make_model(args.model, store, args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        best, stale, curves = -1.0, 0, []
        best_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_best.pt"
        last_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_last.pt"
        for epoch in range(1, MAX_EPOCHS + 1):
            try:
                train_stats = train_epoch(
                    model, dataset, optimizer, args.device, epoch,
                    args.batch_size, deadline=deadline,
                )
                proxy = _proxy_evaluate(
                    model, requests, store, candidates, packed, args.device, deadline
                )
            except DeadlineExceeded:
                save_torch_atomic({
                    "state_dict": model.state_dict(), "model": args.model,
                    "seed": args.seed, "epoch": epoch, "deadline_aborted": True,
                }, last_path)
                save_json(OUT / f"metrics/timing/{args.model}_seed{args.seed}_deadline.json", {
                    "stage": "train", "epoch": epoch,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                raise
            row = {"model": args.model, "seed": args.seed, "epoch": epoch, **train_stats, **proxy}
            curves.append(row)
            checkpoint = {
                "state_dict": model.state_dict(), "model": args.model,
                "variant": VARIANTS[args.model].to_dict(), "seed": args.seed,
                "epoch": epoch, "proxy_recall500": proxy["proxy_Recall@500"],
                "parameters": parameter_report(model),
            }
            save_torch_atomic(checkpoint, last_path)
            if proxy["proxy_Recall@500"] > best:
                best, stale = proxy["proxy_Recall@500"], 0
                save_torch_atomic(checkpoint, best_path)
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        curve_path = OUT / f"metrics/training_curves/{args.model}_seed{args.seed}.csv"
        config_path = OUT / f"configs/{args.model}_seed{args.seed}.json"
        save_csv_atomic(pd.DataFrame(curves), curve_path)
        save_json(config_path, {
            "model": args.model, "variant": VARIANTS[args.model].to_dict(),
            "seed": args.seed, "best_proxy_recall500": best,
            "fixed_proxy_requests": len(requests),
            "fixed_proxy_candidates": len(candidates),
            "loss": "in-batch InfoNCE + 0.5 Phase-5 TF-IDF pairwise HN",
            "parameters": parameter_report(model), "test_opened": False,
        })
        if not best_path.exists() or not last_path.exists() or not curves:
            raise RuntimeError("normal training did not produce complete artifacts")
        save_json(completion, {
            "complete": True, "deadline_aborted": False,
            "model": args.model, "seed": args.seed,
            "epochs_completed": len(curves),
            "curve": str(curve_path.relative_to(ROOT)),
            "config": str(config_path.relative_to(ROOT)),
            "checkpoint": str(best_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(best_path),
            "test_opened": False,
        })
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        active.unlink(missing_ok=True)


def verified_checkpoint(name: str, seed: int) -> tuple[Path, dict, str]:
    path = _completion(name, seed)
    if not path.exists():
        raise SystemExit("training completion marker is required")
    value = json.loads(path.read_text())
    checkpoint = ROOT / value["checkpoint"]
    curve = ROOT / value["curve"]
    config = ROOT / value["config"]
    if not value.get("complete") or value.get("deadline_aborted"):
        raise SystemExit("training was not completed normally")
    if not checkpoint.exists() or not curve.exists() or not config.exists():
        raise SystemExit("training completion marker references missing artifacts")
    if len(pd.read_csv(curve)) != int(value["epochs_completed"]):
        raise SystemExit("training curve length disagrees with completion marker")
    actual = sha256_file(checkpoint)
    if actual != value["checkpoint_sha256"]:
        raise SystemExit("checkpoint SHA-256 changed after training completion")
    return checkpoint, value, actual


def load_best(name, seed, store, device):
    path, _, digest = verified_checkpoint(name, seed)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = make_model(name, store, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint, digest


def _load_e0_as_author(store, device):
    checkpoint = torch.load(_old_checkpoint(), map_location=device, weights_only=False)
    model = make_model(E0, store, device)
    incompat = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_unexpected = {"beta_item_raw", "beta_user_raw"}
    if incompat.missing_keys or set(incompat.unexpected_keys) != allowed_unexpected:
        raise RuntimeError(
            f"E0 author-wrapper mismatch: missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    model.eval()
    return model, checkpoint, sha256_file(_old_checkpoint())


def _evaluate_mode(model, requests, store, candidates, item_vectors, device, mode, deadline):
    item_enabled = mode not in {"item_id_off", "all_id_off"}
    user_enabled = mode not in {"user_id_off", "all_id_off"}
    queries = encode_queries(
        model, requests, store, device,
        item_id_enabled=item_enabled, user_id_enabled=user_enabled,
        deadline=deadline,
    )
    _, raw, seconds, backend, timing = gpu_exact_search(
        item_vectors, queries, OVERFETCH, device, query_batch=512, deadline=deadline
    )
    rankings = filter_history(raw, candidates, [r.history for r in requests], TOPK, deadline)
    validate_topk(rankings, candidates, [r.history for r in requests], deadline=deadline)
    targets = set(map(int, store.train_target_item_ids))
    vocab = set(map(int, store.train_item_id_vocab))
    users = set(map(int, store.train_user_ids))
    old_exposed, old_clicked, old_users = legacy_evaluator_sets()
    metrics, status, per = phase6_status_metrics(
        requests, rankings, targets, vocab, users, mode,
        old_exposed, old_clicked, old_users, deadline,
    )
    return metrics, status, per, {
        "search_seconds": seconds, "search_backend": backend, **timing,
    }


def _full_validation_impl(name: str, seed: int, device: str, e0: bool = False) -> None:
    done = marker(f"full_validation_{name}_seed{seed}.json")
    if done.exists():
        raise SystemExit("completed full validation is immutable")
    ensure_layout()
    started, deadline = time.perf_counter(), time.monotonic() + VALIDATION_BUDGET_SECONDS
    train_frame, valid, store = resources()
    requests = grouped_requests(valid)
    if e0:
        model, checkpoint, digest = _load_e0_as_author(store, device)
        checkpoint_path = _old_checkpoint()
    else:
        model, checkpoint, digest = load_best(name, seed, store, device)
        checkpoint_path = OUT / f"checkpoints/{name}_seed{seed}_best.pt"
    candidates = np.asarray(store.item_ids)
    if len(candidates) != CORPUS_ITEMS:
        raise RuntimeError(f"candidate universe changed: {len(candidates)}")
    if e0:
        # E0 is a frozen anchor: reuse its immutable Experiment-02 vectors and
        # formal normal result instead of recomputing the baseline.
        old_embedding = (
            EXP02_OUT
            / "embeddings/m1_direct_concat_id64/seed42_item_vectors.f32.npy"
        )
        item_on = np.load(old_embedding, mmap_mode="r")
        old_result = json.loads(
            (
                EXP02_OUT
                / "validation/m1_direct_concat_id64_seed42.json"
            ).read_text()
        )
    else:
        item_on = encode_items(model, candidates, store, device, deadline=deadline)
    item_off = encode_items(
        model, candidates, store, device, item_id_enabled=False, deadline=deadline
    )
    embedding_path = OUT / f"embeddings/{name}/seed{seed}_item_vectors_id_on.f32.npy"
    if not e0:
        save_npy_atomic(embedding_path, item_on)
    else:
        embedding_path = old_embedding
    embedding_sha256 = sha256_file(embedding_path)
    modes = {
        "normal": item_on,
        "item_id_off": item_off,
        "user_id_off": item_on,
        "all_id_off": item_off,
    }
    results, timing = {}, {}
    for mode, vectors in modes.items():
        if e0 and mode == "normal":
            results[mode] = {
                "metrics": old_result["metrics"],
                "temporal_item_status": old_result["temporal_item_status"],
                "reused_experiment02": True,
            }
            timing[mode] = {"reused_experiment02": True}
            continue
        metrics, status, per, mode_timing = _evaluate_mode(
            model, requests, store, candidates, vectors, device, mode, deadline
        )
        results[mode] = {"metrics": metrics, "temporal_item_status": status}
        timing[mode] = mode_timing
        save_parquet_atomic(
            per, OUT / f"validation/rankings/{name}/seed{seed}_{mode}.parquet"
        )
    stability = representation_stability(
        model, store, train_frame, device, seed=seed, deadline=deadline
    )
    save_json(OUT / f"diagnostics/{name}_seed{seed}_representation_stability.json", stability)
    counts = train_frame.positive_item_id.value_counts().to_dict()
    if e0:
        old_slice = (
            EXP02_OUT
            / "validation/m1_direct_concat_id64_seed42_feature_slices.csv"
        )
        feature_slice_source = str(old_slice.relative_to(ROOT))
    else:
        normal_frame = pd.read_parquet(
            OUT / f"validation/rankings/{name}/seed{seed}_normal.parquet"
        )
        # feature_slices consumes rankings, not the saved per-request frame.
        rankings = np.asarray(normal_frame["retrieved_top500"].tolist(), dtype=np.int64)
        slice_path = OUT / f"validation/{name}_seed{seed}_feature_slices.csv"
        save_csv_atomic(feature_slices(requests, rankings, store, counts), slice_path)
        feature_slice_source = str(slice_path.relative_to(ROOT))
    metadata_path = OUT / f"embeddings/{name}/seed{seed}_metadata.json"
    save_json(metadata_path, {
        "model": name, "seed": seed, "shape": list(item_on.shape),
        "dtype": str(item_on.dtype), "normalized": True,
        "embedding_sha256": embedding_sha256,
        "candidate_count": len(candidates), "embedding_dim": int(item_on.shape[1]),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": digest,
        "candidate_mapping": str((EXP01_OUT / "cache/item_ids.npy").relative_to(ROOT)),
        "candidate_mapping_sha256": sha256_file(EXP01_OUT / "cache/item_ids.npy"),
        "feature_slice_source": feature_slice_source,
        "id_off_embedding_persisted": False, "test_opened": False,
    })
    payload = {
        "model": name, "seed": seed, "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": digest, "modes": results,
        "representation_stability": stability, "parameters": parameter_report(model),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "candidate_universe": len(candidates), "timing": timing,
        "elapsed_seconds": time.perf_counter() - started, "test_opened": False,
    }
    result_path = OUT / f"validation/{name}_seed{seed}.json"
    save_json(result_path, payload)
    save_json(OUT / f"metrics/timing/{name}_seed{seed}_full_validation.json", {
        "elapsed_seconds": payload["elapsed_seconds"], "modes": timing
    })
    save_json(done, {
        "complete": True, "all_artifacts_complete": True,
        "model": name, "seed": seed, "checkpoint_sha256": digest,
        "embedding": str(embedding_path.relative_to(ROOT)),
        "embedding_sha256": embedding_sha256,
        "embedding_metadata": str(metadata_path.relative_to(ROOT)),
        "result": str(result_path.relative_to(ROOT)), "test_opened": False,
    })


def _with_serial_validation_lock(callback) -> None:
    lock = marker("full_validation_active.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit("another full-catalog validation is active") from exc
    try:
        os.write(descriptor, b"active\n")
        os.close(descriptor)
        callback()
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock.unlink(missing_ok=True)


def full_validation(args) -> None:
    _with_serial_validation_lock(
        lambda: _full_validation_impl(args.model, args.seed, args.device)
    )


def e0_diagnostics(args) -> None:
    _with_serial_validation_lock(
        lambda: _full_validation_impl(E0, 42, args.device, e0=True)
    )


def _validation(name: str, seed: int = 42) -> dict:
    path = OUT / f"validation/{name}_seed{seed}.json"
    done = marker(f"full_validation_{name}_seed{seed}.json")
    if not path.exists() or not done.exists():
        raise SystemExit(f"missing completed validation: {name} seed{seed}")
    value, status = json.loads(path.read_text()), json.loads(done.read_text())
    if (
        not status.get("complete")
        or not status.get("all_artifacts_complete")
        or status.get("result") != str(path.relative_to(ROOT))
        or value["checkpoint_sha256"] != status["checkpoint_sha256"]
    ):
        raise SystemExit(f"incomplete or hash-mismatched validation: {name} seed{seed}")
    return value


def _r500(value: dict, mode: str = "normal") -> float:
    return float(value["modes"][mode]["metrics"]["overall"]["Recall@500"])


def select() -> None:
    values = {name: _validation(name) for name in (E0, *MODEL_NAMES)}
    baselines = _formal_baseline_references()
    if not np.isclose(
        _r500(values[E0]), baselines["e0_seed42_recall500"], rtol=0.0, atol=0.0
    ):
        raise SystemExit("reused E0 metric disagrees with its formal source JSON")
    scores = {name: _r500(value) for name, value in values.items()}
    ranked = sorted(FULL_VARIANTS, key=lambda name: scores[name], reverse=True)
    multi = [ranked[0]]
    if scores[ranked[0]] - scores[ranked[1]] <= 0.001:
        multi.append(ranked[1])
    comparisons = {
        "content_128_to_768_old_id": (E0, "e1_content768_only"),
        "id_64_to_32": (E0, "e2_id32_only"),
        "small_init": ("e2_id32_only", "e3_id32_small_init"),
        "dropout_on_old": (E0, "e4_dropout_only_p30"),
        "content_768_under_id32": ("e3_id32_small_init", "e5_content768_id32_small_init"),
        "dropout_p30": ("e5_content768_id32_small_init", "e6_full_p30"),
        "dropout_p50": ("e5_content768_id32_small_init", "e7_full_p50"),
    }
    attribution = {
        label: {
            "base": base, "candidate": candidate,
            "delta_Recall@500": scores[candidate] - scores[base],
            "delta_all_id_off_Recall@500": _r500(values[candidate], "all_id_off") - _r500(values[base], "all_id_off"),
        }
        for label, (base, candidate) in comparisons.items()
    }
    save_json(marker("selected_structures.json"), {
        "seed42_recall500": scores, "full_configuration_rank": ranked,
        "multi_seed_models": multi, "close_threshold": 0.001,
        "attribution": attribution, "formal_baselines": baselines,
        "test_opened": False,
    })


def _h2_result(seed: int = 42) -> dict:
    return json.loads(
        (EXP01_OUT / f"validation/h2_side_features_id_seed{seed}.json").read_text()
    )


def _bootstrap_vs_h2(name: str, requests) -> dict:
    old_exposed, _, _ = legacy_evaluator_sets()
    store = resources()[2]
    targets = set(map(int, store.train_target_item_ids))
    vocab = set(map(int, store.train_item_id_vocab))
    base = pd.read_parquet(
        EXP01_OUT / "validation_rankings/h2_side_features_id_seed42.parquet"
    )
    candidate = pd.read_parquet(
        OUT / f"validation/rankings/{name}/seed42_normal.parquet"
    )
    return paired_bootstrap_segments(
        base, candidate, requests,
        ("overall", "warm", "cold", "train_target_seen", "train_history_only", "completely_unseen"),
        old_exposed, targets, vocab,
    )


def lock_final() -> None:
    selection = json.loads(marker("selected_structures.json").read_text())
    baselines = selection["formal_baselines"]
    _verify_baseline_references(baselines)
    candidates = {}
    for name in selection["multi_seed_models"]:
        rows = [(seed, _validation(name, seed)) for seed in SEEDS]
        candidates[name] = rows
    winner = max(candidates, key=lambda name: np.mean([_r500(value) for _, value in candidates[name]]))
    rows = candidates[winner]
    scores = [_r500(value) for _, value in rows]
    terminal_seed, terminal_value = sorted(rows, key=lambda row: _r500(row[1]))[1]
    _, valid, _ = resources()
    bootstrap = _bootstrap_vs_h2(winner, grouped_requests(valid))
    save_json(OUT / f"validation/{winner}_vs_h2_bootstrap.json", bootstrap)
    seed42 = dict(rows)[42]
    unseen = float(seed42["modes"]["normal"]["temporal_item_status"]["completely_unseen"]["Recall@500"])
    h2 = _h2_result(42)
    h2_status = h2.get("temporal_item_status", h2.get("phase6_item_status"))
    h2_unseen = float(h2_status["completely_unseen"]["Recall@500"])
    id_off = _r500(seed42, "all_id_off")
    normal = _r500(seed42)
    overall_bootstrap = bootstrap["overall"]
    gates = {
        "seed42_at_least_h2": normal >= baselines["h2_seed42_recall500"],
        "three_seed_mean_at_least_h2": (
            float(np.mean(scores)) >= baselines["h2_three_seed_mean_recall500"]
        ),
        "completely_unseen_noninferior": unseen >= h2_unseen,
        # Reviewable operational definition of "not catastrophic": retain at
        # least 80% of normal R@500 when every ID route is disabled.
        "all_id_off_retains_80_percent": id_off >= 0.8 * normal,
        "bootstrap_does_not_show_h2_better": float(overall_bootstrap["ci95_upper"]) >= 0.0,
    }
    checkpoint, completion, digest = verified_checkpoint(winner, terminal_seed)
    validation_marker = json.loads(
        marker(f"full_validation_{winner}_seed{terminal_seed}.json").read_text()
    )
    save_json(marker("final_model.json"), {
        "model": winner, "seed": terminal_seed,
        "three_seed_mean_recall500": float(np.mean(scores)),
        "three_seed_std_recall500": float(np.std(scores)),
        "three_seed_results": [
            {"seed": seed, "Recall@500": _r500(value)} for seed, value in rows
        ],
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": digest,
        "embedding": validation_marker["embedding"],
        "embedding_sha256": validation_marker["embedding_sha256"],
        "training_completion": completion,
        "formal_baselines": baselines,
        "terminal_test_gates": gates,
        "terminal_allowed": all(gates.values()),
        "test_opened": False,
    })


def terminal_test(args) -> None:
    final_path = marker("final_model.json")
    manifest = marker("terminal_test_manifest.json")
    if manifest.exists():
        raise SystemExit("terminal test is one-shot and has already been opened")
    final = json.loads(final_path.read_text())
    if not final.get("terminal_allowed"):
        raise SystemExit("validation gates are No-Go; test remains unopened")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA preflight failed before opening test")
    checkpoint = ROOT / final["checkpoint"]
    embedding = ROOT / final["embedding"]
    metadata_path = embedding.parent / embedding.name.replace(
        "item_vectors_id_on.f32.npy", "metadata.json"
    )
    if not checkpoint.exists() or not embedding.exists() or not metadata_path.exists():
        raise SystemExit("terminal preflight found a missing immutable artifact")
    metadata = json.loads(metadata_path.read_text())
    if sha256_file(checkpoint) != final["checkpoint_sha256"] or metadata["checkpoint_sha256"] != final["checkpoint_sha256"]:
        raise SystemExit("terminal checkpoint/embedding SHA-256 mismatch")
    if (
        sha256_file(embedding) != final["embedding_sha256"]
        or metadata.get("embedding_sha256") != final["embedding_sha256"]
    ):
        raise SystemExit("terminal item-vector SHA-256 mismatch")
    candidate_path = ROOT / metadata["candidate_mapping"]
    if sha256_file(candidate_path) != metadata["candidate_mapping_sha256"]:
        raise SystemExit("candidate mapping changed after validation")
    shape = tuple(metadata["shape"])
    if (
        shape != (CORPUS_ITEMS, 128)
        or int(metadata.get("candidate_count", -1)) != CORPUS_ITEMS
        or int(metadata.get("embedding_dim", -1)) != 128
        or metadata.get("dtype") != "float32"
    ):
        raise SystemExit(f"unexpected item embedding shape: {shape}")
    inspected = np.load(embedding, mmap_mode="r")
    if inspected.shape != shape or inspected.dtype != np.float32:
        raise SystemExit("item-vector file shape/dtype disagrees with metadata")
    # Only now is the one-shot test opened.
    save_json(manifest, {
        "state": "running", "model": final["model"], "seed": final["seed"],
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    deadline = time.monotonic() + TEST_BUDGET_SECONDS
    requests = QilinData(ROOT).load_test_requests()
    _, _, store = resources()
    model, _, _ = load_best(final["model"], final["seed"], store, args.device)
    candidates = np.load(candidate_path, mmap_mode="r")
    vectors = np.load(embedding, mmap_mode="r")
    metrics, status, per, timing = _evaluate_mode(
        model, requests, store, candidates, vectors, args.device,
        "normal", deadline,
    )
    result_path = OUT / "metrics/terminal_test.json"
    save_json(result_path, {
        "model": final["model"], "seed": final["seed"],
        "metrics": metrics, "temporal_item_status": status,
        "timing": timing, "terminal": True,
    })
    save_parquet_atomic(per, OUT / "validation/rankings/terminal_test.parquet")
    save_json(manifest, {
        "state": "complete", "model": final["model"], "seed": final["seed"],
        "checkpoint_sha256": final["checkpoint_sha256"],
        "result": str(result_path.relative_to(ROOT)),
    })


def report() -> None:
    rows = []
    for name in (E0, *MODEL_NAMES):
        path = OUT / f"validation/{name}_seed42.json"
        completed = marker(f"full_validation_{name}_seed42.json")
        if not path.exists() and not completed.exists():
            continue
        if path.exists() != completed.exists():
            raise SystemExit(
                f"refusing incomplete validation artifact in report: {name} seed42"
            )
        value = _validation(name, 42)
        variant = E0_VARIANT if name == E0 else VARIANTS[name]
        normal = value["modes"]["normal"]
        status = normal["temporal_item_status"]
        rows.append({
            "model": name, "content": variant.content_input_dim,
            "id_dim": variant.id_dim, "init": variant.id_init_std,
            "drop": variant.id_dropout,
            "R@100": normal["metrics"]["overall"]["Recall@100"],
            "R@500": normal["metrics"]["overall"]["Recall@500"],
            "MRR@100": normal["metrics"]["overall"]["MRR@100"],
            "seen": status["train_target_seen"]["Recall@500"],
            "history_only": status["train_history_only"]["Recall@500"],
            "unseen": status["completely_unseen"]["Recall@500"],
            "all_id_off": value["modes"]["all_id_off"]["metrics"]["overall"]["Recall@500"],
        })
    lines = [
        "# Phase 7 Experiment 03：作者式 Concat 与 ID 正则消融", "",
        "> 本文件仅汇总已经完成且带 completion marker 的结果；缺失实验显示为未运行。", "",
        "| Model | Content | ID Dim | Init Std | Drop P | R@100 | R@500 | MRR@100 | Seen | History-only | Unseen | All-ID-off |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['content']} | {row['id_dim']} | {row['init']:.2f} | {row['drop']:.1f} | "
            f"{row['R@100']:.4%} | {row['R@500']:.4%} | {row['MRR@100']:.4%} | "
            f"{row['seen']:.4%} | {row['history_only']:.4%} | {row['unseen']:.4%} | {row['all_id_off']:.4%} |"
        )
    selection_path = marker("selected_structures.json")
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        lines.extend(["", "## 单因素增量归因", "", "| 因素 | 对照 | ΔR@500 | ΔAll-ID-off |", "|---|---|---:|---:|"])
        for label, value in selection["attribution"].items():
            lines.append(
                f"| {label} | {value['candidate']} − {value['base']} | "
                f"{value['delta_Recall@500']:+.4%} | {value['delta_all_id_off_Recall@500']:+.4%} |"
            )
    final_path = marker("final_model.json")
    if final_path.exists():
        final = json.loads(final_path.read_text())
        lines.extend([
            "", "## Validation 决策", "",
            f"- 最佳结构：`{final['model']}`",
            f"- 三 seed R@500：{final['three_seed_mean_recall500']:.4%} ± {final['three_seed_std_recall500']:.4%}",
            f"- Terminal test：{'GO' if final['terminal_allowed'] else 'NO-GO'}",
            f"- Gates：`{json.dumps(final['terminal_test_gates'], ensure_ascii=False)}`",
        ])
    else:
        lines.extend(["", "## 当前状态", "", "代码待 review，尚未产生实验结果，也未读取 test。"])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    authorize(args)
    if args.stage == "plan":
        plan()
        return
    if args.stage != "static-audit" and not torch.cuda.is_available() and args.stage not in {"select", "lock", "report"}:
        raise SystemExit("CUDA is required; refusing silent CPU fallback")
    actions = {
        "static-audit": static_audit,
        "smoke": smoke,
        "train": train,
        "e0-diagnostics": e0_diagnostics,
        "full-validation": full_validation,
        "select": lambda _: select(),
        "lock": lambda _: lock_final(),
        "terminal-test": terminal_test,
        "report": lambda _: report(),
    }
    actions[args.stage](args)


if __name__ == "__main__":
    main()
