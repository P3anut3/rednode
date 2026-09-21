#!/usr/bin/env python3
"""Phase 7 feature-enhanced hybrid two-tower.

The default command is a non-mutating plan. Every data scan, cache build,
training, full validation, lock, or terminal test requires ``--confirm-run``.
This guard exists because Phase 7 must not run before human inspection.
"""

from __future__ import annotations

import argparse
import json
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
    load_temporal_frames,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (  # noqa: E402
    DeadlineExceeded,
    HYBRID_MODELS,
    ID_MODELS,
    LEGACY_EVALUATOR_CACHE,
    MODEL_NAMES,
    OUT,
    PHASE6,
    PROTOCOL,
    ROOT,
    check_deadline,
    ensure_output_layout,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import (  # noqa: E402
    Phase7Dataset,
    load_training_frame,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.evaluation import (  # noqa: E402
    feature_slices,
    paired_bootstrap,
    paired_bootstrap_segments,
    phase6_status_metrics,
    route_contribution,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.features import (  # noqa: E402
    FeatureStore,
    build_feature_cache,
    build_h3_dense_cache,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.retrieval import (  # noqa: E402
    deterministic_proxy_candidates,
    filter_history,
    gpu_exact_search,
    validate_topk,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.trainer import (  # noqa: E402
    encode_items,
    encode_queries,
    make_model,
    pack_item_features,
    parameter_report,
    smoke_gradient_and_cold_audit,
    train_epoch,
)


MUTATING_STAGES = {
    "stage-a",
    "h3-cache",
    "smoke",
    "train",
    "full-validation",
    "refresh-bootstrap",
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
            "stage-a",
            "h3-cache",
            "smoke",
            "train",
            "full-validation",
            "refresh-bootstrap",
            "select",
            "lock",
            "terminal-test",
            "report",
        ),
        default="plan",
    )
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--family", choices=("id", "hybrid"), help="Required by --stage select"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Required for every stage that reads experiment data or trains",
    )
    return parser.parse_args()


def json_default(item):
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, Path):
        return str(item)
    raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
            default=json_default,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def save_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_torch_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def require_authorization(args) -> None:
    if args.stage in MUTATING_STAGES and not args.confirm_run:
        raise SystemExit(
            f"REFUSED: --stage {args.stage} requires explicit --confirm-run after review"
        )
    if args.stage in {"smoke", "train", "full-validation"} and not args.model:
        raise SystemExit(f"--stage {args.stage} requires --model")
    if args.stage == "select" and not args.family:
        raise SystemExit("--stage select requires --family id|hybrid")
    if args.stage in {"smoke", "train", "full-validation"} and args.seed not in {
        42,
        43,
        44,
    }:
        raise SystemExit("formal Phase 7 seeds are fixed to 42/43/44")
    if args.stage in {"train", "full-validation"} and args.seed in {43, 44}:
        family = "id" if args.model in ID_MODELS else "hybrid"
        selection = OUT / f"locks/selected_{family}_structure.json"
        if (
            not selection.exists()
            or json.loads(selection.read_text()).get("model") != args.model
        ):
            raise SystemExit(
                "seeds 43/44 are allowed only for the validation-selected structure"
            )
    if args.model == "id_b3_all" and args.stage in {"train", "full-validation"}:
        decision = OUT / "locks/id_b3_eligible.json"
        if not decision.exists() or not json.loads(decision.read_text()).get(
            "eligible"
        ):
            raise SystemExit(
                "ID-B3 is conditional: B1a, B1b, or B2 must pass the matched-B0 "
                "warm-recall bootstrap gate"
            )
    if args.model == "h3_anonymous_dense":
        decision = OUT / "locks/h3_eligible.json"
        if not decision.exists() or not json.loads(decision.read_text()).get(
            "eligible"
        ):
            raise SystemExit("H3 is conditional: H2 must first show a validation gain")
        if (
            args.stage in {"smoke", "train", "full-validation"}
            and not (OUT / "cache/user_dense.npy").exists()
        ):
            raise SystemExit("H3 requires the separately authorized --stage h3-cache")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plan() -> None:
    payload = {
        "status": "implementation_only_not_authorized_to_run",
        "protocol": PROTOCOL.to_dict(),
        "models": list(MODEL_NAMES),
        "execution_order": [
            "stage-a",
            "smoke",
            "train",
            "full-validation",
            "select",
            "best-structure seeds 43/44",
            "lock",
            "terminal-test",
            "report",
        ],
        "guards": [
            "all execution stages require --confirm-run",
            "ID-B3 requires validation eligibility lock",
            "H3 requires H2 eligibility lock",
            "terminal test requires final_model.json and is one-shot",
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def resources():
    train, valid = load_temporal_frames(ROOT)
    store = FeatureStore()
    if len(train) != PROTOCOL.train_samples or len(valid) != PROTOCOL.valid_samples:
        raise AssertionError("temporal split drift")
    return train, valid, store


def target_logq(frame: pd.DataFrame) -> dict[int, float]:
    count = frame.positive_item_id.value_counts()
    total = len(frame)
    return {
        int(note): float(np.log(value / total + 1e-12)) for note, value in count.items()
    }


def legacy_evaluator_sets() -> tuple[set[int], set[int], set[int]]:
    paths = {
        "exposed": LEGACY_EVALUATOR_CACHE / "train_exposed.npy",
        "clicked": LEGACY_EVALUATOR_CACHE / "train_clicked.npy",
        "users": LEGACY_EVALUATOR_CACHE / "train_users.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen evaluator cache: {missing}")
    return tuple(set(map(int, np.load(path, mmap_mode="r"))) for path in paths.values())


def select_requests(requests, count: int, seed: int):
    if len(requests) <= count:
        return requests
    rows = np.sort(
        np.random.default_rng(seed).choice(len(requests), count, replace=False)
    )
    return [requests[int(row)] for row in rows]


def proxy_evaluate(
    model,
    requests,
    store,
    device: str,
    candidate_ids: np.ndarray,
    deadline: float | None = None,
    proxy_ids: np.ndarray | None = None,
    packed_items: dict[str, np.ndarray] | None = None,
) -> dict:
    if proxy_ids is None:
        positives = set().union(*(set(r.ground_truth) for r in requests))
        eligible_positive = positives & set(map(int, candidate_ids))
        proxy_ids = deterministic_proxy_candidates(
            candidate_ids, eligible_positive, PROTOCOL.proxy_candidates, PROTOCOL.seed
        )
    item_vectors = encode_items(
        model, proxy_ids, store, device, deadline=deadline, packed=packed_items
    )
    queries = encode_queries(model, requests, store, device, deadline=deadline)
    _, rows, seconds, backend, search_timing = gpu_exact_search(
        item_vectors,
        queries,
        min(PROTOCOL.overfetch, len(proxy_ids)),
        device,
        query_batch=512,
        deadline=deadline,
    )
    rankings = filter_history(
        rows,
        proxy_ids,
        [r.history for r in requests],
        PROTOCOL.topk,
        deadline=deadline,
    )
    validate_topk(rankings, proxy_ids, [r.history for r in requests], deadline=deadline)
    metrics, _ = evaluate_rankings(
        requests, rankings, set(), set(), "proxy", deadline=deadline
    )
    return {
        "proxy_Recall@100": metrics["overall"]["Recall@100"],
        "proxy_Recall@500": metrics["overall"]["Recall@500"],
        "proxy_MRR@100": metrics["overall"]["MRR@100"],
        "proxy_candidates": len(proxy_ids),
        "proxy_search_seconds": seconds,
        "proxy_search_backend": backend,
        "proxy_index_build_seconds": search_timing["index_build_seconds"],
        "proxy_query_search_seconds": search_timing["query_search_seconds"],
    }


def _ranking_column(frame: pd.DataFrame) -> str:
    for name in ("retrieved_top1000", "retrieved_top500"):
        if name in frame.columns:
            return name
    raise ValueError("ranking parquet has no supported ranking column")


def fixed_quota_validation(requests, id_rankings, deadline: float | None = None):
    base = PHASE6 / "validation_rankings"
    content_frame = pd.read_parquet(base / "current_content_phase5_top1000.parquet")
    tfidf_frame = pd.read_parquet(base / "tfidf_lexical.parquet")
    content_col, tfidf_col = (
        _ranking_column(content_frame),
        _ranking_column(tfidf_frame),
    )
    content = {
        int(row.request_idx): list(map(int, getattr(row, content_col)))
        for row in content_frame.itertuples(index=False)
    }
    tfidf = {
        int(row.request_idx): list(map(int, getattr(row, tfidf_col)))
        for row in tfidf_frame.itertuples(index=False)
    }
    merged = []
    for request_index, (request, id_values) in enumerate(zip(requests, id_rankings)):
        if request_index % 250 == 0:
            check_deadline(deadline, "fixed quota validation")
        selected, seen = [], set()
        for values, quota in (
            (content[request.request_idx], 300),
            (id_values, 100),
            (tfidf[request.request_idx], 100),
        ):
            count = 0
            for note in values:
                note = int(note)
                if note in seen:
                    continue
                selected.append(note)
                seen.add(note)
                count += 1
                if count == quota:
                    break
        for note in content[request.request_idx]:
            if len(selected) == 500:
                break
            if int(note) not in seen:
                selected.append(int(note))
                seen.add(int(note))
        merged.append(selected[:500])
    return merged


def smoke(args) -> None:
    started = time.perf_counter()
    deadline = time.monotonic() + 600
    seed_all(args.seed)
    train, valid, store = resources()
    hybrid = args.model in HYBRID_MODELS
    frame = load_training_frame(hybrid)
    dataset = Phase7Dataset(frame, store, hybrid, limit=PROTOCOL.smoke_train_samples)
    model = make_model(args.model, store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    stats = train_epoch(
        model,
        dataset,
        optimizer,
        args.device,
        min(args.batch_size, 256),
        1,
        target_logq(train),
        max_batches=40,
        deadline=deadline,
    )
    requests = select_requests(
        grouped_requests(valid), PROTOCOL.smoke_valid_requests, PROTOCOL.seed
    )
    catalog = store.train_target_item_ids if args.model in ID_MODELS else store.item_ids
    proxy = proxy_evaluate(
        model,
        requests,
        store,
        args.device,
        np.asarray(catalog),
        deadline=deadline,
    )
    elapsed = time.perf_counter() - started
    branch_audit = smoke_gradient_and_cold_audit(model, dataset, args.device)
    padding_zero = all(
        torch.count_nonzero(module.weight[0]).item() == 0
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding) and module.padding_idx == 0
    )
    gradient_present = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    payload = {
        "model": args.model,
        "seed": args.seed,
        "train": stats,
        "proxy": proxy,
        "parameters": parameter_report(model),
        "elapsed_seconds": elapsed,
        "budget_seconds": 600,
        "within_budget": elapsed <= 600,
        "checks": {
            "finite_loss": np.isfinite(stats["loss"]),
            "bge_frozen_external_memmap": True,
            "oov_padding_rows_zero": padding_zero,
            "trainable_gradient_present": gradient_present,
            **branch_audit,
            "numeric_cache_finite": bool(
                np.isfinite(store.user_numeric).all()
                and np.isfinite(store.item_numeric).all()
            ),
            "top500_unique_and_history_filtered": True,
            "test_opened": False,
        },
    }
    checks = payload["checks"]
    payload["smoke_passed"] = bool(
        payload["within_budget"]
        and checks["finite_loss"]
        and checks["oov_padding_rows_zero"]
        and checks["padding_embedding_gradient_zero"]
        and checks["all_required_branches_have_gradient"]
        and checks["numeric_cache_finite"]
        and checks["cold_item_id_invariance_max_abs"] <= 1e-6
        and checks["cold_user_id_invariance_max_abs"] <= 1e-6
    )
    save_json(OUT / f"smoke/{args.model}_seed{args.seed}.json", payload)
    if not payload["smoke_passed"]:
        raise RuntimeError("smoke failed; formal training remains blocked")
    print(json.dumps(payload, indent=2, default=json_default))


def train(args) -> None:
    smoke_path = OUT / f"smoke/{args.model}_seed{args.seed}.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get(
        "smoke_passed"
    ):
        raise SystemExit("successful smoke is required before formal training")
    started = time.perf_counter()
    deadline = time.monotonic() + 3600
    seed_all(args.seed)
    train_frame, valid, store = resources()
    hybrid = args.model in HYBRID_MODELS
    frame = load_training_frame(hybrid)
    dataset = Phase7Dataset(frame, store, hybrid)
    requests = select_requests(
        grouped_requests(valid), PROTOCOL.proxy_requests, PROTOCOL.seed
    )
    catalog = np.asarray(store.train_target_item_ids if not hybrid else store.item_ids)
    requests_early = requests[:2000]
    proxy_ids_early = deterministic_proxy_candidates(
        catalog,
        set().union(*(set(request.ground_truth) for request in requests_early))
        & set(map(int, catalog)),
        PROTOCOL.proxy_candidates,
        PROTOCOL.seed,
    )
    proxy_ids_full = deterministic_proxy_candidates(
        catalog,
        set().union(*(set(request.ground_truth) for request in requests))
        & set(map(int, catalog)),
        PROTOCOL.proxy_candidates,
        PROTOCOL.seed,
    )
    packed_early = pack_item_features(proxy_ids_early, store, hybrid)
    packed_full = pack_item_features(proxy_ids_full, store, hybrid)
    model = make_model(args.model, store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    logq = target_logq(train_frame)
    best, stale, curves = -1.0, 0, []
    best_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_best.pt"
    last_path = OUT / f"checkpoints/{args.model}_seed{args.seed}_last.pt"
    for epoch in range(1, PROTOCOL.epochs + 1):
        epoch_start = time.perf_counter()
        try:
            stats = train_epoch(
                model,
                dataset,
                optimizer,
                args.device,
                args.batch_size,
                epoch,
                logq,
                deadline=deadline,
            )
            proxy_requests = requests_early if epoch <= 2 else requests
            epoch_proxy_ids = proxy_ids_early if epoch <= 2 else proxy_ids_full
            epoch_packed = packed_early if epoch <= 2 else packed_full
            proxy = proxy_evaluate(
                model,
                proxy_requests,
                store,
                args.device,
                catalog,
                deadline=deadline,
                proxy_ids=epoch_proxy_ids,
                packed_items=epoch_packed,
            )
        except DeadlineExceeded:
            save_torch_atomic(
                {
                    "state_dict": model.state_dict(),
                    "model": args.model,
                    "seed": args.seed,
                    "epoch": epoch,
                    "aborted_by_deadline": True,
                    "protocol": PROTOCOL.to_dict(),
                },
                last_path,
            )
            save_json(
                OUT / f"timing/{args.model}_seed{args.seed}_deadline.json",
                {
                    "stage": "train",
                    "epoch": epoch,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            raise
        row = {
            "model": args.model,
            "seed": args.seed,
            "epoch": epoch,
            **stats,
            **proxy,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        curves.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "model": args.model,
            "seed": args.seed,
            "epoch": epoch,
            "proxy_recall500": proxy["proxy_Recall@500"],
            "protocol": PROTOCOL.to_dict(),
            "parameters": parameter_report(model),
        }
        save_torch_atomic(checkpoint, last_path)
        if proxy["proxy_Recall@500"] > best:
            best, stale = proxy["proxy_Recall@500"], 0
            save_torch_atomic(checkpoint, best_path)
        else:
            stale += 1
            if stale >= PROTOCOL.patience:
                break
    save_csv_atomic(
        pd.DataFrame(curves),
        OUT / f"training_curves/{args.model}_seed{args.seed}.csv",
    )
    save_json(
        OUT / f"configs/{args.model}_seed{args.seed}.json",
        {
            "model": args.model,
            "seed": args.seed,
            "best_proxy_recall500": best,
            "best_checkpoint": str(best_path.relative_to(ROOT)),
            "last_checkpoint": str(last_path.relative_to(ROOT)),
            "loss": "Phase6 logQ in-batch" if not hybrid else "Phase5 TF-IDF HN fixed",
            "test_opened": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def load_best(model_name: str, seed: int, store: FeatureStore, device: str):
    path = OUT / f"checkpoints/{model_name}_seed{seed}_best.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = make_model(model_name, store, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def full_validation(args) -> None:
    marker = OUT / f"locks/full_validation_{args.model}_seed{args.seed}.json"
    if marker.exists():
        raise SystemExit(
            "full validation already exists; refusing repeated full-corpus scan"
        )
    comparators = {
        "id_b1a_stable_profile": "id_b0_user_history_control",
        "id_b1b_full_profile": "id_b0_user_history_control",
        "id_b2_item_features": "id_b0_user_history_control",
        "id_b3_all": "id_b0_user_history_control",
        "h1_side_features": "h0_content_control",
        "h2_side_features_id": "h1_side_features",
        "h3_anonymous_dense": "h2_side_features_id",
    }
    comparator = comparators.get(args.model)
    if comparator and args.seed == 42:
        comparator_path = OUT / f"validation_rankings/{comparator}_seed42.parquet"
        comparator_marker = OUT / f"locks/full_validation_{comparator}_seed42.json"
        comparator_complete = (
            comparator_marker.exists()
            and json.loads(comparator_marker.read_text()).get("complete") is True
        )
        if not comparator_path.exists() or not comparator_complete:
            raise SystemExit(
                f"seed42 {args.model} requires {comparator} seed42 full-validation first"
            )
    started = time.perf_counter()
    budget = 600 if args.model in ID_MODELS else 1200
    deadline = time.monotonic() + budget
    train, valid, store = resources()
    requests = grouped_requests(valid)
    model, checkpoint = load_best(args.model, args.seed, store, args.device)
    candidate_ids = np.asarray(
        store.train_target_item_ids if args.model in ID_MODELS else store.item_ids
    )
    item_encode_started = time.perf_counter()
    items = encode_items(model, candidate_ids, store, args.device, deadline=deadline)
    item_encode_seconds = time.perf_counter() - item_encode_started
    query_encode_started = time.perf_counter()
    queries = encode_queries(model, requests, store, args.device, deadline=deadline)
    query_encode_seconds = time.perf_counter() - query_encode_started
    _, rows, search_seconds, backend, search_timing = gpu_exact_search(
        items, queries, PROTOCOL.overfetch, args.device, deadline=deadline
    )
    rankings = filter_history(
        rows, candidate_ids, [r.history for r in requests], deadline=deadline
    )
    validate_topk(
        rankings, candidate_ids, [r.history for r in requests], deadline=deadline
    )
    targets, vocab, users = (
        set(map(int, store.train_target_item_ids)),
        set(map(int, store.train_item_id_vocab)),
        set(map(int, store.train_user_ids)),
    )
    old_exposed, old_clicked, old_users = legacy_evaluator_sets()
    metrics, status, per = phase6_status_metrics(
        requests,
        rankings,
        targets,
        vocab,
        users,
        args.model,
        old_exposed,
        old_clicked,
        old_users,
        deadline=deadline,
    )
    per_path = OUT / f"validation_rankings/{args.model}_seed{args.seed}.parquet"
    save_parquet_atomic(per, per_path)
    counts = train.positive_item_id.value_counts().to_dict()
    save_csv_atomic(
        feature_slices(requests, rankings, store, counts),
        OUT / f"validation/{args.model}_seed{args.seed}_feature_slices.csv",
    )
    comparisons = {}
    for reference_name, reference_path in {
        "current_content": PHASE6
        / "validation_rankings/current_content_phase5.parquet",
        "pure_id": PHASE6 / "validation_rankings/pure_d1_logq.parquet",
    }.items():
        if reference_path.exists():
            comparisons[reference_name] = route_contribution(
                requests, rankings, pd.read_parquet(reference_path)
            )
    save_json(
        OUT / f"validation/{args.model}_seed{args.seed}_contribution.json", comparisons
    )
    elapsed = time.perf_counter() - started
    payload = {
        "model": args.model,
        "seed": args.seed,
        "metrics": metrics,
        "phase6_item_status": status,
        "checkpoint_epoch": checkpoint["epoch"],
        "candidate_universe": len(candidate_ids),
        "search_seconds": search_seconds,
        "search_backend": backend,
        **search_timing,
        "item_encode_seconds": item_encode_seconds,
        "query_encode_seconds": query_encode_seconds,
        "learned_alpha": model.alpha_values() if hasattr(model, "alpha_values") else {},
        "elapsed_seconds": elapsed,
        "budget_seconds": budget,
        "within_budget": elapsed <= budget,
        "test_opened": False,
    }
    save_json(OUT / f"validation/{args.model}_seed{args.seed}.json", payload)
    if args.seed == 42 and comparator:
        base_path = OUT / f"validation_rankings/{comparator}_seed42.parquet"
        base_per = pd.read_parquet(base_path)
        bootstrap = paired_bootstrap_segments(
            base_per,
            per,
            requests,
            (
                "overall",
                "warm",
                "cold",
                "train_target_seen",
                "train_history_only",
                "completely_unseen",
            ),
            old_exposed,
            targets,
            vocab,
            deadline=deadline,
        )
        save_json(
            OUT
            / f"validation/{args.model}_seed{args.seed}_vs_{comparator}_bootstrap.json",
            bootstrap,
        )
    # Conditional locks are based only on validation.
    update_conditional_locks()
    elapsed = time.perf_counter() - started
    payload["elapsed_seconds"] = elapsed
    payload["within_budget"] = elapsed <= budget
    save_json(OUT / f"validation/{args.model}_seed{args.seed}.json", payload)
    save_json(
        marker,
        {
            "complete": True,
            "result": f"validation/{args.model}_seed{args.seed}.json",
            "all_dependent_artifacts_complete": True,
            "within_budget": elapsed <= budget,
        },
    )
    print(json.dumps(payload, indent=2, default=json_default))
    if elapsed > budget:
        raise RuntimeError("formal validation exceeded protection budget")


def validation_recall(path: Path) -> float:
    return float(json.loads(path.read_text())["metrics"]["overall"]["Recall@500"])


def update_conditional_locks() -> None:
    b0_path = OUT / "validation/id_b0_user_history_control_seed42.json"
    if b0_path.exists():
        b0 = json.loads(b0_path.read_text())
        b0_warm = float(b0["metrics"]["warm_item"]["Recall@500"])
        values = {}
        for model in (
            "id_b1a_stable_profile",
            "id_b1b_full_profile",
            "id_b2_item_features",
        ):
            result_path = OUT / f"validation/{model}_seed42.json"
            bootstrap_path = (
                OUT
                / f"validation/{model}_seed42_vs_id_b0_user_history_control_bootstrap.json"
            )
            if result_path.exists() and bootstrap_path.exists():
                result = json.loads(result_path.read_text())
                bootstrap = json.loads(bootstrap_path.read_text())
                warm = float(result["metrics"]["warm_item"]["Recall@500"])
                values[model] = {
                    "warm_recall500": warm,
                    "warm_delta": warm - b0_warm,
                    "warm_ci95_lower": bootstrap["warm"]["ci95_lower"],
                    "eligible": warm - b0_warm >= 0.001
                    and bootstrap["warm"]["ci95_lower"] > 0,
                }
        if values:
            save_json(
                OUT / "locks/id_b3_eligible.json",
                {
                    "eligible": any(value["eligible"] for value in values.values()),
                    "minimum_practical_warm_delta": 0.001,
                    "baseline": "id_b0_user_history_control",
                    "baseline_warm_recall500": b0_warm,
                    "validation": values,
                    "test_opened": False,
                },
            )
    h1, h2 = (
        OUT / "validation/h1_side_features_seed42.json",
        OUT / "validation/h2_side_features_id_seed42.json",
    )
    if h1.exists() and h2.exists():
        bootstrap_path = (
            OUT
            / "validation/h2_side_features_id_seed42_vs_h1_side_features_bootstrap.json"
        )
        bootstrap = (
            json.loads(bootstrap_path.read_text()) if bootstrap_path.exists() else {}
        )
        h2_go = bool(
            bootstrap
            and bootstrap["overall"]["point_delta"] > 0
            and bootstrap["train_target_seen"]["point_delta"] > 0
            and bootstrap["completely_unseen"]["ci95_lower"]
            >= PROTOCOL.cold_noninferiority_margin
        )
        save_json(
            OUT / "locks/h3_eligible.json",
            {
                "eligible": h2_go,
                "h1": validation_recall(h1),
                "h2": validation_recall(h2),
                "comparison": "h2_side_features_id vs h1_side_features",
                "warm_segment": "train_target_seen",
                "cold_segment": "completely_unseen",
                "cold_noninferiority_margin": PROTOCOL.cold_noninferiority_margin,
                "test_opened": False,
            },
        )


def refresh_bootstrap_artifacts(model_filter: str | None = None) -> None:
    """Recompute paired CIs from saved rankings without repeating retrieval."""
    _, valid, store = resources()
    requests = grouped_requests(valid)
    old_exposed, _, _ = legacy_evaluator_sets()
    targets = set(map(int, store.train_target_item_ids))
    vocab = set(map(int, store.train_item_id_vocab))
    comparators = {
        "id_b1a_stable_profile": "id_b0_user_history_control",
        "id_b1b_full_profile": "id_b0_user_history_control",
        "id_b2_item_features": "id_b0_user_history_control",
        "id_b3_all": "id_b0_user_history_control",
        "h1_side_features": "h0_content_control",
        "h2_side_features_id": "h1_side_features",
        "h3_anonymous_dense": "h2_side_features_id",
    }
    refreshed = []
    for model, comparator in comparators.items():
        if model_filter is not None and model != model_filter:
            continue
        candidate_path = OUT / f"validation_rankings/{model}_seed42.parquet"
        base_path = OUT / f"validation_rankings/{comparator}_seed42.parquet"
        if not candidate_path.exists() or not base_path.exists():
            continue
        candidate = pd.read_parquet(candidate_path)
        base = pd.read_parquet(base_path)
        bootstrap = paired_bootstrap_segments(
            base,
            candidate,
            requests,
            (
                "overall",
                "warm",
                "cold",
                "train_target_seen",
                "train_history_only",
                "completely_unseen",
            ),
            old_exposed,
            targets,
            vocab,
        )
        path = OUT / f"validation/{model}_seed42_vs_{comparator}_bootstrap.json"
        save_json(path, bootstrap)
        refreshed.append(str(path.relative_to(ROOT)))
    update_conditional_locks()
    print(json.dumps({"refreshed": refreshed}, indent=2))


def select_structure(family: str) -> None:
    allowed = ID_MODELS if family == "id" else HYBRID_MODELS
    rows = []
    for path in sorted((OUT / "validation").glob("*_seed42.json")):
        value = json.loads(path.read_text())
        model = value.get("model")
        marker = OUT / f"locks/full_validation_{model}_seed42.json"
        complete = marker.exists() and json.loads(marker.read_text()).get("complete")
        if "metrics" in value and model in allowed and complete:
            rows.append((float(value["metrics"]["overall"]["Recall@500"]), model))
    if not rows:
        raise SystemExit(f"no seed42 full validation for {family}")
    score, model = max(rows)
    save_json(
        OUT / f"locks/selected_{family}_structure.json",
        {
            "model": model,
            "seed42_validation_recall500": score,
            "selection_metric": "full validation Recall@500",
            "test_opened": False,
        },
    )


def lock_final(args) -> None:
    results = []
    for path in sorted((OUT / "validation").glob("*.json")):
        value = json.loads(path.read_text())
        if "metrics" in value and value.get("model") in HYBRID_MODELS:
            results.append(
                (
                    value["metrics"]["overall"]["Recall@500"],
                    value["model"],
                    int(value["seed"]),
                    path,
                )
            )
    grouped = {}
    for score, model, seed, path in results:
        grouped.setdefault(model, []).append((score, seed, path))
    eligible = {
        model: rows
        for model, rows in grouped.items()
        if {seed for _, seed, _ in rows} >= {42, 43, 44}
    }
    if not eligible:
        raise SystemExit(
            "final lock requires one Hybrid structure with full validation seeds 42/43/44"
        )
    chosen_model, chosen_rows = max(
        eligible.items(), key=lambda item: np.mean([x[0] for x in item[1]])
    )
    chosen_rows = sorted(chosen_rows, key=lambda row: row[0])
    best = chosen_rows[
        len(chosen_rows) // 2
    ]  # terminal uses median seed, not lucky seed.
    save_json(
        OUT / "locks/final_model.json",
        {
            "model": chosen_model,
            "seed": best[1],
            "validation_recall500": best[0],
            "three_seed_mean_recall500": float(np.mean([x[0] for x in chosen_rows])),
            "three_seed_results": [
                {"seed": seed, "Recall@500": score} for score, seed, _ in chosen_rows
            ],
            "checkpoint": f"checkpoints/{chosen_model}_seed{best[1]}_best.pt",
            "selection_metric": "best three-seed mean; terminal checkpoint uses median seed",
            "test_opened": False,
            "quota_retuned": False,
        },
    )


def terminal_test(args) -> None:
    manifest = OUT / "terminal_test/manifest.json"
    if manifest.exists():
        raise SystemExit("terminal test manifest exists; one-shot test is locked")
    lock = json.loads((OUT / "locks/final_model.json").read_text())
    save_json(manifest, {"status": "running", "test_reads": 1, "locked_model": lock})
    _, _, store = resources()
    data = QilinData(ROOT)  # only test-reading path in Phase 7
    requests = data.load_test_requests()
    old_exposed, old_clicked, old_users = legacy_evaluator_sets()
    model, checkpoint = load_best(lock["model"], lock["seed"], store, args.device)
    candidate_ids = np.asarray(
        store.train_target_item_ids if lock["model"] in ID_MODELS else store.item_ids
    )
    item_started = time.perf_counter()
    items = encode_items(model, candidate_ids, store, args.device)
    item_encode_seconds = time.perf_counter() - item_started
    query_started = time.perf_counter()
    queries = encode_queries(model, requests, store, args.device)
    query_encode_seconds = time.perf_counter() - query_started
    _, rows, seconds, backend, search_timing = gpu_exact_search(
        items, queries, PROTOCOL.overfetch, args.device
    )
    rankings = filter_history(rows, candidate_ids, [r.history for r in requests])
    validate_topk(rankings, candidate_ids, [r.history for r in requests])
    targets, vocab, users = (
        set(map(int, store.train_target_item_ids)),
        set(map(int, store.train_item_id_vocab)),
        set(map(int, store.train_user_ids)),
    )
    metrics, status, per = phase6_status_metrics(
        requests,
        rankings,
        targets,
        vocab,
        users,
        lock["model"],
        old_exposed,
        old_clicked,
        old_users,
    )
    save_parquet_atomic(per, OUT / "terminal_test/per_request.parquet")
    result = {
        "model": lock["model"],
        "seed": lock["seed"],
        "metrics": metrics,
        "phase6_item_status": status,
        "search_seconds": seconds,
        "search_backend": backend,
        **search_timing,
        "item_encode_seconds": item_encode_seconds,
        "query_encode_seconds": query_encode_seconds,
        "learned_alpha": model.alpha_values() if hasattr(model, "alpha_values") else {},
        "checkpoint_epoch": checkpoint["epoch"],
        "test_reads": 1,
    }
    save_json(OUT / "terminal_test/result.json", result)
    save_json(manifest, {"status": "complete", "test_reads": 1, "locked_model": lock})


def report() -> None:
    training_seconds = {}
    for path in sorted((OUT / "configs").glob("*_seed*.json")):
        value = json.loads(path.read_text())
        training_seconds[(value["model"], int(value["seed"]))] = float(
            value.get("elapsed_seconds", float("nan"))
        )
    rows = []
    for path in sorted((OUT / "validation").glob("*.json")):
        value = json.loads(path.read_text())
        if "metrics" not in value or "model" not in value:
            continue
        marker = (
            OUT / f"locks/full_validation_{value['model']}_seed{value['seed']}.json"
        )
        if not marker.exists() or not json.loads(marker.read_text()).get("complete"):
            continue
        metrics = value["metrics"]
        rows.append(
            {
                "model": value["model"],
                "seed": int(value["seed"]),
                "r100": metrics["overall"]["Recall@100"],
                "r500": metrics["overall"]["Recall@500"],
                "warm": metrics["warm_item"]["Recall@500"],
                "cold": metrics["cold_item"]["Recall@500"],
                "warm_user": metrics["warm_user"]["Recall@500"],
                "cold_user": metrics["cold_user"]["Recall@500"],
                "target_seen": value["phase6_item_status"]["train_target_seen"][
                    "Recall@500"
                ],
                "history_only": value["phase6_item_status"]["train_history_only"][
                    "Recall@500"
                ],
                "unseen": value["phase6_item_status"]["completely_unseen"][
                    "Recall@500"
                ],
                "alpha": value.get("learned_alpha", {}),
                "item_encode": value.get("item_encode_seconds", float("nan")),
                "query_encode": value.get("query_encode_seconds", float("nan")),
                "index_build": value.get("index_build_seconds", float("nan")),
                "query_search": value.get("query_search_seconds", float("nan")),
                "training": training_seconds.get(
                    (value["model"], int(value["seed"])), float("nan")
                ),
            }
        )
    order = {name: index for index, name in enumerate(MODEL_NAMES)}
    rows.sort(key=lambda row: (order.get(row["model"], 999), row["seed"]))
    lines = [
        "# Phase 7：Feature-enhanced Hybrid Two-Tower",
        "",
        "> 本报告由已授权运行产生的落盘结果生成；缺失阶段不会被推测。",
        "",
        "| Model | Seed | R@100 | R@500 |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row['model']} | {row['seed']} | {row['r100']:.4%} | {row['r500']:.4%} |"
        for row in rows
    ]
    lines += [
        "",
        "## Temporal item status",
        "",
        "| Model | Seed | Target-seen | History-only | Completely-unseen |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row['model']} | {row['seed']} | {row['target_seen']:.4%} | "
        f"{row['history_only']:.4%} | {row['unseen']:.4%} |"
        for row in rows
    ]

    lines += [
        "",
        "## 三 seed 稳定性",
        "",
        "| Model | Seeds | Mean R@500 | Std R@500 |",
        "| --- | --- | ---: | ---: |",
    ]
    for model in MODEL_NAMES:
        scores = [row["r500"] for row in rows if row["model"] == model]
        seeds = [row["seed"] for row in rows if row["model"] == model]
        if scores:
            lines.append(
                f"| {model} | {','.join(map(str, seeds))} | {np.mean(scores):.4%} | {np.std(scores):.4%} |"
            )

    lines += [
        "",
        "## 逐级受控 Bootstrap",
        "",
        "| Comparison | Segment | Delta | CI95 |",
        "| --- | --- | ---: | --- |",
    ]
    for path in sorted((OUT / "validation").glob("*_bootstrap.json")):
        value = json.loads(path.read_text())
        for segment in (
            "overall",
            "train_target_seen",
            "train_history_only",
            "completely_unseen",
        ):
            if segment in value:
                row = value[segment]
                lines.append(
                    f"| {path.stem} | {segment} | {row['point_delta']:.4%} | "
                    f"[{row['ci95_lower']:.4%}, {row['ci95_upper']:.4%}] |"
                )

    lines += [
        "",
        "## Learned alpha 与耗时",
        "",
        "| Model | Seed | Alpha | Train(s) | Item encode(s) | Query encode(s) | Index build(s) | Query search(s) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['seed']} | `{json.dumps(row['alpha'])}` | "
            f"{row['training']:.1f} | {row['item_encode']:.1f} | "
            f"{row['query_encode']:.1f} | {row['index_build']:.1f} | "
            f"{row['query_search']:.1f} |"
        )
    lines += [
        "",
        "## 固定参考",
        "",
        "- Validation：Phase 5 Content 9.0943%，Pure-ID logQ 2.1812%。",
        "- Terminal test：Phase 5 Content 8.1282%，Pure-ID logQ 1.5147%。",
        "- 本阶段按当前实验约束只比较双塔模型本身，不执行或讨论融合。",
        "- temporal validation 上旧全量-train warm/cold 会退化为全 warm；模型选择使用 temporal train-target-seen / completely-unseen。",
        "",
        "## 状态",
        "",
        "- BGE 始终冻结；test 只允许 terminal-test 路径读取。",
    ]
    id_lock = OUT / "locks/selected_id_structure.json"
    hybrid_lock = OUT / "locks/final_model.json"
    b3_lock = OUT / "locks/id_b3_eligible.json"
    h3_lock = OUT / "locks/h3_eligible.json"
    lines += ["", "## 自动决策依据", ""]
    lines.append(
        f"- 独立 ID selected structure：`{id_lock.read_text().strip() if id_lock.exists() else '尚未锁定'}`"
    )
    lines.append(
        f"- B3 eligibility：`{b3_lock.read_text().strip() if b3_lock.exists() else '尚未判断'}`"
    )
    lines.append(
        f"- H3 eligibility：`{h3_lock.read_text().strip() if h3_lock.exists() else '尚未判断'}`"
    )
    lines.append(
        f"- 最终 Hybrid lock：`{hybrid_lock.read_text().strip() if hybrid_lock.exists() else '尚未锁定'}`"
    )
    lines += ["", "## 自动决策", ""]
    id_decision = "尚不可判断：缺少结构锁或三 seed 正式 validation。"
    if id_lock.exists():
        selected_id = json.loads(id_lock.read_text())["model"]
        selected_rows = [row for row in rows if row["model"] == selected_id]
        if {row["seed"] for row in selected_rows} >= {42, 43, 44}:
            b0 = next(
                (
                    row
                    for row in rows
                    if row["model"] == "id_b0_user_history_control"
                    and row["seed"] == 42
                ),
                None,
            )
            selected_42 = next(
                (row for row in selected_rows if row["seed"] == 42), None
            )
            warm_gain = bool(
                b0 is not None
                and selected_42 is not None
                and selected_42["warm"] > b0["warm"]
            )
            seed_scores = [row["r500"] for row in selected_rows]
            if warm_gain and np.std(seed_scores) <= 0.002:
                id_decision = (
                    f"GO：`{selected_id}` 相对匹配 B0 的 warm recall 提升，"
                    f"且三 seed 方向稳定（std={np.std(seed_scores):.4%}）。"
                )
            else:
                id_decision = (
                    f"NO-GO：`{selected_id}` 未同时满足 warm 增益与三 seed 稳定性。"
                )
    lines.append(f"- 独立 ID route：{id_decision}")

    residual_decision = "尚不可判断：缺少 H2 vs H1 bootstrap。"
    residual_path = (
        OUT / "validation/h2_side_features_id_seed42_vs_h1_side_features_bootstrap.json"
    )
    if residual_path.exists():
        residual = json.loads(residual_path.read_text())
        residual_go = (
            residual["overall"]["point_delta"] > 0
            and residual["train_target_seen"]["point_delta"] > 0
            and residual["completely_unseen"]["ci95_lower"]
            >= PROTOCOL.cold_noninferiority_margin
        )
        residual_decision = (
            "GO：H2 相对 H1 有 overall/temporal-warm 正增益，且 completely-unseen 满足 non-inferiority。"
            if residual_go
            else "NO-GO：H2 相对 H1 未通过增量与 temporal cold non-inferiority 门槛。"
        )
    lines.append(f"- Hybrid 中 ID residual：{residual_decision}")

    replacement_decision = "尚不可判断：缺少三 seed final Hybrid lock。"
    if hybrid_lock.exists():
        locked = json.loads(hybrid_lock.read_text())
        mean_recall = float(locked["three_seed_mean_recall500"])
        content_validation = 0.09094312597039826
        delta = mean_recall - content_validation
        replacement_decision = (
            f"单 Hybrid 三 seed mean R@500={mean_recall:.4%}，"
            f"相对既有 Content validation 增量 {delta:+.4%}。"
        )
    lines.append(f"- 特征增强 Hybrid 相对 Content：{replacement_decision}")
    terminal_path = OUT / "terminal_test/result.json"
    if terminal_path.exists():
        terminal = json.loads(terminal_path.read_text())
        terminal_overall = terminal["metrics"]["overall"]
        terminal_status = terminal["phase6_item_status"]
        lines += [
            "",
            "## Terminal Test（配置锁定后唯一一次）",
            "",
            f"- Model：`{terminal['model']}`，seed={terminal['seed']}，checkpoint epoch={terminal['checkpoint_epoch']}。",
            f"- Recall@100：{terminal_overall['Recall@100']:.4%}。",
            f"- Recall@500：{terminal_overall['Recall@500']:.4%}。",
            f"- Train-target-seen Recall@500：{terminal_status['train_target_seen']['Recall@500']:.4%}。",
            f"- History-only Recall@500：{terminal_status['train_history_only']['Recall@500']:.4%}。",
            f"- Completely-unseen Recall@500：{terminal_status['completely_unseen']['Recall@500']:.4%}。",
            f"- GPU resident exact search：{terminal['search_seconds']:.3f}s；item encode：{terminal['item_encode_seconds']:.3f}s；query encode：{terminal['query_encode_seconds']:.3f}s。",
            "- Test 读取次数：1；结果未用于重新调参。",
        ]
    temporary = OUT / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(OUT / "summary.md")


def main() -> None:
    args = parse_args()
    require_authorization(args)
    if args.stage == "plan":
        return plan()
    ensure_output_layout()
    if args.stage == "stage-a":
        print(json.dumps(build_feature_cache(), indent=2))
    elif args.stage == "h3-cache":
        print(json.dumps(build_h3_dense_cache(), indent=2))
    elif args.stage == "smoke":
        smoke(args)
    elif args.stage == "train":
        train(args)
    elif args.stage == "full-validation":
        full_validation(args)
    elif args.stage == "refresh-bootstrap":
        refresh_bootstrap_artifacts(args.model)
    elif args.stage == "select":
        select_structure(args.family)
    elif args.stage == "lock":
        lock_final(args)
    elif args.stage == "terminal-test":
        terminal_test(args)
    elif args.stage == "report":
        report()


if __name__ == "__main__":
    main()
