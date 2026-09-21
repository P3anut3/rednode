#!/usr/bin/env python3
"""Phase 6: Pure-ID retrieval and frozen-content ID residual experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import joblib
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import (  # noqa: E402
    Phase6Dataset, audit_train_valid, build_contract, collate_phase6, grouped_requests,
    load_temporal_frames,
)
from experiments.phase_06.experiment_01_id_two_tower_retrieval.losses import (  # noqa: E402
    phase6_inbatch_loss,
)
from experiments.phase_06.experiment_01_id_two_tower_retrieval.models import (  # noqa: E402
    ContentIdResidualTower, PureIdTower,
)
from experiments.phase_01.experiment_01_full_corpus_baselines.recall.itemcf import ItemCF  # noqa: E402
from experiments.phase_04.experiment_01_learnable_user_tower_n10.models.user_tower import Phase4SingleAttention  # noqa: E402
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.trainer import phase4_encode_requests  # noqa: E402
from experiments.common.dense_recall import make_row_lookup, rows_to_filtered_rankings  # noqa: E402
from experiments.phase_05.experiment_01_request_hard_negative_mining.mining import _required_history_texts, _top_sparse_with_scores  # noqa: E402

OUT = ROOT / "results/phase_06/experiment_01_id_two_tower_retrieval"
P25 = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark"
NOTE_IDS_PATH = P25 / "note_ids.npy"
BGE_PATH = P25 / "embeddings/bge_base_zh/basic_clean/embeddings.f16"
FROZEN = (ROOT / "results/phase_04/experiment_02_history_n20_multi_interest/"
          "baseline_releases/v1_frozen_bge_n20_single_attention")
PHASE5_BEST = (ROOT / "results/phase_05/experiment_02_bias_aware_hard_negative_training/"
               "checkpoints/tfidf_hard_best.pt")
TFIDF_CACHE = ROOT / "results/phase_05/experiment_01_request_hard_negative_mining/cache/tfidf"
PURE_MODELS = ("m0_user_mf", "m1_history_mean", "m1a_history_attention",
               "m1b_user_history_attention")
DIM, BGE_DIM, HISTORY_N = 128, 768, 20


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "pure-id", "pure-seeds",
                                             "baselines", "m2", "negatives", "fusion",
                                             "lock", "terminal-test", "report", "all"),
                        default="audit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--valid-requests", type=int, default=0,
                        help="0 uses every temporal-validation request")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--only-model", choices=PURE_MODELS, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-m2", choices=("m2_0_content_projection", "m2_1_binary_residual",
                                               "m2_2_frequency_gate", "m2_3_dropout_02",
                                               "m2_3_dropout_05"), default=None)
    parser.add_argument("--negative-route", choices=("pure", "m2"), default=None)
    parser.add_argument("--negative-objective", choices=("logq", "uniform8", "uniform16"),
                        default=None)
    return parser.parse_args()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_for(value: str) -> str:
    if value == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def resources():
    note_ids = np.load(NOTE_IDS_PATH, mmap_mode="r")
    train, valid = load_temporal_frames(ROOT)
    contract = build_contract(train, note_ids)
    return note_ids, train, valid, contract


def audit() -> dict:
    note_ids, train, valid, contract = resources()
    payload = audit_train_valid(train, valid, contract)
    payload.update({"catalog_items": int(len(note_ids)), "embedding_dim": BGE_DIM,
                    "id_dim": DIM, "history_n": HISTORY_N,
                    "test_fields_read": False})
    save_json(OUT / "audit/data_audit_train_valid.json", payload)
    target_counts = train.positive_item_id.value_counts().rename_axis("note_id").rename(
        "train_target_count").reset_index()
    target_counts.to_parquet(OUT / "audit/train_target_frequency.parquet", index=False,
                             compression="zstd")
    np.save(OUT / "audit/id_vocab_note_ids.npy", contract.vocab_ids)
    np.save(OUT / "audit/pure_id_candidate_note_ids.npy", contract.candidate_ids)
    np.save(OUT / "audit/train_user_ids.npy", contract.train_users)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def subset_requests(requests: list[TestRequest], size: int, seed: int = 42):
    if not size or size >= len(requests): return requests
    positions = np.sort(np.random.default_rng(seed).choice(len(requests), size, replace=False))
    return [requests[int(pos)] for pos in positions]


def batch_to(batch: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def train_epoch(model, dataset, optimizer, device: str, batch_size: int, epoch: int,
                target_probability: dict[int, float] | None = None,
                max_batches: int | None = None, uniform_count: int = 0,
                uniform_catalog: np.ndarray | None = None,
                user_known: dict[int, set[int]] | None = None,
                contract=None, note_lookup: np.ndarray | None = None,
                bge=None) -> dict:
    generator = torch.Generator().manual_seed(42 + epoch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=2, persistent_workers=True, pin_memory=True,
                        collate_fn=collate_phase6, generator=generator)
    model.train(); losses, uniform_losses, masks = [], [], np.zeros(3, dtype=np.int64)
    rng = np.random.default_rng(42 + epoch)
    for step, cpu in enumerate(loader):
        if max_batches is not None and step >= max_batches: break
        batch = batch_to(cpu, device)
        query, target = model(batch)
        log_q = None
        if target_probability is not None:
            log_q = torch.as_tensor([np.log(target_probability[int(note)] + 1e-12)
                                     for note in batch["target_note_id"].detach().cpu().tolist()],
                                    dtype=query.dtype, device=device)
        loss, stats = phase6_inbatch_loss(query, target, batch, log_q=log_q)
        if uniform_count:
            if uniform_catalog is None or user_known is None:
                raise ValueError("uniform negative resources missing")
            sampled = np.empty((len(query), uniform_count), dtype=np.int64)
            target_cpu = batch["target_note_id"].detach().cpu().numpy()
            history_cpu = batch["history_note_ids"].detach().cpu().numpy()
            users_cpu = batch["user_id"].detach().cpu().numpy()
            same_request = batch["same_request_positive_ids"]
            for row in range(len(query)):
                blocked = set(map(int, history_cpu[row][history_cpu[row] >= 0]))
                blocked.update(map(int, same_request[row])); blocked.add(int(target_cpu[row]))
                blocked.update(user_known.get(int(users_cpu[row]), set()))
                chosen = []
                while len(chosen) < uniform_count:
                    draws = rng.choice(uniform_catalog, size=(uniform_count-len(chosen))*3,
                                       replace=True)
                    for note in draws:
                        note = int(note)
                        if note not in blocked and note not in chosen:
                            chosen.append(note)
                            if len(chosen) == uniform_count: break
                sampled[row] = chosen
            sampled_gpu = torch.from_numpy(sampled).to(device)
            if isinstance(model, PureIdTower):
                mapped = contract.item_to_vocab[sampled]
                neg = model.target(torch.from_numpy(mapped + 1).to(device))
            else:
                rows = note_lookup[sampled]
                content = torch.from_numpy(np.asarray(bge[rows], dtype=np.float32)).to(device)
                neg = model.represent(content, sampled_gpu)
            pos_score = (query * target).sum(-1, keepdim=True)
            neg_score = torch.einsum("bd,bnd->bn", query, neg)
            explicit = F.cross_entropy(torch.cat((pos_score, neg_score), dim=1) / 0.05,
                                       torch.zeros(len(query), dtype=torch.long, device=device))
            loss = loss + explicit
            uniform_losses.append(float(explicit.detach()))
        if not torch.isfinite(loss): raise FloatingPointError("non-finite Phase 6 loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
        masks += [stats["masked_duplicate"], stats["masked_history"],
                  stats["masked_same_request"]]
    return {"train_loss": float(np.mean(losses)), "batches": len(losses),
            "uniform_loss": float(np.mean(uniform_losses)) if uniform_losses else 0.0,
            "uniform_negatives": uniform_count,
            "masked_duplicate": int(masks[0]), "masked_history": int(masks[1]),
            "masked_same_request": int(masks[2])}


def request_batch_arrays(requests, contract, note_lookup=None, bge=None):
    n = len(requests)
    history_vocab = np.zeros((n, HISTORY_N), dtype=np.int64)
    history_note = np.full((n, HISTORY_N), -1, dtype=np.int64)
    id_mask = np.zeros((n, HISTORY_N), dtype=np.bool_)
    user_row = np.zeros(n, dtype=np.int64)
    content, content_mask = None, None
    if bge is not None:
        content = np.zeros((n, HISTORY_N, BGE_DIM), dtype=np.float32)
        content_mask = np.zeros((n, HISTORY_N), dtype=np.bool_)
    for row, request in enumerate(requests):
        user_row[row] = contract.user_to_row.get(int(request.user_idx), -1) + 1
        history = request.history[-HISTORY_N:]
        for col, note in enumerate(history):
            history_note[row, col] = note
            mapped = int(contract.item_to_vocab[note]) if 0 <= note < len(contract.item_to_vocab) else -1
            if mapped >= 0:
                history_vocab[row, col] = mapped + 1; id_mask[row, col] = True
            if bge is not None:
                note_row = int(note_lookup[note]) if 0 <= note < len(note_lookup) else -1
                if note_row >= 0:
                    content[row, col] = bge[note_row]; content_mask[row, col] = True
    return history_vocab, history_note, id_mask, user_row, content, content_mask


@torch.inference_mode()
def encode_pure_queries(model, requests, contract, device: str, batch_size: int = 1024):
    arrays = request_batch_arrays(requests, contract)
    output = []
    model.eval()
    for start in range(0, len(requests), batch_size):
        stop = start + batch_size
        history = torch.from_numpy(arrays[0][start:stop]).to(device)
        mask = torch.from_numpy(arrays[2][start:stop]).to(device)
        users = torch.from_numpy(arrays[3][start:stop]).to(device)
        query = model.query(history, mask, users)
        # A closed branch must not create arbitrary tie rankings from a zero query.
        active = users.ne(0) if model.kind == "m0_user_mf" else mask.any(1)
        query = query * active.unsqueeze(-1)
        output.append(query.cpu().numpy())
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


def filtered_rankings(requests, raw_rows: np.ndarray, candidate_ids: np.ndarray,
                      active: np.ndarray, k: int = 500) -> list[list[int]]:
    rankings = []
    for request, rows, enabled in zip(requests, raw_rows, active):
        if not enabled:
            rankings.append([]); continue
        blocked = set(map(int, request.history)); selected, seen = [], set()
        for row in rows:
            if row < 0: continue
            note = int(candidate_ids[int(row)])
            if note in blocked or note in seen: continue
            selected.append(note); seen.add(note)
            if len(selected) == k: break
        rankings.append(selected)
    return rankings


def exact_search(vectors: np.ndarray, queries: np.ndarray, topk: int, threads: int):
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    begin = time.perf_counter(); _, rows = index.search(queries, min(topk, len(vectors)))
    return rows, time.perf_counter() - begin


def phase_status_metrics(requests, rankings, contract, method):
    target = set(map(int, contract.candidate_ids))
    vocab = set(map(int, contract.vocab_ids)); history_only = vocab - target
    metrics, per = evaluate_rankings(requests, rankings, target,
                                     set(map(int, contract.train_users)), method,
                                     clicked_items=target)
    extra = {}
    for name, allowed in (("train_target_seen", target), ("train_history_only", history_only),
                          ("completely_unseen", None)):
        remapped = []
        for request in requests:
            truth = set(request.ground_truth)
            selected = truth - vocab if allowed is None else truth & allowed
            if selected:
                remapped.append(TestRequest(request.request_idx, request.user_idx,
                                            request.history, frozenset(selected)))
        ranking_map = {r.request_idx: rank for r, rank in zip(requests, rankings)}
        values = [ranking_map[r.request_idx] for r in remapped]
        result, _ = evaluate_rankings(remapped, values, set(), set(), method + "_" + name)
        extra[name] = result["overall"]
    return metrics, extra, per


def evaluate_pure(model, requests, contract, device, threads, method, save_rankings=False):
    queries = encode_pure_queries(model, requests, contract, device)
    candidate_vocab = contract.item_to_vocab[contract.candidate_ids].astype(np.int64) + 1
    with torch.inference_mode():
        vectors = model.target(torch.from_numpy(candidate_vocab).to(device)).cpu().numpy()
    active = np.linalg.norm(queries, axis=1) > 0
    rows, seconds = exact_search(vectors, queries, min(700, len(vectors)), threads)
    rankings = filtered_rankings(requests, rows, contract.candidate_ids, active)
    metrics, status, per = phase_status_metrics(requests, rankings, contract, method)
    if save_rankings:
        path = OUT / "validation_rankings" / f"{method}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True); per.to_parquet(path, index=False,
                                                                      compression="zstd")
    return {"method": method, "search_seconds": seconds, "active_requests": int(active.sum()),
            "candidate_universe_size": len(contract.candidate_ids), "metrics": metrics,
            "phase6_item_status": status}, rankings


def save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def train_pure_one(kind: str, seed: int, config, smoke: bool = False) -> dict:
    seed_all(seed); note_ids, train, valid, contract = resources()
    if smoke:
        train = train.iloc[:5000].copy(); valid = valid.iloc[:1000].copy()
    all_requests = grouped_requests(valid)
    selection_size = config.valid_requests or 5000
    requests = subset_requests(all_requests, selection_size, seed=42)
    dataset = Phase6Dataset(train, contract, note_ids, history_n=HISTORY_N)
    device = device_for(config.device)
    model = PureIdTower(kind, len(contract.vocab_ids), len(contract.train_users), DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, stale, curves = -1.0, 0, []
    epochs = 2 if smoke else config.epochs
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_log = train_epoch(model, dataset, optimizer, device, config.batch_size, epoch,
                                max_batches=12 if smoke else None)
        payload, _ = evaluate_pure(model, requests, contract, device, config.threads,
                                   f"{kind}_seed{seed}")
        overall = payload["metrics"]["overall"]
        row = {"model": kind, "seed": seed, "epoch": epoch, **train_log,
               "Recall@100": overall["Recall@100"], "Recall@500": overall["Recall@500"],
               "MRR@100": overall["MRR@100"], "epoch_seconds": time.perf_counter() - started}
        curves.append(row); print(json.dumps(row), flush=True)
        if row["Recall@500"] > best:
            best, stale = row["Recall@500"], 0
            save_checkpoint(OUT / "checkpoints" / f"{kind}_seed{seed}_best.pt", model,
                            {**row, "dim": DIM, "history_n": HISTORY_N,
                             "candidate_universe_size": len(contract.candidate_ids)})
        else:
            stale += 1
            if stale >= config.patience: break
    curve_path = OUT / ("smoke" if smoke else "training_curves") / f"{kind}_seed{seed}.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(curves).to_csv(curve_path, index=False)
    checkpoint = torch.load(OUT / "checkpoints" / f"{kind}_seed{seed}_best.pt",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final_requests = requests if smoke else all_requests
    result, _ = evaluate_pure(model, final_requests, contract, device, config.threads,
                              f"{kind}_seed{seed}", save_rankings=not smoke)
    result.update({"seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "best_epoch": checkpoint["metadata"]["epoch"],
                   "selection_validation_requests": len(requests),
                   "full_validation_requests": len(final_requests)})
    save_json(OUT / ("smoke" if smoke else "validation") / f"{kind}_seed{seed}.json", result)
    return result


def run_pure(config, smoke=False):
    audit()
    selected = ((config.only_model,) if config.only_model else
                (PURE_MODELS if not smoke else ("m1_history_mean", "m1a_history_attention")))
    results = [train_pure_one(kind, config.seed, config, smoke=smoke) for kind in selected]
    if not smoke:
        rows = []
        for value in results:
            metric = value["metrics"]["overall"]
            rows.append({"method": value["method"], "seed": value["seed"],
                         **{f"Recall@{k}": metric[f"Recall@{k}"] for k in (10,50,100,200,500)},
                         "MRR@100": metric["MRR@100"],
                         "Warm-item Recall@500": value["phase6_item_status"]["train_target_seen"]["Recall@500"],
                         "Warm-user Recall@500": value["metrics"]["warm_user"]["Recall@500"],
                         "Cold-user Recall@500": value["metrics"]["cold_user"]["Recall@500"],
                         "parameters": value["parameters"]})
        result_path = OUT / "pure_id_structure_validation.csv"
        if result_path.exists():
            old = pd.read_csv(result_path)
            frame = pd.concat((old, pd.DataFrame(rows)), ignore_index=True)
            frame = frame.drop_duplicates(["method", "seed"], keep="last")
        else:
            frame = pd.DataFrame(rows)
        frame = frame.sort_values("Recall@500", ascending=False)
        frame.to_csv(result_path, index=False)
        seed42 = frame[frame.seed == 42]
        if set(value.rsplit("_seed", 1)[0] for value in seed42.method) == set(PURE_MODELS):
            lock = {"best_structure": seed42.iloc[0].method.rsplit("_seed", 1)[0],
                    "selection_metric": "validation Recall@500", "seed": 42,
                    "test_read": False}
            save_json(OUT / "locks/pure_id_structure.json", lock)
    return results


def pure_seeds(config):
    lock = json.loads((OUT / "locks/pure_id_structure.json").read_text())
    kind = lock["best_structure"]
    results = []
    for seed in (42, 43, 44):
        path = OUT / "validation" / f"{kind}_seed{seed}.json"
        results.append(json.loads(path.read_text()) if path.exists()
                       else train_pure_one(kind, seed, config))
    rows = [{"seed": value["seed"], "Recall@100": value["metrics"]["overall"]["Recall@100"],
             "Recall@500": value["metrics"]["overall"]["Recall@500"],
             "MRR@100": value["metrics"]["overall"]["MRR@100"]} for value in results]
    pd.DataFrame(rows).to_csv(OUT / "pure_id_seed_stability.csv", index=False)
    median_seed = int(pd.DataFrame(rows).sort_values("Recall@500").iloc[1].seed)
    save_json(OUT / "locks/pure_id_final.json", {"structure": kind, "seeds": [42,43,44],
              "terminal_checkpoint_seed": median_seed,
              "policy": "median validation seed avoids lucky-seed selection"})


def residual_contract_arrays(train: pd.DataFrame, contract):
    """Map only temporal-train positive targets to residual rows; history-only stays zero."""
    lookup = np.zeros(len(contract.item_to_vocab), dtype=np.int64)
    lookup[contract.candidate_ids] = np.arange(1, len(contract.candidate_ids) + 1,
                                                dtype=np.int64)
    counts = train.positive_item_id.value_counts()
    frequency = np.zeros(len(contract.candidate_ids) + 1, dtype=np.float32)
    frequency[1:] = np.asarray([counts.get(int(note), 0) for note in contract.candidate_ids],
                               dtype=np.float32)
    return lookup, frequency


def make_m2(name: str, train: pd.DataFrame, contract, device: str):
    lookup, frequency = residual_contract_arrays(train, contract)
    residual = name != "m2_0_content_projection"
    gate = np.ones(len(frequency), dtype=np.float32)
    if name == "m2_2_frequency_gate":
        gate[1:] = np.log1p(frequency[1:]) / max(np.log1p(frequency[1:].max()), 1.0)
    dropout = 0.2 if name.endswith("dropout_02") else (0.5 if name.endswith("dropout_05") else 0.0)
    model = ContentIdResidualTower(
        len(contract.candidate_ids), len(contract.train_users),
        torch.from_numpy(lookup), DIM, BGE_DIM, residual=residual,
        frequency_gate=torch.from_numpy(gate), id_dropout=dropout).to(device)
    if not residual:
        for module in (model.item_residual, model.user, model.user_projection):
            for parameter in module.parameters(): parameter.requires_grad_(False)
        model.alpha_raw.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_m2_queries(model, requests, contract, note_ids, bge, device: str,
                      batch_size: int = 256):
    max_id = int(note_ids.max()); note_lookup = np.full(max_id + 1, -1, dtype=np.int32)
    note_lookup[note_ids] = np.arange(len(note_ids), dtype=np.int32)
    arrays = request_batch_arrays(requests, contract, note_lookup, bge)
    output = []
    model.eval()
    for start in range(0, len(requests), batch_size):
        stop = start + batch_size
        content = torch.from_numpy(arrays[4][start:stop]).to(device)
        notes = torch.from_numpy(arrays[1][start:stop]).to(device)
        mask = torch.from_numpy(arrays[5][start:stop]).to(device)
        users = torch.from_numpy(arrays[3][start:stop]).to(device)
        output.append(model.query(content, notes, mask, users).cpu().numpy())
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


@torch.inference_mode()
def build_m2_index(model, note_ids, bge, device: str, cache_name: str,
                   threads: int, batch_size: int = 4096):
    cache = OUT / "m2_vectors" / f"{cache_name}.f16"
    cache.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(note_ids), DIM)
    vectors = np.memmap(cache, mode="w+", dtype=np.float16, shape=shape)
    model.eval(); encode_start = time.perf_counter()
    for start in range(0, len(note_ids), batch_size):
        stop = min(start + batch_size, len(note_ids))
        content = torch.from_numpy(np.asarray(bge[start:stop], dtype=np.float32)).to(device)
        notes = torch.from_numpy(np.asarray(note_ids[start:stop], dtype=np.int64)).to(device)
        vectors[start:stop] = model.represent(content, notes).cpu().numpy().astype(np.float16)
    vectors.flush(); encode_seconds = time.perf_counter() - encode_start
    faiss.omp_set_num_threads(threads); index = faiss.IndexFlatIP(DIM)
    index_start = time.perf_counter()
    for start in range(0, len(note_ids), 100_000):
        index.add(np.ascontiguousarray(vectors[start:start+100_000], dtype=np.float32))
    return index, vectors, {"vector_encode_seconds": encode_seconds,
                            "index_build_seconds": time.perf_counter() - index_start,
                            "vector_file": str(cache.relative_to(ROOT)),
                            "vector_bytes": cache.stat().st_size}


def evaluate_m2(model, requests, train, contract, note_ids, bge, device, threads,
                method, cache_name, save_rankings=False):
    queries = encode_m2_queries(model, requests, contract, note_ids, bge, device)
    index, _, timing = build_m2_index(model, note_ids, bge, device, cache_name, threads)
    begin = time.perf_counter(); raw = []
    for start in range(0, len(queries), 128):
        _, rows = index.search(queries[start:start+128], 650); raw.append(rows)
    timing["search_seconds"] = time.perf_counter() - begin
    active = np.linalg.norm(queries, axis=1) > 0
    rankings = filtered_rankings(requests, np.vstack(raw), note_ids, active)
    metrics, status, per = phase_status_metrics(requests, rankings, contract, method)
    if save_rankings:
        path = OUT / "validation_rankings" / f"{method}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True); per.to_parquet(path, index=False,
                                                                      compression="zstd")
    return {"method": method, "metrics": metrics, "phase6_item_status": status,
            "candidate_universe_size": len(note_ids), **timing}, rankings


def train_m2_one(name: str, config) -> dict:
    seed_all(config.seed); note_ids, train, valid, contract = resources()
    device = device_for(config.device)
    bge = np.memmap(BGE_PATH, mode="r", dtype=np.float16, shape=(len(note_ids), BGE_DIM))
    all_requests = grouped_requests(valid)
    selection = subset_requests(all_requests, config.valid_requests or 5000, seed=42)
    dataset = Phase6Dataset(train, contract, note_ids, BGE_PATH, BGE_DIM, HISTORY_N)
    model = make_m2(name, train, contract, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4, weight_decay=1e-4)
    best, stale, curves = -1.0, 0, []
    for epoch in range(1, min(config.epochs, 6) + 1):
        started = time.perf_counter()
        log = train_epoch(model, dataset, optimizer, device, min(config.batch_size, 256), epoch)
        payload, _ = evaluate_m2(model, selection, train, contract, note_ids, bge, device,
                                 config.threads, name, f"selection_{name}_epoch{epoch}")
        metric = payload["metrics"]["overall"]
        row = {"model": name, "epoch": epoch, **log,
               "Recall@100": metric["Recall@100"], "Recall@500": metric["Recall@500"],
               "MRR@100": metric["MRR@100"], "alpha": float(model.alpha.detach()),
               "epoch_seconds": time.perf_counter() - started,
               **{key: payload[key] for key in ("vector_encode_seconds", "index_build_seconds",
                                                "search_seconds")}}
        curves.append(row); print(json.dumps(row), flush=True)
        if row["Recall@500"] > best:
            best, stale = row["Recall@500"], 0
            save_checkpoint(OUT / "checkpoints" / f"{name}_best.pt", model,
                            {**row, "history_n": HISTORY_N, "dim": DIM,
                             "cold_noninferiority_margin_absolute": -0.002})
        else:
            stale += 1
            if stale >= config.patience: break
    path = OUT / "training_curves" / f"{name}.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(path, index=False)
    checkpoint = torch.load(OUT / "checkpoints" / f"{name}_best.pt", map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    result, _ = evaluate_m2(model, all_requests, train, contract, note_ids, bge, device,
                            config.threads, name, f"best_{name}", save_rankings=True)
    norm_stats = None
    if model.residual_enabled:
        sample_ids = contract.candidate_ids[:min(20_000, len(contract.candidate_ids))]
        note_lookup = np.full(int(note_ids.max()) + 1, -1, dtype=np.int32)
        note_lookup[note_ids] = np.arange(len(note_ids), dtype=np.int32)
        with torch.inference_mode():
            content = torch.from_numpy(np.asarray(bge[note_lookup[sample_ids]], dtype=np.float32)).to(device)
            ids_gpu = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64)).to(device)
            projected = model.content_projection(content)
            vocab = model.item_to_vocab[ids_gpu]
            residual = model.alpha * model.frequency_gate[vocab].unsqueeze(-1) * model.item_residual(vocab)
            norm_stats = {"sample_items": len(sample_ids),
                          "mean_content_projection_norm": float(projected.norm(dim=-1).mean()),
                          "mean_scaled_residual_norm": float(residual.norm(dim=-1).mean()),
                          "residual_to_content_norm_ratio": float(
                              residual.norm(dim=-1).mean()/projected.norm(dim=-1).mean())}
    result.update({"best_epoch": checkpoint["metadata"]["epoch"],
                   "alpha": float(model.alpha.detach()),
                   "norm_diagnostics": norm_stats,
                   "total_parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "validation_requests": len(all_requests), "test_read": False})
    save_json(OUT / "validation" / f"{name}.json", result)
    return result


def paired_bootstrap_delta(base_path: Path, treatment_path: Path, requests,
                           contract, segment: str, seed: int = 42, replicates: int = 10_000):
    base = pd.read_parquet(base_path); treatment = pd.read_parquet(treatment_path)
    base_map = {int(row.request_idx): row for row in base.itertuples(index=False)}
    trt_map = {int(row.request_idx): row for row in treatment.itertuples(index=False)}
    target_seen = set(map(int, contract.candidate_ids)); vocab = set(map(int, contract.vocab_ids))
    deltas = []
    for request in requests:
        truth = set(request.ground_truth)
        if segment == "warm": truth &= target_seen
        elif segment == "cold": truth -= vocab
        if not truth: continue
        def recall(row):
            ranking = set(row.retrieved_top500)
            return len(ranking & truth) / len(truth)
        deltas.append(recall(trt_map[request.request_idx]) - recall(base_map[request.request_idx]))
    values = np.asarray(deltas, dtype=np.float64); rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 500):
        size = min(500, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start+size] = values[indices].mean(1)
    return {"segment": segment, "eligible_requests": len(values), "point_delta": float(values.mean()),
            "ci95_lower": float(np.quantile(means, .025)),
            "ci95_upper": float(np.quantile(means, .975)), "replicates": replicates}


def run_m2(config):
    names = [config.only_m2] if config.only_m2 else ["m2_0_content_projection",
                                                      "m2_1_binary_residual"]
    for name in names: train_m2_one(name, config)
    base = OUT / "validation_rankings/m2_0_content_projection.parquet"
    residual = OUT / "validation_rankings/m2_1_binary_residual.parquet"
    if base.exists() and residual.exists():
        _, _, valid, contract = resources(); requests = grouped_requests(valid)
        bootstrap = {segment: paired_bootstrap_delta(base, residual, requests, contract, segment)
                     for segment in ("warm", "cold", "overall")}
        save_json(OUT / "m2_residual_bootstrap.json", bootstrap)
        go = (bootstrap["warm"]["ci95_lower"] > 0 and
              bootstrap["cold"]["ci95_lower"] > -0.002 and
              bootstrap["overall"]["point_delta"] >= 0)
        save_json(OUT / "locks/m2_residual_decision.json", {
            "go": go, "cold_noninferiority_margin_absolute": -0.002,
            "criteria": "warm CI lower > 0; cold CI lower > -0.002; overall point delta >= 0",
            "bootstrap": bootstrap, "test_read": False})


def temporal_popularity(train: pd.DataFrame) -> list[int]:
    counts = train.positive_item_id.value_counts()
    return sorted(map(int, counts.index), key=lambda note: (-int(counts[note]), note))


def static_filtered(rank: list[int], requests, k=500):
    output = []
    for request in requests:
        blocked = set(map(int, request.history)); selected = [note for note in rank if note not in blocked][:k]
        output.append(selected)
    return output


def build_itemcf(train: pd.DataFrame, requests, popularity):
    interactions = [set(map(int, group.positive_item_id))
                    for _, group in train.groupby("user_id", sort=False)]
    model = ItemCF(neighbors_per_item=200).fit(interactions)
    native = model.recommend(requests, 500, True, None)
    fallback = model.recommend(requests, 500, True, popularity)
    stats = vars(model.stats) if model.stats is not None else {}
    return native, fallback, stats


def _gpu_content_worker(rank: int, world_size: int, query_path: str, query_count: int,
                        topk: int, output_dir: str):
    torch.cuda.set_device(rank); device = f"cuda:{rank}"
    note_ids = np.load(NOTE_IDS_PATH, mmap_mode="r")
    embeddings = np.memmap(BGE_PATH, mode="r", dtype=np.float16,
                           shape=(len(note_ids), BGE_DIM))
    corpus = torch.from_numpy(np.asarray(embeddings)).to(device=device, dtype=torch.float32)
    queries = np.memmap(query_path, mode="r", dtype=np.float32,
                        shape=(query_count, BGE_DIM))
    indices = np.array_split(np.arange(query_count), world_size)[rank]
    output = np.empty((len(indices), topk), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, len(indices), 16):
            selected = indices[start:start+16]
            query = torch.from_numpy(np.asarray(queries[selected], dtype=np.float32)).to(device)
            scores = query @ corpus.T
            output[start:start+len(selected)] = torch.topk(scores, topk, dim=1, sorted=True).indices.cpu().numpy()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(output_dir) / f"part_{rank}.npz", positions=indices, rows=output)


def current_content_validation(requests, note_ids, device, threads, cache_tag="validation"):
    embeddings = np.memmap(BGE_PATH, mode="r", dtype=np.float16,
                           shape=(len(note_ids), BGE_DIM))
    lookup = make_row_lookup(note_ids)
    model = Phase4SingleAttention(BGE_DIM).to(device)
    checkpoint = torch.load(PHASE5_BEST, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    queries = phase4_encode_requests(model, requests, embeddings, lookup, device,
                                     history_n=20, batch_size=256)[:, 0]
    del model
    if device.startswith("cuda"): torch.cuda.empty_cache()
    cache = OUT / "cache/gpu_content_exact" / cache_tag
    cache.mkdir(parents=True, exist_ok=True)
    query_path = cache / "queries.f32"
    query_file = np.memmap(query_path, mode="w+", dtype=np.float32, shape=queries.shape)
    query_file[:] = queries; query_file.flush(); del query_file
    world_size = torch.cuda.device_count()
    if world_size < 1: raise RuntimeError("GPU exact content retrieval requires CUDA")
    start = time.perf_counter()
    torch.multiprocessing.spawn(_gpu_content_worker,
        args=(world_size, str(query_path), len(queries), 1100, str(cache / "parts")),
        nprocs=world_size, join=True)
    rows = np.empty((len(queries), 1100), dtype=np.int32)
    for rank in range(world_size):
        value = np.load(cache / "parts" / f"part_{rank}.npz")
        rows[value["positions"]] = value["rows"]
    rankings = rows_to_filtered_rankings(rows, note_ids, requests, 1000)
    return rankings, time.perf_counter() - start


def tfidf_validation(requests, threads: int):
    # Phase 5 already computed exact sparse TF-IDF Top200 for every frozen
    # temporal train/validation request before negative filtering.  Fusion uses
    # at most Top100, so reuse those raw rankings and avoid another 70-minute
    # sparse full-corpus multiplication.
    wanted = {request.request_idx for request in requests}; raw = {}
    begin = time.perf_counter()
    parts = ROOT / "results/phase_05/experiment_01_request_hard_negative_mining/cache/tfidf_parts"
    for path in sorted(parts.glob("part_*.parquet")):
        table = pq.read_table(path, columns=["request_idx", "raw_ids"])
        for request_id, ids in zip(table.column(0).to_pylist(), table.column(1).to_pylist()):
            if int(request_id) in wanted: raw[int(request_id)] = list(map(int, ids))
    if set(raw) != wanted:
        raise AssertionError(f"missing temporal-validation TF-IDF raw rankings: {len(wanted-set(raw))}")
    output = []
    for request in requests:
        blocked = set(map(int, request.history))
        output.append([note for note in raw[request.request_idx] if note not in blocked][:200])
    return output, time.perf_counter() - begin


def save_validation_route(name, requests, rankings, contract, metadata=None):
    metrics, status, per = phase_status_metrics(requests, rankings, contract, name)
    path = OUT / "validation_rankings" / f"{name}.parquet"; path.parent.mkdir(parents=True, exist_ok=True)
    per.to_parquet(path, index=False, compression="zstd")
    payload = {"method": name, "metrics": metrics, "phase6_item_status": status,
               "validation_requests": len(requests), "test_read": False, **(metadata or {})}
    save_json(OUT / "validation" / f"{name}.json", payload)
    return payload


def contribution_analysis(requests, routes: dict[str, list[list[int]]], output_dir: Path | None = None):
    truth_by_request = {request.request_idx: set(request.ground_truth) for request in requests}
    route_hits = {}
    rows = []
    for name, rankings in routes.items():
        pairs, request_hits, unique_items = set(), set(), set()
        for request, ranking in zip(requests, rankings):
            hits = truth_by_request[request.request_idx] & set(ranking[:500])
            if hits: request_hits.add(request.request_idx)
            for note in hits: pairs.add((request.request_idx, note)); unique_items.add(note)
        route_hits[name] = pairs
        rows.append({"route": name, "request_hits": len(request_hits), "positive_pair_hits": len(pairs),
                     "unique_positive_item_hits": len(unique_items)})
    names = list(routes)
    overlaps = []
    for left_index, left in enumerate(names):
        for right in names[left_index+1:]:
            a, b = route_hits[left], route_hits[right]
            candidate_jaccards, candidate_intersections = [], []
            for left_rank, right_rank in zip(routes[left], routes[right]):
                left_set, right_set = set(left_rank[:500]), set(right_rank[:500])
                union = left_set | right_set
                candidate_intersections.append(len(left_set & right_set))
                candidate_jaccards.append(len(left_set & right_set) / len(union) if union else 0.0)
            overlaps.append({"left": left, "right": right, "intersection": len(a & b),
                             "union": len(a | b), "jaccard": len(a & b) / len(a | b) if a | b else 0,
                             "left_only": len(a-b), "right_only": len(b-a),
                             "mean_candidate_intersection_at500": float(np.mean(candidate_intersections)),
                             "mean_candidate_jaccard_at500": float(np.mean(candidate_jaccards))})
    destination = output_dir or OUT
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "route_contribution.csv", index=False)
    pd.DataFrame(overlaps).to_csv(destination / "route_overlap.csv", index=False)
    return {"routes": rows, "pairs": overlaps,
            "union_oracle_positive_pairs": len(set().union(*route_hits.values()))}


def run_baselines(config):
    note_ids, train, valid, contract = resources(); requests = grouped_requests(valid)
    popularity = temporal_popularity(train)
    pop_rankings = static_filtered(popularity, requests)
    item_native, item_fallback, item_stats = build_itemcf(train, requests, popularity)
    device = device_for(config.device)
    content, content_seconds = current_content_validation(requests, note_ids, device, config.threads)
    tfidf, tfidf_seconds = tfidf_validation(requests, min(config.threads, 16))
    routes = {"popularity_temporal_train": pop_rankings,
              "itemcf_temporal_train_native": item_native,
              "itemcf_temporal_train_fallback": item_fallback,
              "current_content_phase5": [row[:500] for row in content],
              "tfidf_lexical": tfidf}
    for name, rankings in routes.items():
        meta = {"itemcf_build": item_stats} if name.startswith("itemcf") else {}
        if name == "current_content_phase5": meta["search_seconds"] = content_seconds
        if name == "tfidf_lexical": meta["search_seconds"] = tfidf_seconds
        save_validation_route(name, requests, rankings, contract, meta)
    # Keep the over-fetched content route for deterministic fusion refill.
    pd.DataFrame({"request_idx": [r.request_idx for r in requests],
                  "retrieved_top1000": content}).to_parquet(
                      OUT / "validation_rankings/current_content_phase5_top1000.parquet",
                      index=False, compression="zstd")
    save_json(OUT / "route_contribution.json", contribution_analysis(requests, routes))


def known_user_positives(train: pd.DataFrame) -> dict[int, set[int]]:
    return {int(user): set(map(int, group.positive_item_id))
            for user, group in train.groupby("user_id", sort=False)}


def train_negative_variant(route: str, objective: str, config):
    seed_all(42); note_ids, train, valid, contract = resources(); device = device_for(config.device)
    requests_all = grouped_requests(valid); selection = subset_requests(requests_all, config.valid_requests or 5000)
    is_m2 = route == "m2"
    bge = (np.memmap(BGE_PATH, mode="r", dtype=np.float16,
                     shape=(len(note_ids), BGE_DIM)) if is_m2 else None)
    dataset = Phase6Dataset(train, contract, note_ids, BGE_PATH if is_m2 else None,
                            BGE_DIM, HISTORY_N)
    model = (make_m2("m2_0_content_projection", train, contract, device) if is_m2 else
             PureIdTower("m0_user_mf", len(contract.vocab_ids), len(contract.train_users), DIM).to(device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4 if is_m2 else 1e-3,
                                  weight_decay=1e-4 if is_m2 else 1e-5)
    counts = train.positive_item_id.value_counts(); total = len(train)
    probability = ({int(note): float(count / total) for note, count in counts.items()}
                   if objective == "logq" else None)
    uniform_count = int(objective.removeprefix("uniform")) if objective.startswith("uniform") else 0
    catalog = np.asarray(note_ids if is_m2 else contract.candidate_ids, dtype=np.int64)
    users = known_user_positives(train)
    note_lookup = None
    if is_m2:
        note_lookup = np.full(int(note_ids.max()) + 1, -1, dtype=np.int32)
        note_lookup[note_ids] = np.arange(len(note_ids), dtype=np.int32)
    name = f"{route}_{'d1_logq' if objective == 'logq' else 'd2_' + objective}"
    curves, best, stale = [], -1.0, 0
    max_epochs = min(config.epochs, 6 if is_m2 else 8)
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        log = train_epoch(model, dataset, optimizer, device,
                          min(config.batch_size, 256) if is_m2 else config.batch_size,
                          epoch, target_probability=probability,
                          uniform_count=uniform_count, uniform_catalog=catalog,
                          user_known=users, contract=contract, note_lookup=note_lookup, bge=bge)
        if is_m2:
            payload, _ = evaluate_m2(model, selection, train, contract, note_ids, bge, device,
                                     config.threads, name, f"selection_{name}_epoch{epoch}")
        else:
            payload, _ = evaluate_pure(model, selection, contract, device, config.threads, name)
        metric = payload["metrics"]["overall"]
        row = {"model": name, "epoch": epoch, **log, "Recall@100": metric["Recall@100"],
               "Recall@500": metric["Recall@500"], "MRR@100": metric["MRR@100"],
               "epoch_seconds": time.perf_counter()-started}
        curves.append(row); print(json.dumps(row), flush=True)
        if row["Recall@500"] > best:
            best, stale = row["Recall@500"], 0
            save_checkpoint(OUT / "checkpoints" / f"{name}_best.pt", model,
                            {**row, "route": route, "objective": objective,
                             "candidate_universe_size": len(catalog)})
        else:
            stale += 1
            if stale >= config.patience: break
    curve_path = OUT / "training_curves" / f"{name}.csv"; pd.DataFrame(curves).to_csv(curve_path, index=False)
    checkpoint = torch.load(OUT / "checkpoints" / f"{name}_best.pt", map_location=device,
                            weights_only=False); model.load_state_dict(checkpoint["state_dict"])
    if is_m2:
        result, _ = evaluate_m2(model, requests_all, train, contract, note_ids, bge, device,
                                config.threads, name, f"best_{name}", save_rankings=True)
    else:
        result, _ = evaluate_pure(model, requests_all, contract, device, config.threads,
                                  name, save_rankings=True)
    result.update({"best_epoch": checkpoint["metadata"]["epoch"], "route": route,
                   "objective": objective, "test_read": False})
    save_json(OUT / "validation" / f"{name}.json", result)


def run_negatives(config):
    if not config.negative_route or not config.negative_objective:
        raise ValueError("negatives stage requires --negative-route and --negative-objective")
    train_negative_variant(config.negative_route, config.negative_objective, config)


def load_ranking_map(path: Path, column: str = "retrieved_top500") -> dict[int, list[int]]:
    frame = pd.read_parquet(path, columns=["request_idx", column])
    return {int(request): list(map(int, ranking))
            for request, ranking in zip(frame.request_idx, frame[column])}


def merge_quota(requests, maps, quotas):
    output = []
    for request in requests:
        selected, seen = [], set()
        for route, quota in quotas.items():
            for note in maps[route][request.request_idx][:quota]:
                if note not in seen:
                    selected.append(note); seen.add(note)
        for note in maps["content"][request.request_idx]:
            if note not in seen:
                selected.append(note); seen.add(note)
            if len(selected) >= 500: break
        output.append(selected[:500])
    return output


def merge_rrf(requests, maps, include_itemcf=False, k=60):
    names = ["content", "id", "tfidf"] + (["itemcf"] if include_itemcf else [])
    output = []
    for request in requests:
        scores = defaultdict(float)
        for name in names:
            for rank, note in enumerate(maps[name][request.request_idx][:500], 1):
                scores[int(note)] += 1.0 / (k + rank)
        ranking = [note for note, _ in sorted(scores.items(), key=lambda value: (-value[1], value[0]))[:500]]
        if len(ranking) < 500:
            seen = set(ranking)
            for note in maps["content"][request.request_idx]:
                if note not in seen: ranking.append(note); seen.add(note)
                if len(ranking) == 500: break
        output.append(ranking)
    return output


def fusion_displacement(requests, content, fused):
    added = displaced = 0
    for request, base, result in zip(requests, content, fused):
        truth = set(request.ground_truth); base_hits = truth & set(base[:500]); fused_hits = truth & set(result)
        added += len(fused_hits - base_hits); displaced += len(base_hits - fused_hits)
    return {"positive_pairs_added_vs_content": added,
            "content_positive_pairs_displaced": displaced,
            "net_positive_pairs": added-displaced}


def run_fusion(config):
    _, _, valid, contract = resources(); requests = grouped_requests(valid)
    negative_lock_path = OUT / "locks/negative_sampling.json"
    if negative_lock_path.exists():
        pure_method = json.loads(negative_lock_path.read_text())["pure"]["method"]
    else:
        pure_lock = json.loads((OUT / "locks/pure_id_final.json").read_text())
        pure_method = f"m0_user_mf_seed{pure_lock['terminal_checkpoint_seed']}"
    maps = {
        "content": load_ranking_map(OUT / "validation_rankings/current_content_phase5_top1000.parquet",
                                    "retrieved_top1000"),
        "id": load_ranking_map(OUT / "validation_rankings" / f"{pure_method}.parquet"),
        "tfidf": load_ranking_map(OUT / "validation_rankings/tfidf_lexical.parquet"),
        "itemcf": load_ranking_map(OUT / "validation_rankings/itemcf_temporal_train_native.parquet"),
    }
    configurations = {
        "quota_400_50_50": {"content": 400, "id": 50, "tfidf": 50},
        "quota_300_100_100": {"content": 300, "id": 100, "tfidf": 100},
        "quota_250_150_100": {"content": 250, "id": 150, "tfidf": 100},
    }
    content_rankings = [maps["content"][request.request_idx][:500] for request in requests]
    content_payload = save_validation_route("fusion_content_only_reference", requests,
                                            content_rankings, contract)
    rows = [{"method": "content_only", "Recall@500":
             content_payload["metrics"]["overall"]["Recall@500"],
             "positive_pairs_added_vs_content": 0, "content_positive_pairs_displaced": 0,
             "net_positive_pairs": 0}]
    rankings_by_name = {}
    for name, quota in configurations.items():
        ranking = merge_quota(requests, maps, quota); rankings_by_name[name] = ranking
        payload = save_validation_route(name, requests, ranking, contract,
                                        {"quota": quota, **fusion_displacement(
                                            requests, [maps['content'][r.request_idx] for r in requests], ranking)})
        rows.append({"method": name, "Recall@500": payload["metrics"]["overall"]["Recall@500"],
                     **fusion_displacement(requests,
                         [maps['content'][r.request_idx] for r in requests], ranking)})
    # RRF is evaluated because route contribution analysis is already available.
    for include_itemcf in (False, True):
        name = "rrf_k60_with_itemcf" if include_itemcf else "rrf_k60"
        ranking = merge_rrf(requests, maps, include_itemcf); rankings_by_name[name] = ranking
        payload = save_validation_route(name, requests, ranking, contract,
                                        {"rrf_k": 60, "include_itemcf": include_itemcf})
        rows.append({"method": name, "Recall@500": payload["metrics"]["overall"]["Recall@500"],
                     **fusion_displacement(requests,
                         [maps['content'][r.request_idx] for r in requests], ranking)})
    frame = pd.DataFrame(rows).sort_values("Recall@500", ascending=False)
    frame.to_csv(OUT / "fusion_validation.csv", index=False)
    best = frame.iloc[0]
    if best.method == "content_only":
        bootstrap = {"overall": {"point_delta": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}}
    else:
        bootstrap = {segment: paired_bootstrap_delta(
            OUT / "validation_rankings/fusion_content_only_reference.parquet",
            OUT / "validation_rankings" / f"{best.method}.parquet",
            requests, contract, segment) for segment in ("overall", "warm", "cold")}
    save_json(OUT / "fusion_paired_bootstrap.json", bootstrap)
    save_json(OUT / "locks/final_fusion.json", {
        "method": best.method, "selection_metric": "validation Recall@500",
        "Recall@500": float(best["Recall@500"]), "test_read": False,
        "quota": configurations.get(best.method), "rrf_k": 60 if best.method.startswith("rrf") else None,
        "include_itemcf": best.method == "rrf_k60_with_itemcf",
        "paired_bootstrap_vs_content": bootstrap})


def metric_row(path: Path) -> dict:
    payload = json.loads(path.read_text())
    metric = payload["metrics"]["overall"]
    return {"method": payload["method"], "Recall@100": metric["Recall@100"],
            "Recall@500": metric["Recall@500"], "MRR@100": metric["MRR@100"],
            "path": str(path.relative_to(ROOT))}


def lock_configs():
    # Complete the residual magnitude audit from the already selected M2-1
    # checkpoint without retraining or re-running retrieval.
    residual_result_path = OUT / "validation/m2_1_binary_residual.json"
    if residual_result_path.exists():
        residual_result = json.loads(residual_result_path.read_text())
        if residual_result.get("norm_diagnostics") is None:
            note_ids, train_frame, _, contract_value = resources()
            device = device_for("auto")
            model = make_m2("m2_1_binary_residual", train_frame, contract_value, device)
            checkpoint = torch.load(OUT / "checkpoints/m2_1_binary_residual_best.pt",
                                    map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"]); model.eval()
            bge = np.memmap(BGE_PATH, mode="r", dtype=np.float16,
                            shape=(len(note_ids), BGE_DIM))
            note_lookup = np.full(int(note_ids.max()) + 1, -1, dtype=np.int32)
            note_lookup[note_ids] = np.arange(len(note_ids), dtype=np.int32)
            sample_ids = contract_value.candidate_ids[:20_000]
            with torch.inference_mode():
                content = torch.from_numpy(np.asarray(bge[note_lookup[sample_ids]],
                                                       dtype=np.float32)).to(device)
                ids_gpu = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64)).to(device)
                projected = model.content_projection(content)
                vocab_rows = model.item_to_vocab[ids_gpu]
                residual_vectors = (model.alpha * model.frequency_gate[vocab_rows].unsqueeze(-1)
                                    * model.item_residual(vocab_rows))
                content_norm = projected.norm(dim=-1).mean()
                residual_norm = residual_vectors.norm(dim=-1).mean()
            residual_result["norm_diagnostics"] = {
                "sample_items": len(sample_ids),
                "mean_content_projection_norm": float(content_norm),
                "mean_scaled_residual_norm": float(residual_norm),
                "residual_to_content_norm_ratio": float(residual_norm/content_norm)}
            save_json(residual_result_path, residual_result)
    pure_lock = json.loads((OUT / "locks/pure_id_final.json").read_text())
    pure_d0 = OUT / "validation" / f"m0_user_mf_seed{pure_lock['terminal_checkpoint_seed']}.json"
    pure_paths = [pure_d0] + sorted((OUT / "validation").glob("pure_d*.json"))
    m2_paths = [OUT / "validation/m2_0_content_projection.json"] + sorted(
        (OUT / "validation").glob("m2_d*.json"))
    pure_rows = [metric_row(path) for path in pure_paths if path.exists()]
    m2_rows = [metric_row(path) for path in m2_paths if path.exists()]
    pd.DataFrame([{**row, "route": "pure"} for row in pure_rows] +
                 [{**row, "route": "m2"} for row in m2_rows]).to_csv(
                     OUT / "negative_sampling_validation.csv", index=False)
    pure_best = max(pure_rows, key=lambda row: row["Recall@500"])
    m2_best = max(m2_rows, key=lambda row: row["Recall@500"])
    payload = {"pure": pure_best, "m2": m2_best,
               "selection_metric": "validation Recall@500", "test_read": False,
               "d0_references": {"pure": pure_d0.stem, "m2": "m2_0_content_projection"}}
    save_json(OUT / "locks/negative_sampling.json", payload)
    # Recompute the route audit with the selected Pure-ID checkpoint included.
    _, _, valid, _ = resources(); requests = grouped_requests(valid)
    paths = {
        "content": OUT / "validation_rankings/current_content_phase5.parquet",
        "pure_id": OUT / "validation_rankings" / f"{pure_best['method']}.parquet",
        "itemcf": OUT / "validation_rankings/itemcf_temporal_train_native.parquet",
        "popularity": OUT / "validation_rankings/popularity_temporal_train.parquet",
        "tfidf": OUT / "validation_rankings/tfidf_lexical.parquet",
    }
    if all(path.exists() for path in paths.values()):
        rankings = []
        routes = {}
        for name, path in paths.items():
            mapping = load_ranking_map(path)
            routes[name] = [mapping[request.request_idx] for request in requests]
        analysis = contribution_analysis(requests, routes, OUT / "selected_route_analysis")
        content_pairs = set(); id_pairs = set()
        target_seen = set(map(int, np.load(OUT / "audit/pure_id_candidate_note_ids.npy")))
        for request, content_rank, id_rank in zip(requests, routes["content"], routes["pure_id"]):
            truth = set(request.ground_truth)
            content_pairs.update((request.request_idx, note) for note in truth & set(content_rank))
            id_pairs.update((request.request_idx, note) for note in truth & set(id_rank))
        analysis["content_vs_id"] = {
            "content_only_positive_pairs": len(content_pairs-id_pairs),
            "id_only_positive_pairs": len(id_pairs-content_pairs),
            "both_positive_pairs": len(content_pairs&id_pairs),
            "neither_positive_pairs": sum(len(r.ground_truth) for r in requests)-len(content_pairs|id_pairs),
            "warm_id_only_positive_pairs": sum(note in target_seen for _, note in id_pairs-content_pairs),
        }
        save_json(OUT / "selected_route_analysis/summary.json", analysis)
    # Target-frequency diagnostics for all four Pure-ID structures and the
    # selected negative-sampling variant. Buckets are request-level macro.
    train, valid = load_temporal_frames(ROOT); requests = grouped_requests(valid)
    counts = train.positive_item_id.value_counts().to_dict()
    bucket_sets = {
        "1": {int(note) for note, count in counts.items() if count == 1},
        "2-3": {int(note) for note, count in counts.items() if 2 <= count <= 3},
        "4-10": {int(note) for note, count in counts.items() if 4 <= count <= 10},
        "11+": {int(note) for note, count in counts.items() if count >= 11},
    }
    frequency_rows = []
    methods = [f"{name}_seed42" for name in PURE_MODELS] + [pure_best["method"]]
    for method in dict.fromkeys(methods):
        ranking_path = OUT / "validation_rankings" / f"{method}.parquet"
        if not ranking_path.exists(): continue
        mapping = load_ranking_map(ranking_path)
        for bucket, allowed in bucket_sets.items():
            subset = [TestRequest(r.request_idx, r.user_idx, r.history,
                                  frozenset(set(r.ground_truth) & allowed))
                      for r in requests if set(r.ground_truth) & allowed]
            result, _ = evaluate_rankings(subset, [mapping[r.request_idx] for r in subset],
                                          set(), set(), method + "_freq_" + bucket)
            frequency_rows.append({"method": method, "frequency_bucket": bucket,
                                   "eligible_requests": len(subset),
                                   "Recall@100": result["overall"]["Recall@100"],
                                   "Recall@500": result["overall"]["Recall@500"]})
    pd.DataFrame(frequency_rows).to_csv(OUT / "pure_id_frequency_buckets.csv", index=False)
    print(json.dumps(payload, indent=2))


def save_terminal_metrics(name, requests, rankings, eval_splits, contract, metadata=None):
    metrics, per = evaluate_rankings(requests, rankings, eval_splits[0], eval_splits[2], name,
                                     clicked_items=eval_splits[1])
    _, status, _ = phase_status_metrics(requests, rankings, contract, name)
    path = OUT / "terminal_test/per_request" / f"{name}.parquet"; path.parent.mkdir(parents=True, exist_ok=True)
    per.to_parquet(path, index=False, compression="zstd")
    payload = {"method": name, "metrics": metrics, "phase6_item_status": status,
               "test_requests": len(requests), **(metadata or {})}
    save_json(OUT / "terminal_test/metrics" / f"{name}.json", payload)
    return payload


def load_model_for_method(method: str, route: str, train, contract, device):
    if route == "pure":
        model = PureIdTower("m0_user_mf", len(contract.vocab_ids), len(contract.train_users), DIM).to(device)
    else:
        model = make_m2("m2_0_content_projection", train, contract, device)
    checkpoint = torch.load(OUT / "checkpoints" / f"{method}_best.pt", map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model, checkpoint["metadata"]


def terminal_test(config):
    manifest_path = OUT / "terminal_test/manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "complete":
            raise RuntimeError("Phase 6 terminal test already completed; refusing a second test evaluation")
    locks = {
        "pure_structure": json.loads((OUT / "locks/pure_id_final.json").read_text()),
        "negative": json.loads((OUT / "locks/negative_sampling.json").read_text()),
        "fusion": json.loads((OUT / "locks/final_fusion.json").read_text()),
        "residual": json.loads((OUT / "locks/m2_residual_decision.json").read_text()),
    }
    save_json(manifest_path, {"status": "running", "started_at": time.time(),
                              "locked_configs": locks, "test_reads": 1})
    note_ids, train, _, contract = resources(); device = device_for(config.device)
    data = QilinData(ROOT)
    requests = data.load_test_requests()  # The only Phase 6 path that opens test.
    eval_splits = (data.train_exposed_items, set(data.train_click_counts), data.train_users)
    train_users = set(map(int, contract.train_users)); test_users = {r.user_idx for r in requests}
    target_items = set(map(int, contract.candidate_ids)); vocab = set(map(int, contract.vocab_ids))
    test_positive = []
    for request in requests:
        for note in request.ground_truth: test_positive.append(note)
    status_counts = {name: 0 for name in ("train_target_seen", "train_history_only", "completely_unseen")}
    for note in test_positive:
        if note in target_items: status_counts["train_target_seen"] += 1
        elif note in vocab: status_counts["train_history_only"] += 1
        else: status_counts["completely_unseen"] += 1
    audit_payload = {"test_requests": len(requests), "test_positive_interactions": len(test_positive),
                     "test_unique_users": len(test_users), "test_unseen_users": len(test_users-train_users),
                     "test_unseen_user_rate": len(test_users-train_users)/len(test_users),
                     "positive_status": {key: {"count": value, "rate": value/len(test_positive)}
                                         for key, value in status_counts.items()},
                     "read_after_all_configs_locked": True}
    save_json(OUT / "terminal_test/data_audit_test.json", audit_payload)

    pure_method = locks["negative"]["pure"]["method"]
    pure_model, pure_meta = load_model_for_method(pure_method, "pure", train, contract, device)
    pure_payload, pure_rankings = evaluate_pure(pure_model, requests, contract, device,
                                                config.threads, pure_method)

    m2_method = locks["negative"]["m2"]["method"]
    m2_model, m2_meta = load_model_for_method(m2_method, "m2", train, contract, device)
    bge = np.memmap(BGE_PATH, mode="r", dtype=np.float16, shape=(len(note_ids), BGE_DIM))
    m2_payload, m2_rankings = evaluate_m2(m2_model, requests, train, contract, note_ids, bge,
                                          device, config.threads, m2_method,
                                          f"terminal_{m2_method}")

    # Current Content is rerun exact to retain Top1000 for deterministic quota refill.
    content_top1000, content_seconds = current_content_validation(requests, note_ids, device,
                                                                  config.threads, "terminal_test")
    content_rankings = [ranking[:500] for ranking in content_top1000]
    tfidf_path = (ROOT / "results/phase_01/experiment_01_full_corpus_baselines/full/per_request/"
                  "tfidf_history20__with_history_filter.parquet")
    tfidf_map = load_ranking_map(tfidf_path); tfidf_rankings = [tfidf_map[r.request_idx] for r in requests]
    popularity = temporal_popularity(train); pop_rankings = static_filtered(popularity, requests)
    item_native, item_fallback, item_stats = build_itemcf(train, requests, popularity)

    route_rankings = {pure_method: pure_rankings, m2_method: m2_rankings,
                      "current_content_phase5": content_rankings, "tfidf_lexical": tfidf_rankings,
                      "popularity_temporal_train": pop_rankings,
                      "itemcf_temporal_train_native": item_native,
                      "itemcf_temporal_train_fallback": item_fallback}
    payloads = {}
    for name, ranking in route_rankings.items():
        metadata = {}
        if name == pure_method: metadata["checkpoint_metadata"] = pure_meta
        if name == m2_method: metadata.update({"checkpoint_metadata": m2_meta,
                                               "retrieval": {k: m2_payload[k] for k in
                                                ("vector_encode_seconds", "index_build_seconds", "search_seconds")}})
        if name == "current_content_phase5": metadata["search_seconds"] = content_seconds
        if name.startswith("itemcf"): metadata["itemcf_build"] = item_stats
        payloads[name] = save_terminal_metrics(name, requests, ranking, eval_splits, contract, metadata)

    maps = {"content": {r.request_idx: ranking for r, ranking in zip(requests, content_top1000)},
            "id": {r.request_idx: ranking for r, ranking in zip(requests, pure_rankings)},
            "tfidf": {r.request_idx: ranking for r, ranking in zip(requests, tfidf_rankings)},
            "itemcf": {r.request_idx: ranking for r, ranking in zip(requests, item_native)}}
    fusion = locks["fusion"]
    if fusion["method"] == "content_only":
        fused = content_rankings
    elif fusion["method"].startswith("quota"):
        fused = merge_quota(requests, maps, fusion["quota"])
    else:
        fused = merge_rrf(requests, maps, bool(fusion.get("include_itemcf")), 60)
    payloads[fusion["method"]] = save_terminal_metrics(fusion["method"], requests, fused,
                                                        eval_splits, contract,
                                                        {"locked_validation_config": fusion,
                                                         **fusion_displacement(requests, content_rankings, fused)})
    contribution = contribution_analysis(requests, {
        "content": content_rankings, "pure_id": pure_rankings,
        "itemcf": item_native, "tfidf": tfidf_rankings}, OUT / "terminal_test")
    save_json(OUT / "terminal_test/route_contribution.json", contribution)
    rows = []
    for name, payload in payloads.items():
        metric = payload["metrics"]["overall"]
        rows.append({"method": name, **{f"Recall@{k}": metric[f"Recall@{k}"]
                                        for k in (10,50,100,200,500)},
                     "MRR@100": metric["MRR@100"],
                     "Warm Recall@500": payload["metrics"]["warm_item"]["Recall@500"],
                     "Cold Recall@500": payload["metrics"]["cold_item"]["Recall@500"],
                     "Warm-user Recall@500": payload["metrics"]["warm_user"]["Recall@500"],
                     "Cold-user Recall@500": payload["metrics"]["cold_user"]["Recall@500"]})
    frame = pd.DataFrame(rows).sort_values("Recall@500", ascending=False)
    frame.to_csv(OUT / "terminal_test/results.csv", index=False)
    save_json(manifest_path, {"status": "complete", "completed_at": time.time(),
                              "locked_configs": locks, "test_reads": 1,
                              "test_requests": len(requests),
                              "best_method": frame.iloc[0].method,
                              "best_recall500": float(frame.iloc[0]["Recall@500"])})


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: f"{value:.6f}")
    return "\n".join(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join(["---"] * len(columns)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |"
                        for row in data.itertuples(index=False, name=None))])


def write_report():
    audit_value = json.loads((OUT / "audit/data_audit_train_valid.json").read_text())
    structures = pd.read_csv(OUT / "pure_id_structure_validation.csv")
    seeds = pd.read_csv(OUT / "pure_id_seed_stability.csv")
    negative = pd.read_csv(OUT / "negative_sampling_validation.csv")
    fusion = pd.read_csv(OUT / "fusion_validation.csv")
    test = pd.read_csv(OUT / "terminal_test/results.csv")
    residual = json.loads((OUT / "locks/m2_residual_decision.json").read_text())
    route = json.loads((OUT / "selected_route_analysis/summary.json").read_text())
    fusion_lock = json.loads((OUT / "locks/final_fusion.json").read_text())
    m2_rows = []
    for name in ("m2_0_content_projection", "m2_1_binary_residual"):
        value = json.loads((OUT / "validation" / f"{name}.json").read_text())
        m2_rows.append({"Method": name,
                        "R@100": value["metrics"]["overall"]["Recall@100"],
                        "R@500": value["metrics"]["overall"]["Recall@500"],
                        "Target-seen R@500": value["phase6_item_status"]["train_target_seen"]["Recall@500"],
                        "History-only R@500": value["phase6_item_status"]["train_history_only"]["Recall@500"],
                        "Unseen R@500": value["phase6_item_status"]["completely_unseen"]["Recall@500"]})
    m2_frame = pd.DataFrame(m2_rows)
    pure_columns = ["method", "Recall@100", "Recall@500", "MRR@100",
                    "Warm-item Recall@500", "Cold-user Recall@500"]
    terminal_columns = ["method", "Recall@100", "Recall@500", "MRR@100",
                        "Warm Recall@500", "Cold Recall@500"]
    best = test.sort_values("Recall@500", ascending=False).iloc[0]
    content_id = route.get("content_vs_id", {})
    id_warm = float(negative[(negative.route == "pure")].sort_values("Recall@500",
                    ascending=False).iloc[0]["Recall@500"])
    lines = [
        "# Phase 6：ID 协同召回与 Content-ID Residual", "",
        "## 实验协议", "",
        "- 完全复用 Phase 4/5 temporal split；训练/验证为 254,583 / 47,733 个 positive samples。",
        "- 所有结构、负采样和融合均只用 validation 选择；Phase 6 test 由 terminal gate 只读取一次。",
        "- Pure-ID 候选仅含 temporal-train positive targets；M2 始终保留全 1,983,938 item 内容表示。",
        "- BGE encoder 与原始 768d item embedding 冻结；M2 每个 checkpoint 均重建自己的 128d 全库 Exact IndexFlatIP。", "",
        "## Data Audit", "",
        f"- Train unique users：{audit_value['train_unique_users']:,}；validation unseen users：{audit_value['valid_unseen_users']:,}（{audit_value['valid_unseen_user_rate']:.2%}）。",
        f"- Train target/history/vocabulary items：{audit_value['train_target_unique_items']:,} / {audit_value['train_history_unique_items']:,} / {audit_value['history_union_target_items']:,}。",
        f"- History-only items：{audit_value['history_only_items']:,}；Pure-ID candidate universe：{audit_value['pure_id_candidate_universe']:,}。",
        f"- Validation positives：target-seen {audit_value['valid_positive_status']['train_target_seen']['rate']:.2%}，history-only {audit_value['valid_positive_status']['train_history_only']['rate']:.2%}，completely-unseen {audit_value['valid_positive_status']['completely_unseen']['rate']:.2%}。",
        "- 已有 Phase 5 parquet 保存 per-positive timestamp，而冻结 split 按 request 最早 positive timestamp 形成；多正样本 request 会令展示时间范围重叠，但 request membership 不跨 split。", "",
        "## Pure-ID 结构（Validation）", "", md_table(structures, pure_columns), "",
        "M0/M1/M1a/M1b 均使用相同 dim=128、tied item embedding、batch、optimizer、seed 与 false-negative mask。M0 最好，但 cold user 的 ID 分支关闭，因此 recall 为 0。", "",
        "### M0 Seed 稳定性", "", md_table(seeds, ["seed", "Recall@100", "Recall@500", "MRR@100"]), "",
        "## Content-ID Residual（Validation）", "", md_table(m2_frame, list(m2_frame.columns)), "",
        f"M2-1 相对 M2-0 的 warm delta={residual['bootstrap']['warm']['point_delta']:.6f}，95% CI [{residual['bootstrap']['warm']['ci95_lower']:.6f}, {residual['bootstrap']['warm']['ci95_upper']:.6f}]；cold delta={residual['bootstrap']['cold']['point_delta']:.6f}。",
        "Residual 的 warm CI 下界不大于 0，违反预注册 Go 条件，因此不执行 frequency gate 和 ID dropout，并判定 M2 residual No-Go。", "",
        "## D0 / logQ / Uniform Negatives（Validation）", "",
        md_table(negative.sort_values(["route", "Recall@500"], ascending=[True, False]),
                 ["route", "method", "Recall@100", "Recall@500", "MRR@100"]), "",
        "Pure-ID 的 logQ correction 明显有效；M2 的 uniform-16 只有很小总体增益。首轮没有组合 logQ 与 uniform，也没有引入 Phase 5 hard negatives。", "",
        "## 路线互补性", "",
        f"- Content-only / ID-only / both / neither positive pairs：{content_id.get('content_only_positive_pairs', 0):,} / {content_id.get('id_only_positive_pairs', 0):,} / {content_id.get('both_positive_pairs', 0):,} / {content_id.get('neither_positive_pairs', 0):,}。",
        f"- Warm positive 中 ID-only hits：{content_id.get('warm_id_only_positive_pairs', 0):,}。",
        f"- 多路 union oracle positive pairs：{route.get('union_oracle_positive_pairs', 0):,}（仅互补性分析，不作为可部署结果）。", "",
        "## Fixed-budget Fusion（Validation）", "", md_table(fusion,
                 ["method", "Recall@500", "positive_pairs_added_vs_content",
                  "content_positive_pairs_displaced", "net_positive_pairs"]), "",
        f"锁定配置：`{fusion_lock['method']}`；选择只看 validation。", "",
        "## Terminal Test（一次）", "", md_table(test, terminal_columns), "",
        f"最终最高 Recall@500 为 `{best['method']}` 的 {best['Recall@500']:.4%}。", "",
        "## 最终决策", "",
        "- **独立 ID route：保留为实验候选，但仅服务 warm/seen-user。** logQ 后 warm 能力和 Content-miss/ID-hit 证明其有独立信号；是否进入线上固定预算由 fusion bootstrap 决定，不能用于 cold user/item。",
        "- **M2 ID residual：不保留。** 它未提升 warm，反而显著降低 target-seen recall；总体小升来自 unseen，不符合本阶段目标。",
        "- **下一阶段建议：** 优先研究验证集上被证明有效的固定预算 Content/TF-IDF/ID 路由组合；若 ID 在固定预算 bootstrap 下不显著，则先改善 ID 序列建模与频率校正，不进入粗排/精排，也不解冻 BGE。", "",
        "本阶段已停止，没有自动开始新的召回或排序实验。", "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    summary_path = ROOT / "experiments/SUMMARY_ZH.md"
    addition = ("\n\n## Phase 6：ID 协同召回与 Content-ID Residual\n\n"
                f"- 目标：验证 Pure-ID 独立路线与 frozen-content ID residual。\n"
                f"- Pure-ID 最佳 validation R@500：{negative.loc[negative.route == 'pure', 'Recall@500'].max():.4%}。\n"
                f"- M2 residual：No-Go；warm paired-bootstrap CI 为负。\n"
                f"- Terminal 最佳 R@500：{best['Recall@500']:.4%}（{best['method']}）。\n"
                "- 产物：`results/phase_06/experiment_01_id_two_tower_retrieval/summary.md`。\n")
    text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else "# 实验汇总\n"
    if "## Phase 6：ID 协同召回与 Content-ID Residual" not in text:
        summary_path.write_text(text.rstrip() + addition, encoding="utf-8")


def main():
    config = args(); OUT.mkdir(parents=True, exist_ok=True)
    if config.stage == "audit": audit()
    elif config.stage == "smoke": run_pure(config, smoke=True)
    elif config.stage == "pure-id": run_pure(config)
    elif config.stage == "pure-seeds": pure_seeds(config)
    elif config.stage == "baselines": run_baselines(config)
    elif config.stage == "m2": run_m2(config)
    elif config.stage == "negatives": run_negatives(config)
    elif config.stage == "fusion": run_fusion(config)
    elif config.stage == "lock": lock_configs()
    elif config.stage == "terminal-test": terminal_test(config)
    elif config.stage == "report": write_report()
    else:
        raise NotImplementedError(f"stage {config.stage} is added after Pure-ID audit passes")


if __name__ == "__main__":
    main()
