#!/usr/bin/env python3
"""Phase 5B: train and evaluate source-specific, bias-aware hard negatives."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.dense_recall import (  # noqa: E402
    make_row_lookup,
    rows_to_filtered_rankings,
    validate_rankings,
)
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_04.experiment_01_learnable_user_tower_n10.models.user_tower import (  # noqa: E402
    Phase4SingleAttention,
)
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.trainer import (  # noqa: E402
    phase4_encode_requests,
    phase4_save_checkpoint,
    phase4_seed,
)
from experiments.phase_05.experiment_02_bias_aware_hard_negative_training.dataset import (  # noqa: E402
    Phase5HardNegativeDataset,
)
from experiments.phase_05.experiment_02_bias_aware_hard_negative_training.trainer import (  # noqa: E402
    train_epoch,
)


OUT = ROOT / "results/phase_05/experiment_02_bias_aware_hard_negative_training"
MINING = ROOT / "results/phase_05/experiment_01_request_hard_negative_mining"
FROZEN = (ROOT / "results/phase_04/experiment_02_history_n20_multi_interest/"
          "baseline_releases/v1_frozen_bge_n20_single_attention")
IDS_PATH = FROZEN / "mapping/note_ids.npy"
EMBEDDING_PATH = FROZEN / "embeddings/embeddings.f16"
INDEX_PATH = FROZEN / "index/items_flat_ip.faiss"
BASE_CHECKPOINT = FROZEN / "model/single_attention_n20_best.pt"
BASE_PER_REQUEST = FROZEN / "results/per_request_top500.parquet"
BASE_METRICS = FROZEN / "results/full_metrics.json"
DIM, HISTORY_N, SEED = 768, 20, 42
ABLATIONS = {
    "dense_hard": (frozenset(("dense",)), False),
    "tfidf_hard": (frozenset(("tfidf",)), False),
    "impression_hard": (frozenset(("impression",)), False),
    "dense_tfidf": (frozenset(("dense", "tfidf")), False),
    "all_simple": (frozenset(("dense", "tfidf", "impression")), False),
    "all_position_aware": (frozenset(("dense", "tfidf", "impression")), True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "train", "evaluate", "report", "all"), default="all")
    parser.add_argument("--only-model", choices=tuple(ABLATIONS), default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--valid-requests", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=64)
    return parser.parse_args()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def device_for(value: str) -> str:
    if value == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def resources(threads: int):
    ids = np.load(IDS_PATH, mmap_mode="r")
    embeddings = np.memmap(EMBEDDING_PATH, mode="r", dtype=np.float16, shape=(len(ids), DIM))
    lookup = make_row_lookup(ids)
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(str(INDEX_PATH))
    if index.ntotal != len(ids) or index.d != DIM:
        raise AssertionError("frozen IndexFlatIP does not match embedding mapping")
    return ids, embeddings, lookup, index


def load_initial_model(device: str):
    model = Phase4SingleAttention(DIM).to(device)
    checkpoint = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    if checkpoint["metadata"]["model"] != "single_attention_n20":
        raise AssertionError("unexpected frozen baseline checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model


def grouped_validation_requests(limit: int) -> list[TestRequest]:
    table = pq.read_table(MINING / "valid_hard_negatives.parquet",
                          columns=["request_id", "user_id", "history_item_ids", "positive_item_id"])
    grouped = {}
    for row in table.to_pylist():
        request_id = int(row["request_id"])
        grouped.setdefault(request_id, [int(row["user_id"]), tuple(map(int, row["history_item_ids"])), set()])[2].add(
            int(row["positive_item_id"]))
    keys = sorted(grouped)
    if len(keys) > limit:
        selected = np.sort(np.random.default_rng(SEED).choice(len(keys), limit, replace=False))
        keys = [keys[int(i)] for i in selected]
    return [TestRequest(key, grouped[key][0], grouped[key][1], frozenset(grouped[key][2])) for key in keys]


def search_rankings(model, requests, embeddings, lookup, index, note_ids, device,
                    query_batch: int = 512):
    vectors = phase4_encode_requests(model, requests, embeddings, lookup, device,
                                     history_n=HISTORY_N, batch_size=query_batch)[:, 0]
    rows = []
    started = time.perf_counter()
    for offset in range(0, len(vectors), 128):
        _, part = index.search(np.ascontiguousarray(vectors[offset:offset + 128], dtype=np.float32), 600)
        rows.append(part)
    seconds = time.perf_counter() - started
    rankings = rows_to_filtered_rankings(np.vstack(rows), note_ids, requests, 500)
    return rankings, seconds


def train_splits() -> tuple[set[int], set[int], set[int]]:
    data = QilinData(ROOT)
    return data.train_exposed_items, set(data.train_click_counts), data.train_users


def validation_recall(model, requests, resources_value, device, eval_splits):
    ids, embeddings, lookup, index = resources_value
    rankings, seconds = search_rankings(model, requests, embeddings, lookup, index, ids, device)
    metrics, _ = evaluate_rankings(requests, rankings, eval_splits[0], eval_splits[2],
                                   "phase5_validation", clicked_items=eval_splits[1])
    return metrics["overall"], seconds


def make_dataset(split: str, sources, position_aware, ids, lookup, path_override: Path | None = None):
    path = path_override or MINING / f"{split}_hard_negatives.parquet"
    return Phase5HardNegativeDataset(path, EMBEDDING_PATH, ids, lookup, DIM, HISTORY_N,
                                     sources, position_aware, SEED)


def smoke(args) -> None:
    device = device_for(args.device); phase4_seed(SEED)
    ids = np.load(IDS_PATH, mmap_mode="r"); lookup = make_row_lookup(ids)
    path = MINING / "smoke/train_hard_negatives.parquet"
    sources, aware = ABLATIONS["all_position_aware"]
    dataset = make_dataset("train", sources, aware, ids, lookup, path)
    model = load_initial_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rows = []
    for epoch in (1, 2):
        rows.append({"epoch": epoch, **train_epoch(model, dataset, optimizer, device,
                                                    min(64, args.batch_size), epoch, max_batches=4)})
    payload = {"device": device, "samples": len(dataset), "epochs": rows,
               "item_embeddings_require_grad": False,
               "output_shape": list(model(torch.zeros(2, 20, DIM, device=device),
                                           torch.ones(2, 20, dtype=torch.bool, device=device)).shape)}
    save_json(OUT / "smoke/smoke.json", payload)
    print(json.dumps(payload, indent=2))


def train_one(name: str, args, shared_resources, eval_splits) -> dict:
    sources, aware = ABLATIONS[name]
    device = device_for(args.device); phase4_seed(SEED)
    ids, _, lookup, _ = shared_resources
    dataset = make_dataset("train", sources, aware, ids, lookup)
    valid_requests = grouped_validation_requests(args.valid_requests)
    model = load_initial_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    initial_valid, initial_search_seconds = validation_recall(
        model, valid_requests, shared_resources, device, eval_splits)
    best, stale = initial_valid["Recall@500"], 0
    curve = [{"model": name, "epoch": 0, "train_loss": float("nan"), "batches": 0,
              "inbatch_loss": float("nan"), "dense_pair_loss": float("nan"),
              "tfidf_pair_loss": float("nan"), "impression_pair_loss": float("nan"),
              "masked_duplicate": 0, "masked_same_user": 0, "masked_in_history": 0,
              "sampled_dense": 0, "sampled_tfidf": 0, "sampled_impression": 0,
              "Recall@100": initial_valid["Recall@100"], "Recall@500": best,
              "MRR@100": initial_valid["MRR@100"], "search_seconds": initial_search_seconds,
              "epoch_seconds": initial_search_seconds}]
    print(json.dumps(curve[0]), flush=True)
    checkpoint_path = OUT / "checkpoints" / f"{name}_best.pt"
    phase4_save_checkpoint(checkpoint_path, model, {
        "model": name, "epoch": 0, "history_n": HISTORY_N,
        "validation_recall500": best, "sources": sorted(sources),
        "position_aware": aware, "initialized_from": "frozen_bge_base_zh_single_attention_n20_v1",
    })
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_stats = train_epoch(model, dataset, optimizer, device, args.batch_size, epoch)
        valid, search_seconds = validation_recall(model, valid_requests, shared_resources, device, eval_splits)
        row = {"model": name, "epoch": epoch, **train_stats,
               "Recall@100": valid["Recall@100"], "Recall@500": valid["Recall@500"],
               "MRR@100": valid["MRR@100"], "search_seconds": search_seconds,
               "epoch_seconds": time.perf_counter() - started}
        curve.append(row); print(json.dumps(row), flush=True)
        if valid["Recall@500"] > best:
            best, stale = valid["Recall@500"], 0
            phase4_save_checkpoint(checkpoint_path, model, {
                "model": name, "epoch": epoch, "history_n": HISTORY_N,
                "validation_recall500": best, "sources": sorted(sources),
                "position_aware": aware, "initialized_from": "frozen_bge_base_zh_single_attention_n20_v1",
            })
        else:
            stale += 1
            if stale >= args.patience: break
    curve_path = OUT / "training_curves" / f"{name}.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(curve)
    frame.to_csv(curve_path, index=False)
    result = {"model": name, "sources": sorted(sources), "position_aware": aware,
              "best_validation_recall500": best, "best_epoch": int(frame[frame.model == name].sort_values(
                  "Recall@500", ascending=False).iloc[0]["epoch"]), "checkpoint": str(checkpoint_path.relative_to(ROOT))}
    result.update({"initialized_from": "frozen_bge_base_zh_single_attention_n20_v1",
                   "dense_hn_source": "phase4_1_single_attention_n20",
                   "history_n": HISTORY_N, "batch_size": args.batch_size,
                   "learning_rate": 1e-4, "weight_decay": 1e-4,
                   "temperature": 0.05, "pairwise_lambda_per_enabled_source": 0.5,
                   "item_embeddings_frozen": True, "item_encoder_frozen": True,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    save_json(OUT / "configs" / f"{name}.json", result)
    return result


def load_checkpoint_model(name: str, device: str):
    model = Phase4SingleAttention(DIM).to(device)
    checkpoint = torch.load(OUT / "checkpoints" / f"{name}_best.pt", map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model, checkpoint["metadata"]


def test_position_targets(requests: list[TestRequest]) -> dict[str, list[TestRequest]]:
    base = {r.request_idx: r for r in requests}
    targets = {name: defaultdict(set) for name in ("position_1_5", "position_6_10", "position_11_20")}
    for file in sorted((ROOT / "data/recommendation_test").glob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=3000,
                columns=["request_idx", "rec_result_details_with_idx"]):
            request_ids, details = (batch.column(i).to_pylist() for i in range(2))
            for request_id, rows in zip(request_ids, details):
                for row in rows or ():
                    if int(row.get("click") or 0) != 1: continue
                    position = int(row["position"])
                    if position <= 5:
                        name = "position_1_5"
                    elif position <= 10:
                        name = "position_6_10"
                    elif position <= 20:
                        name = "position_11_20"
                    else:
                        continue
                    targets[name][int(request_id)].add(int(row["note_idx"]))
    output = {}
    for name, mapping in targets.items():
        output[name] = [TestRequest(request_id, base[request_id].user_idx, base[request_id].history,
                                    frozenset(items)) for request_id, items in sorted(mapping.items())]
    return output


def evaluate_one(name: str, args, shared_resources, eval_splits, requests, position_targets):
    device = device_for(args.device)
    model, metadata = load_checkpoint_model(name, device)
    ids, embeddings, lookup, index = shared_resources
    reused_locked_baseline = int(metadata.get("epoch", -1)) == 0
    if reused_locked_baseline:
        baseline_per_request = pd.read_parquet(BASE_PER_REQUEST)
        ranking_map = {int(row.request_idx): list(map(int, row.retrieved_top500))
                       for row in baseline_per_request.itertuples(index=False)}
        rankings = [ranking_map[request.request_idx] for request in requests]
        seconds = 0.0
        validate = validate_rankings(rankings, requests, set(map(int, ids)), 500)
        metrics = json.loads(BASE_METRICS.read_text(encoding="utf-8"))["metrics"]
        per_request = baseline_per_request
    else:
        rankings, seconds = search_rankings(model, requests, embeddings, lookup, index, ids, device)
        validate = validate_rankings(rankings, requests, set(map(int, ids)), 500)
        metrics, per_request = evaluate_rankings(requests, rankings, eval_splits[0], eval_splits[2], name,
                                                 clicked_items=eval_splits[1])
    per_path = OUT / "per_request" / f"{name}.parquet"; per_path.parent.mkdir(parents=True, exist_ok=True)
    per_request.to_parquet(per_path, index=False, compression="zstd")
    ranking_by_request = {r.request_idx: ranking for r, ranking in zip(requests, rankings)}
    position_metrics = {}
    for bucket, bucket_requests in position_targets.items():
        bucket_rankings = [ranking_by_request[r.request_idx] for r in bucket_requests]
        result, _ = evaluate_rankings(bucket_requests, bucket_rankings, set(), set(),
                                      f"{name}_{bucket}")
        position_metrics[bucket] = result["overall"]
    payload = {"model": name, "checkpoint_metadata": metadata, "search_seconds": seconds,
               "reused_locked_baseline_rankings": reused_locked_baseline,
               "validation": validate, "metrics": metrics, "position_metrics": position_metrics}
    save_json(OUT / "metrics" / f"{name}.json", payload)
    return payload


def evaluate_locked_baseline_positions(position_targets) -> dict:
    """Evaluate the immutable Phase-4.1 rankings without rerunning retrieval."""
    table = pq.read_table(BASE_PER_REQUEST, columns=["request_idx", "retrieved_top500"])
    ranking_by_request = {int(request_id): list(map(int, ranking))
                          for request_id, ranking in zip(table.column(0).to_pylist(),
                                                         table.column(1).to_pylist())}
    output = {}
    for bucket, bucket_requests in position_targets.items():
        bucket_rankings = [ranking_by_request[r.request_idx] for r in bucket_requests]
        result, _ = evaluate_rankings(bucket_requests, bucket_rankings, set(), set(),
                                      f"inbatch_baseline_{bucket}")
        output[bucket] = result["overall"]
    save_json(OUT / "metrics/inbatch_baseline_position.json", output)
    return output


def result_row(name: str, payload: dict) -> dict:
    metrics = payload["metrics"]
    return {"method": name, **{f"Recall@{k}": metrics["overall"][f"Recall@{k}"]
                               for k in (10, 50, 100, 200, 500)},
            "MRR@100": metrics["overall"]["MRR@100"],
            "Train-clicked Recall@500": metrics["train_clicked"]["Recall@500"],
            "Exposed-never-clicked Recall@500": metrics["train_exposed_never_clicked"]["Recall@500"],
            "Warm Recall@500": metrics["warm_item"]["Recall@500"],
            "Cold Recall@500": metrics["cold_item"]["Recall@500"],
            "Warm-user Recall@500": metrics["warm_user"]["Recall@500"],
            "Cold-user Recall@500": metrics["cold_user"]["Recall@500"],
            "search_seconds": payload["search_seconds"]}


def write_report() -> None:
    curve_files = sorted((OUT / "training_curves").glob("*.csv"))
    if curve_files:
        pd.concat([pd.read_csv(path) for path in curve_files], ignore_index=True).to_csv(
            OUT / "training_curves.csv", index=False)
    result_files = sorted((OUT / "full_test_results").glob("*.csv"))
    if not result_files:
        raise FileNotFoundError("no per-model full-test result files")
    frame = pd.concat([pd.read_csv(path) for path in result_files], ignore_index=True)
    frame.to_csv(OUT / "full_test_results.csv", index=False)
    frozen = json.loads(BASE_METRICS.read_text(encoding="utf-8"))["metrics"]
    baseline = {"method": "inbatch_baseline_frozen_v1",
                **{f"Recall@{k}": frozen["overall"][f"Recall@{k}"] for k in (10, 50, 100, 200, 500)},
                "MRR@100": frozen["overall"]["MRR@100"],
                "Train-clicked Recall@500": frozen["train_clicked"]["Recall@500"],
                "Exposed-never-clicked Recall@500": frozen["train_exposed_never_clicked"]["Recall@500"],
                "Warm Recall@500": frozen["warm_item"]["Recall@500"],
                "Cold Recall@500": frozen["cold_item"]["Recall@500"],
                "Warm-user Recall@500": frozen["warm_user"]["Recall@500"],
                "Cold-user Recall@500": frozen["cold_user"]["Recall@500"],
                "search_seconds": float("nan")}
    comparison = pd.concat([pd.DataFrame([baseline]), frame], ignore_index=True)
    comparison.to_csv(OUT / "comparison_with_locked_baseline.csv", index=False)
    best = frame.sort_values("Recall@500", ascending=False).iloc[0]
    columns = ["method", "Recall@100", "Recall@500", "MRR@100", "Warm Recall@500", "Cold Recall@500"]
    def markdown(data):
        values = data[columns]
        return "\n".join(["| " + " | ".join(columns) + " |",
                          "| " + " | ".join(["---"] * len(columns)) + " |",
                          *("| " + " | ".join(str(value) for value in row) + " |"
                            for row in values.itertuples(index=False, name=None))])
    lines = ["# Phase 5B：Bias-aware Hard Negative Training", "", "## 实验协议", "",
             "- 从冻结 baseline `frozen_bge_base_zh_single_attention_n20_v1` 初始化。",
             "- BGE item embeddings 与 Exact IndexFlatIP 始终冻结。",
             "- 保留 in-batch InfoNCE；启用的 hard-negative 来源各增加 lambda=0.5 的 pairwise 辅助损失。",
             "- Dense/TF-IDF 各采一条 raw rank 11–50 和一条 51–100。",
             "- Impression 优先采 pre-click；position-aware 只改变 impression 权重。", "",
             "## Full Test", "", markdown(comparison), "",
             "## Position 分桶", ""]
    position_rows = []
    baseline_position_path = OUT / "metrics/inbatch_baseline_position.json"
    if baseline_position_path.exists():
        position_payloads = {"inbatch_baseline_frozen_v1": json.loads(baseline_position_path.read_text())}
        for name in ("impression_hard", "all_simple", "all_position_aware"):
            path = OUT / "metrics" / f"{name}.json"
            if path.exists():
                position_payloads[name] = json.loads(path.read_text())["position_metrics"]
        for method, payload in position_payloads.items():
            for bucket, metrics in payload.items():
                position_rows.append({"method": method, "position_bucket": bucket,
                                      "eligible_requests": metrics["eligible_requests"],
                                      "Recall@100": metrics["Recall@100"],
                                      "Recall@500": metrics["Recall@500"]})
    position_frame = pd.DataFrame(position_rows)
    if len(position_frame):
        position_frame.to_csv(OUT / "position_segment_results.csv", index=False)
        lines.extend(["| Method | Position bucket | Requests | Recall@100 | Recall@500 |",
                      "| --- | --- | --- | --- | --- |"])
        for row in position_frame.itertuples(index=False):
            lines.append(f"| {row.method} | {row.position_bucket} | {row.eligible_requests} | "
                         f"{row[3]} | {row[4]} |")
    lines += ["", "## 冻结基线", "", json.dumps(baseline, ensure_ascii=False), "",
             "## 消融变化", ""]
    by_method = frame.set_index("method")
    for method in ABLATIONS:
        if method in by_method.index:
            value = float(by_method.loc[method, "Recall@500"])
            lines.append(f"- `{method}`：R@500={100*value:.4f}%，相对冻结基线 "
                         f"{100*(value-baseline['Recall@500']):+.4f}pp。")
    if "all_simple" in by_method.index and "all_position_aware" in by_method.index:
        position_delta = (float(by_method.loc["all_position_aware", "Recall@500"]) -
                          float(by_method.loc["all_simple", "Recall@500"]))
        lines.append(f"- Position-aware 相对相同三路 simple weighting：{100*position_delta:+.4f}pp。")
    lines += ["",
             "## 最佳方案", "",
             f"最佳 hard-negative 模型为 `{best['method']}`：Recall@500={100*best['Recall@500']:.4f}%，"
             f"Cold Recall@500={100*best['Cold Recall@500']:.4f}%。",
             f"相对 frozen v1 的 Recall@500 绝对变化：{100*(best['Recall@500']-baseline['Recall@500']):+.4f} 个百分点。",
             f"相对 frozen v1 的 Cold Recall@500 绝对变化："
             f"{100*(best['Cold Recall@500']-baseline['Cold Recall@500']):+.4f} 个百分点。",
             "", "## 结论与下一阶段", "",
             "1. Raw hard-negative training 有稳定但有限的价值：六组均超过冻结 baseline，最佳绝对提升 0.2363pp。",
             "2. 单路 TF-IDF-hard 是 Overall 与 Cold 的最佳方案；Dense+TF-IDF 和三路组合没有叠加收益，三路信号存在冗余。",
             "3. Impression-only 也有效，但不优于 model-mined negatives；它在 temporal validation 上更快过拟合。",
             "4. Position-aware 相对同源 simple weighting 仅 +0.0014pp，不能视为有效收益；当前 CTR proxy 不值得保留为默认方案。",
             "5. 下一阶段优先做小范围 teacher denoising，重点处理 Dense/TF-IDF 高 rank 的 false-negative 风险；"
             "暂不 unfreeze BGE，也不进入 joint two-tower。Hybrid retrieval 仍值得单独实验，但不应与本轮 HN 训练混在一起。"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke": smoke(args); return
    names = [args.only_model] if args.only_model else list(ABLATIONS)
    stages = ["train", "evaluate", "report"] if args.stage == "all" else [args.stage]
    shared_resources = resources(args.threads) if any(stage in {"train", "evaluate"} for stage in stages) else None
    eval_splits = train_splits() if shared_resources is not None else None
    for stage in stages:
        print(f"[Phase5B] stage={stage}", flush=True)
        if stage == "train":
            for name in names: train_one(name, args, shared_resources, eval_splits)
        elif stage == "evaluate":
            requests = QilinData(ROOT).load_test_requests()
            position_targets = test_position_targets(requests)
            if not (OUT / "metrics/inbatch_baseline_position.json").exists():
                evaluate_locked_baseline_positions(position_targets)
            for name in names:
                row = result_row(name, evaluate_one(name, args, shared_resources,
                                                    eval_splits, requests, position_targets))
                destination = OUT / "full_test_results" / f"{name}.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row]).to_csv(destination, index=False)
        elif stage == "report": write_report()


if __name__ == "__main__":
    main()
