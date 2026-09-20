#!/usr/bin/env python3
"""Phase 5A: request-level Dense, TF-IDF, and impression hard-negative mining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_05.experiment_01_request_hard_negative_mining.mining import (  # noqa: E402
    MiningPaths,
    audit_pools,
    finalize_dense,
    finalize_tfidf,
    merge_sample_pools,
    mine_dense,
    mine_tfidf,
    prepare_samples,
    validate_dense_faiss_equivalence,
)

BASE_OUT = ROOT / "results/phase_05/experiment_01_request_hard_negative_mining"
FROZEN = (ROOT / "results/phase_04/experiment_02_history_n20_multi_interest/"
          "baseline_releases/v1_frozen_bge_n20_single_attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "dense", "finalize-dense", "validate-dense-faiss", "tfidf",
                                             "finalize-tfidf", "merge", "audit", "report", "all"),
                        default="all")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--tfidf-batch-size", type=int, default=4)
    parser.add_argument("--tfidf-threads", type=int, default=8)
    parser.add_argument("--tfidf-part-requests", type=int, default=1000)
    parser.add_argument("--tfidf-worker-index", type=int, default=0)
    parser.add_argument("--tfidf-worker-count", type=int, default=1)
    return parser.parse_args()


def write_report(paths: MiningPaths) -> None:
    stats = json.loads((paths.out / "mining_stats.json").read_text())
    prepare = json.loads((paths.out / "prepare_stats.json").read_text())
    dense = json.loads((paths.out / "dense_finalize_stats.json").read_text())
    tfidf = json.loads((paths.out / "tfidf_finalize_stats.json").read_text())
    equivalence_path = paths.out / "dense_faiss_equivalence.json"
    equivalence = json.loads(equivalence_path.read_text()) if equivalence_path.exists() else None
    overlap = pd.DataFrame(stats["overlap_means"])
    rank = pd.DataFrame(stats["rank_audit"])
    position = pd.read_csv(paths.out / "position_confidence_weights.csv")

    def markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        values = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        return "\n".join(["| " + " | ".join(columns) + " |",
                          "| " + " | ".join(["---"] * len(columns)) + " |",
                          *("| " + " | ".join(row) + " |" for row in values)])
    lines = [
        "# Phase 5A：Request-level Hard Negative Mining 审计", "",
        "## 实验协议", "",
        "- 样本与时间切分完全沿用 Phase 4.1，history N=20。",
        "- Dense 来源锁定为 `frozen_bge_base_zh_single_attention_n20_v1`。",
        "- TF-IDF 沿用 Phase 1 的 char-bigram exact 配置。",
        "- Impression 候选先按精确 request timestamp 分组，再比较 position。",
        "- Position proxy 只使用 temporal training partition，validation/test 均不参与估计。",
        "- 三路均过滤 history、同 request 全部正例、当前正例、重复项和该用户所有 train-known positives。",
        "- Mining 未读取 test 数据。", "",
        "## 数据规模", "",
        f"- Train/valid 样本：{prepare['train_samples']:,} / {prepare['valid_samples']:,}。",
        f"- Unique request context：{prepare['requests']:,}。",
        f"- Dense 可用负例/request：均值 {dense['mean_available']:.3f}，不足 100 的 request：{dense['requests_below_100']:,}。",
        f"- TF-IDF 可用负例/request：均值 {tfidf['mean_available']:.3f}，不足 100 的 request：{tfidf['requests_below_100']:,}。",
        f"- Impression 负例 occurrence：{stats['impression_total']:,}。", "",
        "## Impression 类型", "",
    ]
    total_imp = max(1, stats["impression_total"])
    for name, count in sorted(stats["impression_types"].items()):
        lines.append(f"- {name}: {count:,} ({100*count/total_imp:.2f}%).")
    if equivalence:
        lines += ["", "## Dense 与冻结 Faiss 一致性", "",
                  f"- 固定 {equivalence['requests']} request 的 Top100 set overlap："
                  f"{100*equivalence['top100_mean_set_overlap']:.4f}%。",
                  f"- Top200 set overlap：{100*equivalence['top200_mean_set_overlap']:.4f}%。",
                  f"- Top100 顺序完全一致 request：{equivalence['exact_top100_order_requests']}/"
                  f"{equivalence['requests']}。"]
    lines += ["", "## 三路负例重叠", "", markdown(overlap),
              "", "## 来源覆盖率", "", markdown(pd.DataFrame(stats["sample_source_coverage"])),
              "", "## Rank 与 False-negative 风险", "", markdown(rank),
              "", "## Position Confidence Proxy", "",
              markdown(position.head(20)), "",
              "> CTR(position) 只作为 train log 的 position-confidence proxy，不宣称它是因果 examination propensity。",
              "", "## False-negative 结构抽样", "",
              "固定 seed、未经人工挑选的每来源 150 条样本保存在 `false_negative_diagnostic.parquet`，"
              "包含标题以及可取得的 Dense/TF-IDF score。",
              f"- 过滤后 blocked overlap：`{stats['blocked_overlap_after_filter']}`。",
              "", "## Phase 5B 准入", "",
              "Mining 产物已通过不读取 test 和显式正例过滤检查；Phase 5B 只读使用这些固定负例池，不覆盖它们。"]
    lines += ["", "## Skip Top-rank 判断", ""]
    for source in ("dense", "tfidf"):
        source_rank = rank[rank.source == source].set_index("rank_bucket")
        if {"1-10", "11-20", "21-50"}.issubset(source_rank.index):
            lines.append(
                f"- {source}：known-positive overlap 从 Top1–10 的 "
                f"{100*source_rank.loc['1-10','known_positive_rate']:.3f}% 降至 11–20 的 "
                f"{100*source_rank.loc['11-20','known_positive_rate']:.3f}% 和 21–50 的 "
                f"{100*source_rank.loc['21-50','known_positive_rate']:.3f}%；"
                "首轮训练跳过 Top10，保留 11–50 与 51–100 两层。")
    (paths.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = BASE_OUT / "smoke" if args.mode == "smoke" else BASE_OUT
    paths = MiningPaths(ROOT, out, FROZEN)
    paths.out.mkdir(parents=True, exist_ok=True)
    limit = 100 if args.mode == "smoke" else None
    stages = (["prepare", "dense", "finalize-dense", "validate-dense-faiss", "tfidf", "finalize-tfidf",
               "merge", "audit", "report"] if args.stage == "all" else [args.stage])
    for stage in stages:
        print(f"[Phase5A] stage={stage} mode={args.mode}", flush=True)
        if stage == "prepare": prepare_samples(paths, limit_requests=limit)
        elif stage == "dense": mine_dense(paths, args.gpus, args.dense_batch_size,
                                            limit_requests=limit)
        elif stage == "finalize-dense": finalize_dense(paths)
        elif stage == "validate-dense-faiss": validate_dense_faiss_equivalence(paths)
        elif stage == "tfidf": mine_tfidf(paths, args.tfidf_batch_size, args.tfidf_threads,
                                            part_requests=args.tfidf_part_requests,
                                            limit_requests=limit,
                                            corpus_limit=10_000 if args.mode == "smoke" else None,
                                            worker_index=args.tfidf_worker_index,
                                            worker_count=args.tfidf_worker_count)
        elif stage == "finalize-tfidf": finalize_tfidf(paths)
        elif stage == "merge": merge_sample_pools(paths)
        elif stage == "audit": audit_pools(paths)
        elif stage == "report": write_report(paths)


if __name__ == "__main__":
    main()
