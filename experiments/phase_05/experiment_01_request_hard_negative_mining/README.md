# Experiment 01：Request-level Hard Negative Mining

- 目标：按 Phase-4.1 training sample/request 挖掘 Dense Top100、TF-IDF Top100 与同 timestamp session impression negatives，完成 false-negative、rank、overlap 和 position-confidence audit。
- 锁定 Dense 来源：`frozen_bge_base_zh_single_attention_n20_v1`。
- 单进程运行：`python experiments/phase_05/experiment_01_request_hard_negative_mining/run.py --stage all --mode full`
- TF-IDF 支持按 `--tfidf-worker-index/--tfidf-worker-count` 划分互斥分片；已有 parquet 分片自动跳过，可断点恢复。
- 结果：`results/phase_05/experiment_01_request_hard_negative_mining/`。
- 边界：仅使用 train recommendation；test 不用于 mining。Phase 5A 完成 audit 前不启动 Phase 5B。
- 结论：Dense/TF-IDF Top1–10 的 known-positive overlap 分别为 7.973%/35.079%，跳过 Top10 有依据；三路过滤后 blocked overlap 均为 0。完整统计见对应结果目录 `summary.md`。
