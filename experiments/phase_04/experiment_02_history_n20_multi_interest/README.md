# Experiment 02：History N=20 与 Multi-interest 复验

- 目标：检验先前截断为 10 条是否丢失信号，并比较 N=20 Single Attention、K=4 + diversity loss 与 adaptive multi-interest。
- 结果：5K K4 仅比 Single +0.0851pp 且仍 collapse；Full N=20 Single Attention R@100/R@500=3.4709%/7.8919%，Cold=11.1411%，已超过 TF-IDF。
- 运行：`python experiments/phase_04/experiment_02_history_n20_multi_interest/run.py --help`
- 产物：`results/phase_04/experiment_02_history_n20_multi_interest/`。

## Frozen baseline

`baseline_releases/v1_frozen_bge_n20_single_attention/` 是已冻结的正式 baseline，包含全库 BGE embedding、Exact Faiss、mapping、epoch-2 checkpoint、配置、full-test 结果、代码快照和 SHA-256 manifest。该版本不得覆盖；后续变更必须创建 v2。
