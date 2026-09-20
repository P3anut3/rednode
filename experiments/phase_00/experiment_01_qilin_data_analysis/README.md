# Experiment 01：Qilin 数据分析

- 目标：检查真实 parquet schema、行为稀疏性、item 长尾、cold start、history 覆盖和内容可用性，判断是否可以设计约 200 万 note 全库召回。
- 结果：点击仅覆盖全库 9.50%，test 正互动 44.44% 属于 train-unexposed item，但 99.82% note 有文本。可做全库评估，不应以纯 ID retrieval 作为全库主方案。
- 运行：`python experiments/phase_00/experiment_01_qilin_data_analysis/run.py`
- 产物：`results/phase_00/experiment_01_qilin_data_analysis/`。
