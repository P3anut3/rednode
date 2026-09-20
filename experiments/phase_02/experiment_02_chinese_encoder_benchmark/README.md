# Experiment 02：Chinese Encoder Benchmark

- 目标：严格固定检索协议，验证中文 BGE 能够修复多少 English E5 的语言错配。
- 结果：BGE 中文 UNK rate=0.0242%（E5=75.3000%）；full R@500=2.6622%，Cold=3.7798%。Basic Clean 略优，删除 emoji 无收益。
- 运行：`python experiments/phase_02/experiment_02_chinese_encoder_benchmark/run.py --stage diagnostics|encode|evaluate|report|all`
- 产物：`results/phase_02/experiment_02_chinese_encoder_benchmark/`。
