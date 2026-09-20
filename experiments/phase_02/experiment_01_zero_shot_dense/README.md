# Experiment 01：English E5 Zero-shot Dense

- 目标：验证 zero-shot dense representation 质量及 Faiss exact/ANN 全库检索成本。
- 结果：E5 Exact R@500=0.8847%、Cold=1.2369%；ANN R@500=0.5641%，相对 exact 损失 36.24%。英文 tokenizer 对中文有严重 `[UNK]` 错配。
- 运行：`python experiments/phase_02/experiment_01_zero_shot_dense/run.py --stage smoke|encode|indices|ablations|full|report|all`
- 产物：`results/phase_02/experiment_01_zero_shot_dense/`。
