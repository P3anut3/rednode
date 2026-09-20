# Experiment 02：Bias-aware Hard Negative Training

- 目标：从冻结 N=20 Single Attention baseline 继续训练，比较 Dense、TF-IDF、impression 三路 hard negatives 及 position-aware weighting。
- 损失：保留 in-batch InfoNCE，叠加 source-specific pairwise `softplus(score_neg-score_pos)`；三路初始 lambda 均为 0.5。
- Item tower：frozen BGE-base-zh，不微调。
- GPU smoke：`python experiments/phase_05/experiment_02_bias_aware_hard_negative_training/run.py --stage smoke --device cuda`
- 单项训练：追加 `--stage train --only-model <ablation> --device cuda`；正式评估固定使用全库 Exact IndexFlatIP。
- 结果：`results/phase_05/experiment_02_bias_aware_hard_negative_training/`。
- 结论：单路 TF-IDF hard 最佳，Full R@500=8.1282%、Cold=11.4368%；position-aware 相对 simple 仅 +0.0014pp，不保留为默认配置。
