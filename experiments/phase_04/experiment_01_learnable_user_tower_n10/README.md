# Experiment 01：Learnable User Tower（History N=10）

- 目标：冻结 BGE item embedding，仅训练 Mean+MLP、Single Attention 与 K=4 Multi-interest User Tower。
- 结果：Full R@500 分别为 5.6782%/5.9759%/5.8425%。K=4 inter-interest cosine 约 0.9999，发生严重 collapse；Single Attention 最好。
- 运行：`python experiments/phase_04/experiment_01_learnable_user_tower_n10/run.py --help`
- 产物：`results/phase_04/experiment_01_learnable_user_tower_n10/`。
