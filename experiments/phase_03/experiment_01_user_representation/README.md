# Experiment 01：User Representation Benchmark

- 目标：固定中文 BGE item embedding，比较 mean、recent-only、Top-k history 和 request-local KMeans multi-interest。
- 结果：Full Mean/Top3/KMeans-K4 R@500=2.6622%/4.0568%/4.8467%；KMeans Cold=6.9938%，且 equal-budget 下仍有收益。
- 运行：`python experiments/phase_03/experiment_01_user_representation/run.py --help`
- 产物：`results/phase_03/experiment_01_user_representation/`。
