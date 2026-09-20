# Experiment 01：Full-Corpus Recall Baselines

- 目标：用统一 evaluator 比较 Random、Global/Category Popularity、ItemCF 和 TF-IDF。
- 结果：TF-IDF 最强，Overall R@500=7.2394%，Cold R@500=9.9427%；Popularity=2.8371%，ItemCF 最好=1.1994%。TF-IDF exact 检索 2419.3s，峰值内存 21.621 GiB。
- 运行：`python experiments/phase_01/experiment_01_full_corpus_baselines/run.py --mode smoke|full`
- 产物：`results/phase_01/experiment_01_full_corpus_baselines/full/`；smoke 在同级 `smoke/`。
