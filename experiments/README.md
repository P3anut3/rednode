# Qilin 实验目录

本目录收纳尚未上线的所有实验代码。当前没有任何实现被正式提升为生产代码，因此原 `src/` 实验模块已全部回归本目录。

## 命名规则

```text
experiments/
├── common/                              # 跨阶段共用，但仍属实验性质的数据/评估工具
├── phase_XX/
│   └── experiment_YY_<short_name>/
│       ├── run.py                    # 统一入口
│       ├── README.md                 # 目标、协议、结果、产物映射
│       └── recall|models|training/   # 该次实验专属实现
└── SUMMARY_ZH.md                         # 全部实验中文总结
```

- 阶段统一使用 `phase_00`、`phase_01` 等两位编号。
- 同一阶段的新实验必须新建 `experiment_YY_<short_name>` 目录，不再向 `experiments/` 根目录添加编号脚本。
- 每个实验的结果必须写入同名镜像目录 `results/phase_XX/experiment_YY_<short_name>/`。checkpoint、embedding 和 Faiss 索引不放入 Git。
- smoke 结果放在该实验结果目录的 `smoke/`，不另建根级结果目录。
- 只有经过明确上线决策、补齐测试和稳定接口后，才从实验目录提升到 `src/`。

## 当前目录

| 阶段 | 实验 | 主入口 |
| --- | --- | --- |
| Phase 0 | Qilin 数据分析 | `phase_00/experiment_01_qilin_data_analysis/run.py` |
| Phase 1 | 全库传统召回 baseline | `phase_01/experiment_01_full_corpus_baselines/run.py` |
| Phase 2 | English E5 zero-shot dense | `phase_02/experiment_01_zero_shot_dense/run.py` |
| Phase 2.5 | 中文 BGE encoder benchmark | `phase_02/experiment_02_chinese_encoder_benchmark/run.py` |
| Phase 3 | 用户表示与无监督多兴趣 | `phase_03/experiment_01_user_representation/run.py` |
| Phase 4 | N=10 可学习 User Tower | `phase_04/experiment_01_learnable_user_tower_n10/run.py` |
| Phase 4.1 | N=20 与多兴趣必要性复验 | `phase_04/experiment_02_history_n20_multi_interest/run.py` |

完整结果见 [SUMMARY_ZH.md](SUMMARY_ZH.md)。
