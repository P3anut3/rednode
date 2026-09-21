# Qilin 全库推荐召回实验汇总

## 统一协议

- 候选全库：`data/notes` 中 1,983,938 个 unique note。
- 测试集：11,115 个 request，44,685 条 click 正互动；多正样本 request-level macro Recall。
- 主评估过滤 recent history，但不从 ground truth 删除正样本。
- Cold item 指未在 train recommendation 曝光过的 note。
- 测试 click 只用于评估，不用于候选生成、表示构建或训练。

## 阶段结果总览

| 阶段 / 实验 | 要验证的问题 | 关键结果 | 阶段结论 |
| --- | --- | --- | --- |
| Phase 0 / 数据分析 | Qilin 是否足以支持约 200 万 item 全库召回？ | train 曝光覆盖 29.40%，点击覆盖 9.50%；test 正互动中 44.44% 是 train-unexposed item；99.82% note 有文本。 | 可以评估全库召回，但纯 ID 方案无法充分覆盖长尾与 cold item，内容召回必要。 |
| Phase 1 / 传统 baseline | Random、Popularity、Category、ItemCF、TF-IDF 在统一全库协议下谁最强？ | TF-IDF R@500 **7.2394%**、Cold R@500 **9.9427%**；Popularity 2.8371%；ItemCF 最好 1.1994%；Random 0.0443%。TF-IDF 检索 2419.3s，峰值 21.621 GiB。 | 词法内容信号是强 baseline，但 exact sparse retrieval 昂贵。 |
| Phase 2 / English E5 | zero-shot dense + Faiss 能否超过 TF-IDF，ANN 能否加速？ | Exact R@100/R@500 = 0.3854%/0.8847%，Cold 1.2369%；ANN R@500 0.5641%，相对损失 36.24%；ANN 约比 TF-IDF exact 快 1262.9×。 | 速度问题可解，但 English E5 对中文语料表示失效，不能据此否定 dense。 |
| Phase 2 / 中文 encoder | 换成真正支持中文的 encoder 后，zero-shot dense 能提升多少？ | E5 中文 UNK 75.3000%，BGE-base-zh 为 0.0242%。BGE full R@100/R@500 = 1.2251%/2.6622%，Cold 3.7798%，为 E5 R@500 的 3.01×，仍低于 TF-IDF。 | 语言匹配是必要条件，但简单 history mean 成为新的主要瓶颈。 |
| Phase 3 / 用户表示 | 不训练 item encoder，多兴趣表示是否优于 single mean？ | Full Mean/Top3/KMeans-K4 R@500 = 2.6622%/4.0568%/**4.8467%**；KMeans Cold = **6.9938%**。K=4 在 equal budget 下仍提升，high-diversity 人群相对收益最大。 | Mean 确是重要瓶颈；有必要验证可学习 user tower。 |
| Phase 4 / N=10 User Tower | 冻结 item embedding，仅用行为监督学习聚合能否超过 KMeans？ | Mean+MLP 5.6782%；Single Attention **5.9759%**；Learned K4 equal 5.8425%。K4 inter-interest cosine 均值 0.9999，严重 collapse。 | 监督聚合有效，但收益来自 single attention，不是有效的 learned multi-interest。 |
| Phase 4 / N=20 复验 | 更长历史是否比继续复杂化多兴趣更重要？ | 5K：Single/K4/Adaptive R@500 = 8.1486%/8.2337%/8.1426%；K4 仅 +0.0851pp 且仍 collapse。Full N=20 Single Attention R@100/R@500 = **3.4709%/7.8919%**，Cold = **11.1411%**。 | N=20 Single Attention 超过 TF-IDF；当前无必要继续 fixed-K 多兴趣，先保留较完整历史更有价值。 |
| Phase 5A / Hard-negative mining | Dense、TF-IDF、session impression 三路 request-level HN 的规模、重叠与 false-negative 风险如何？ | 83,437 requests；Dense 100/request，TF-IDF 99.687/request，Impression 359,831 条；Dense/TF-IDF Top1–10 known-positive overlap 为 7.973%/35.079%；过滤后 blocked overlap 全为 0。 | 跳过 Top10 有直接依据；三路交集极少，但 Dense/TF-IDF 平均重叠约 9.52 条/request。 |
| Phase 5B / Bias-aware HN training | Raw HN 与 position-aware weighting 能否超过 7.8919%？ | 最佳 TF-IDF-hard R@100/R@500 = **3.5414%/8.1282%**，Cold = **11.4368%**；position-aware 相对同源 simple 仅 +0.0014pp。 | Raw HN 有小幅稳定收益；TF-IDF 单路最佳，组合无叠加，position proxy 无实质价值。 |

## 关键方法演进

| 方法 | Full R@500 | Cold R@500 | 相对当时基线的解读 |
| --- | ---: | ---: | --- |
| Random | 0.0443% | 0.0320% | 仅用于验证 evaluator |
| Global Popularity | 2.8371% | 0.0000% | 无法原生召回 train-unexposed item |
| TF-IDF history-20 | 7.2394% | 9.9427% | 最强词法 baseline，但 exact 慢 |
| E5 English mean-10 | 0.8847% | 1.2369% | 语言/词表严重错配 |
| BGE Chinese mean-10 | 2.6622% | 3.7798% | 中文 encoder 显著修复 item representation |
| BGE Top3-history-10 | 4.0568% | 5.6149% | 减少兴趣稀释有效 |
| BGE KMeans-K4-10 | 4.8467% | 6.9938% | 无监督多兴趣有效 |
| Learned Single Attention-10 | 5.9759% | 8.3835% | 行为监督显著改善聚合 |
| Learned Multi-interest-K4-10 | 5.8425% | 8.1459% | 低于 single attention，且 slot collapse |
| Learned Single Attention-20 | 7.8919% | 11.1411% | Phase 4.1 frozen baseline |
| **Single Attention-20 + TF-IDF Hard Negatives** | **8.1282%** | **11.4368%** | Phase 5 当前最佳；相对 frozen baseline +0.2363pp |

## 当前总结

1. **当前最佳线路是中文 BGE frozen item embedding + history N=20 + Single Attention + TF-IDF hard-negative training。** Full R@500 为 8.1282%，Cold 为 11.4368%；相对 Phase 4.1 frozen baseline 分别 +0.2363pp/+0.2957pp。
2. **多兴趣现象真实存在，但当前 learned fixed-K 实现没有证明其价值。** Phase 3 KMeans 证明分离兴趣可以帮助召回；但 Phase 4 的 K=4 注意力向量几乎完全重合，加小幅 diversity loss 仍未解决。
3. **前 20 条历史比前 10 条更有价值。** N=20 Single Attention 比 N=10 的 Full R@500 从 5.9759% 提高到 7.8919%，说明之前的 N=10 截断丢失了有用兴趣信号。
4. **尚不应直接将实验代码移入 `src/`。** 当前最佳模型虽已超过 TF-IDF，但还需要确认可重复训练稳定性、线上候选预算/ANN 损失、checkpoint 加载接口与生产数据契约。
5. **下一步建议：** 先做小范围 teacher-denoised hard negatives，重点处理 Dense/TF-IDF 高 rank 的 false-negative 风险；暂不 unfreeze BGE 或 joint two-tower。Hybrid retrieval 作为独立实验，不与 teacher filtering 同时引入。

## 已冻结正式 Baseline

- Baseline ID：`frozen_bge_base_zh_single_attention_n20_v1`
- 路径：`results/phase_04/experiment_02_history_n20_multi_interest/baseline_releases/v1_frozen_bge_n20_single_attention/`
- 内容：1,983,938-item BGE embedding、Exact `IndexFlatIP`、note mapping、N=20 Single Attention epoch-2 checkpoint、配置、full-test 结果、代码快照与 SHA-256 manifest。
- 状态：只读、append-only；不得覆盖，后续变更必须创建 v2。

## 历史产物映射

| 实验目录 | 已有结果目录 |
| --- | --- |
| `phase_00/experiment_01_qilin_data_analysis/` | `results/phase_00/experiment_01_qilin_data_analysis/` |
| `phase_01/experiment_01_full_corpus_baselines/` | `results/phase_01/experiment_01_full_corpus_baselines/` |
| `phase_02/experiment_01_zero_shot_dense/` | `results/phase_02/experiment_01_zero_shot_dense/` |
| `phase_02/experiment_02_chinese_encoder_benchmark/` | `results/phase_02/experiment_02_chinese_encoder_benchmark/` |
| `phase_03/experiment_01_user_representation/` | `results/phase_03/experiment_01_user_representation/` |
| `phase_04/experiment_01_learnable_user_tower_n10/` | `results/phase_04/experiment_01_learnable_user_tower_n10/` |
| `phase_04/experiment_02_history_n20_multi_interest/` | `results/phase_04/experiment_02_history_n20_multi_interest/` |
| `phase_05/experiment_01_request_hard_negative_mining/` | `results/phase_05/experiment_01_request_hard_negative_mining/` |
| `phase_05/experiment_02_bias_aware_hard_negative_training/` | `results/phase_05/experiment_02_bias_aware_hard_negative_training/` |

## Phase 6：ID 协同召回与 Content-ID Residual

- 目标：验证 Pure-ID 独立路线与 frozen-content ID residual。
- Pure-ID 最佳 validation R@500：2.1812%。
- M2 residual：No-Go；warm paired-bootstrap CI 为负。
- Terminal 最佳 R@500：8.4774%（quota_300_100_100）。
- 产物：`results/phase_06/experiment_01_id_two_tower_retrieval/summary.md`。

## Phase 7：Feature-enhanced Hybrid Two-Tower（待执行）

- 目标：在冻结 BGE、固定 128 维和既有 loss 协议下，以匹配的 `user ID + history-ID Attention` B0 为控制组，分别验证稳定用户画像、fans/follows、物品结构特征和受 mask 约束的 ID residual。
- 受控比较：B1a/B1b/B2/B3 均相对 B0；H1 相对 H0、H2 相对 H1、H3 相对 H2。Stage B 额外屏蔽 same-user batch false negatives。
- 状态：代码实现和合成自检已完成，尚未生成真实缓存、运行真实 smoke、训练或读取 test；所有执行 stage 需要人工审核后显式传入 `--confirm-run`。
- 代码：`experiments/phase_07/experiment_01_feature_hybrid_two_tower/`。
- 结果：`results/phase_07/experiment_01_feature_hybrid_two_tower/`。
