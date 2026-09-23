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

## Phase 7 / Experiment 01：Feature-enhanced Hybrid Two-Tower

- 目标：在冻结 BGE、固定 128 维和既有 loss 协议下，验证画像、物品结构特征和受 mask 约束的 ID residual。
- Pure-ID：B3 三 seed validation R@500 均值 3.6733%，明显优于匹配 B0 的 2.5014%；物品结构特征贡献大于用户画像。
- Hybrid：H0/H1/H2 seed42 validation R@500 为 8.5410%/8.8315%/9.5106%；H2 三 seed均值 9.5441%。H3 匿名 dense 特征降至 9.2747%，不保留。
- 唯一一次 terminal test：锁定 H2 seed42，R@100/R@500 为 3.8618%/8.7028%；completely-unseen R@500 为 10.4802%。
- 工程：128 维 198 万 corpus 常驻单卡 GPU，terminal exact search 1.053 秒；本实验未运行融合。
- 代码：`experiments/phase_07/experiment_01_feature_hybrid_two_tower/`。
- 结果：`results/phase_07/experiment_01_feature_hybrid_two_tower/summary.md`。

## Phase 7 / Experiment 02：Concat Fusion 消融

- 目标：在其余协议冻结的情况下比较 ID64 Direct Concat MLP 与 ID64 Concat MLP + Residual，判断 residual 是否保护内容与 cold recall。
- 控制：输出和全库索引仍为 128 维；history N=20、Frozen BGE、side features、Phase 5 TF-IDF hard-negative loss均不变；不使用图片、匿名 dense、位置或全周期行为统计。
- Seed42 validation：M1 Direct Concat R@500=0.5506%；M2 Residual Concat R@500=6.5028%，M2-M1 paired-bootstrap 95% CI 为 `[+5.6003,+6.3276]pp`。
- M2 三 seed validation R@500=6.5028%/6.4646%/6.4711%，均值 6.4795%、std 0.0167pp，稳定但明显低于现有 H2 三 seed均值 9.5441%。
- 条件控制：因 M1/M2 均低于 H2，补跑 Add-ID64 seed42；R@500=8.9520%。相对 H2 seed42，overall -0.5586pp、target-seen +0.2187pp、completely-unseen -0.9476pp。
- 决策：Concat 结构 No-Go；ID64 Add 控制也未达到替换标准，继续保留 Experiment 01 的 Add-ID128 H2。没有读取 terminal test，避免为落后候选消耗 test。
- 代码：`experiments/phase_07/experiment_02_concat_fusion_ablation/`。
- 结果：`results/phase_07/experiment_02_concat_fusion_ablation/summary.md`。

## Phase 7 / Experiment 03：作者式 Concat 与 ID 正则消融

- 目标：解释 Experiment 02 Direct Concat 的 0.5506% R@500 是否来自融合前的 Content 768→128 信息瓶颈，并独立验证 ID64→ID32、小初始化和 Structured ID Dropout。
- 结构：E1–E7 受控消融；完整作者式物品塔直接拼接 BGE768 + metadata128 + ID32 + seen flag，经 929→512→128 MLP；用户塔为 history128 + profile128 + ID32 + seen flag，经 289→512→128 MLP。
- 诊断：为每个 best checkpoint 固定计算 Item-ID-off、User-ID-off、All-ID-off、按互斥 temporal/frequency bucket 的 ID-on/off 表示稳定性，以及每 epoch target/history/user 的实际整路 dropout 比例；dropout 同时关闭 ID 向量和 seen flag，以匹配真实 cold 输入。
- 安全门禁：E0 只读取 Experiment 02，不重训；所有执行阶段需 `--confirm-run`；训练/验证通过 completion marker 与 SHA-256 绑定；仅验证 gate 全部通过才允许一次 terminal test。
- Seed42 validation：E0/E1/E2/E3/E4/E5/E6/E7 R@500 分别为 0.5506%/1.0487%/0.5305%/0.5497%/2.5115%/0.5015%/2.8738%/3.2898%。Raw BGE768、ID32 和小初始化均未单独修复 Direct Concat；Structured ID Dropout 是主要有效因素。
- 最佳 E7 Full-P50 三 seed R@500 为 3.2898%/3.2585%/3.4463%，均值 3.3315%、std 0.0821pp，明显低于 H2 三 seed均值 9.5441%。E7 vs H2 overall paired delta = -6.2208pp，95% CI `[−6.6265,−5.8409]pp`。
- ID-off：E7 normal/Item-ID-off/User-ID-off/All-ID-off R@500 = 3.2898%/3.7674%/3.2302%/3.7511%，说明模型没有依靠 ID shortcut，但 Direct Concat 主结构本身远弱于 Add H2。
- 决策：**No-Go**。terminal gate 仅 All-ID-off 保留率通过，其余四项失败；未读取 test，继续保留 Experiment 01 的 H2。
- 代码：`experiments/phase_07/experiment_03_author_concat_regularization_ablation/`。
- 结果：`results/phase_07/experiment_03_author_concat_regularization_ablation/summary.md`。

## Phase 7 / Experiment 04：H2 检索维度消融

- 目标：用 H0/H1/H2 的 128d/256d 2×3 对照判断最终 retrieval space 是否存在容量瓶颈。
- 控制：仅训练 H0/H1/H2-256；128d 直接读取 Experiment 01 正式结果和 rankings。ID embedding保持128d，类别embedding保持16d，numeric hidden保持32d。
- 协议：完全复用 temporal train、N20、Frozen BGE、side features、TF-IDF HN、InfoNCE、proxy和全库GPU exact evaluator。
- Seed42 validation：H0/H1/H2-256 R@500 = 9.4083%/9.8353%/9.8098%，相对对应128d提升 +0.8672pp/+1.0038pp/+0.2992pp，三种结构均受益，说明128d存在普遍容量瓶颈。
- H2-256 三 seed R@500 = 9.8098%/9.9023%/9.9521%，均值9.8881%、std 0.0590pp；overall paired-bootstrap 95% CI `[+0.0325,+0.5736]pp`。
- 唯一 terminal test：锁定中位 seed43，R@100/R@500 = 3.9854%/9.0231%，相对 H2-128 terminal 8.7028%提升 +0.3203pp。
- 工程：256d全库item vectors为1.892 GiB，validation exact search约1.60s，峰值显存约4.22 GiB；全部预注册gate通过，结论为 **GO，升级256d**。
- 代码：`experiments/phase_07/experiment_04_h2_retrieval_dimension_ablation/`。
- 结果：`results/phase_07/experiment_04_h2_retrieval_dimension_ablation/summary.md`。
