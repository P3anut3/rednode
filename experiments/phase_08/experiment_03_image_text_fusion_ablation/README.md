# Phase 8 / Experiment 03：图文融合与受控内容残差

## 目标

检验 Phase 8-02 图片残差的有图收益、无图退化是否可通过逐物品门控和图文交互改善。候选全集、temporal train/validation、history N=20、Frozen BGE/SigLIP、mean_all 图片、H2-256 side features/ID、TF-IDF hard-negative loss 和 GPU exact 评估协议均不变。不要使用 test 选模型。

## 模型与控制

- C0：Phase 7-04 H2-256；A0：Phase 8-02 Mean All 加法，均直接读取正式结果，不重训。
- F1：共同图片投影 `P_img(x)` 加逐物品 gate。
- F2：F1 加 `MLP([text256,image256])` 零初始化交互修正。
- F3：F1 加 `MLP([BGE768,SigLIP768])` 零初始化交互修正。
- 三者共同使用 `z=text256+available×alpha_image×gate×(P_img(x)+correction)`；然后叠加 H2 metadata 和 item-ID 残差并 L2 normalize。history、target、negative、candidate 共享物品塔。
- Gate 最后一层零初始化、初始值为1；共同参数从 A0 初值显式复制。合成自检与 smoke 必须确认 F1/F2/F3 初始 item/query 等于 A0，缺图图片增量严格为零。

## 执行顺序

```text
plan → self_check → audit → F1/F2/F3 单卡 smoke
→ F1/F2/F3 seed42 train → 串行全库 validation
→ bootstrap-seed42 → select
→ 仅最佳结构 seed43/44 train + validation → lock
→ diagnose（不参与选型）
→ 仅 GO：freeze-vector → 一次 terminal-test
→ report → 停止
```

运行入口：`experiments/phase_08/experiment_03_image_text_fusion_ablation/run.py`。默认 `plan` 只读；其他阶段需 `--confirm-run`。训练与检索阶段需 CUDA，禁止 CPU fallback。示例：

```bash
PY=/home/cmj/.conda/envs/qilin-onerec/bin/python
RUN=experiments/phase_08/experiment_03_image_text_fusion_ablation/run.py
$PY "$RUN" --stage plan
$PY experiments/phase_08/experiment_03_image_text_fusion_ablation/self_check.py
$PY "$RUN" --stage audit --confirm-run
$PY "$RUN" --stage smoke --model f1_gate --seed 42 --device cuda:0 --confirm-run
```

Seed42 先排除无图或 completely-unseen 相对 C0 点估计下降超过0.1pp 的结构，再按 Overall R@500 选择。仅最佳结构补 seed43/44。三 seed 最终 GO 要求：相对 C0 平均增益至少0.1pp、request-level CI 下界>0、相对 A0 总体不退步、有图正收益、无图及 completely-unseen CI 下界均≥-0.1pp、三个 seed 均正向。旧协议 validation cold_item 无样本，报告 N/A；同时报告有图×completely-unseen 交叉切片。Bootstrap CI 反映 request 波动，不能代替 seed 不确定性。

诊断仅使用 validation：对 A0 和最终 F 同一 seed 分别关闭历史/候选/双侧图片；最终 F 额外置零图片向量但保留 available flag。比较 Top500 组成、positive 增减、有图/无图召回。推理干预属分布外机制线索，不参与选型或 GO。

## 资源与结果

磁盘预预算：五组 best/last checkpoint、rankings、一次锁定向量、临时发布和其他 slack 共约8.5 GiB，之外至少保留16 GiB。未选中结构的全库向量不长期落盘；只有 validation GO 后才生成并 hash 绑定 terminal 向量。默认 `num_workers=0`；正式训练可最多两模型并发，全库 validation 与诊断扫描串行。进程树内存、主机 available、GPU 分配与磁盘空间均有保护线。Phase 8-01/02 产物只读、不清理或覆盖。

结果目录：`results/phase_08/experiment_03_image_text_fusion_ablation/`。正式结论见其 `summary.md`；运行期间不得用 proxy 召回替代全库结果。

## 最终结论

Audit 与三个单卡 smoke 均通过；F1/F2/F3 起始 item/query 和 A0 的最大差异约 `3e-8`，无图图片增量为0。三个结构均完成 seed42 训练与 1,983,938 全库 exact validation。R@500 依次为 F1 9.8747%、F2 9.8314%、F3 10.0350%；同 seed C0 为9.8098%、A0 为9.9032%。F3 有图×completely-unseen 提升，但无图相对 C0 下降0.7281pp；F1/F2 也分别下降0.5841pp/1.0629pp。三者均未通过预注册的无图初筛，因此正式选型为空、No-Go；未补 seed43/44，recommendation test 未读取。

仅做机制诊断的 F3 seed42 在关闭历史侧图片时，无图 R@500 从7.6760%升至8.9025%；关闭候选侧时为8.2152%。这些是分布外推理干预，不能作正式模型结果或严格因果解释。保留实验代码与图片资产，不将新融合接入当前 H2-256。完整配对 CI、交叉切片、参数和耗时见镜像结果目录的 `summary.md`。本阶段不自动微调 BGE/SigLIP，不做图片独立召回、搜索多任务或粗排。
