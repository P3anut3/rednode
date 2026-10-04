# Phase 8 / Experiment 02：冻结图片特征对 H2-256 召回的增益

## 验证目标

只检验将 Phase 8-01 的冻结 SigLIP 物品图片向量接入现有 H2-256 物品塔，能否提高 1,983,938 全库推荐召回，同时保护无图与冷物品。没有图片独立召回、搜索多任务或粗排。

## 固定协议与对照

- C0：直接引用 Phase 7-04 `h2_256` 同 seed 的正式 validation JSON 与 per-request ranking，不重训、不覆盖。
- Temporal train 254,583 positives，validation 47,733 positives；最近 20 条历史；BGE 和 SigLIP 向量均冻结。
- Item/User ID embedding 仍为 128d，metadata/profile、attention、ID adapter 和 H2-256 其他结构不变；最终检索空间为 256d。
- “双塔不变”指结构与训练协议不变：BGE/SigLIP 原始 embedding 冻结，H2 原有可训练层与新增图片投影一起从相同 seed 重新训练；不是冻结 Phase 7-04 的整个 H2 checkpoint。新增层在旧层初始化完成后创建，旧层的同 seed 初值保持一致。
- 训练目标完全沿用 Phase 7-04：in-batch InfoNCE + 0.5 TF-IDF hard-negative pairwise，temperature 0.05；AdamW 1e-4、batch 512、最多 6 epoch、patience 2。
- 每 epoch 固定 5,000 validation requests、100,000 proxy candidates；best checkpoint 才做一次全库 GPU exact validation。
- 只用 temporal train 拟合旧特征 vocabulary、数值统计；不读取 test 来选图片策略、seed 或 checkpoint。

## 唯一新增分支

`image_available × α_image × Linear(SigLIP_768 → 256, bias=False)` 加到原 H2 物品向量归一化之前。`α_image` 以 0.05 初始化并学习。历史 item、训练 target、TF-IDF negatives、全库 candidate 共用同一个 item tower。缺图时该分支严格为零，但最终 query 和其他模型参数仍可能改变，因此必须实测无图召回。

| 名称 | 冻结图片输入 |
|---|---|
| C0 | 无；读取 Phase 7-04 既有结果 |
| `i1_first` | `first_image` |
| `i2_top3` | `mean_top3` |
| `i3_all` | `mean_all` |

图片产物以只读 mmap 按 canonical row 读取；一次进程只使用当前策略。不会重新解码图片或运行 SigLIP。Phase 8-01 的 SHA-256、shape、dtype、availability mask 与 H2 note ID mapping 在 audit 和正式运行前核对。

## 阶段与门禁

默认 `plan` 只打印计划，不读大数据、不写产物。其他阶段均需 `--confirm-run`。Audit、三种图片策略的单卡 smoke、正式训练与 validation 已完成；预注册门禁判定 No-Go，terminal test 未打开。

```text
plan → 人工 review → audit → 三模型单卡 smoke
→ I1/I2/I3 seed42 train → seed42 full-validation（串行）
→ bootstrap-seed42 → select → 最佳策略 seed43/44 train + full-validation
→ lock（三 seed 配对 bootstrap 与 Go/No-Go）
→ 仅 GO 时 terminal-test 一次 → report
```

运行入口：

```bash
PY=/home/cmj/.conda/envs/qilin-onerec/bin/python
RUN=experiments/phase_08/experiment_02_image_hybrid_recall/run.py
$PY "$RUN" --stage plan
# review 通过后再按阶段运行；例如：
# $PY "$RUN" --stage audit --confirm-run
# $PY "$RUN" --stage smoke --model i1_first --seed 42 --device cuda:0 --confirm-run
```

训练前 audit 统计 train/validation 正样本有图率及 temporal item 分层。Smoke 检查冻结输入、图片投影梯度、缺图严格零分支、history/target/candidate 共用 item tower、256d 单位范数、Top500 唯一性与 history filtering，并记录吞吐、data wait fraction、进程树 RSS 和峰值显存。先单卡通过 smoke，再决定正式并发；正式 full-validation 必须串行。首次以 2 workers、prefetch 1 运行时进程树 RSS 达 16.01 GiB，被保护线安全终止；改成 0 worker 后三种策略全部通过。因此 DataLoader 默认改为 0 workers、prefetch 1；样本抽样、batch、loss 与 shuffle seed 不变。

执行保护线默认：主机 available ≥48 GiB、进程树 RSS ≤16 GiB、GPU allocated peak ≤20 GiB、磁盘 free ≥16 GiB；训练每 batch 检查内存/GPU、每 20 batch 检查磁盘，越线立即停止且不写完成 marker。全库向量验证前还额外预留一份 256d float32 item vectors 的空间。

Full-validation 使用同实验排他锁。异常退出后只允许显式 `--recover-stale-validation-lock`，并校验锁属于同主机且 PID 已不存在；不会自动覆盖已完成的 validation marker。

主指标是同 seed、同 request 的 validation Recall@500。报告 R@100、MRR@100、有图/无图、旧 warm/cold、train-target-seen/history-only/completely-unseen 分层，及相对 C0 的 request-level paired bootstrap。现有 H2-256 三个 validation seed 的旧协议 `cold_item.eligible_requests=0`，该切片固定报告 **N/A**，不用于门禁，也不从 test 补样本；冷启动保护使用有 11,499 个 eligible validation requests 的 `completely_unseen`。选策略只看 seed42 validation；最佳策略补 seed43/44。GO 门槛：三 seed mean R@500 至少提升 0.1pp，overall paired 95% CI 下界 > 0，无图和 completely-unseen 的 paired 95% CI 下界均不低于 -0.1pp。未通过则不读取 test。

通过 validation gate 后仅做一次 terminal test；报告将图片模型与已锁定 H2-256 terminal R@500=9.0231% 及差值并列。Terminal 数值只用于最终观察，不参与策略或 checkpoint 选择。

若资源限制迫使正式 batch 或 loss 协议变化，本实验结果不能直接与既有 C0 做图片因果比较；必须先另建匹配无图对照。本入口直接拒绝正式 batch != 512。

## 结果路径与当前结论

运行后全部新产物写入 `results/phase_08/experiment_02_image_hybrid_recall/`，含 audit、smoke、checkpoint、proxy/training curve、full validation、per-request ranking、item vectors、bootstrap、决策和 summary。Phase 7-04 与 Phase 8-01 产物只读。

最终结论：Audit 显示 temporal train 的 254,583 个正样本中 115,012 个有图（45.18%）；validation 的 47,733 个正样本中 20,738 个有图（43.45%）。三种策略的 seed42 单卡 smoke 均通过。Seed42 全库 validation R@500：首图 9.8304%、Top3 9.8677%、Mean All 9.9032%，同 seed C0 为 9.8098%；按预定规则选择 Mean All 补 seed43/44。Mean All 三 seed 平均 R@500 为 9.9757%，相对同 seed C0 平均仅 +0.0876pp；配对 bootstrap overall 95% CI 为 [-0.0370,+0.2091]pp，跨零。有图正样本约 +0.6668pp，但无图正样本约 -0.4689pp；completely-unseen 非劣效门槛亦未通过。**No-Go：不将图片分支并入当前 H2-256；test 未读取。** 详细表和资源统计见镜像目录的 `summary.md`。
