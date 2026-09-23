# Phase 7 Experiment 03：作者式 Concat 与 ID 正则消融

## 当前状态

**实验已完成并判定 No-Go；terminal gate 未通过，因此没有读取 test。**

- seed42 最佳：E7 Full-P50，validation R@500 = 3.2898%；
- E7 三 seed R@500 = 3.2898% / 3.2585% / 3.4463%，均值 3.3315%；
- H2 三 seed均值 = 9.5441%；
- E7 vs H2 overall paired delta = -6.2208pp，95% CI
  `[-6.6265, -5.8409]pp`；
- Structured ID Dropout 是唯一显著有效因素，但不足以挽救 Direct Concat。

本实验不修改 Experiment 02。E0 固定读取 Experiment 02 的
`m1_direct_concat_id64` checkpoint、正式 validation 指标和全库 embedding，
不会重新训练；E1–E7 写入本实验独立结果目录。

## 验证目标

实验仅拆解以下四个变量：

1. Fusion MLP 看前置投影后的 128d content，还是完整 BGE 768d；
2. User/Item ID embedding 使用 64d 还是 32d；
3. ID 初始化标准差为 0.02 还是 0.01；
4. 是否采用无 rescale 的整路 Structured ID Dropout。

除以上变量外，继续固定 temporal split、history N=20、Frozen BGE、现有安全
side features、Phase 5 TF-IDF hard-negative loss、temperature=0.05、batch=512、
AdamW 1e-4、最多 6 epoch、patience=2、固定 proxy 和 1,983,938 全库 exact
evaluator。

## 消融矩阵

| Model | Content 输入 | ID dim | ID init std | ID dropout |
|---|---:|---:|---:|---:|
| E0 `e0_old_direct_concat` | 128 | 64 | 0.02 | 0 |
| E1 `e1_content768_only` | 768 | 64 | 0.02 | 0 |
| E2 `e2_id32_only` | 128 | 32 | 0.02 | 0 |
| E3 `e3_id32_small_init` | 128 | 32 | 0.01 | 0 |
| E4 `e4_dropout_only_p30` | 128 | 64 | 0.02 | 0.3 |
| E5 `e5_content768_id32_small_init` | 768 | 32 | 0.01 | 0 |
| E6 `e6_full_p30` | 768 | 32 | 0.01 | 0.3 |
| E7 `e7_full_p50` | 768 | 32 | 0.01 | 0.5 |

raw-content 模型的物品输入为 `768 + 128 + ID + 1`：ID32 时正好 929d；
用户输入为 `128 + 128 + ID + 1`：ID32 时为 289d。输出与索引始终为 128d。

## Structured ID Dropout 契约

- mask 形状是每个 ID 单元一个 `[..., 1]`，整条 ID 向量同时保留或清零；
- 不做普通 dropout 的 `1/(1-p)` 放大；
- target item、history item、user ID 三侧均使用同一规则；
- OOV/padding row 0 永远是零；eval/inference 自动关闭 dropout；
- 使用 `raw_seen` 统计原始词表成员身份；送入 Fusion MLP 的是
  `effective_seen = raw_seen & keep`，ID-off/dropout 会同时清零 ID 向量与 seen flag，
  从而与真实 cold serving 分布一致；
- 一个 target 在 in-batch candidate matrix 中只编码一次，不能按 query 重采 mask；
- 每 epoch 记录 target/history/user 的 seen、OOV、dropped-seen 数量和实际比例。

## 分阶段门禁

默认命令只打印计划，不读取数据：

```bash
python experiments/phase_07/experiment_03_author_concat_regularization_ablation/run.py
```

所有会读取 Qilin 数据或写产物的阶段必须显式加 `--confirm-run`。建议 review 后按：

```text
static-audit
→ synthetic self_check.py
→ E1–E7 smoke
→ E1–E7 seed42 train
→ 串行 full-validation
→ select
→ 仅 E5/E6/E7 最佳（及≤0.1pp第二名）补 seed43/44
→ lock
→ 满足全部 gate 时 one-shot terminal-test
→ report
```

历史执行示例：

```bash
python experiments/phase_07/experiment_03_author_concat_regularization_ablation/run.py \
  --stage smoke --model e6_full_p30 --seed 42 --device cuda:0 --confirm-run
```

训练必须先存在同 model/seed 的成功 smoke。Full validation 必须先验证 training
completion marker、完整 curve/config 和 checkpoint SHA-256；完成 validation 后同名
checkpoint 不可覆盖。全库 validation 有串行锁。

## Validation 诊断

每个 best checkpoint 除正常检索外，还运行：

- Item-ID-off：target/history Item ID 全部屏蔽，User ID 保留；
- User-ID-off：User ID 屏蔽，Item/history ID 保留；
- All-ID-off：两侧 ID 均屏蔽。

三种 ID-off 都同时关闭对应 ID 向量和 seen flag，模拟真实 OOV/cold 输入。另固定抽样计算
`cos(item_ID-on, item_ID-off)`、向量 norm 和 raw ID norm，按 target-seen、
history-only、completely-unseen 与频次桶输出 mean/P10/P50/P90。

E0 normal 指标和 embedding 直接引用 Experiment 02；本实验只补它的 ID-off 与表示
稳定性诊断。

## 多 seed 与 test gate

E5/E6/E7 seed42 最高者补 seed43/44；若第二名差距不超过 0.1pp，也补三 seed。
Terminal test 只有同时满足以下条件才会开放：

1. seed42 R@500 ≥ H2 seed42 9.5106%；
2. 三 seed mean ≥ H2 三 seed mean 9.5441%；
3. completely-unseen 不低于 H2；
4. All-ID-off 至少保留 normal R@500 的 80%（将“无灾难下降”固化为可 review 阈值）；
5. paired bootstrap 的 95% CI 不支持 H2 明显更好。

E0/H2 gate 数值不硬编码；运行 selection 时从已有正式 validation JSON 读取完整
浮点精度，并把每个来源文件的 SHA-256 写入 selection/final lock。Full validation
同样记录全库 item-vector 文件 SHA-256、shape、dtype、candidate count 和维度，
terminal preflight 必须逐项复核。

若任一 gate 不满足，`terminal-test` 会拒绝读取 test。

E1–E4 只有 seed42，定位是单因素初筛，微小差异不得解释成稳定因果结论。
Content 128→768 同时改变输入宽度、非 ID 参数量和随机数消耗；报告只能表述为
“与移除前置信息瓶颈一致”，不能声称完全排除容量或随机性的影响。正式结论只使用
按规则补齐 42/43/44 的最佳完整配置。

## 产物

代码：`experiments/phase_07/experiment_03_author_concat_regularization_ablation/`

结果：`results/phase_07/experiment_03_author_concat_regularization_ablation/`

其中 checkpoint、embedding、metrics、validation rankings、dropout/表示稳定性诊断和
completion marker 均按 model/seed 独立命名，不会覆盖 Experiment 02。
