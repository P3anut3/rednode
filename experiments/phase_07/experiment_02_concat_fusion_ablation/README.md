# Phase 7 Experiment 02：Concat Fusion 消融

## 当前状态

**仅完成代码实现，等待人工 review。尚未读取真实实验数据、运行 smoke、训练、full validation 或 test。**

所有会产生实验产物的 stage 都要求显式传入 `--confirm-run`。默认 `--stage plan` 只打印冻结协议。正式结果具有 completion marker，terminal test 使用 one-shot manifest 防止重复读取。

## 目标与受控变量

本实验只比较两个 128 维全库双塔：

- `m1_direct_concat_id64`：Content 128 + Metadata/Profile 128 + ID 64 + seen flag，经 321→512→128 MLP。
- `m2_concat_residual_id64`：与 M1 完全相同，但输出为 `base + softplus(beta) * delta`；beta 初始 0.05，fusion 最后一层零初始化。

共同冻结项：temporal train/validation split、history N=20、BGE 输入、train-only vocabulary/normalization、Phase 5 TF-IDF hard-negative loss、temperature 0.05、batch 512、AdamW 1e-4、最多 6 epoch、patience 2。图片、匿名 dense features、位置和全周期行为统计均不使用。

M1/M2 相对现有 H2 同时改变融合方式和 ID 维度，因此不能单独归因于 ID64。只有两者都比 H2 validation 低至少 0.1pp，selection lock 才将 `add_id64_eligible` 标为 true；不满足条件时不得启动该控制实验。

该条件已在 seed 42 validation 触发，因此实现 `add_id64_control`：完全保留 H2 的加法结构，将 User/Item ID 降为 64 维，并分别通过无偏置 `Linear(64→128)` 接入 128 维空间。它只作为 seed 42 归因控制，不进入无条件多 seed sweep。

## 复用与边界

直接复用 Experiment 01 已审计的：

- train-only feature cache 与 `FeatureStore`；
- 列式 `Phase7Dataset` 和 batch BGE memmap gather；
- request-specific TF-IDF hard negatives及其 rank fallback；
- same-user、duplicate-target、history-positive mask；
- 统一 evaluator、temporal item status、feature/frequency slices；
- exact retrieval、history filtering 与 Top500 校验。

本实验新增的结构和 orchestration 全部保留在当前实验目录，不进入 `src/`。

## 文件

- `config.py`：冻结协议、路径和 deadline。
- `models.py`：M1/M2、ID64、shared item tower、positive beta。
- `losses.py`：只转接既有固定 TF-IDF-HN objective。
- `trainer.py`：2-worker 训练、proxy/full 编码、参数与 residual diagnostics。
- `run.py`：授权门、串行 full validation 锁、bootstrap、多 seed 和 one-shot test。
- `self_check.py`：仅合成张量，不读取项目数据。

## Review 后的建议执行顺序

当前不要执行以下命令；它们仅记录审核通过后的运行协议。

```bash
# 1. 合成自检（不读数据）
python -m experiments.phase_07.experiment_02_concat_fusion_ablation.self_check

# 2. 两个结构分别 smoke
python experiments/phase_07/experiment_02_concat_fusion_ablation/run.py \
  --stage smoke --model m1_direct_concat_id64 --seed 42 --device cuda:0 --confirm-run
python experiments/phase_07/experiment_02_concat_fusion_ablation/run.py \
  --stage smoke --model m2_concat_residual_id64 --seed 42 --device cuda:0 --confirm-run

# 3. 两卡并行训练 seed42；每 epoch 仅固定 proxy
# GPU0: M1 --stage train
# GPU1: M2 --stage train

# 4. M1/M2 full validation 必须串行
# --stage full-validation --model ... --seed 42

# 5. 锁定 seed42 结构；胜者（若差距<=0.1pp则两者）补 seed43/44
python experiments/phase_07/experiment_02_concat_fusion_ablation/run.py \
  --stage select --confirm-run

# 6. 三 seed结果齐备后 lock；之后 terminal test 只能执行一次
python experiments/phase_07/experiment_02_concat_fusion_ablation/run.py \
  --stage lock --confirm-run
```

## 本轮 Review 修复

- Cold-item 与 cold-user invariance 已拆开检查：前者只扰动 Item ID，后者只扰动 User ID，历史 Item ID 保持不变。
- 所有 epoch 使用同一组 5,000 validation requests 和同一组 100,000 candidates，不再跨不同 proxy 做 early stopping。
- 只有正常训练结束、curve/config/best/last 全部落盘后才写 training completion marker；deadline 中断不会生成 marker。
- Full validation、结构选择、最终锁定与 terminal preflight 均校验 checkpoint SHA-256；embedding metadata 同时绑定 checkpoint 与 candidate mapping。
- Terminal test 在读取 test 前完成 GPU、checkpoint、embedding、mapping、shape/dtype/norm 检查，检索和评估受 deadline 保护。
- M2 residual diagnostics 改为 validation 分桶统计，并增加 `cos(delta, base)`；覆盖 target-seen、history-only、completely-unseen、seen-user 与 unseen-user。
- `FusionMLP` 的输出维度由构造参数控制，不再写死 128。

## 检索与性能保护

虽然原实验描述写了 GPU chunked matmul，本实现延续已确认更快的 Phase 7 策略：FAISS GPU FlatIP 可用时优先使用，否则将约 198 万 × 128 float32 corpus 一次上传并常驻单张 GPU，再分 query batch exact search。不会重复通过 PCIe 搬运 corpus。

- 每 epoch 禁止全库检索；固定 100k proxy 仅用于 early stopping。
- 每个 best checkpoint 最多生成一次全库向量，并保存到 `embeddings/<model>/`；terminal test 复用锁定向量，禁止再次编码。
- M1/M2 的 embedding 与 validation ranking 分别落在模型子目录；`full_validation_active.lock` 强制串行扫描。
- 全部 epoch 使用完全相同的 5,000 requests + 100,000 candidates proxy，early-stopping 分数可直接比较。
- 正常训练结束后才写 `training_complete_<model>_seed<seed>.json`；full validation 校验 training curve、config 和 checkpoint SHA-256。
- embedding metadata 绑定 checkpoint SHA-256 和 candidate mapping SHA-256；训练完成或 validation 后拒绝覆盖 checkpoint。
- terminal test 在写 one-shot manifest、读取 test 前完成 CUDA、checkpoint、embedding、dtype/shape、mapping 和 SHA-256 preflight，并设置 20 分钟 deadline。
- timeout 不写 completion marker；训练 timeout 保存 last checkpoint。
- 每次只搜索 Top600，再统一过滤 history 并派生全部 K 指标。
- 不运行融合。

## 输出

所有产物只能写入：

`results/phase_07/experiment_02_concat_fusion_ablation/`

正式结果需包含 Recall@10/50/100/200/500、MRR@100、legacy warm/cold、temporal target-seen/history-only/completely-unseen、frequency slices、参数量、ID 参数量、checkpoint 大小、GPU 峰值、data wait、item/query encoding 与 exact-search 时间。M2 另外在 validation 上按 target-seen/history-only/completely-unseen 和 seen/unseen user 输出 beta、residual/base norm ratio、delta/base cosine 与 final/base cosine。
