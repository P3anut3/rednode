# Phase 7：Feature-enhanced Hybrid Two-Tower

## 状态

实验已完成。配置在 validation 上锁定后，terminal test 只读取了一次；不得覆盖或据 test 回调参数。

- 最佳结构：`h2_side_features_id`，三 seed validation R@500 均值 9.5441%。
- Terminal test：R@100 3.8618%，R@500 8.7028%。
- 全库 exact fallback 将约 198 万条 128 维 float32 corpus 一次上传并常驻单张 GPU；test exact search 1.053 秒。
- 本轮没有运行融合。

## 验证目标

1. ID-B0/B1a/B1b/B2/B3 在同一个 `user ID + history-ID Attention` 控制结构上，分别验证稳定画像、粉丝/关注数、物品结构特征及其组合的增量。
2. H0/H1/H2 验证冻结 BGE 后，side features 与受 mask 约束的 ID residual 能否形成单一全库 Hybrid 索引。
3. H3 仅在 H2 通过 validation Go 标准后，单独引入匿名 `dense_feat1~40`。

固定协议：254,583 train positives、47,733 validation positives、history N=20、128 维输出、BGE 完全冻结。ID 使用 Phase 6 logQ in-batch loss；Hybrid 固定使用 Phase 5 TF-IDF hard-negative loss。

## 代码结构

- `config.py`：冻结协议、字段白名单、路径和预算。
- `features.py`：train-only vocab/归一化、mmap 特征缓存及可选 H3 dense 缓存。
- `dataset.py`：数组索引数据集；DataLoader 不读取 Parquet/解析字符串。
- `models.py`：匹配控制组 ID-B0、ID-B1a/B1b/B2/B3 与 H0/H1/H2/H3。
- `losses.py`：固定 logQ 与 TF-IDF-hard objective。
- `retrieval.py`：FAISS GPU 优先；fallback 将完整 128 维 corpus 常驻单卡 GPU；一次 Top600 后过滤历史。
- `evaluation.py`：统一 evaluator、temporal item status、paired bootstrap。
- `run.py`：有条件执行、预算保护、validation lock 和 one-shot terminal test。
- `self_check.py`：只使用合成张量，不读取 Qilin 数据。

## 特征契约

Stage A 只读取明确白名单。任何曝光、点击、点赞、收藏、评论、分享、播放累计字段均禁止进入缓存。

- 用户类别：`gender/platform/age`，temporal-train vocab，0 为 OOV。
- 用户数值：`fans_num/follows_num`，非负 `log1p` 后按 temporal-train 用户 z-score。
- 物品类别：`note_type/taxonomy1_id/taxonomy2_id/commercial_flag`，temporal-train item vocab，0 为 OOV。
- 物品数值：`log1p(video_duration)`、裁剪后的宽高比、`log1p(image_num/content_length)`，按 temporal-train item 标准化。
- ID vocabulary：temporal-train history 与 positive target 的并集；OOV/cold ID residual 固定为零。
- H3 dense feature 不在 Stage A 生成，只有 H2 validation 产生 eligibility lock 后才能单独缓存。

除题目指定六个文件外，缓存还保存 `user_ids.npy`、`item_ids.npy` 和三份 train-only ID 映射，以保证数组行号可审计。

## 审核后运行顺序

以下命令保留为可复现实验说明；现有结果已有完成锁，程序会拒绝覆盖关键正式产物：

```bash
# 1. Stage A audit/cache（不训练）
python experiments/phase_07/experiment_01_feature_hybrid_two_tower/run.py \
  --stage stage-a --confirm-run

# 2. 每个结构必须先 smoke
python experiments/phase_07/experiment_01_feature_hybrid_two_tower/run.py \
  --stage smoke --model id_b0_user_history_control --confirm-run

# 3. 正式训练；每 epoch 只使用固定 proxy validation
python experiments/phase_07/experiment_01_feature_hybrid_two_tower/run.py \
  --stage train --model id_b0_user_history_control --seed 42 --confirm-run

# 4. 最佳 checkpoint 仅做一次 full validation
python experiments/phase_07/experiment_01_feature_hybrid_two_tower/run.py \
  --stage full-validation --model id_b0_user_history_control --seed 42 --confirm-run

# 5. seed42 结构选择后才允许最佳结构补跑 seed43/44
python experiments/phase_07/experiment_01_feature_hybrid_two_tower/run.py \
  --stage select --family id --confirm-run
```

ID-B3、H3 均有条件锁。最终 test 还要求先执行 `--stage lock`，并由 `terminal_test/manifest.json` 保证只读一次。

本轮实验只比较各双塔自身的 validation/test 指标；不运行固定预算融合、RRF 或融合 quota 调整。

## 受控比较与执行门槛

- `B1a - B0`：仅稳定画像 `gender/platform/age` 的增量。
- `B1b - B0`：稳定画像再加入 `fans/follows` 的总增量。
- `B2 - B0`：仅物品结构特征的增量。
- `B3 - B0`：全部 side features 的增量；只有 B1a/B1b/B2 至少一个 warm R@500 提升不低于 0.1pp 且 paired-bootstrap warm CI 下界大于 0 才能运行。
- `H1 - H0`：side features 增量；`H2 - H1`：ID residual 增量；`H3 - H2`：匿名 dense features 增量。
- Stage B 的 logQ loss 会同时屏蔽 duplicate target、history target、same-request positive 和 same-user positive，避免同用户 batch 内 false negative。

正式训练使用列式 NumPy 缓存，不做逐样本 Pandas `iloc`；BGE 在 collator 中按 batch gather。训练、item encoding、GPU corpus upload、query search 和 bootstrap 都在安全 batch 边界检查 deadline。

## 结果目录

所有缓存、checkpoint、曲线、validation ranking、锁、test 和报告严格写入：

`results/phase_07/experiment_01_feature_hybrid_two_tower/`

## 当前结论

- 结构特征对 Pure-ID 的增益大于用户画像；B3 三 seed R@500 均值 3.6733%。
- H1 side features 相对 H0 提升 0.2904pp；H2 再加入 ID residual 后提升 0.6791pp。
- H3 匿名 dense features 相对 H2 显著下降 0.2359pp，不保留。
- 锁定 H2 后 terminal test R@500 8.7028%，高于既有 Content 8.1282%。
- 详细结果见 `results/phase_07/experiment_01_feature_hybrid_two_tower/summary.md`。
