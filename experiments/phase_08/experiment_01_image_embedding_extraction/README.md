# Phase 8 / Experiment 01：冻结图片 Embedding 提取

## 验证目标

本实验只建设可复用的图片特征基础设施：基于本地缓存的 `google/siglip-base-patch16-224`，将 Qilin `notes.image_path` 中的全部图片编码为冻结的逐图片 embedding，并派生 `first_image`、`mean_top3`、`mean_all` 三种确定性 item pooling。

本实验不训练推荐模型、不接入 H2、不读取 recommendation test 标签、不下载或切换视觉模型。正式图片编码只允许 CUDA。

## 数据契约

- canonical item 顺序复用 Phase 7 配置引用的冻结 baseline `note_ids.npy`；Phase 8 的 mapping 是指向该文件的相对符号链接，并校验 SHA-256。
- `image_path` 是唯一可信图片列表；不使用 `image_num` 建映射。
- CSR 映射由 `note_ids.npy + image_offsets.npy` 表示，manifest 保留原始图片顺序。
- 每 100,000 张图片一个 manifest/embedding shard；逐图 embedding 为 L2-normalized float16 `[N, 768]`。
- 解码失败写零向量和 `valid=False`，错误记录不会静默丢弃。
- 完成 shard 是 append-only；marker、manifest、embedding、valid mask 必须通过 hash/shape/dtype 校验才能断点跳过。

## 运行顺序

默认仅打印计划，不扫描全量 notes、图片或模型：

```bash
/home/cmj/.conda/envs/qilin-onerec/bin/python \
  experiments/phase_08/experiment_01_image_embedding_extraction/run.py
```

本次恢复验证已完成 audit、GPU smoke 和单卡 shard canary；正式全量提取仍未完成。默认采用保守配置 `batch_size=64`、`num_workers=0`、`prefetch_factor=1`。10k 对照显示每卡 2 workers 可提高吞吐，因此单卡 shard canary 使用 2 workers；每次扩展并发都应重新观察 RSS、host available、swap 和吞吐。

运行前先执行合成检查和保守 GPU smoke：

```bash
PY=/home/cmj/.conda/envs/qilin-onerec/bin/python
EXP=experiments/phase_08/experiment_01_image_embedding_extraction

$PY $EXP/self_check.py
$PY $EXP/run.py --stage audit --confirm-run
$PY $EXP/run.py --stage smoke --device cuda:0 --batch-size 64 --num-workers 0 --prefetch-factor 1 --smoke-autocast-only --confirm-run

# 可选：在同一个 10k 样本上比较 1、2 个 DataLoader worker
$PY $EXP/run.py --stage smoke --device cuda:0 --batch-size 64 --num-workers 1 --prefetch-factor 1 --smoke-autocast-only --confirm-run
$PY $EXP/run.py --stage smoke --device cuda:0 --batch-size 64 --num-workers 2 --prefetch-factor 1 --smoke-autocast-only --confirm-run

# 首先单卡、单 shard canary；max-shards 防止误启动全量任务
CUDA_VISIBLE_DEVICES=0 $PY $EXP/run.py --stage extract --worker-rank 0 --world-size 1 --max-shards 1 --device cuda:0 --batch-size 64 --num-workers 2 --prefetch-factor 1 --min-host-available-gb 32 --max-process-tree-rss-gb 16 --recover-stale-locks --confirm-run

# 单卡 canary 稳定后，逐步增加并发。每个 worker 在独立会话中启动；
# 每个 worker 使用 batch=64、num_workers=2、prefetch=1 和资源上限。
# 先验证双卡，再评估是否扩到四卡；吞吐增益不足时维持较低并发。
# 只在全部 50 个 shard 都有完整且 hash 校验通过的 marker 后，才能运行：
# $PY $EXP/run.py --stage finalize-extraction --confirm-run
# $PY $EXP/run.py --stage pool-first --confirm-run
# $PY $EXP/run.py --stage pool-top3 --confirm-run
# $PY $EXP/run.py --stage pool-all --confirm-run
# $PY $EXP/run.py --stage validate --confirm-run
# $PY $EXP/run.py --stage report --confirm-run
```

`CUDA_VISIBLE_DEVICES` 隔离后，每个独立 worker 都使用其进程内的 `cuda:0`；shard 分配规则是 `shard_id % world_size == worker_rank`。

## 结果目录

所有结果写入：

```text
results/phase_08/experiment_01_image_embedding_extraction/
```

目录只在对应阶段实际产生产物时创建，不预建无意义空目录。大体积 embedding、manifest、mask 和临时文件不应提交 Git。

## 断点与安全

- Audit 先写入 `audit_building/`，完整验证且 `unsafe_path_count=0` 后才发布正式 mapping；中断后只有显式 `--recover-audit` 可清理重建。
- `audit_complete.json` 绑定 manifest metadata 与固定 10k smoke sample；smoke 结果再绑定 audit marker、sample、模型 fingerprint、推理协议和最终 batch size。正式 worker 会重新加载模型并与 smoke fingerprint 比较。
- 已完成 shard 在复用时同样必须匹配当前 smoke fingerprint/protocol；finalize 和 extraction global marker 还会绑定整个 `smoke_result.json`，因此重跑 smoke 后旧 shard 不会被静默接受。
- `extract` 不写全局 complete marker；只有 `finalize-extraction` 验证全部 shard 后才写。
- marker 不存在时重做该 shard；marker 存在但文件/hash 不匹配时停止。
- 每个 marker 同时绑定 embedding、valid mask、manifest、decode-failure 表和模型 fingerprint；pooling 前会重新校验完整逐图资产链。
- 默认拒绝覆盖已完成正式 shard；显式 `--overwrite-debug` 才允许恢复性重生成，新资产完整写完后才发布，旧资产以 hard-link 方式保存在 `diagnostics/overwritten_shards/`。
- stale lock 只能使用 `--recover-stale-locks` 显式恢复，并且必须是同一 hostname 且原 PID 已不存在。
- 临时 `.tmp` 文件不被视为正式产物；完成后原子 rename。
- `SIGTERM` 会让当前未完成 shard 保持非正式状态，已完成 shard不受影响。
- unsafe path 会阻塞 smoke/extract。
- Smoke 在相同 10,000 图片上比较 FP32 与 CUDA float16 autocast，记录吞吐、显存、有效图片一致性和 embedding cosine；正式提取使用 autocast，输出与 L2 normalize 转回 FP32。
- DataLoader worker 内完成图像预处理；collate 返回行号、格式、错误元数据和 pixel tensor，不会把原始 PIL 图片送回主进程。正常异常退出时显式关闭 DataLoader worker。
- smoke 和 shard 逐批记录进程树 RSS、主机可用内存、swap、子进程数；默认主机可用内存安全线 32 GiB、单 worker 进程树 RSS 上限 32 GiB，可通过 CLI 调整。
- 每种 pooling 先完整写入 `<strategy>_building/`，校验并写 completion marker 后才原子发布目录；中断产物只有显式 `--recover-pooling` 可以清理。已完成 pooling 保持不可覆盖。

## 当前结论

当前进度（2026-09-24）：audit 已完成；保守 GPU smoke 的 10k 全部成功；单卡 shard 0 的 100k canary 已完成并通过 marker/hash 校验。双卡 canary 各处理约 30k 后按运行时长要求中止，没有写完成 marker；其临时 shard 文件仍保留且不会被当作正式产物。全量图片提取、finalize、pooling 和 validate 尚未完成，recommendation test 未读取。
