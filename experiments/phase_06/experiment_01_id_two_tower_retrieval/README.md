# Phase 6：ID 协同召回与 Content-ID Residual

## 验证目标

在严格复用 Phase 4/5 temporal split 的前提下，验证 Pure-ID 是否值得保留为独立召回路，以及仅对 temporal-train target-seen item 添加 ID residual 能否提升 warm recall 并保护 cold recall。

## 核心控制变量

- Train/validation 固定为 254,583 / 47,733 个 positive samples。
- Pure-ID 词表为 temporal-train history 与 positive target 的并集；候选仅为 temporal-train positive targets。
- Content-ID 使用冻结 BGE-base-zh 内容向量，BGE encoder 不训练。
- 全部结构、负采样与融合选择只看 validation；test 仅由带落盘闸门的 terminal stage 读取一次。
- 不使用图片、用户画像、Phase 5 hard negatives，也不进入粗排/精排。

## 运行方式

```bash
python experiments/phase_06/experiment_01_id_two_tower_retrieval/run.py --stage audit
python experiments/phase_06/experiment_01_id_two_tower_retrieval/run.py --stage smoke --device cuda
python experiments/phase_06/experiment_01_id_two_tower_retrieval/run.py --stage pure-id --device cuda
```

后续阶段由 `run.py --help` 列出的显式 stage 执行。所有结果只写入镜像目录：

`results/phase_06/experiment_01_id_two_tower_retrieval/`

## Test 闸门

除 `terminal-test` 外的代码路径不会调用 test loader。终测开始前写入 manifest；完成后标记 `complete`，默认拒绝重复运行。
