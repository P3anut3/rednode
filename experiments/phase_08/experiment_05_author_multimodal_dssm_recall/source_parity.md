# 作者源码对齐契约（Phase 8-05，待 review）

来源固定为 [WBoxian/Rednote-Qilin-Search-Rec-System](https://github.com/WBoxian/Rednote-Qilin-Search-Rec-System/tree/39b7767372e46dfa5cd20b689d1c31a6d608ea69)，commit `39b7767372e46dfa5cd20b689d1c31a6d608ea69`。

`source/` 保存五个未经修改的原始文件，`source_manifest.json` 固定 SHA-256。原脚本不是实验入口：`source_contract.py` 只提取模型和纯函数 AST，避免执行原脚本的目录创建、数据读取和模型下载。实际塔直接继承锁定源码的 `DSSMModel`，不是 Phase7 AuthorConcatTower，也不依赖已撤回的 Phase8-04 模型/指标。

## 已核对的源码

| 原始文件 | 核对范围 |
|---|---|
| `src/recall/dssm_trainer.py` | DSSMModel、数据/collate、历史公式、字段列表、easy/hard 执行路径、优化器、初始化、session split、归一化、AUC 选型和导出 |
| `src/recall/build_multiroute_recall.py` | 历史截取、逐历史近邻、180 深度、反向遍历和 decay、融合及其他通道 |
| `src/preprocess/build_features.py` | 真实字段、类别映射、比率分母、原脚本的字段 fallback、SQL join、时间列 |
| `src/preprocess/build_note_text_emb.py` | BGE 模型、title/content 格式、CLS/normalize、512 token、float16 |
| `src/preprocess/build_note_image_emb.py` | SigLIP、实际图片路径、缺图零值、多图均值、原脚本 mean 后未再次 normalize |

## 塔结构：保留原作者计算路径

物品：`ID32 + type4 + tax1/2/3 各16 + numeric39 + text768 + image768`，输入 **1659d**，`Linear(1659,512) → ReLU → Linear(512,128) → normalize`。

用户：`ID32 + gender4 + platform4 + age8 + location16 + fans/follows2 + dense40 + summary768 + sequence128`，输入 **1002d**，`Linear(1002,512) → ReLU → Linear(512,128) → normalize`。

不增加 seen 标志、metadata encoder、LayerNorm、GELU、shortcut 或新 attention。本文的 structured dropout 只改变原有 ID 向量，不增加输入槽。

### 历史内容

- 冻结内容序列：`0.72 × BGE + 0.28 × SigLIP`；缺图填零。T0/T1 的所有图片输入填零，不把文本权重改成1，因此维度、公式和参数完全一致。
- 最近20条，保留原训练代码 `ids[-20:]` 与原有序列顺序，最后一条作为最近行为。沿用已冻结的 Phase4/5 history，不从 target 构造序列。
- 摘要：`normalize(.34*recency + .26*recent5 + .18*last + .14*mean + .08*max)`；recency 权重由 `.35..1.0` 线性递增后归一化。摘要始终768d。
- 序列：`Linear768→128 + BiGRU(hidden64×2) + Linear128→1 + masked softmax attention`。历史不调用 trainable item tower。
- **明确适配**：全空历史序列输出零，而不是原实现全 mask softmax 平均出的 GRU bias；有有效历史时保留作者 padded BiGRU 的原计算，不改成 packed GRU。未知目录外历史内容为零且不参与有效 mask。
- 数据无逐条历史时间戳，不能重新证明 history 事件先后；继承旧协议的请求前历史语义，audit 记录来源、重叠与每个 request 的一致性。训练 history-target overlap 如非零，直接阻断，待用户检查。

### ID、类别与初始化

| 项 | 作者源码 | 本地适配 |
|---|---|---|
| Item ID | 原始整数索引，表大小覆盖 full_note_ids；easy in-batch 下并不会因为构建了 full negative pool 就更新所有目录 ID | canonical 全库每条占一个 row，row0 保留 OOV；目录只来自 canonical，不读 validation/test 行为建表 |
| User ID | `train max(user_idx)+1`；越界被 clamp | train-only 紧凑 vocab；未知用户 row0 |
| 类别 | search/rec train 共同构建映射，可能晚于本地 temporal cutoff | 只用 temporal-train 用户及历史∪target 物品类别；不引入 search |
| 非法索引 | `_safe_index` 裁剪到合法边界，会映射到已知 ID | 越界统一映射 OOV0，不 clamp 到已知 ID |
| 默认初始化 | PyTorch Embedding/Linear/GRU 默认，无 small init | 已知 row 完全保留默认初始化、相同创建顺序；只将 user/item OOV row0 置零且梯度为零 |
| Note type | `note_type-1` 再 clamp，Emb(3,4) | 保留1/2转换；异常 raw type 在数据层记录并映射0，经模型 OOV 规则处理 |

所有组同 seed 构造后 state_dict 完全一致；ID dropout 使用独立 CPU `torch.Generator`，不改变 torch 全局初始化 RNG 或 numpy shuffle。每个 batch 的 unique target 只编码一次，重复 item 使用同一 mask；用户同一 ID 也共享当次 mask。整向量 Bernoulli 清零，不乘 `1/(1-p)`，不看标签；eval/offline catalog 不 dropout。作者历史内容没有 ID，**没有 history ID dropout**。

这是“已有目录、行为稀疏”的 transductive 设置，不等于未来发布的未知物品已学到 ID。未获行为监督的目录 ID 保留默认随机向量，是本轮必须诊断的风险，不能暗中将它们设为0或加入 seen 输入。

## 特征与时间协议

本地 parquet schema 已只读核对：notes 有 note_type、taxonomy1/2/3、六个静态数值以及作者要求的累计计数；user_feat 有 location、fans/follows、dense_feat1..40。正式缺失率、norm 和 vocab 仍需用户 review 后执行 audit，不在实现阶段宣称已审计通过。

作者物品39数值 = static6 + counts20 + rates13。`config.py` 精确固定顺序，不能使用不相关字段替换。rates 使用 `build_features.py` 的同名分子/分母推导，分母0时为0。原脚本某些缺列会使用其他累计列 fallback；本实验**不沿用这种替代**，缺槽填0并在 manifest 明示。

- **F**：原作者字段的本地快照参考。用户计数/dense 与 item 累计统计可以输入，但时间口径未验证；只用 temporal-train fitting rows 的均值/样本 std，counts/static/fans/follows 先 log1p，rates/dense 不 log。负数计数裁剪0、nonfinite 清零，是明确数值稳定适配。
- **S**：保持全部39/42槽。启用静态 video_duration/height/width、image_num、content_length、commercial_flag。用户 fans/follows/dense40、物品累计 counts/rates 全部置0。不把“temporal-train 结束的统计值”提供给更早训练请求；当前没有带历史时间的合法计数缓存，所以不假装重建了请求前统计。
- 画像/类目保留，按静态快照类别假设解释，但 effective date 不明；S **不宣称已经严格验证所有类别和内容快照的历史可用时间**。S 的保证主要是未引入时间未验证的动态数值及未来行为统计。
- 固定资产 BGE basic-clean + taxonomy 与作者的 `title：...\ncontent：...` 并不相同；本地 SigLIP mean_all 用全部有效图、FP32 mean 后再次normalize，作者 mean 后没有再次normalize，且图片异常处理不同。两套 F/S 共用本地资产，不重新提取。

因此 F 标为 **“作者原结构/原字段的本地适配参考”**，不能宣称复现作者公开指标；S 也必须披露上述静态快照假设。

## 实际训练路径与受控差异

| 项 | 锁定源码实际执行 | 本实验 |
|---|---|---|
| Loss | main 两个 loader 均 `use_inbatch_neg=True`，easy 没有 real hard，CE diagonal；顶部 triplet 注释过时 | 同 unmasked in-batch CE，temperature .07；不加 Phase5 HN、不悄悄 mask 修正 |
| False negatives | duplicate target、同 request、历史正例、同用户正例均未屏蔽 | 六组一致保留原 objective，同时每 epoch 统计四类 overlap。不能将 loss 下降等同内容学习 |
| Optimizer | AdamW lr1e-3 / wd1e-2；AMP | 相同；不更改初始化、batch768或负采样 |
| Scheduler | ReduceLROnPlateau min(val_loss), patience2, factor.5 | 保留，使用固定 proxy 对应的 positive interactions 计算 easy val_loss |
| Epoch | 3，上限；early stop2 | 同上限3/patience2，但 best 按固定 proxy request-macro R500，而非 AUC |
| Split | 随机 session split20%；归一化在切分前估计 | 固定旧 temporal 254583 train /47733 valid positives；13594 validation requests；norm 只在 temporal-train 拟合 |
| AUC | 采样或 B² in-batch AUC用于选型 | 有界、固定位置的正/负 cosine AUC仅辅助，不是作者全量 AUC复现；正式用 full-corpus Recall |

所有组使用相同 proxy5000 requests、100000 candidates（包含该 proxy 全部正例），不能跨 epoch 改 proxy。训练每 epoch 保存模型权重，不重复 best 体积；中断 checkpoint 单独命名，不写正常完成 marker。

## 双路策略对齐与差异

DI 重用 D 的128d item vectors、同索引；每条历史单独检索，作者 `_recall_ann_item_history` 的 `sum(sim/(1+idx))` 保留。报告180深度原口径的不足Top500比例，并固定 overfetch 为 `600+max(original_history_unique_count)` 保障非空历史的合法候选；不依据 label 调 KNN 深度。C 使用冻结 BGE768，采用完全相同聚合。

正式DI/C只在180口径不足500时追加深层候选，保留原180排序及完整前缀；已经有500条的请求绝不重排。完整深度聚合重排另列`deep-rerank`消融，质量与覆盖率单列，不参与正式route/fusion选优。

安全D按三seed均值选择。T seed42若达到同seed最佳B，则必须补43/44后才能进入最终均值比较。多路seed42仅筛选，稳定性由对应43/44同名配置验证；C冻结KNN跨seed共享，DI绑定D模型/seed/vector hash。

作者多路脚本外层 `_resolve_hist_items` 使用 `[:20]`，内部还融合 Swing、UserCF、sequence transition 等，且 min-max/配额/权重策略复杂。本实验按已冻结最近20条协议统一为 `[-20:]`，不引入这些其他通道。**D+DI 是作者两条 DSSM 路径的受控对比，不是作者整个 multiroute pipeline 的完整复现。**

主/次分数不直接跨模型相加。固定450/50、400/100、350/150预算、去重和主路补齐；RRF固定k60单独标记。空历史I2I输出空，不凭空加入target/用户未来行为或监督补齐。union oracle单列，绝不当固定Top500成绩。

H2用同seed已有 validation ranking/JSON，维度256；作者模型128。这一横向对比同时改变架构、训练协议和维度，不给出“仅concat导致差异”的因果解释。

## 当前可验证证据

合成同权重 eval：作者与本地 item/user 前向最大差均 **0.0**；历史摘要对齐容差1e-6。默认初始化六组相同、整路/无 rescale/shared-candidate dropout、OOV和空历史、所有塔/GRU/类别/ID分支梯度检查通过。

这些是代码级证据，不是10k数据 smoke、feature audit或正式质量证据。本轮停止在 review 门口，**未读取 test、未运行真实数据 smoke或训练/validation**。
