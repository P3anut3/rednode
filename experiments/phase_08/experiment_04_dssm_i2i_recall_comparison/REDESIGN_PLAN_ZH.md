# Phase 8-05 重做方案：作者图文 DSSM 结构与双路召回

状态：方案；尚未实现或启动。旧 Phase8-04 的模型结果已撤回、产物已删除。本方案应交给 Codex 在新实验目录实现，先停在代码 review 门口。

## 研究目标与来源

比较作者实际图文 concat DSSM 结构与当前 H2-256，并验证 DSSM 用户塔召回 + 逐历史 I2I 在固定 Top500 下的增量。复现应以源码为准，不能将 Phase7 的 AuthorConcatTower 改名后直接替代。

锁定作者仓库 commit：`39b7767372e46dfa5cd20b689d1c31a6d608ea69`。

- [DSSM、历史构造、训练](https://github.com/WBoxian/Rednote-Qilin-Search-Rec-System/blob/39b7767372e46dfa5cd20b689d1c31a6d608ea69/src/recall/dssm_trainer.py)
- [逐历史近邻与多路融合](https://github.com/WBoxian/Rednote-Qilin-Search-Rec-System/blob/39b7767372e46dfa5cd20b689d1c31a6d608ea69/src/recall/build_multiroute_recall.py)

新代码与结果镜像目录：

`experiments/phase_08/experiment_05_author_multimodal_dssm_recall/`

`results/phase_08/experiment_05_author_multimodal_dssm_recall/`

## 第一道门：源码与输入审计

生成 source_parity.md，逐项记录原作者模型、历史处理、词表、归一化、loss、optimizer 与本地适配。逐字段记录来源、维度、缺失率、请求前可用性，不能静默用其他特征替换。

本文要求复现原作者结构，但保留我们的 temporal split：train 254,583 positives；validation 47,733 positives / 13,594 requests；全库 1,983,938 notes；最近20条历史；相同 history exclusion 与 request-macro 指标。作者原代码的随机 session split、切分前统计归一化、采样 AUC 不作为主评估协议。

复用已提取 BGE768 和 SigLIP mean_all768，冻结缓存；不得重新编码图片。必须注明图片 encoder/pooling 与作者资产可能不同，本轮只能称“作者结构 + 本地冻结特征 + temporal 协议”，不能宣称作者完整原实验复现。图片缺失使用零向量，mask 用于内部读取与正确性检查，不增加新的 seen/is_available 输入列。

作者 user 原始类别字段为 gender/platform/age/location；物品字段为 type/tax1/tax2/tax3。location 必须从原始数据核查，不能因为 Phase7 cache 没有而省略。数值字段包含用户计数、dense_feat1~40、物品静态特征与累计统计。请求前时间无法核实的字段在主对照对应输入槽置零、明确列出原因；若可从 temporal-train 日志按截止时间重建则使用重建值。40维 dense_feat、累计统计等来源不能核实前不得进入主结果。零槽只保持结构维度，不能称这些特征已被使用。最终先公布 feature manifest，再 review 决定是否额外做快照特征研究。

## 第二道门：按原作者实现塔结构

Item tower：

`concat(item_ID32, type4, tax1/2/3各16, 原始数值向量D, BGE768, SigLIP768)`

`1620+D → Linear512 → ReLU → Linear128 → L2 normalize`。

不得提前把元数据转换成128d，不得提前把文本/图片压成128d，不增加 LayerNorm、GELU 或融合残差。没有显式 seen 标志。

User tower：

`concat(user_ID32, gender4, platform4, age8, location16, fans/follows2, dense40, 多模态历史摘要768, 序列表示128)`

`1002 → Linear512 → ReLU → Linear128 → L2 normalize`。

历史内容先按原作者 `0.72*text + 0.28*image` 形成768d序列。768d摘要保留源码的 recency/recent/last/mean/max 组合；序列另一分支为 `Linear768→128 → BiGRU(hidden64×2) → masked attention128`。User history 不调用 trainable item tower，不能用此前128d item attention替代这一分支。禁止查询读取当前 target。

保存原作者默认初始化协议；小初始化只可作为后续单独变量。需要修复的 OOV、全空历史输出、非法索引处理等必须在 parity 文档注明，不将陌生ID clamp成任意已知用户/物品。

在相同合成输入、相同权重和 eval 模式下，与锁定源码的纯模型类验证 item/user前向数值等价。原始字段填零仍可验证结构一致性。对 padding 和空历史做边界测试；若改用 packed GRU，明确这改变了原作者 padding行为，不能在结构基线中静默替换。

## 词表协议与核心消融

作者物品ID表覆盖已有全库目录，和此前 train history∪target vocab 不同。目录ID只来自已有 canonical corpus，不使用 validation/test标签、频次或行为；属于已知目录的 transductive 设置，不等于未来新发布物品已经有训练信息。用户ID词表和类别词表/归一化统计仍 train-only。

先 seed42 跑以下四组，除注明因素外共享结构、loss、batch 与优化器：

| 模型 | 图文输入 | Item ID词表 | structured ID dropout |
|---|---|---|---|
| S0 | 文本+图片 | 作者式已知全库目录 | 0 |
| S1 | 文本；图片槽及历史图片置零 | 同S0 | 0 |
| S2 | 同S0 | train history∪target；外部统一零OOV | 0 |
| S3 | 同S2 | 同S2 | 0.5 |

S0−S1回答图片作用；S0−S2回答ID目录协议作用；S3−S2回答对训练ID缺失的适应能力。S0是结构参考，不应在加入ID dropout、小初始化后再声称完全沿用原作者训练。

S3仅随机整体屏蔽用户/物品ID向量，不改变内容、类别、数值字段；每个batch中的同一candidate编码共享，采样mask不能读取正负标签。验证与全库编码关闭dropout。新模型不增加显式seen输入。

## 训练、早停与捷径诊断

首轮使用作者当前实际默认路径：in-batch InfoNCE、temperature0.07、AdamW lr1e-3/weight_decay1e-2、batch768、最多3epoch。原脚本实际启用in-batch，不能仅凭顶部旧注释改成triplet或额外easy negatives。先采用easy模式，无外部hard negatives；作者的hard pool来自其他阶段，无法直接等同Phase5 TF-IDF负例。本轮不引入teacher或ranking挖掘。

修正 duplicate target/same-request positive/history positive 的负例mask并披露差异；validation checkpoint选型使用相同固定proxy requests/candidates，正式质量使用全库Recall，AUC只作训练诊断。可记录原作者loss/AUC最大checkpoint，不能用AUC替代全库结果。

每epoch记录train loss、正负cosine、梯度/更新范数、训练target与负例的ID可用性；正式检索记录Top500中train-target-seen/history-only/completely-unseen占比、各组Recall、ID-off诊断。若出现所有cold结果被排斥，先审计content/ID/输入尺度与loss，不直接宣称架构失败。必要时只加一组相同作者结构采用H2训练目标的对照，单独归因训练协议；不随意追加网格搜索。

## 召回与固定预算对照

对最佳DSSM checkpoint生成完整128d物品向量，用相同checkpoint查询向量做GPU exact检索，产出D Top500。

DI使用同一D item vectors，对最近20条历史逐条取近邻，按作者recency decay加权；不能称DI是独立纯内容encoder。C使用冻结BGE768构造真正纯内容I2I。仅复现作者双路的D+DI；Swing、UserCF、序列转移路不混入本轮。

报告 H2、最佳D、DI、C、D+DI、D+C、H2+C。每路历史过滤、候选全集和Top500预算一致。I2I默认180近邻；不足500时采用预声明overfetch补位，空历史报告覆盖率并由融合主路补齐。融合只选少量450/50、400/100、350/150配额；若次路明显更强，也允许对调主次路并披露。可以固定RRF k=60作一种预注册对照。报告新增命中、挤掉命中、union oracle和paired bootstrap，不能用Top1000 union代替Top500。

先完成seed42结构筛选，再对胜出结构补43/44，读取相同seed H2已有rankings比较。自训练DSSM单路与H2的差异包含输出维度/结构/训练协议，不能把所有增益称作concat的因果效应。全文只使用validation；新test执行仍需锁定配置后的用户指令。

## 性能与执行顺序

源码/字段audit → 前向parity与self-check → 单卡10k训练smoke+固定检索smoke → S0/S1/S2/S3 seed42 → 串行full-validation → 胜出模型三seed → 固定预算融合 → 中文报告 → 停止。

num_workers默认0，只读mmap，GPU分批编码/exact检索，不复制多份全库图文数组。unique历史item批量KNN只算一次并缓存；设置deadline、进程树RSS、显存和磁盘监控。各checkpoint完整向量只编码一次；保存source/model/mapping/vector hash及断点marker。需要全库ID表时提前预算embedding和Adam状态，不照搬原脚本大规模内存DataLoader。

先实现并交付review，不在代码对齐和字段清单通过前启动正式训练。报告明确分开“原作者结构已对齐”“特征资产差异”“数据协议差异”和“ID目录设定”；不能再将简化模型结果当作作者模型效果。
