# Phase 8-05：作者图文 DSSM 与 ID Dropout 消融

状态：**正式validation实验完成并停止。14组训练/全库validation/诊断、三seed多路评估及报告完成；test未读取。**

## 最终结论

三seed平均R@500：F参考A0/A1=4.6205%/4.3071%，安全B0/B1=2.5840%/2.3582%；安全D选择B0，明显低于H2-256的9.8881%。ID dropout0.5在F/S两组都降低总体及completely-unseen Recall。T0/T1仅seed42，图片在无dropout时有局部收益，不能推广为跨seed稳定结论。

DI=2.5088%，BGE-I2I C=5.5671%；D+DI RRF=2.7468%，D+C RRF=4.7253%，均弱于H2。H2+C 450/50平均9.9310%，相对H2仅+0.0429pp，95%CI[−0.0422,+0.1266]pp；扩大C配额和固定RRF均退化。存在独有命中，但尚无足够固定预算收益证据。**不替换H2，暂不新增正式I2I路，不读取test。**

2026-10-02重新验证14组training/validation/diagnostics与3组routes的marker及文件hash、报告46项依赖通过；AMP数值修复记录保留。中文结论见结果目录`analysis_summary_zh.md`，完整逐seed指标见`summary.md`。

这是新实验，Phase8-04 已撤回的 checkpoint、指标和缓存都不复用。源码锁定 `39b7767372e46dfa5cd20b689d1c31a6d608ea69`，模型直接复用经 hash 校验的作者纯模型类，详见 [source_parity.md](source_parity.md)。不读取 test，不自动替换 H2。

## 目标与矩阵

在作者真实图文 DSSM、相同初始化/数据顺序/训练协议下，只改变 F/S 字段协议、图片和 ID dropout。Item1659→512→128，User1002→512→128；只有 Linear/ReLU，没有见过标志、metadata encoder或融合残差。历史来自冻结图文768序列，摘要768与 BiGRU128，不调用 trainable item tower。

| 模型 | 字段协议 | 图片 | 整路 ID dropout |
|---|---|---|---:|
| A0 | F作者快照字段参考 | mean_all | 0 |
| A1 | F作者快照字段参考 | mean_all | .5 |
| B0 | S保守数值安全协议 | mean_all | 0 |
| B1 | S保守数值安全协议 | mean_all | .5 |
| T0 | S | 图片槽/历史图片置0 | 0 |
| T1 | S | 图片槽/历史图片置0 | .5 |

F动态快照时间口径未验证，不作为无泄漏/上线证据。S对未知动态数值置零，不用训练末尾计数冒充训练请求前计数；类别和冻结内容仍有明确静态快照假设。所有槽位及顺序由 config.py 固定，实际启用/缺失/置零在 audit 的 feature_manifest.json 披露。

全库 canonical ID表（含OOV0）；user vocab train-only；类别vocab和norm只用temporal-train。目录内但行为未见的item仍有原默认ID初始化，不暗中关闭其ID。独立 dropout RNG、不rescale、共享candidate只编码一次、eval关闭dropout。历史内容分支没有ID输入。

## 固定协议

- train254583、valid47733 positives /13594 requests，N20，候选1983938。
- BGE768/SigLIP mean_all768只读复用；输出128。对照H2-256同seed validation；9.8881%为三seed validation参考，不引用9.0231% terminal test选型。
- 作者easy in-batch CE、不加HN；temperature .07、AdamW1e-3/wd1e-2、batch768、最多3epoch、patience2、固定proxy5000/100000。
- 为真正控制变量，先保留作者 unmasked easy loss；每epoch审计 duplicate/same-request/history-positive/same-user false negatives，不只修改dropout组。
- 正式只有GPU resident FP32 exact，无CPU fallback。workers0，显存/RSS/磁盘/deadline保护，串行full catalog。完成marker最后写，hash绑定source、feature、checkpoint、vector、mapping和ranking。

## 代码审查重点

`source_contract.py` AST加载、`models.py` OOV/全空历史/dropout适配；`data.py` F/S字段与train-only fitting；`trainer.py` 原easy loss、unique target共享mask；`run.py` 正常完成与hash门禁；`routes.py` 逐历史KNN、固定预算与oracle区别。

`self_check.py` 只用合成数据。真实10k smoke 六组全部通过，同 seed 初始化 hash 一致；进程峰值 RSS 3.6–5.0 GiB，GPU分配峰值1.46 GiB，整路ID dropout实际比例约49%。当前正式训练/全库验证队列串行运行，日志位于对应结果目录 `logs/`。

## 运行方式

所有路径相对项目根；以下用已知环境python。plan不读大数据、不写文件：

```bash
/home/cmj/.conda/envs/qilin-onerec/bin/python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage plan
/home/cmj/.conda/envs/qilin-onerec/bin/python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/self_check.py
```

审查批准后，先audit，再**逐组单卡smoke**，每组10000 train samples/1000 validation requests/100000候选，1epoch、600秒deadline。数据/产物阶段必须显式 `--confirm-run`：

```bash
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage audit --confirm-run
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage smoke --model B0 --device cuda:0 --confirm-run
```

把model依次改为A0/A1/B0/B1/T0/T1，不把smoke当正式指标。六组smoke通过、核对F/S manifest后，再对各组seed42 train→validate→diagnostics：

```bash
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage train --model B0 --seed 42 --device cuda:0 --confirm-run
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage validate --model B0 --seed 42 --device cuda:0 --confirm-run
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage diagnostics --model B0 --seed 42 --device cuda:0 --confirm-run
```

A0/A1/B0/B1每组再补43/44。B0/B1按三seed均值选择；T0/T1先seed42筛选。任何T的seed42达到或超过同seed最佳B，select只写`selection_pending.json`并阻断，不生成最终D；该T须显式`--confirm-image-seeds`补齐43/44的train→validate→diagnostics，再按三seed均值参加最终选型。非胜出的T若因图片归因补齐三seed，也可按均值参与。F最佳参考A同样按三seed均值。

select要求已纳入的训练/validation/诊断正常完成、source与feature一致、初始化与proxy hash一致；单seed T不能成为最终D。最终`selection.json`冻结后不再允许新增训练；不允许凭一个已有best.pt提前选型。

```bash
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage select --confirm-run
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage routes --device cuda:0 --confirm-run
python experiments/phase_08/experiment_05_author_multimodal_dssm_recall/run.py --stage report --confirm-run
```

`routes --seed 42`是单seed多路筛选，不能宣称稳定收益。结构及同名quota/RRF配置的稳定性需要补对应`routes --seed 43`和`--seed 44`，分别写入`routes/seed42|seed43|seed44/`。DI的KNN按模型/seed隔离，C的冻结BGE KNN三seed共享。report只对三seed共同存在的同一融合配置汇总request-averaged bootstrap、每seed差值和std；不将各seed各自最优quota拼成“一个配置”。

正式DI/C保留180邻居聚合得到的原有排序：已有500候选的请求不改动；不足500时按深层结果追加尚未出现的候选。`DI/C-author180`是原口径、`DI/C`是保序补位正式口径、`DI/C-deep-rerank`是完整深度重排诊断。三者分别报告质量与覆盖率，重排消融不参加正式路线选优。

没有terminal-test stage。validation/routes未完成时允许显式 `--resume`；已完成产物拒绝覆盖。KNN按unique历史物品、512查询/chunk缓存并hash校验，避免逐request重复全库扫描。训练中断保存interrupted.pt并拒绝作为正常完成checkpoint；暂不提供改变随机序列的隐式重训/自动resume，需审查后新run。

## 诊断与解释范围

### AMP 数值恢复（2026-10-01）

A1 seed43第2个batch的loss/前向有限，但默认loss scale65536使`user_mlp.2.bias`梯度溢出；同输入较低scale与FP32均正常。原代码在GradScaler自动backoff前直接抛错。现保留默认scale与所有模型/loss/采样设置，让GradScaler拒绝该更新并降scale，随后重放**同一batch、同一ID dropout mask**，不跳样本、不改变mask统计；最多8次重试，前向/loss非有限或重试仍失败立即停止。

合成GradScaler检查证明溢出时参数/optimizer state不更新；真实故障前3batch复现只需1次backoff，scale32768，2304个用户样本计数无重复。每epoch保存overflow事件/最终scale。已完成旧产物不重写marker：通过`debug/numerical_recovery/runtime_patch.json`绑定修复前后精确源码hash，允许的变动仅数值运行/门禁/自检/报告；模型、数据、loss、映射及评估源码保持不变。旧源码和失败训练目录已归档；A1 seed43只完成1次更新，无正常marker，按同seed从头重跑该组，其他已完成组不重训。

复现与修复记录：结果目录`debug/numerical_recovery/`。这项修复属于数值执行保护，不是新的结构/超参数消融；报告需保留该披露。

固定1000 validation requests上的Item-ID-off/User-ID-off/All-ID-off是OOD机制线索，不参加选型，不是正式模型成绩。记录 train正负ID可用性、正负cosine、输入尺度、gradient norms、三类物品在Top500占比、ID实际drop比例。双路固定450/50、400/100、350/150，必要时交换主次，RRF固定60；报告新增/挤掉、重叠/独有命中、request union oracle与paired CI。

三seed差值按request平均后bootstrap；另外报告每seed差值和std。CI反映request波动，不表示已经充分刻画随机初始化不确定性。T seed42小差异只作初筛，不强行图片因果结论。报告自动汇总同seed H2、warm/cold、有图/无图、ID-off、输入尺度/分支梯度、false-negative、训练wall/encoding/index/search/RSS/GPU/cache成本。overall CI为正不自动GO，completely-unseen或无图退化须单独披露；未预注册非劣效margin时，不宣称cold非劣效已证明。

## 结果路径与停止

镜像：`results/phase_08/experiment_05_author_multimodal_dssm_recall/`，audit/feature缓存、smoke、training每epochcheckpoint、validation全库vectors与rankings、diagnostics、ablation_bootstrap、selection、knn、routes、summary均在其中。正式文件只在阶段被授权后创建；当前已完成 audit，后续以各阶段 completion marker 为准。

最终回答F/S dropout、图片、未验证字段影响、ID捷径、D/DI/C及固定预算收益，记录时间/内存/index/cache大小。完成validation研究后停止，不读test、不替换H2、不进入粗排精排。
