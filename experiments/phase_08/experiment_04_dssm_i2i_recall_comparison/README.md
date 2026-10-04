# Phase 8 / Experiment 04：已撤回的简化 DSSM 与 I2I 对照

本版本已按用户要求撤回，结果目录已删除。旧代码仅保留用于审计实现差异，不应继续执行其训练/评估命令，也不能将其作为原作者 DSSM 的复现结果。

## 撤回原因

D 直接复用 Phase7-03 的 AuthorConcatTower，缺少作者实际 DSSM 的图片输入、多模态 768d 历史摘要、BiGRU 序列分支及直接拼接原始类别 embedding 的结构，并额外输入 seen 标志。损失和训练超参数也不同。它不能回答原作者图文 DSSM 加 I2I 是否优于 H2。

训练正样本都有训练 ID，而大部分 TF-IDF 负例没有，且未使用 ID dropout。Top500 对有 ID 物品极端集中，提示 ID 可用性捷径；这是机制线索，未通过因果消融确认。不能把原因归结为已证明的检索代码错误，也不能认为补上图片就必然解决问题。

## 后续方案

详见 [重做方案](REDESIGN_PLAN_ZH.md)。新实验使用独立 Experiment 05 目录；Phase7 H2 与 Phase8 图片资产继续复用。保留 temporal validation 和全库评估口径；本轮只形成方案，尚未实现或启动新实验。

## 结果目录

原镜像目录 `results/phase_08/experiment_04_dssm_i2i_recall_comparison/` 已删除，包括 debug、checkpoint、embedding、KNN、ranking、marker 和 summary。没有保留备份；旧指标从汇总文档撤回。
