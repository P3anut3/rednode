# Phase 7 Experiment 04：H2 检索维度消融

## 目标

在 Phase 7-01 的 H0/H1/H2 结构、数据、特征、loss 和负采样完全不变时，只把
最终 retrieval space 从 128d 扩到 256d，判断近两倍索引与点积成本是否值得。

128d H0/H1/H2 直接读取 Experiment 01 的正式 validation JSON 和 per-request
ranking，不重训。新实验只训练 `h0_256`、`h1_256`、`h2_256`。

## 最终结果

- Validation R@500：H0/H1/H2-256 = 9.4083%/9.8353%/9.8098%；相对对应
  128d 分别 +0.8672pp/+1.0038pp/+0.2992pp；
- H2-256 三 seed = 9.8098%/9.9023%/9.9521%，均值9.8881%、std 0.0590pp；
- H2 overall paired-bootstrap 95% CI = `[+0.0325,+0.5736]pp`；
- 唯一 terminal test 锁定中位 seed43，R@500 = 9.0231%，相对 H2-128
  terminal 8.7028% 提升 +0.3203pp；
- 256d item vectors为1.892 GiB，full validation exact search约1.60s，峰值显存
  约4.22 GiB；全部预注册 gate 通过，结论为 **GO**。

## 256d 结构契约

- Frozen BGE 768→256；
- metadata/profile 输出256，但 categorical embedding仍为16，numeric hidden仍为32；
- User/Item ID embedding固定128，经无 bias adapter映射到256；
- OOV row0 经 adapter仍严格为零；
- history attention和 residual MLP工作在256d；
- alpha均从0.05开始；不加入 dropout、新 loss、新 HN 或新特征。

## 协议

- temporal train 254,583，history N=20；
- Phase 5 TF-IDF HN + in-batch InfoNCE，temperature 0.05；
- batch 512，AdamW 1e-4，最多6 epoch，patience2；
- 固定5,000 proxy requests和100,000 proxy candidates；
- 正式 validation为1,983,938全库 GPU-resident exact IP；
- H0/H1仅seed42，H2只有seed42不低于H2-128超过0.1pp时才补43/44。

## 执行阶段

```text
static-audit → smoke → train → serial full-validation
→ bootstrap → select → conditional H2 seed43/44 → lock → report
```

所有执行阶段需要 `--confirm-run`。Test只有全部validation gate通过才允许打开；否则
保持未读取。

## Go / No-Go

H2-256必须同时满足：三seed mean至少+0.20pp、overall bootstrap下界>0、
completely-unseen下降不超过0.10pp、target-seen无显著下降、资源不超预算、三seed
方差不明显高于H2-128。Seed42若低于H2-128超过0.1pp，直接No-Go且不补多seed。

代码：`experiments/phase_07/experiment_04_h2_retrieval_dimension_ablation/`

结果：`results/phase_07/experiment_04_h2_retrieval_dimension_ablation/`
