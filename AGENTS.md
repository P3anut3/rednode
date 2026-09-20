# Repository Working Rules

## 实验与生产代码边界

- 项目未明确决定上线的代码都属于实验代码，必须放在 `experiments/`。
- `src/` 只能放经过明确上线决策、接口稳定化和必要测试后的生产代码。不得为了便于 import 而把实验模块放入 `src/`。
- 复用的实验数据加载、评估和检索工具放在 `experiments/common/`，仍视为实验代码。

## 实验目录命名

- 实验代码统一使用：

  ```text
  experiments/phase_XX/experiment_YY_<short_name>/
  ```

- `XX` 是两位阶段编号，`YY` 是该阶段内两位实验编号，`short_name` 使用小写 snake_case。
- 每个新实验必须新建 `experiment_YY_<short_name>` 目录；不得在 `experiments/` 根目录新增散落的编号脚本，也不得覆盖另一实验。
- 每个实验目录至少包含 `run.py` 和 `README.md`。README 用中文记录验证目标、主要控制变量、运行方式、结果目录和最终结论。

## 实验结果目录

- 结果目录必须与实验代码目录一一镜像：

  ```text
  experiments/phase_XX/experiment_YY_<short_name>/
  results/phase_XX/experiment_YY_<short_name>/
  ```

- 所有 report、CSV、JSON、per-request 结果、checkpoint、embedding、index 和临时 cache 都必须写入对应实验的 result 目录，不得在 `results/` 根目录创建散落目录。
- smoke 或 debug 产物放在对应实验结果目录的 `smoke/` 或 `debug/`，不把 smoke 命名为独立根级实验。
- 大体积结果不纳入 Git。整理目录时优先在同一文件系统直接移动，不复制、不重新编码、不重新训练。
- 移动实验或结果后，必须同步修改代码中的读写路径、跨阶段依赖、README 和汇总文档。所有路径使用项目相对路径。

## 实验汇总与验证

- 每完成或重新组织一个实验，同步更新 `experiments/SUMMARY_ZH.md`，用中文记录实验目标、关键数字、结论和产物路径。
- 完成路径调整后，至少执行 Python 语法检查、各 `run.py --help` 导入检查（如支持）和 `git diff --check`。
- 不得为验证目录调整而默认重跑全量训练、全库 embedding 或昂贵检索。
