# Changelog

## 1.1.0

一次完整的方法学与工程化升级，`analysis.py` 重写为可参数化的命令行工具。

### 新增

- 命令行参数：`--input` / `--output` / `--train-ratio` / `--min-samples` / `--cohen-d` / `--bootstrap` / `--min-group` / `--seed` / `--result-lower-bound`，均有默认值，不改即可运行。
- 训练/验证集按**测定时间顺序**切分（前 80% 训练、后 20% 验证），替代原先的随机切分。
- 验证集覆盖率评估：输出 `Reference_Intervals_Table1_with_Percentage.xlsx` 与 `Coverage_OK_90` 标记。
- 五法异常值对比表（以 Tukey 为基准的 BR 值）：`Reference_Interval_Comparison_Table3.xlsx`。
- 区间限值的 95% 抽样区间四列（`Lower/Upper 95% CI Low-High`）。
- 年龄分层的 Cohen's d 记录：`Auto_Age_Grouping_Results.xlsx`。
- 森林图 X 轴按实际覆盖率自适应，避免低覆盖率点被裁掉。
- 大样本散点图按性别各抽样 8000 点绘制（图例标注 `subsampled`），定量结果仍基于全量数据。

### 变更

- 区间算法明确为 **Box-Cox 变换（数据含 ≤0 时平移）→ 五法之一剔除异常值 → bootstrap 分位数 → 逆变换**，不再对外描述为"非参数百分位法"。
- 年龄分组算法对齐 Wu et al. *Clin Biochem* 2026;146:111200 §2.4：单岁起始、最小样本量与 Cohen's d 双阈值迭代合并。
- 依赖更正：移除 `pygam`，改用 `statsmodels` 分位数回归（`quantreg`）+ `patsy` B 样条。
- 数据契约明确：必需 `结果`/`测定时间`/`性别`/`年龄` 四列，支持中英文列名回退，缺列直接报错而非静默跳过。
- 图表输出改为定名文件（不再落在 `results_<时间戳>/` 下随机命名），便于流水线引用。

### 修复

- 修复年龄列被误存为日期序列号时的解析问题（按 1899-12-30 反推）。
- 修复小样本组（n≈20）下机器学习类方法标点过少导致区间失真的问题。

## 1.0.0

初版：数据加载、预处理、五法异常值检测、性别/年龄分层、参考区间计算与三类可视化。
