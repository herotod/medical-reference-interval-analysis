---
name: medical-reference-interval-analysis
display_name: 医学参考区间分析
display_name_en: Medical Reference Interval Analysis
description: 临床检验真实世界数据（RWD）间接法参考区间（RI）构建与验证。用于：建立参考区间、用验证集评估参考区间（覆盖率是否 ≥90%）、判定性别/年龄是否需要分组、剔除异常值后估算 2.5th/97.5th 限值、生成 KDE/分位数回归/森林图。触发场景包括 PT/APTT/ALT/AFP/CRP/iCa 等检验项目的参考区间项目、WS/T 402 与 CLSI C28-A3c 间接法、五法异常值检测对比、Harris-Boyd 性别分组、年龄分组、bootstrap 分位数。
description_zh: 临床检验真实世界数据（RWD）间接法参考区间（RI）构建与验证。建立参考区间、用验证集评估覆盖率（≥90%）、判定性别/年龄是否分组、五法剔除异常值后估算 2.5th/97.5th 限值，并生成 KDE、分位数回归与森林图。适用于 WS/T 402 与 CLSI C28-A3c 间接法项目。
description_en: An agent skill that builds and validates indirect reference intervals from real-world laboratory data (five outlier algorithms, Harris-Boyd, age partitioning, bootstrap percentiles, forest plot).
version: 1.1.0
usage:
  - "python analysis.py --input <数据目录> --output <输出目录>"
  - "默认读取当前目录下所有 .xlsx，结果写入 ./results_<时间戳>/"
---

# 医学参考区间分析（间接法）

## 何时用

用户给出一个或多个 `.xlsx` 检验数据（每项目一个文件、含结果/测定时间/性别/年龄），要求：

- 建立某检验项目的参考区间（下限/上限）；
- 判断男/女、年龄是否需要各自建立区间；
- 用独立验证集检查覆盖率（CLSI C28-A3c / WS/T 402 惯例：≥90% 的参考个体落在区间内）；
- 比较多种异常值剔除算法对区间的影响（BR 值）；
- 出 KDE 分布图、年龄–结果分位数回归图、覆盖率森林图。

不适合：仅需单个百分位数点估计（直接用 numpy.percentile 即可）、室间质评/正确度验证、方法学比对（EP09）。

## 运行

```bash
python analysis.py --input "<数据目录>" --output "<输出目录>"
```

- 依赖：pandas、numpy、scipy、statsmodels、patsy、scikit-learn、matplotlib、seaborn、openpyxl（写出 xlsx 必需）。建议 Python ≥3.9。
- 无图形界面时设 `MPLBACKEND=Agg`，否则 `plt.show()` 可能阻塞。
- 参数（都有默认值，不改即可）：`--train-ratio 0.8`、`--min-samples 120`、`--cohen-d 0.2`、`--bootstrap 1000`、`--min-group 20`、`--seed 42`、`--result-lower-bound`（默认剔除结果 ≤0；血气 BE/AG/渗量间隙等可取 0 或负值的指标必须显式指定下限）。

## 数据契约

每个文件一个指标，项目名取文件名（`AFP.xlsx` → `AFP`）。必需四列，中文优先、英文回退：

| 列 | 说明 |
|---|---|
| `结果` / `result` | 数值型检验结果；默认剔除 ≤0 与无法解析的值 |
| `测定时间` / `date` | 日期或 Excel 序列号（按 1899-12-30 起算） |
| `性别` / `gender` | `男/male/m`、`女/female/f`，其它按缺失处理并剔除 |
| `年龄` / `age` | 岁；若整列最大值 >120 视为误存成日期序列号，按 1899-12-30 反推年龄；缺失年龄按缺失剔除 |

缺列时抛 `ValueError` 并列出实际列名，不会静默跳过。**注意表头是繁体（如 `測定時間`）时会被判定为缺列**——中文列名必须精确匹配简体字。

## 输出（写在 `--output` 目录）

| 文件 | 内容 |
|---|---|
| `dataset_split_statistics.xlsx` | 训练/验证拆分统计（样本量、性别构成、年龄、日期范围） |
| `Auto_Age_Grouping_Results.xlsx` | 年龄分层结果（每段 N、Mean/Median/SD/Min/Max、与前一段的 Cohen's d） |
| `Reference_Intervals_Table2.xlsx` | 五种方法的区间限值 + 限值的 95% 抽样区间（Project/Gender/Age Group/Method/Lower/Lower 95% CI Low-High/Upper/Upper 95% CI Low-High/N） |
| `Reference_Interval_Comparison_Table3.xlsx` | 以 Tukey 为基准的 BR（\|BR\| ≥ 0.375 记为显著差异） |
| `Reference_Intervals_Table1_with_Percentage.xlsx` | 验证集覆盖率 + `Coverage_OK_90` 标记 |
| `Combined_AFP_KDE_plot_with_HarrisBoyd.png` | 各指标性别 KDE + Harris-Boyd / M-W / Cohen's d / 偏度注释 |
| `Combined_AFP_GAM_plot.png` | 年龄–结果散点 + B 样条分位数回归（50th、2.5/97.5th） |
| `SCI_quality_combined_methods_reference_interval_forest_plot.png` | 五法 × 各分组的覆盖率森林图（90% 红色参考线） |
| `train/`、`val/` | 拆分后的数据（保留原文件名） |

## 方法学口径（写论文时必须照这个口径描述）

1. 切分：按 `date` **时间顺序**前 80% 训练、后 20% 验证（非随机）。日期列必须是真正解析成时间戳的，否则切分失去时间外验证意义。
2. 区间算法：**Box-Cox 变换（数据含 ≤0 时先平移）→ 五法之一剔除异常值 → bootstrap 1000 次取 2.5th/97.5th 分位数的中位数 → 逆变换回原始尺度**。这不是纯非参数分位数法，方法学段落不能写成"non-parametric percentile method"。
3. bootstrap 给两样东西：限值的点估计（bootstrap 分位数分布的中位数），以及该点估计的 95% 抽样区间（`Lower/Upper 95% CI Low-High` 四列）。论文写法为"下限(95% CI)"，不要把点估计当成置信区间，也不要把 CI 当成区间限值本身。
4. 异常值五法：Tukey(1.5 IQR)、Z-score(|z|>3)、GMM(2 成分，score 1% 分位)、Isolation Forest(contamination=0.01)、LOF(contamination=0.01)。样本量小时（n≈20）四种机器学习法实际只能标 1 个点，与 Tukey 差别很小，别在文中夸大"五法差异"。
5. 性别：Harris-Boyd 的 Z 与 Z*（Z*=3√((n1+n2)/240)）只作为**判断依据的参考**；脚本仍会对男、女各算一套区间。若 |Z| ≤ Z*，论文里要说明"按 CLSI 要求男女分别建立，HB 检验未提示差异"，不要写成"HB 检验证明需要分组"。
6. 年龄（**必须与 Wu et al. Clin Biochem 2026;146:111200 §2.4 的算法一致**）：以**单岁**为初始组，仅用训练集、男女分别做；每轮顺序遍历相邻组对，任一组 n < `--min-samples`（默认 120）则无条件合并，两组均达标则算 Cohen's d、|d| ≤ `--cohen-d`（默认 0.2）则合并；合并后就地与下一组继续比较；一整趟无合并且收敛或达 10 轮停止。输出即"每组 n ≥120、相邻组 |d| >0.2"的离散年龄段，无需预设候选切点。
   - `--min-samples` 不要为了"少出几段"而调大（例如 200000）：那会让算法退化成整段合并，等于放弃论文的核心方法，报告中也不能再写"数据驱动分层"。
   - 单岁样本量普遍达不到 120 的队列（如只有一两千例、跨 80 岁），结果会合并成一段——这属于方法本身的样本量限制，报告里写"按最小样本量与效应量合并，未触发年龄分割"，不能写成"验证了年龄无需分组"。
7. 覆盖率：用 `Reference_Intervals_Table1_with_Percentage.xlsx` 的 `Percentage` 与 `Coverage_OK_90` 判定；<90% 的组会在运行日志里逐条列出，必须回到样本量/分组合理性上找原因。

## 已知限制（结论里要交代）

- 区间下限可能低于生物学零界（逆变换后出现极小正值），必要时在报告里截断并说明。
- 分组依据（性别/HB、年龄/Cohen's d）与最终输出区间的对应关系需要人工复核，脚本不做自动"合并男女"。
- 图使用 Arial 字体，中文标题会缺字；指标名保持英文或拉丁缩写。
- 森林图 X 轴按实际覆盖率自适应（下限 = 最小覆盖率 − 1.5，且至少留 3 个百分点），不再固定 85–100，以免低覆盖率的点被裁掉。
- 出图规模：森林图图幅随面板数与标签行数自适应、单边像素 ≤5800（约 300 dpi 可排版）；散点图在样本 >20000 时按性别各抽样 8000 点绘制（图例标注 `(subsampled)`），**定量结果仍基于全量数据**，只有散点做了抽样。

## 维护

- `analysis.py` 是唯一实现，改动后必须用合成数据跑通全流程（8 个产物齐全）再交付。
- 版本变更记录见 `CHANGELOG.md`；配套论文作者与单位信息见 `README.md`。
- 复现性：`--seed` 统一控制 GMM / Isolation Forest / bootstrap 的随机性，同参数同数据结果可复现。
