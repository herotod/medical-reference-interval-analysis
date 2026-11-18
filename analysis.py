# -*- coding: utf-8 -*-
"""
医学参考区间分析系统
包含数据集分割、性别差异分析、年龄分组、参考区间计算和验证功能
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, skew, norm, gaussian_kde
from scipy import stats
from statsmodels.formula.api import quantreg
from patsy import bs
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
import math
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 配置参数（默认值见 CONFIG，可由命令行覆盖，见文件末尾 argparse）
# =============================================================================

CONFIG = {
    "train_ratio": 0.8,            # 训练集比例（按测定时间顺序切分，前 80% 训练、后 20% 验证）
    "min_samples": 120,            # 年龄分组：单个年龄组的最小样本量
    "cohen_d_threshold": 0.2,      # 年龄分组：相邻年龄组 Cohen's d 合并阈值
    "bootstrap_iterations": 1000,  # 参考区间 bootstrap 次数
    "min_group_for_ri": 20,        # 单个性别-年龄组计算参考区间的最低样本量
    "random_seed": 42,             # 随机种子（GMM / IsolationForest / bootstrap 复现）
    "result_lower_bound": None,    # 结果值下限过滤；None 表示沿用 result > 0 的默认口径
}

# 运行期路径（由 configure() 赋值）
source_path = "."
output_base_path = None
train_path = None
val_path = None


def configure(input_dir=".", output_dir=None, **overrides):
    """设置输入/输出目录与分析参数；目录不存在时自动创建。

    脚本读写用的是模块级全局变量，因此这里必须显式声明 global 后再赋值，
    否则函数内的赋值只会产生局部变量，main() 仍会读到旧路径。
    """
    global source_path, output_base_path, train_path, val_path
    source_path = os.path.abspath(input_dir)
    if output_dir:
        output_base_path = os.path.abspath(output_dir)
    else:
        output_base_path = os.path.join(
            os.getcwd(), "results_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    train_path = os.path.join(output_base_path, "train")
    val_path = os.path.join(output_base_path, "val")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in CONFIG:
            raise KeyError(f"未知配置项: {key}")
        CONFIG[key] = value
    return output_base_path

# =============================================================================
# 2. 数据预处理和分割函数
# =============================================================================

def excel_date_to_age(excel_date):
    """将Excel日期格式转换为年龄"""
    try:
        # Excel日期基准是1899-12-30
        base_date = datetime(1899, 12, 30)
        date_value = base_date + timedelta(days=excel_date)
        current_date = datetime.now()
        age = current_date.year - date_value.year
        if (current_date.month, current_date.day) < (date_value.month, date_value.day):
            age -= 1
        return age
    except:
        # 如果不是Excel日期格式，直接返回原值
        return excel_date

def preprocess_file(file_path):
    """预处理单个文件的数据"""
    try:
        # 读取文件
        df = pd.read_excel(file_path)
        
        # 获取项目名称（从文件名获取；用 splitext 而不是 split('.')，避免文件名自带点号时错位）
        filename = os.path.basename(file_path)
        project_name = os.path.splitext(filename)[0]
        
        # 预处理 - 使用中英文列名（中文优先，缺失时回退英文）
        result_col = '结果' if '结果' in df.columns else ('result' if 'result' in df.columns else None)
        date_col = '测定时间' if '测定时间' in df.columns else ('date' if 'date' in df.columns else None)
        gender_col = '性别' if '性别' in df.columns else ('gender' if 'gender' in df.columns else None)
        age_col = '年龄' if '年龄' in df.columns else ('age' if 'age' in df.columns else None)
        
        missing_cols = [name for name, col in [
            ('结果/result', result_col), ('测定时间/date', date_col),
            ('性别/gender', gender_col), ('年龄/age', age_col)
        ] if col is None]
        if missing_cols:
            raise ValueError(f"文件缺少必需列: {', '.join(missing_cols)}；实际列名为 {list(df.columns)}")
        
        # 创建新的DataFrame
        new_df = df[[result_col, date_col, gender_col, age_col]].copy()
        new_df.columns = ['result', 'date', 'gender', 'age']
        
        # 处理日期列（Excel 序列号按 1899-12-30 起算，避免被误解析为 1970 年附近）
        if pd.api.types.is_numeric_dtype(new_df['date']) and new_df['date'].notna().any():
            new_df['date'] = pd.Timestamp('1899-12-30') + pd.to_timedelta(new_df['date'], unit='D')
        else:
            new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        
        # 处理性别列
        def convert_gender(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x).strip().lower()
            if x_str in ['男', 'male', 'm']:
                return 'male'
            elif x_str in ['女', 'female', 'f']:
                return 'female'
            else:
                return np.nan
        
        new_df['gender'] = new_df['gender'].apply(convert_gender)
        new_df.dropna(subset=['gender'], inplace=True)
        
        # 处理年龄列
        new_df['age'] = pd.to_numeric(new_df['age'], errors='coerce')
        
        # 检查年龄是否为Excel日期格式（年龄大于120的可能是日期）
        if new_df['age'].max() > 120:
            new_df['age'] = new_df['age'].apply(excel_date_to_age)
        
        new_df.dropna(subset=['age'], inplace=True)
        
        # 处理结果列
        new_df['result'] = pd.to_numeric(new_df['result'], errors='coerce')
        lower_bound = CONFIG.get('result_lower_bound')
        if lower_bound is None:
            # 默认口径：剔除 <=0 的结果（适用于浓度类指标；若指标本身可为 0 或负值，请用 --result-lower-bound 显式指定）
            new_df = new_df[new_df['result'] > 0]
        else:
            new_df = new_df[new_df['result'] >= lower_bound]
        new_df.dropna(subset=['result'], inplace=True)
        
        # 按日期升序排序
        new_df = new_df.sort_values('date').reset_index(drop=True)
        
        return {
            'project_name': project_name,
            'filename': filename,
            'data': new_df,
            'male_data': new_df[new_df['gender'] == 'male']['result'].dropna().values,
            'female_data': new_df[new_df['gender'] == 'female']['result'].dropna().values,
            'total_data': new_df['result'].dropna().values
        }
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def split_dataset_by_ratio(data_dict, train_ratio=0.8):
    """按4:1比例分割数据集（训练集占80%）"""
    data = data_dict['data'].copy()
    total_samples = len(data)
    
    if total_samples < 5:
        # 如果样本太少，全部作为训练集
        train_data = data
        val_data = pd.DataFrame(columns=data.columns)
    else:
        # 计算分割点
        split_idx = int(total_samples * train_ratio)
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
    
    return train_data, val_data

# =============================================================================
# 3. 性别差异分析函数
# =============================================================================

def harris_boyd_test(male_data, female_data):
    """应用Harris-Boyd方法判断两个亚组是否需要划分"""
    n1 = len(male_data)
    n2 = len(female_data)
    
    # 检查样本量是否足够
    if n1 < 30 or n2 < 30:
        return np.nan, np.nan, "样本量不足(n<30)", False
    
    # 计算均值
    mean1 = np.mean(male_data)
    mean2 = np.mean(female_data)
    
    # 计算标准差
    std1 = np.std(male_data, ddof=1)
    std2 = np.std(female_data, ddof=1)
    
    # 计算Z值
    z_value = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
    
    # 计算Z*临界值
    z_star = 3 * np.sqrt((n1 + n2) / 240)
    
    # 判断是否需要划分
    if abs(z_value) > z_star:
        recommendation = "|Z| > Z*"
        significant = True
    else:
        recommendation = "|Z| ≤ Z*"
        significant = False
    
    return z_value, z_star, recommendation, significant

def calculate_cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # 计算合并标准差
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # 计算Cohen's d
    cohen_d = (mean1 - mean2) / pooled_std
    return cohen_d

def calculate_mann_whitney_u(group1, group2):
    """计算Mann-Whitney U检验的P值"""
    n1, n2 = len(group1), len(group2)
    if n1 < 3 or n2 < 3:
        return np.nan, "Insufficient data"
    
    try:
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        if p_value < 0.001:
            significance = "*** (P < 0.001)"
        elif p_value < 0.01:
            significance = "** (P < 0.01)"
        elif p_value < 0.05:
            significance = "* (P < 0.05)"
        else:
            significance = "Not significant (P ≥ 0.05)"
        
        return p_value, significance
    except Exception as e:
        print(f"Error in Mann-Whitney U test: {e}")
        return np.nan, "Test failed"

def calculate_skewness_with_interpretation(data):
    """计算偏度并提供解释"""
    if len(data) < 3:
        return np.nan, "Insufficient data"
    
    skewness_value = skew(data)
    
    if abs(skewness_value) < 0.5:
        interpretation = "Approx. symmetric"
    elif 0.5 <= skewness_value < 1.0:
        interpretation = "Moderately right-skewed"
    elif skewness_value >= 1.0:
        interpretation = "Highly right-skewed"
    elif -1.0 < skewness_value <= -0.5:
        interpretation = "Moderately left-skewed"
    else:
        interpretation = "Highly left-skewed"
    
    return skewness_value, interpretation

def fit_gam_model(data, gender):
    """为特定性别拟合GAM模型"""
    gender_data = data[data['gender'] == gender].copy()
    
    if len(gender_data) < 10:
        return None, None, None, None
    
    gender_data = gender_data.sort_values('age')
    
    try:
        formula_median = f"result ~ bs(age, df=4)"
        model_median = quantreg(formula_median, gender_data).fit(q=0.5)
        model_lower = quantreg(formula_median, gender_data).fit(q=0.025)
        model_upper = quantreg(formula_median, gender_data).fit(q=0.975)
        
        age_range = np.linspace(gender_data['age'].min(), gender_data['age'].max(), 100)
        pred_data = pd.DataFrame({'age': age_range})
        
        pred_median = model_median.predict(pred_data)
        pred_lower = model_lower.predict(pred_data)
        pred_upper = model_upper.predict(pred_data)
        
        return age_range, pred_median, pred_lower, pred_upper
        
    except Exception as e:
        print(f"Error fitting GAM model: {e}")
        return None, None, None, None

# =============================================================================
# 4. 年龄分组函数
# =============================================================================

def calculate_cohens_d_age(group1, group2):
    """计算Cohen's d效应量（用于年龄分组）"""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohen_d = (mean1 - mean2) / pooled_std
    return cohen_d

def auto_age_grouping_improved(data, min_samples=120, cohen_d_threshold=0.2, max_iterations=10):
    """自动年龄分层（与 Wu et al. Clin Biochem 2026, §2.4 一致）

    算法（论文原文口径，逐字对应）：
      1. 以**单岁**为初始组，仅用训练集，男女分别进行；
      2. 每轮顺序遍历所有相邻组对 (i, i+1)：
         - 任一组 n < min_samples → 无条件合并（样本量准则）；
         - 两组均 n ≥ min_samples → 计算 Cohen's d，|d| ≤ cohen_d_threshold 则合并（效应量准则）；
      3. **合并后就地继续**：新合成的组立即与下一组比较（论文的"顺序遍历"是
         单趟凝聚，不是每轮只配一轮对，否则收敛要慢好几轮、还容易留下
         由配对顺序导致的碎段）；
      4. 一整趟遍历无任何合并（收敛）或达到 max_iterations 轮即停止；
      5. 输出满足"每组 n ≥ min_samples、相邻组 |d| > 阈值"的离散年龄段。

    论文默认参数：min_samples = 120、cohen_d_threshold = 0.2、max_iterations = 10。
    注意：本算法从单岁起点做相邻合并，因此只有当单岁样本量能达到 min_samples
    时才会真正形成年龄段；样本被摊到几十个单岁上去时（例如 1-89 岁、1000 例），
    几乎所有单岁组都低于阈值，最终会合并成一整段——这是方法本身的限制，报告
    里必须如实写"按最小样本量与效应量合并，未触发年龄分割"。
    """
    if len(data) == 0 or 'age' not in data.columns:
        return []

    # 年龄取整：先四舍五入再转整型（astype(int) 是截断，会把 17.9 岁算成 17 岁）；
    # NaN 年龄先剔除，否则 astype(int) 会抛 IntCastingNaNError 并中断整条流程。
    age_value = pd.to_numeric(data['age'], errors='coerce')
    data = data.assign(age_int=age_value.round())
    data = data[data['age_int'].notna()].copy()
    data['age_int'] = data['age_int'].astype(int)

    if data.empty:
        return []

    min_age = int(data['age_int'].min())
    max_age = int(data['age_int'].max())

    # 创建初始的单岁年龄组
    groups = []
    for low in range(min_age, max_age + 1):
        group_df = data[data['age_int'] == low]
        groups.append({
            'range': (low, low),
            'data': group_df['result'].values,
            'size': len(group_df)
        })

    iteration = 0
    work = groups          # 就地凝聚的工作副本

    while iteration < max_iterations:
        changed = False

        # 单趟顺序遍历：合并后就地继续比较，不跳过下一组
        i = 0
        while i < len(work) - 1:
            cur, nxt = work[i], work[i + 1]

            if cur['size'] < min_samples or nxt['size'] < min_samples:
                merge = True      # 样本量准则：无条件合并
            else:
                cohen_d = calculate_cohens_d_age(cur['data'], nxt['data'])
                merge = (not np.isnan(cohen_d)) and abs(cohen_d) <= cohen_d_threshold

            if merge:
                merged_data = np.concatenate([cur['data'], nxt['data']])
                work[i] = {
                    'range': (cur['range'][0], nxt['range'][1]),
                    'data': merged_data,
                    'size': len(merged_data)
                }
                del work[i + 1]
                changed = True
            else:
                i += 1

        # 末尾残组：不足 min_samples 时向前并，避免留下不可靠的一小段
        while len(work) > 1 and work[-1]['size'] < min_samples:
            last = work.pop()
            prev = work[-1]
            merged_data = np.concatenate([prev['data'], last['data']])
            work[-1] = {
                'range': (prev['range'][0], last['range'][1]),
                'data': merged_data,
                'size': len(merged_data)
            }
            changed = True

        iteration += 1
        if not changed:
            break

    final_ranges = [g['range'] for g in work]
    return final_ranges

# =============================================================================
# 5. 参考区间计算函数
# =============================================================================

def detect_outliers_tukey(data):
    """Tukey方法检测异常值"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_gmm(data):
    """GMM方法检测异常值"""
    if len(data) < 10:
        return np.zeros(len(data), dtype=bool)
    data_reshaped = data.reshape(-1, 1)
    # 固定随机种子：GMM 默认随机初始化，不设种子时同一份数据两次运行会给出不同区间，
    # 论文结果无法复现。
    gmm = GaussianMixture(n_components=2, random_state=CONFIG.get('random_seed', 42))
    gmm.fit(data_reshaped)
    scores = gmm.score_samples(data_reshaped)
    threshold = np.percentile(scores, 1)
    return scores < threshold

def detect_outliers_zscore(data):
    """Z-score方法检测异常值"""
    z_scores = np.abs(stats.zscore(data))
    return z_scores > 3

def detect_outliers_isolation_forest(data):
    """Isolation Forest方法检测异常值"""
    data_reshaped = data.reshape(-1, 1)
    clf = IsolationForest(contamination=0.01, random_state=42)
    preds = clf.fit_predict(data_reshaped)
    return preds == -1

def detect_outliers_lof(data):
    """Local Outlier Factor方法检测异常值"""
    data_reshaped = data.reshape(-1, 1)
    lof = LocalOutlierFactor(contamination=0.01)
    preds = lof.fit_predict(data_reshaped)
    return preds == -1

def boxcox_transform(data):
    """Box-Cox变换使数据近似正态分布"""
    min_val = np.min(data)
    if min_val <= 0:
        shift = 1 - min_val
        data = data + shift
    else:
        shift = 0
    
    transformed, lmbda = stats.boxcox(data)
    return transformed, lmbda, shift

def inverse_boxcox(transformed, lmbda, shift):
    """Box-Cox逆变换"""
    # λ 用容差判断：MLE 优化得到的 λ 极少恰好等于 0，而是 ±1e-9 量级的数，
    # 此时 (λx+1)**(1/λ) 会数值溢出，得到 inf 或 nan。
    if abs(lmbda) < 1e-6:
        original = np.exp(transformed)
    else:
        original = (lmbda * transformed + 1) ** (1 / lmbda)
    
    return original - shift

def bootstrap_reference_interval(data, n_bootstrap=1000, alpha=0.05, seed=None):
    """自举法估计 2.5th/97.5th 分位数及其 95% 置信区间

    返回 6 元组：(下限, 上限, 下限 CI 低, 下限 CI 高, 上限 CI 低, 上限 CI 高)。
    前两个是 bootstrap 重复抽样下分位数的中位数（限值的稳健点估计）；后四个是
    该分位数估计的 95% 抽样区间（对应论文 Table 3 的 "(95% CI)" 括号列）。
    两者不能混为一谈：论文正文把前者当限值、把后者当限值的置信区间报告。
    """
    n = len(data)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed if seed is not None else CONFIG.get('random_seed', 42))
    lower_percentile = alpha * 100 / 2
    upper_percentile = (1 - alpha / 2) * 100

    lower_bounds = []
    upper_bounds = []

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        lower_bounds.append(np.percentile(sample, lower_percentile))
        upper_bounds.append(np.percentile(sample, upper_percentile))

    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    lower_bound = np.median(lower_bounds)
    upper_bound = np.median(upper_bounds)

    ci = alpha * 100 / 2, (1 - alpha / 2) * 100
    lower_ci = np.percentile(lower_bounds, ci)
    upper_ci = np.percentile(upper_bounds, ci)

    return lower_bound, upper_bound, lower_ci[0], lower_ci[1], upper_ci[0], upper_ci[1]

def calculate_brs(ref1_lower, ref1_upper, ref2_lower, ref2_upper):
    """计算BR值（Bias Ratio）"""
    if ref1_lower == ref1_upper:
        return np.nan, np.nan
    
    SDri = (ref1_upper - ref1_lower) / 3.92
    if SDri == 0:
        return np.nan, np.nan
    
    br_lower = abs(ref2_lower - ref1_lower) / SDri
    br_upper = abs(ref2_upper - ref1_upper) / SDri
    return br_lower, br_upper

# =============================================================================
# 6. 内部验证函数
# =============================================================================

def check_age_in_range(age_str, age):
    """检查年龄是否在指定范围内"""
    if pd.isna(age) or age is None:
        return False
    
    try:
        if '-' in age_str:
            min_age, max_age = age_str.split('-')
            min_age = int(min_age.strip())
            max_age = int(max_age.strip())
            return min_age <= age <= max_age
        else:
            return False
    except:
        return False

def build_project_file_map(folder):
    """建立 project 名 -> 实际文件路径 的映射。

    参考区间表中的 Project 来自文件名（去掉扩展名），而拆分出的验证文件名
    保留原文件名。文件名自带点号（如 ALT.2024.xlsx）时，f"{project}.xlsx"
    会拼错路径并静默跳过整组验证，因此这里按目录实际文件名建映射。
    """
    file_map = {}
    for path in glob.glob(os.path.join(folder, "*.xlsx")):
        file_map[os.path.splitext(os.path.basename(path))[0]] = path
    return file_map

def load_validation_frame(file_path, cache=None):
    """读取并预处理验证集文件；cache 用于避免同一文件被反复读上千次。

    原实现对参考区间表的每一行都重新 read_excel 一次（行数 = 分组 × 5 种方法，
    大型项目可达数千行），几十个指标的队列会因此耗时数分钟到数十分钟。
    """
    if cache is not None and file_path in cache:
        return cache[file_path]
    
    val_df = pd.read_excel(file_path)
    required_columns = ['result', 'gender', 'age']
    if not all(col in val_df.columns for col in required_columns):
        raise ValueError(f"验证文件缺少必要列 {required_columns}: {file_path}")
    
    # 统一性别取值（验证集可能来自本方拆分（male/female），也可能是外部原始数据（男/女）
    val_df['gender'] = val_df['gender'].apply(
        lambda x: np.nan if pd.isna(x) else str(x).strip().lower())
    val_df['gender'] = val_df['gender'].replace({'男': 'male', 'm': 'male', '女': 'female', 'f': 'female'})
    
    val_df['result'] = pd.to_numeric(val_df['result'], errors='coerce')
    val_df['age_numeric'] = pd.to_numeric(val_df['age'], errors='coerce')
    if val_df['age_numeric'].max() > 120:
        val_df['age_value'] = val_df['age_numeric'].apply(excel_date_to_age)
    else:
        val_df['age_value'] = val_df['age_numeric']
    
    if cache is not None:
        cache[file_path] = val_df
    return val_df

def calculate_percentage_for_row(row, val_data_folder, cache=None, file_map=None):
    """为参考区间表中的每一行计算百分比（验证集覆盖率）"""
    project = row['Project']
    gender = row['Gender']
    age_group = row['Age Group']
    lower_limit = row['Lower Limit']
    upper_limit = row['Upper Limit']
    
    # 构建对应的验证数据文件路径
    if file_map and project in file_map:
        file_path = file_map[project]
    else:
        file_path = os.path.join(val_data_folder, f"{project}.xlsx")
    
    if not os.path.exists(file_path):
        print(f"验证文件不存在: {file_path}")
        return None
    
    try:
        val_df = load_validation_frame(file_path, cache)
        
        gender_key = str(gender).strip().lower()
        if gender_key in ('male', 'm', '男'):
            gender_key = 'male'
        elif gender_key in ('female', 'f', '女'):
            gender_key = 'female'
        else:
            print(f"未知性别: {gender}")
            return None
        
        filtered_df = val_df[val_df['gender'] == gender_key]
        if len(filtered_df) == 0:
            print(f"没有找到匹配性别的数据: {project}, {gender}")
            return 0.0
        
        age_filter = filtered_df['age_value'].apply(lambda x: check_age_in_range(age_group, x))
        filtered_df = filtered_df[age_filter]
        if len(filtered_df) == 0:
            print(f"没有找到匹配年龄范围的数据: {project}, {gender}, {age_group}")
            return 0.0
        
        result_in_range = (filtered_df['result'] >= lower_limit) & (filtered_df['result'] <= upper_limit)
        percentage = (result_in_range.sum() / len(filtered_df)) * 100
        return round(percentage, 2)
    
    except Exception as e:
        print(f"处理验证文件时出错 {file_path}: {str(e)}")
        return None

# =============================================================================
# 7. 主分析流程
# =============================================================================

def main():
    global output_base_path, train_path, val_path
    # 直接 import 后调用 main()（未经过 configure）时，按默认参数补一次配置，
    # 否则 train_path / val_path 仍为 None，7.1 写文件会直接失败。
    if output_base_path is None:
        configure(input_dir=source_path)

    print("=== 医学参考区间分析系统 ===")
    
    # 获取所有xlsx文件
    file_paths = glob.glob(os.path.join(source_path, "*.xlsx"))
    print(f"找到 {len(file_paths)} 个数据文件")
    
    if len(file_paths) == 0:
        print("未找到任何数据文件，请检查路径")
        return
    
    # =========================================================================
    # 7.1 数据集分割
    # =========================================================================
    print("\n1. 数据集分割...")
    
    all_projects = []
    train_stats = []
    val_stats = []
    
    for file_path in file_paths:
        print(f"处理文件: {os.path.basename(file_path)}")
        
        # 预处理数据
        project_data = preprocess_file(file_path)
        if project_data is None or project_data['data'].empty:
            print(f"  跳过: 数据为空或处理失败")
            continue
        
        # 按4:1比例分割数据集（训练集占80%）
        train_data, val_data = split_dataset_by_ratio(project_data, train_ratio=CONFIG["train_ratio"])
        
        # 保存分割后的数据（保持原始文件名）
        original_filename = project_data['filename']
        
        if not train_data.empty:
            train_filename = os.path.join(train_path, original_filename)
            train_data.to_excel(train_filename, index=False)
        
        if not val_data.empty:
            val_filename = os.path.join(val_path, original_filename)
            val_data.to_excel(val_filename, index=False)
        
        # 统计训练集信息
        if not train_data.empty:
            train_stats.append({
                'Project': project_data['project_name'],
                'Dataset': 'Training',
                'Total_Samples': len(train_data),
                'Male_Samples': len(train_data[train_data['gender'] == 'male']),
                'Female_Samples': len(train_data[train_data['gender'] == 'female']),
                'Age_Mean': train_data['age'].mean(),
                'Age_Std': train_data['age'].std(),
                'Age_Min': train_data['age'].min(),
                'Age_Max': train_data['age'].max(),
                'Date_Min': train_data['date'].min(),
                'Date_Max': train_data['date'].max()
            })
        
        # 统计验证集信息
        if not val_data.empty:
            val_stats.append({
                'Project': project_data['project_name'],
                'Dataset': 'Validation',
                'Total_Samples': len(val_data),
                'Male_Samples': len(val_data[val_data['gender'] == 'male']),
                'Female_Samples': len(val_data[val_data['gender'] == 'female']),
                'Age_Mean': val_data['age'].mean(),
                'Age_Std': val_data['age'].std(),
                'Age_Min': val_data['age'].min(),
                'Age_Max': val_data['age'].max(),
                'Date_Min': val_data['date'].min(),
                'Date_Max': val_data['date'].max()
            })
        
        # 添加到项目列表供后续分析使用
        all_projects.append(project_data)
        
        print(f"  训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")
    
    # 保存统计信息
    if train_stats or val_stats:
        all_stats = train_stats + val_stats
        stats_df = pd.DataFrame(all_stats)
        
        # 计算百分比
        stats_df['Male_Percentage'] = (stats_df['Male_Samples'] / stats_df['Total_Samples'] * 100).round(2)
        stats_df['Female_Percentage'] = (stats_df['Female_Samples'] / stats_df['Total_Samples'] * 100).round(2)
        
        # 重新排列列的顺序
        stats_df = stats_df[[
            'Project', 'Dataset', 'Total_Samples', 
            'Male_Samples', 'Male_Percentage', 'Female_Samples', 'Female_Percentage',
            'Age_Mean', 'Age_Std', 'Age_Min', 'Age_Max', 'Date_Min', 'Date_Max'
        ]]
        
        # 保存统计表格
        stats_filename = os.path.join(output_base_path, "dataset_split_statistics.xlsx")
        stats_df.to_excel(stats_filename, index=False)
        
        print(f"数据集分割统计已保存到: {stats_filename}")
        
        # 汇总统计
        total_train = stats_df[stats_df['Dataset'] == 'Training']['Total_Samples'].sum()
        total_val = stats_df[stats_df['Dataset'] == 'Validation']['Total_Samples'].sum()
        
        print(f"训练集总样本数: {total_train}")
        print(f"验证集总样本数: {total_val}")
        print(f"总样本数: {total_train + total_val}")
        if total_val > 0:
            print(f"训练集/验证集比例: {total_train/total_val:.2f}:1")
    else:
        print("没有找到有效数据进行分割")
        return
    
    # =========================================================================
    # 7.2 性别差异分析
    # =========================================================================
    print("\n2. 性别差异分析...")
    
    # 加载训练集数据进行性别差异分析
    train_projects = []
    for file_path in glob.glob(os.path.join(train_path, "*.xlsx")):
        project_data = preprocess_file(file_path)
        if project_data:
            # 计算Cohen's d
            project_data['cohen_d'] = calculate_cohens_d(
                project_data['male_data'], 
                project_data['female_data']
            )
            
            # 计算Mann-Whitney U检验
            project_data['mw_pvalue'], project_data['mw_significance'] = calculate_mann_whitney_u(
                project_data['male_data'], 
                project_data['female_data']
            )
            
            # 计算偏度
            project_data['skewness_total'], project_data['skewness_interpretation_total'] = calculate_skewness_with_interpretation(project_data['total_data'])
            project_data['skewness_male'], project_data['skewness_interpretation_male'] = calculate_skewness_with_interpretation(project_data['male_data'])
            project_data['skewness_female'], project_data['skewness_interpretation_female'] = calculate_skewness_with_interpretation(project_data['female_data'])
            
            # 计算Harris-Boyd检验
            project_data['hb_z'], project_data['hb_z_star'], project_data['hb_recommendation'], project_data['hb_significant'] = harris_boyd_test(
                project_data['male_data'], 
                project_data['female_data']
            )
            
            train_projects.append(project_data)
    
    print(f"成功处理 {len(train_projects)} 个项目进行性别差异分析")
    
    # 创建KDE图 - 动态调整布局
    if train_projects:
        print("生成KDE图...")
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.0
        
        # 显示所有项目，不再限制数量
        display_projects = train_projects
        
        # 动态计算子图布局 - 根据项目数量智能调整
        n_projects = len(display_projects)
        
        # 动态计算子图布局：SCI 风格统一单行横排，避免 3 个项目被折到第二行
        n_projects = len(display_projects)
        n_cols = n_projects
        n_rows = 1

        print(f"KDE图布局: {n_rows}行 × {n_cols}列，共{n_projects}个项目")

        # 创建图形：单行横排，每个面板约 4.6 英寸宽
        fig_height = 4.6
        fig_width = 4.6 * n_cols
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=600)
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        
        for i, project in enumerate(display_projects):
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # 获取样本量信息
            total_count = len(project['total_data'])
            male_count = len(project['male_data'])
            female_count = len(project['female_data'])
            
            # 绘制总人群KDE
            if len(project['total_data']) > 0:
                sns.kdeplot(project['total_data'], label='Total', color='black', linewidth=2.5, fill=False)
            
            # 绘制男性KDE
            if len(project['male_data']) > 0:
                sns.kdeplot(project['male_data'], label='Male', color='blue', linewidth=2, fill=True, alpha=0.3)
            
            # 绘制女性KDE  
            if len(project['female_data']) > 0:
                sns.kdeplot(project['female_data'], label='Female', color='red', linewidth=2, fill=True, alpha=0.3)
            
            # 设置子图属性
            plt.xlabel('Result', fontsize=11, fontweight='bold')
            plt.ylabel('Density', fontsize=11, fontweight='bold')
            plt.title(f"{project['project_name']}", fontsize=12, fontweight='bold', pad=10)
            plt.legend(fontsize=9, frameon=True, fancybox=False, shadow=False, framealpha=0.9, edgecolor='0.8')
            
            # 样本量信息放左下角：原来放在顶部中央会被右上角图例压住，数字还被截断
            sample_text = f"n = {total_count:,}  (M {male_count:,} / F {female_count:,})"
            plt.text(0.02, 0.03, sample_text, transform=ax.transAxes,
                     fontsize=8, fontweight='bold', verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
            
            # 在右下角添加统计信息
            stats_text = ""
            
            # Harris-Boyd检验结果
            if not np.isnan(project['hb_z']):
                hb_text = f"Harris-Boyd Test:\n"
                hb_text += f"Z = {project['hb_z']:.4f}\n"
                hb_text += f"Z* = {project['hb_z_star']:.4f}\n"
                hb_text += f"{project['hb_recommendation']}\n\n"
                stats_text += hb_text
            else:
                stats_text += "Harris-Boyd: 样本量不足(n<30)\n\n"
            
            # Mann-Whitney U检验P值
            if not np.isnan(project['mw_pvalue']):
                mw_text = f"M-W U test P value: {project['mw_pvalue']:.4f}\n"
                mw_text += f"{project['mw_significance']}\n\n"
                stats_text += mw_text
            else:
                stats_text += "M-W U test: Insufficient data\n\n"
            
            # Cohen's d值
            if not np.isnan(project['cohen_d']):
                cohen_d_text = f"Cohen's d = {project['cohen_d']:.3f}\n"
                if abs(project['cohen_d']) > 0.2:
                    cohen_d_text += "(Significant, >0.2)\n"
                else:
                    cohen_d_text += "(Not significant, ≤0.2)\n"
                stats_text += cohen_d_text + "\n"
            else:
                stats_text += "Cohen's d: Insufficient data\n\n"
            
            # 偏度信息
            if not np.isnan(project['skewness_total']):
                stats_text += f"Skewness:\n"
                stats_text += f"Total: {project['skewness_total']:.3f}\n"
                stats_text += f"({project['skewness_interpretation_total']})\n"
                
                if not np.isnan(project['skewness_male']):
                    stats_text += f"Male: {project['skewness_male']:.3f}\n"
                
                if not np.isnan(project['skewness_female']):
                    stats_text += f"Female: {project['skewness_female']:.3f}"
            else:
                stats_text += "Skewness: Insufficient data"
            
            plt.text(0.95, 0.05, stats_text, transform=ax.transAxes, 
                     fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, 
                              edgecolor='black', pad=0.5))
        
        plt.suptitle('Kernel Density Estimation of Results by Project and Gender (with Harris-Boyd Test)', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_base_path, "Combined_AFP_KDE_plot_with_HarrisBoyd.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        # plt.show()
        
        # 创建GAM图 - 动态调整布局
        print("生成GAM图...")
        # 动态计算子图布局（与KDE图相同）
        n_projects = len(display_projects)
        
        # 动态计算子图布局：SCI 风格统一单行横排
        n_projects = len(display_projects)
        n_cols = n_projects
        n_rows = 1

        print(f"GAM图布局: {n_rows}行 × {n_cols}列，共{n_projects}个项目")

        # 创建图形：单行横排
        fig_height = 4.6
        fig_width = 4.6 * n_cols
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=600)
        sns.set_style("whitegrid")
        
        for i, project in enumerate(display_projects):
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            gam_data = project['data'][['age', 'result', 'gender']].copy().dropna()
            
            # 获取样本量信息
            total_count = len(gam_data)
            male_count = len(gam_data[gam_data['gender']=='male'])
            female_count = len(gam_data[gam_data['gender']=='female'])
            
            # 绘制原始数据点
            # 样本量上万时逐点绘制会糊成一块色板（16 万点即如此），失去可读性：
            # 超过 20000 点时按性别随机抽样至 8000 点绘制，并缩小点径。
            male_points = gam_data[gam_data['gender'] == 'male']
            female_points = gam_data[gam_data['gender'] == 'female']
            max_plot_points = 8000
            point_size = 15 if len(gam_data) <= 20000 else 6
            point_alpha = 0.1 if len(gam_data) <= 20000 else 0.25
            rng = np.random.RandomState(CONFIG['random_seed'])

            if len(male_points) > max_plot_points:
                male_points = male_points.sample(n=max_plot_points, random_state=rng)
            if len(female_points) > max_plot_points:
                female_points = female_points.sample(n=max_plot_points, random_state=rng)

            if len(male_points) > 0:
                plt.scatter(male_points['age'], male_points['result'],
                            alpha=point_alpha, color='blue', label='Male data' + (' (subsampled)' if male_count > max_plot_points else ''), s=point_size)

            if len(female_points) > 0:
                plt.scatter(female_points['age'], female_points['result'],
                            alpha=point_alpha, color='red', label='Female data' + (' (subsampled)' if female_count > max_plot_points else ''), s=point_size)
            
            # 为男性和女性分别拟合GAM模型
            age_range_male, pred_median_male, pred_lower_male, pred_upper_male = fit_gam_model(gam_data, 'male')
            age_range_female, pred_median_female, pred_lower_female, pred_upper_female = fit_gam_model(gam_data, 'female')

            # 稀疏年龄（如 85 岁以上只有几十例）会把样条外推成明显上扬的假趋势，
            # 只在「该性别样本量 ≥20 例」的年龄区间内绘制拟合曲线，避免图上出现
            # 由外推造成的伪拐点。
            def _dense_span(points, min_n=20):
                cnt = points['age'].round().value_counts()
                cnt = cnt[cnt >= min_n]
                return (int(cnt.index.min()), int(cnt.index.max())) if len(cnt) else None

            span_male = _dense_span(gam_data[gam_data['gender'] == 'male'])
            span_female = _dense_span(gam_data[gam_data['gender'] == 'female'])

            if age_range_male is not None and span_male is not None:
                keep = (age_range_male >= span_male[0]) & (age_range_male <= span_male[1])
                age_range_male, pred_median_male = age_range_male[keep], pred_median_male[keep]
                pred_lower_male, pred_upper_male = pred_lower_male[keep], pred_upper_male[keep]

            if age_range_female is not None and span_female is not None:
                keep = (age_range_female >= span_female[0]) & (age_range_female <= span_female[1])
                age_range_female, pred_median_female = age_range_female[keep], pred_median_female[keep]
                pred_lower_female, pred_upper_female = pred_lower_female[keep], pred_upper_female[keep]
            
            # 绘制男性样条分位数回归曲线
            # 注意：这里拟合的是 quantreg + B 样条基（bs(age, df=4)）的分位数回归，
            # 不是 GAM；阴影带是 2.5th–97.5th 分位数带（即参考区间随年龄的变化），
            # 不是"中位数估计的 95% 置信区间"。图和正文都必须按此表述，否则方法学写错。
            if age_range_male is not None:
                plt.plot(age_range_male, pred_median_male, color='darkblue', linewidth=2.5, label='Male median')
                plt.fill_between(age_range_male, pred_lower_male, pred_upper_male, 
                                color='blue', alpha=0.2, label='Male 2.5th–97.5th pct')
            
            # 绘制女性样条分位数回归曲线
            if age_range_female is not None:
                plt.plot(age_range_female, pred_median_female, color='darkred', linewidth=2.5, label='Female median')
                plt.fill_between(age_range_female, pred_lower_female, pred_upper_female, 
                                color='red', alpha=0.2, label='Female 2.5th–97.5th pct')
            
            # 设置子图属性
            plt.xlabel('Age', fontsize=11, fontweight='bold')
            plt.ylabel('Result', fontsize=11, fontweight='bold')
            plt.title(f"{project['project_name']}", fontsize=12, fontweight='bold', pad=10)
            plt.legend(loc='upper right', fontsize=8, frameon=True, fancybox=False, shadow=False, framealpha=0.9, edgecolor='0.8')
            
            # 样本量信息放左下角：原来放在顶部中央会被右上角图例压住，数字还被截断
            sample_text = f"n = {total_count:,}  (M {male_count:,} / F {female_count:,})"
            plt.text(0.02, 0.03, sample_text, transform=ax.transAxes,
                     fontsize=8, fontweight='bold', verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
            
            # 设置y轴范围
            if len(gam_data) > 0:
                y_min = gam_data['result'].quantile(0.01)
                y_max = gam_data['result'].quantile(0.99)
                plt.ylim(y_min, y_max)
        
        plt.suptitle('Spline quantile regression of results by age and gender (50th, 2.5th/97.5th percentiles)', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # 图幅本身已是 10 英寸见方，再乘 600 dpi 会得到约 5900×5900 像素的超大 PNG（无法预览），
        # 统一按 300 dpi 出图，兼顾印刷精度与文件可用性。
        plt.savefig(os.path.join(output_base_path, "Combined_AFP_GAM_plot.png"),
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        # plt.show()
    
    # =========================================================================
    # 7.3 年龄分组
    # =========================================================================
    print("\n3. 年龄分组分析...")
    
    # 创建结果数据框
    age_results = []
    
    # 分析每个项目
    for project in train_projects:
        project_name = project['project_name']
        print(f"分析项目: {project_name}")
        
        # 处理男性数据
        if len(project['male_data']) > 0:
            male_groups = auto_age_grouping_improved(
                project['data'][project['data']['gender'] == 'male'],
                min_samples=CONFIG["min_samples"],
                cohen_d_threshold=CONFIG["cohen_d_threshold"])
            
            for i, (low, high) in enumerate(male_groups):
                group_data = project['data'][
                    (project['data']['gender'] == 'male') & 
                    (project['data']['age'] >= low) & 
                    (project['data']['age'] <= high)
                ]['result'].values
                
                if len(group_data) > 0:
                    prev_cohen_d = np.nan
                    if i > 0:
                        prev_group_data = project['data'][
                            (project['data']['gender'] == 'male') & 
                            (project['data']['age'] >= male_groups[i-1][0]) & 
                            (project['data']['age'] <= male_groups[i-1][1])
                        ]['result'].values
                        prev_cohen_d = calculate_cohens_d_age(prev_group_data, group_data)
                    
                    age_results.append({
                        'Project': project_name,
                        'Gender': 'Male',
                        'Age Group': f"{low}-{high}",
                        'N': len(group_data),
                        'Mean': np.mean(group_data),
                        'Median': np.median(group_data),
                        'Std': np.std(group_data),
                        'Min': np.min(group_data),
                        'Max': np.max(group_data),
                        "Cohen's d vs Previous": prev_cohen_d
                    })
        
        # 处理女性数据
        if len(project['female_data']) > 0:
            female_groups = auto_age_grouping_improved(
                project['data'][project['data']['gender'] == 'female'],
                min_samples=CONFIG["min_samples"],
                cohen_d_threshold=CONFIG["cohen_d_threshold"])
            
            for i, (low, high) in enumerate(female_groups):
                group_data = project['data'][
                    (project['data']['gender'] == 'female') & 
                    (project['data']['age'] >= low) & 
                    (project['data']['age'] <= high)
                ]['result'].values
                
                if len(group_data) > 0:
                    prev_cohen_d = np.nan
                    if i > 0:
                        prev_group_data = project['data'][
                            (project['data']['gender'] == 'female') & 
                            (project['data']['age'] >= female_groups[i-1][0]) & 
                            (project['data']['age'] <= female_groups[i-1][1])
                        ]['result'].values
                        prev_cohen_d = calculate_cohens_d_age(prev_group_data, group_data)
                    
                    age_results.append({
                        'Project': project_name,
                        'Gender': 'Female',
                        'Age Group': f"{low}-{high}",
                        'N': len(group_data),
                        'Mean': np.mean(group_data),
                        'Median': np.median(group_data),
                        'Std': np.std(group_data),
                        'Min': np.min(group_data),
                        'Max': np.max(group_data),
                        "Cohen's d vs Previous": prev_cohen_d
                    })
    
    # 创建结果数据框
    if age_results:
        age_results_df = pd.DataFrame(age_results)
        
        # 保存结果到Excel
        age_output_path = os.path.join(output_base_path, "Auto_Age_Grouping_Results.xlsx")
        age_results_df.to_excel(age_output_path, index=False)
        print(f"年龄分组结果已保存到: {age_output_path}")
    else:
        print("未生成年龄分组结果")
        age_results_df = pd.DataFrame()
    
    # =========================================================================
    # 7.4 参考区间计算
    # =========================================================================
    print("\n4. 参考区间计算...")
    
    if age_results_df.empty:
        print("年龄分组结果为空，跳过参考区间计算")
    else:
        reference_intervals = []
        train_file_map = build_project_file_map(train_path)
        
        # 对每个项目、每个性别、每个年龄分组进行处理
        for (project_name, gender), group_data in age_results_df.groupby(['Project', 'Gender']):
            print(f"处理项目: {project_name}, 性别: {gender}")
            
            # 找到对应的训练数据文件（按目录实际文件名匹配，避免文件名带点号时路径拼错）
            train_file_path = train_file_map.get(project_name)
            if train_file_path is None or not os.path.exists(train_file_path):
                print(f"  训练文件不存在: {project_name}")
                continue
            
            # 加载训练数据
            try:
                train_df = pd.read_excel(train_file_path)
                train_df = train_df[['gender', 'age', 'result']].copy()
                
                # 转换性别格式
                train_df['gender'] = train_df['gender'].apply(
                    lambda x: 'male' if str(x).lower() in ['男', 'male', 'm'] else 
                             ('female' if str(x).lower() in ['女', 'female', 'f'] else None)
                )
                
                # 过滤性别
                gender_df = train_df[train_df['gender'] == gender.lower()]
                
                if gender_df.empty:
                    print(f"  没有找到{gender}数据")
                    continue
                
            except Exception as e:
                print(f"  读取训练数据失败: {e}")
                continue
            
            # 处理每个年龄组
            for _, group_row in group_data.iterrows():
                age_group = group_row['Age Group']
                low, high = map(int, age_group.split('-'))
                
                # 筛选该年龄组的数据
                group_result_data = gender_df[
                    (gender_df['age'] >= low) & 
                    (gender_df['age'] <= high)
                ]['result'].values
                
                if len(group_result_data) < CONFIG["min_group_for_ri"]:
                    print(f"  年龄组 {age_group}: 样本量不足 ({len(group_result_data)} < {CONFIG['min_group_for_ri']})")
                    continue
                    
                print(f"  处理 {gender} 组 {age_group}: {len(group_result_data)} 个样本")
                
                # 进行Box-Cox变换
                try:
                    transformed_data, lmbda, shift = boxcox_transform(group_result_data)
                except Exception as e:
                    print(f"  Box-Cox变换失败: {e}")
                    continue
                
                # 五种异常值检测方法
                methods = {
                    'Tukey': detect_outliers_tukey,
                    'GMM': detect_outliers_gmm,
                    'Z-score': detect_outliers_zscore,
                    'Isolation Forest': detect_outliers_isolation_forest,
                    'Local Outlier Factor': detect_outliers_lof
                }
                
                for method_name, method_func in methods.items():
                    try:
                        # 检测异常值
                        outliers = method_func(transformed_data)
                        clean_data = transformed_data[~outliers]
                        
                        # 使用自举法计算参考区间（含限值的 95% 抽样区间）
                        (lower_bound, upper_bound,
                         lower_ci_lo, lower_ci_hi,
                         upper_ci_lo, upper_ci_hi) = bootstrap_reference_interval(
                            clean_data,
                            n_bootstrap=CONFIG["bootstrap_iterations"],
                            seed=CONFIG["random_seed"])

                        # 逆变换回原始尺度
                        original_lower = inverse_boxcox(lower_bound, lmbda, shift)
                        original_upper = inverse_boxcox(upper_bound, lmbda, shift)

                        # 限值置信区间同样逆变换；数值失败时退回点估计，保证不出现 nan
                        def _inv_safe(x, fallback):
                            try:
                                v = inverse_boxcox(x, lmbda, shift)
                                return float(v) if np.isfinite(v) else fallback
                            except Exception:
                                return fallback

                        original_lower_ci_lo = _inv_safe(lower_ci_lo, original_lower)
                        original_lower_ci_hi = _inv_safe(lower_ci_hi, original_lower)
                        original_upper_ci_lo = _inv_safe(upper_ci_lo, original_upper)
                        original_upper_ci_hi = _inv_safe(upper_ci_hi, original_upper)

                        # 保序：CI 端点不得越过点估计（个别病态分布下会出现交叉）
                        lo_ci = (min(original_lower_ci_lo, original_lower),
                                 max(original_lower_ci_hi, original_lower))
                        hi_ci = (min(original_upper_ci_lo, original_upper),
                                 max(original_upper_ci_hi, original_upper))

                        reference_intervals.append({
                            'Project': project_name,
                            'Gender': gender,
                            'Age Group': age_group,
                            'Method': method_name,
                            'Lower Limit': original_lower,
                            'Lower 95% CI Low': lo_ci[0],
                            'Lower 95% CI High': lo_ci[1],
                            'Upper Limit': original_upper,
                            'Upper 95% CI Low': hi_ci[0],
                            'Upper 95% CI High': hi_ci[1],
                            'N': len(clean_data)
                        })
                    except Exception as e:
                        print(f"  方法 {method_name} 失败: {e}")
                        continue
        
        # 创建参考区间表格
        if reference_intervals:
            reference_df = pd.DataFrame(reference_intervals)
            reference_output_path = os.path.join(output_base_path, "Reference_Intervals_Table2.xlsx")
            reference_df.to_excel(reference_output_path, index=False)
            print(f"参考区间表已保存到: {reference_output_path}")
            
            # =================================================================
            # 7.5 参考区间比较
            # =================================================================
            print("\n5. 参考区间比较...")
            
            comparison_results = []
            
            # 对每个项目、性别、年龄分组，以Tukey方法为基准进行比较
            for (project, gender, age_group), group_df in reference_df.groupby(['Project', 'Gender', 'Age Group']):
                # 获取Tukey方法的结果
                tukey_rows = group_df[group_df['Method'] == 'Tukey']
                if tukey_rows.empty:
                    continue
                    
                tukey_row = tukey_rows.iloc[0]
                tukey_lower = tukey_row['Lower Limit']
                tukey_upper = tukey_row['Upper Limit']
                
                # 与其他方法比较
                for _, row in group_df.iterrows():
                    if row['Method'] == 'Tukey':
                        continue
                    
                    method = row['Method']
                    method_lower = row['Lower Limit']
                    method_upper = row['Upper Limit']
                    
                    # 计算BR值
                    br_lower, br_upper = calculate_brs(tukey_lower, tukey_upper, method_lower, method_upper)
                    
                    # 判断差异是否有意义
                    significant_lower = abs(br_lower) >= 0.375
                    significant_upper = abs(br_upper) >= 0.375
                    
                    comparison_results.append({
                        'Project': project,
                        'Gender': gender,
                        'Age Group': age_group,
                        'Comparison Method': method,
                        'Tukey Lower': tukey_lower,
                        'Tukey Upper': tukey_upper,
                        'Method Lower': method_lower,
                        'Method Upper': method_upper,
                        'BR Lower': br_lower,
                        'BR Upper': br_upper,
                        'Significant Lower': significant_lower,
                        'Significant Upper': significant_upper
                    })
            
            # 创建比较表格
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                comparison_output_path = os.path.join(output_base_path, "Reference_Interval_Comparison_Table3.xlsx")
                comparison_df.to_excel(comparison_output_path, index=False)
                print(f"参考区间比较表已保存到: {comparison_output_path}")
            else:
                print("未生成参考区间比较表")
        else:
            print("未生成参考区间表")
            reference_df = pd.DataFrame()
    
    # =========================================================================
    # 7.6 内部验证
    # =========================================================================
    print("\n6. 内部验证...")
    
    # 检查参考区间表是否存在
    reference_table_path = os.path.join(output_base_path, "Reference_Intervals_Table2.xlsx")
    if not os.path.exists(reference_table_path):
        print(f"参考区间表不存在: {reference_table_path}")
        print("跳过内部验证")
    else:
        try:
            ref_df = pd.read_excel(reference_table_path)
            print(f"成功读取参考区间表，共 {len(ref_df)} 行数据")
            
            # 为每个项目计算百分比
            percentages = []
            val_cache = {}                      # 同一验证集文件只读一次
            val_file_map = build_project_file_map(val_path)
            
            for idx, row in ref_df.iterrows():
                if idx % 50 == 0:
                    print(f"处理第 {idx+1}/{len(ref_df)} 行...")
                
                percentage = calculate_percentage_for_row(
                    row, val_path, cache=val_cache, file_map=val_file_map)
                percentages.append(percentage)
            
            # 添加百分比列到参考区间表
            ref_df['Percentage'] = percentages
            # 90% 覆盖率是 CLSI C28-A3c / WS/T 402 的通行验收线，单独标注便于筛查不达标组
            ref_df['Coverage_OK_90'] = [None if p is None else (p >= 90) for p in percentages]
            
            # 添加百分比列到参考区间表
            ref_df['Percentage'] = percentages
            
            # 保存结果
            output_path = os.path.join(output_base_path, "Reference_Intervals_Table1_with_Percentage.xlsx")
            ref_df.to_excel(output_path, index=False)
            print(f"带百分比的参考区间表已保存到: {output_path}")
            
            # 显示统计信息
            valid_percentages = [p for p in percentages if p is not None]
            print(f"\n验证统计信息:")
            print(f"总行数: {len(ref_df)}")
            print(f"成功计算百分比的行数: {len(valid_percentages)}")
            print(f"失败的行数: {len(ref_df) - len(valid_percentages)}")
            
            if valid_percentages:
                print(f"百分比范围: {min(valid_percentages)}% - {max(valid_percentages)}%")
                print(f"平均百分比: {np.mean(valid_percentages):.2f}%")
                below = [p for p in valid_percentages if p < 90]
                print(f"覆盖率 <90% 的行数: {len(below)}/{len(valid_percentages)}")
                if below:
                    fail_df = ref_df[ref_df['Percentage'].apply(lambda v: v is not None and v < 90)]
                    for _, r in fail_df.iterrows():
                        print(f"  未达标: {r['Project']} {r['Gender']} {r['Age Group']} "
                              f"{r['Method']} -> {r['Percentage']}%")
            
        except Exception as e:
            print(f"内部验证失败: {str(e)}")
    
    # =========================================================================
    # 7.7 生成森林图
    # =========================================================================
    print("\n7. 生成森林图...")
    
    # 检查带百分比的文件是否存在
    percentage_file = os.path.join(output_base_path, "Reference_Intervals_Table1_with_Percentage.xlsx")
    if not os.path.exists(percentage_file):
        print(f"带百分比的文件不存在: {percentage_file}")
        print("跳过森林图生成")
    else:
        try:
            df = pd.read_excel(percentage_file)
            
            # 获取所有方法和项目
            methods = df['Method'].unique()
            projects = df['Project'].unique()
            
            # 创建统一的Y轴结构
            y_labels = []
            project_positions = []
            project_end_positions = []
            
            current_pos = 0
            for project_idx, project in enumerate(projects):
                project_positions.append(current_pos)
                
                project_data = df[df['Project'] == project]
                combinations = project_data[['Gender', 'Age Group']].drop_duplicates()
                
                male_combinations = combinations[combinations['Gender'] == 'Male']
                female_combinations = combinations[combinations['Gender'] == 'Female']
                
                # 处理男性组合
                for _, combo in male_combinations.iterrows():
                    gender = combo['Gender']
                    age_group = combo['Age Group']
                    label = f"{project}-M({age_group})"
                    y_labels.append(label)
                    current_pos += 1
                
                # 处理女性组合
                for _, combo in female_combinations.iterrows():
                    gender = combo['Gender']
                    age_group = combo['Age Group']
                    label = f"{project}-F({age_group})"
                    y_labels.append(label)
                    current_pos += 1
                
                project_end_positions.append(current_pos - 1)
                
                if project_idx < len(projects) - 1:
                    y_labels.append('')
                    current_pos += 1
            
            # 创建森林图
            # 图幅按标签数量与面板数量自适应：原实现 figsize=(6*5, 15) + 600 dpi 会输出
            # 17760×8586 像素的 PNG，既无法预览也无法用于投稿。这里限制单边像素 ≤5800，
            # 并按行高缩放字号，避免标签互相压字。
            n_rows = max(1, len(y_labels))
            fig_w = 2.8 * len(methods) + 2.2
            # 行高加大：0.42 英寸/行，让各组人群纵向拉开、标签不挤
            fig_h = max(5.5, 0.42 * n_rows + 2.4)
            save_dpi = 300
            while max(fig_w, fig_h) * save_dpi > 5800:
                save_dpi -= 25
            font_size = int(max(9, min(15, (fig_h * 72.0 / (n_rows + 2)) * 0.62)))

            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.dpi'] = save_dpi
            plt.rcParams['font.size'] = font_size
            plt.rcParams['axes.linewidth'] = 1.0
            plt.rcParams['xtick.major.width'] = 1.0
            plt.rcParams['ytick.major.width'] = 1.0
            plt.rcParams['font.weight'] = 'bold'

            fig, axes = plt.subplots(1, len(methods), figsize=(fig_w, fig_h), sharey=True)
            if len(methods) == 1:
                axes = [axes]

            fig.suptitle('Validation sets for RIs established by different algorithms',
                         fontsize=font_size + 6, y=0.99, fontweight='bold')
            
            # 为每个方法创建子图
            for method_idx, method in enumerate(methods):
                ax = axes[method_idx]
                
                method_data = df[df['Method'] == method]
                plot_data = []
                
                y_pos = 0
                for label in y_labels:
                    if label == '':
                        y_pos += 1
                        continue
                    
                    if '-M(' in label:
                        parts = label.split('-M(')
                        project = parts[0]
                        age_group = parts[1][:-1]
                        
                        subset = method_data[(method_data['Project'] == project) & 
                                           (method_data['Gender'] == 'Male') & 
                                           (method_data['Age Group'] == age_group)]
                        
                        if not subset.empty:
                            plot_data.append({
                                'y_pos': y_pos,
                                'percentage': subset.iloc[0]['Percentage'],
                                'project': project,
                                'gender': 'M'
                            })
                    elif '-F(' in label:
                        parts = label.split('-F(')
                        project = parts[0]
                        age_group = parts[1][:-1]
                        
                        subset = method_data[(method_data['Project'] == project) & 
                                           (method_data['Gender'] == 'Female') & 
                                           (method_data['Age Group'] == age_group)]
                        
                        if not subset.empty:
                            plot_data.append({
                                'y_pos': y_pos,
                                'percentage': subset.iloc[0]['Percentage'],
                                'project': project,
                                'gender': 'F'
                            })
                    
                    y_pos += 1
                
                # 绘制数据点
                for data in plot_data:
                    if pd.notna(data['percentage']):
                        color = 'darkblue' if data['gender'] == 'M' else 'darkred'
                        ax.scatter(data['percentage'], len(y_labels) - 1 - data['y_pos'], 
                                  color=color, s=180, alpha=0.9, edgecolors='black', linewidth=1.2)
                
                # 设置Y轴
                ax.set_yticks(range(len(y_labels)))
                reversed_labels = y_labels[::-1]
                ax.set_yticklabels(reversed_labels, fontsize=font_size + 2, fontweight='bold')
                
                # X 轴固定 85–100%（用户要求；本队列最低覆盖率 89.76%，无点被裁）
                ax.set_xlim(85, 100)
                # X 轴标题只在中间面板给一次，五份重复文字会互相压字
                if method_idx == len(methods) // 2:
                    ax.set_xlabel('Validation percentage (%)',
                                  fontsize=font_size + 2, fontweight='bold', labelpad=8)
                ax.grid(axis='x', alpha=0.4, linestyle=':', linewidth=0.8)
                ax.set_title(method.replace(' ', '\n'), fontsize=font_size + 3, pad=8, fontweight='bold')
                
                # 添加项目分隔线
                for i in range(len(projects) - 1):
                    sep_pos = project_end_positions[i] + 1.5
                    actual_sep_pos = len(y_labels) - 1 - sep_pos
                    ax.axhline(y=actual_sep_pos, color='darkgray', linestyle='-', linewidth=2.0, alpha=0.9)
                
                # 添加背景色条
                for i in range(len(y_labels)):
                    if i % 2 == 0 and y_labels[len(y_labels) - 1 - i] != '':
                        actual_pos = len(y_labels) - 1 - i
                        ax.axhspan(actual_pos - 0.5, actual_pos + 0.5, facecolor='lightgray', alpha=0.2)
                
                # 在90%位置添加红色垂直虚线（标签放顶部，避免压在某一行的数据点上）
                ax.axvline(x=90, color='red', linestyle='-.', linewidth=2.5, alpha=0.9)
                ax.text(90.4, len(y_labels) - 0.6, '90%', fontsize=font_size + 3, fontweight='bold',
                        color='red', verticalalignment='center')
                
                # 设置纵坐标字体
                ax.tick_params(axis='y', which='major', labelsize=font_size)
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')

            marker_size = max(6, font_size * 0.8)
            # 添加图例
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='darkblue', markersize=marker_size,
                                         label='Male', markeredgecolor='black', markeredgewidth=1.2),
                              plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='darkred', markersize=marker_size,
                                         label='Female', markeredgecolor='black', markeredgewidth=1.2)]
            fig.legend(handles=legend_elements, title='Gender',
                      loc='upper center', ncol=2, fontsize=font_size + 2,
                      title_fontsize=font_size + 3,
                      bbox_to_anchor=(0.5, 0.945), frameon=False)

            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.90])

            # 保存图片
            sci_quality_plot_path = os.path.join(output_base_path, 'SCI_quality_combined_methods_reference_interval_forest_plot.png')
            plt.savefig(sci_quality_plot_path, dpi=save_dpi, bbox_inches='tight')
            print(f"SCI期刊质量的合并方法参考区间森林图已保存到: {sci_quality_plot_path}")
            
            # plt.show()
            
        except Exception as e:
            print(f"生成森林图失败: {e}")
    
    print("\n=== 分析完成！ ===")
    print(f"所有结果已保存到: {output_base_path}")
    print("生成的文件包括:")
    print("1. dataset_split_statistics.xlsx")
    print("2. Combined_AFP_KDE_plot_with_HarrisBoyd.png")
    print("3. Combined_AFP_GAM_plot.png") 
    print("4. Auto_Age_Grouping_Results.xlsx")
    print("5. Reference_Intervals_Table2.xlsx")
    print("6. Reference_Interval_Comparison_Table3.xlsx")
    print("7. Reference_Intervals_Table1_with_Percentage.xlsx")
    print("8. SCI_quality_combined_methods_reference_interval_forest_plot.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="间接法参考区间构建与验证（RWD，五法剔除异常值 + Harris-Boyd + 年龄分组 + bootstrap）")
    parser.add_argument("--input", default=".", help="原始 xlsx 数据所在目录（默认当前目录）")
    parser.add_argument("--output", default=None, help="结果输出目录（默认 ./results_<时间戳>）")
    parser.add_argument("--train-ratio", type=float, default=None, help="训练集比例，默认 0.8（按测定时间顺序切分）")
    parser.add_argument("--min-samples", type=int, default=None, help="年龄分组最小样本量，默认 120")
    parser.add_argument("--cohen-d", type=float, default=None, help="年龄分组 Cohen's d 合并阈值，默认 0.2")
    parser.add_argument("--bootstrap", type=int, default=None, help="bootstrap 次数，默认 1000")
    parser.add_argument("--min-group", type=int, default=None, help="计算参考区间的最低样本量，默认 20")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认 42")
    parser.add_argument("--result-lower-bound", type=float, default=None,
                        help="结果值下限；默认剔除 <=0（若指标可取 0 或负值，请显式指定）")
    args = parser.parse_args()

    configure(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        min_samples=args.min_samples,
        cohen_d_threshold=args.cohen_d,
        bootstrap_iterations=args.bootstrap,
        min_group_for_ri=args.min_group,
        random_seed=args.seed,
    )
    if args.result_lower_bound is not None:
        CONFIG["result_lower_bound"] = args.result_lower_bound

    main()
