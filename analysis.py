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
# 1. 配置参数
# =============================================================================

# 设置输出路径
output_base_path = os.path.join(".", "results_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(output_base_path, exist_ok=True)

# 原始数据路径
source_path = "."

# 训练集和验证集路径
train_path = os.path.join(output_base_path, "train")
val_path = os.path.join(output_base_path, "val")
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

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
        
        # 获取项目名称（从文件名获取）
        filename = os.path.basename(file_path)
        project_name = filename.split('.')[0]
        
        # 预处理 - 使用中英文列名
        # 首先检查列名是否存在
        required_columns = []
        for col in ['结果', 'result', '测定时间', 'date', '性别', 'gender', '年龄', 'age']:
            if col in df.columns:
                if '结果' in df.columns or 'result' in df.columns:
                    result_col = '结果' if '结果' in df.columns else 'result'
                if '测定时间' in df.columns or 'date' in df.columns:
                    date_col = '测定时间' if '測定時間' in df.columns else 'date'
                if '性别' in df.columns or 'gender' in df.columns:
                    gender_col = '性别' if '性别' in df.columns else 'gender'
                if '年龄' in df.columns or 'age' in df.columns:
                    age_col = '年龄' if '年龄' in df.columns else 'age'
        
        # 创建新的DataFrame
        new_df = df[[result_col, date_col, gender_col, age_col]].copy()
        new_df.columns = ['result', 'date', 'gender', 'age']
        
        # 处理日期列
        try:
            # 尝试多种日期格式
            new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        except:
            # 如果转换失败，尝试Excel日期格式
            new_df['date'] = new_df['date'].apply(lambda x: pd.Timestamp('1899-12-30') + pd.Timedelta(days=x) 
                                                if isinstance(x, (int, float)) else pd.NaT)
        
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
        new_df = new_df[new_df['result'] > 0]
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
    """改进的自动年龄分组算法"""
    if len(data) == 0:
        return []
    
    # 确保年龄是整数
    data['age_int'] = data['age'].astype(int)
    min_age = int(data['age_int'].min())
    max_age = int(data['age_int'].max())
    
    # 创建初始的单岁年龄组
    age_ranges = []
    for age in range(min_age, max_age + 1):
        age_ranges.append((age, age))
    
    groups = []
    for low, high in age_ranges:
        group_df = data[(data['age_int'] >= low) & (data['age_int'] <= high)]
        groups.append({
            'range': (low, high),
            'data': group_df['result'].values,
            'size': len(group_df)
        })
    
    iteration = 0
    changed = True
    
    while changed and iteration < max_iterations:
        changed = False
        i = 0
        new_groups = []
        
        while i < len(groups):
            if i == len(groups) - 1:
                new_groups.append(groups[i])
                i += 1
            else:
                current_group = groups[i]
                next_group = groups[i+1]
                
                if current_group['size'] < min_samples or next_group['size'] < min_samples:
                    # 合并小样本组
                    new_low = min(current_group['range'][0], next_group['range'][0])
                    new_high = max(current_group['range'][1], next_group['range'][1])
                    merged_data = np.concatenate([current_group['data'], next_group['data']])
                    new_groups.append({
                        'range': (new_low, new_high),
                        'data': merged_data,
                        'size': len(merged_data)
                    })
                    changed = True
                    i += 2
                else:
                    # 计算Cohen's d
                    cohen_d = calculate_cohens_d_age(current_group['data'], next_group['data'])
                    
                    if abs(cohen_d) <= cohen_d_threshold:
                        # 效应量小，合并组
                        new_low = min(current_group['range'][0], next_group['range'][0])
                        new_high = max(current_group['range'][1], next_group['range'][1])
                        merged_data = np.concatenate([current_group['data'], next_group['data']])
                        new_groups.append({
                            'range': (new_low, new_high),
                            'data': merged_data,
                            'size': len(merged_data)
                        })
                        changed = True
                        i += 2
                    else:
                        # 效应量大，保持分开
                        new_groups.append(current_group)
                        i += 1
        
        groups = new_groups
        iteration += 1
    
    final_ranges = [g['range'] for g in groups]
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
    data_reshaped = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2)
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
    if lmbda == 0:
        original = np.exp(transformed)
    else:
        original = (lmbda * transformed + 1) ** (1 / lmbda)
    
    return original - shift

def bootstrap_reference_interval(data, n_bootstrap=1000, alpha=0.05):
    """使用自举法计算参考区间"""
    n = len(data)
    if n < 10:
        return np.nan, np.nan
    
    lower_percentile = alpha * 100 / 2
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bounds = []
    upper_bounds = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        lower_bounds.append(np.percentile(sample, lower_percentile))
        upper_bounds.append(np.percentile(sample, upper_percentile))
    
    lower_bound = np.median(lower_bounds)
    upper_bound = np.median(upper_bounds)
    
    return lower_bound, upper_bound

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

def calculate_percentage_for_row(row, val_data_folder):
    """为参考区间表中的每一行计算百分比"""
    project = row['Project']
    gender = row['Gender']
    age_group = row['Age Group']
    method = row['Method']
    lower_limit = row['Lower Limit']
    upper_limit = row['Upper Limit']
    
    # 构建对应的验证数据文件路径
    file_path = os.path.join(val_data_folder, f"{project}.xlsx")
    
    if not os.path.exists(file_path):
        print(f"验证文件不存在: {file_path}")
        return None
    
    try:
        # 读取验证数据文件
        val_df = pd.read_excel(file_path)
        
        # 检查必要的列是否存在
        required_columns = ['result', 'gender', 'age']
        if not all(col in val_df.columns for col in required_columns):
            print(f"验证文件缺少必要列: {file_path}")
            return None
        
        # 过滤数据：匹配性别
        if gender.lower() == 'male':
            gender_filter = (val_df['gender'].str.lower() == 'male') | (val_df['gender'] == '男')
        elif gender.lower() == 'female':
            gender_filter = (val_df['gender'].str.lower() == 'female') | (val_df['gender'] == '女')
        else:
            print(f"未知性别: {gender}")
            return None
        
        # 应用性别过滤
        filtered_df = val_df[gender_filter].copy()
        
        if len(filtered_df) == 0:
            print(f"没有找到匹配性别的数据: {project}, {gender}")
            return 0.0
        
        # 处理年龄数据
        filtered_df['age_numeric'] = pd.to_numeric(filtered_df['age'], errors='coerce')
        
        # 检查年龄是否为Excel日期格式
        if filtered_df['age_numeric'].max() > 120:
            filtered_df['age_value'] = filtered_df['age_numeric'].apply(excel_date_to_age)
        else:
            filtered_df['age_value'] = filtered_df['age_numeric']
        
        # 应用年龄范围过滤
        age_filter = filtered_df['age_value'].apply(lambda x: check_age_in_range(age_group, x))
        filtered_df = filtered_df[age_filter]
        
        if len(filtered_df) == 0:
            print(f"没有找到匹配年龄范围的数据: {project}, {gender}, {age_group}")
            return 0.0
        
        # 检查结果值是否在参考区间内
        result_in_range = (filtered_df['result'] >= lower_limit) & (filtered_df['result'] <= upper_limit)
        
        # 计算百分比
        percentage = (result_in_range.sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0.0
        
        return round(percentage, 2)
    
    except Exception as e:
        print(f"处理验证文件时出错 {file_path}: {str(e)}")
        return None

# =============================================================================
# 7. 主分析流程
# =============================================================================

def main():
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
        train_data, val_data = split_dataset_by_ratio(project_data, train_ratio=0.8)
        
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
        
        # 计算合适的列数和行数
        if n_projects <= 4:
            n_cols = min(n_projects, 2)  # 1-4个项目时，最多2列
            n_rows = math.ceil(n_projects / n_cols)
        elif n_projects <= 9:
            n_cols = 3  # 5-9个项目时，使用3列
            n_rows = math.ceil(n_projects / n_cols)
        else:
            n_cols = 4  # 10个以上项目时，使用4列
            n_rows = math.ceil(n_projects / n_cols)
        
        print(f"KDE图布局: {n_rows}行 × {n_cols}列，共{n_projects}个项目")
        
        # 创建图形，根据行数调整高度
        fig_height = 5 * n_rows
        fig_width = 5 * n_cols
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
            plt.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
            
            # 在图表上方中央添加样本量信息
            sample_text = f"Total: {total_count}\nMale: {male_count}\nFemale: {female_count}"
            plt.text(0.5, 0.95, sample_text, transform=ax.transAxes, 
                     fontsize=10, fontweight='bold', verticalalignment='top', horizontalalignment='center',
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
        
        # 计算合适的列数和行数
        if n_projects <= 4:
            n_cols = min(n_projects, 2)  # 1-4个项目时，最多2列
            n_rows = math.ceil(n_projects / n_cols)
        elif n_projects <= 9:
            n_cols = 3  # 5-9个项目时，使用3列
            n_rows = math.ceil(n_projects / n_cols)
        else:
            n_cols = 4  # 10个以上项目时，使用4列
            n_rows = math.ceil(n_projects / n_cols)
        
        print(f"GAM图布局: {n_rows}行 × {n_cols}列，共{n_projects}个项目")
        
        # 创建图形，根据行数调整高度
        fig_height = 5 * n_rows
        fig_width = 5 * n_cols
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
            male_points = gam_data[gam_data['gender']=='male']
            female_points = gam_data[gam_data['gender']=='female']
            
            if len(male_points) > 0:
                plt.scatter(male_points['age'], male_points['result'], 
                           alpha=0.1, color='blue', label='Male data', s=15)
            
            if len(female_points) > 0:
                plt.scatter(female_points['age'], female_points['result'], 
                           alpha=0.1, color='red', label='Female data', s=15)
            
            # 为男性和女性分别拟合GAM模型
            age_range_male, pred_median_male, pred_lower_male, pred_upper_male = fit_gam_model(gam_data, 'male')
            age_range_female, pred_median_female, pred_lower_female, pred_upper_female = fit_gam_model(gam_data, 'female')
            
            # 绘制男性GAM曲线
            if age_range_male is not None:
                plt.plot(age_range_male, pred_median_male, color='darkblue', linewidth=2.5, label='Male median')
                plt.fill_between(age_range_male, pred_lower_male, pred_upper_male, 
                                color='blue', alpha=0.2, label='Male 95% CI')
            
            # 绘制女性GAM曲线
            if age_range_female is not None:
                plt.plot(age_range_female, pred_median_female, color='darkred', linewidth=2.5, label='Female median')
                plt.fill_between(age_range_female, pred_lower_female, pred_upper_female, 
                                color='red', alpha=0.2, label='Female 95% CI')
            
            # 设置子图属性
            plt.xlabel('Age', fontsize=11, fontweight='bold')
            plt.ylabel('Result', fontsize=11, fontweight='bold')
            plt.title(f"{project['project_name']}", fontsize=12, fontweight='bold', pad=10)
            plt.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, shadow=True)
            
            # 在图表上方中央添加样本量信息
            sample_text = f"Total: {total_count}\nMale: {male_count}\nFemale: {female_count}"
            plt.text(0.5, 0.95, sample_text, transform=ax.transAxes, 
                     fontsize=10, fontweight='bold', verticalalignment='top', horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
            
            # 设置y轴范围
            if len(gam_data) > 0:
                y_min = gam_data['result'].quantile(0.01)
                y_max = gam_data['result'].quantile(0.99)
                plt.ylim(y_min, y_max)
        
        plt.suptitle('Generalized Additive Models of Results by Age and Gender', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_base_path, "Combined_AFP_GAM_plot.png"), 
                    dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
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
            male_groups = auto_age_grouping_improved(project['data'][project['data']['gender'] == 'male'])
            
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
            female_groups = auto_age_grouping_improved(project['data'][project['data']['gender'] == 'female'])
            
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
        
        # 对每个项目、每个性别、每个年龄分组进行处理
        for (project_name, gender), group_data in age_results_df.groupby(['Project', 'Gender']):
            print(f"处理项目: {project_name}, 性别: {gender}")
            
            # 找到对应的训练数据文件
            train_file_path = os.path.join(train_path, f"{project_name}.xlsx")
            if not os.path.exists(train_file_path):
                print(f"  训练文件不存在: {train_file_path}")
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
                
                if len(group_result_data) < 20:
                    print(f"  年龄组 {age_group}: 样本量不足 ({len(group_result_data)} < 20)")
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
                        
                        # 使用自举法计算参考区间
                        lower_bound, upper_bound = bootstrap_reference_interval(clean_data)
                        
                        # 逆变换回原始尺度
                        original_lower = inverse_boxcox(lower_bound, lmbda, shift)
                        original_upper = inverse_boxcox(upper_bound, lmbda, shift)
                        
                        reference_intervals.append({
                            'Project': project_name,
                            'Gender': gender,
                            'Age Group': age_group,
                            'Method': method_name,
                            'Lower Limit': original_lower,
                            'Upper Limit': original_upper
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
            
            for idx, row in ref_df.iterrows():
                if idx % 50 == 0:
                    print(f"处理第 {idx+1}/{len(ref_df)} 行...")
                
                percentage = calculate_percentage_for_row(row, val_path)
                percentages.append(percentage)
            
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
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.dpi'] = 600
            plt.rcParams['font.size'] = 20
            plt.rcParams['axes.linewidth'] = 1.5
            plt.rcParams['xtick.major.width'] = 1.5
            plt.rcParams['ytick.major.width'] = 1.5
            plt.rcParams['font.weight'] = 'bold'
            
            fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 15), sharey=True)
            if len(methods) == 1:
                axes = [axes]
            
            fig.suptitle('Validation sets for RIs established by different algorithms', 
                         fontsize=24, y=0.95, fontweight='bold')
            
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
                ax.set_yticklabels(reversed_labels, fontsize=23, fontweight='bold')
                
                # 设置X轴
                ax.set_xlim(85, 100)
                ax.set_xlabel('Validation Percentage (%)', fontsize=20, fontweight='bold')
                ax.grid(axis='x', alpha=0.4, linestyle=':', linewidth=0.8)
                ax.set_title(f'{method}', fontsize=23, pad=25, fontweight='bold')
                
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
                
                # 在90%位置添加红色垂直虚线
                ax.axvline(x=90, color='red', linestyle='-.', linewidth=2.5, alpha=0.9)
                ax.text(90.2, 2, '90%', fontsize=27, fontweight='bold', color='red')
                
                # 设置纵坐标字体
                ax.tick_params(axis='y', which='major', labelsize=20)
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')
            
            # 添加图例
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='darkblue', markersize=15, 
                                         label='Male', markeredgecolor='black', markeredgewidth=1.2),
                              plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='darkred', markersize=15, 
                                         label='Female', markeredgecolor='black', markeredgewidth=1.2)]
            fig.legend(handles=legend_elements, title='Gender', 
                      loc='upper right', fontsize=20, title_fontsize=25,
                      bbox_to_anchor=(0.99, 0.97), frameon=True, fancybox=True, shadow=True)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            
            # 保存图片
            sci_quality_plot_path = os.path.join(output_base_path, 'SCI_quality_combined_methods_reference_interval_forest_plot.png')
            plt.savefig(sci_quality_plot_path, dpi=600, bbox_inches='tight')
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
    main()
