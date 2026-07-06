---
name: medical-reference-interval-analysis
description: A comprehensive tool for calculating and validating medical reference intervals (RIs). It supports automatic data cleaning, outlier detection, gender/age partitioning, and robust statistical analysis.
version: 1.0.0
usage:
  - "Analyze reference intervals for a dataset: `python analysis.py`"
  - "The script automatically processes all .xlsx files in the current directory."
---

# Medical Reference Interval Analysis Skill

This skill provides a robust pipeline for establishing and verifying medical reference intervals from laboratory data. It automates the complex statistical procedures required by CLSI guidelines, making it easier for laboratory professionals to determine accurate reference ranges.

## Features

1.  **Automated Data Loading**: Automatically detects and processes all Excel (`.xlsx`) files in the source directory.
2.  **Data Preprocessing**: Handles column renaming (Sex->gender, Age->age), type conversion, and missing value removal.
3.  **Advanced Outlier Detection**:
    *   Tukey's Fences (IQR method)
    *   Z-Score
    *   Gaussian Mixture Models (GMM)
    *   Isolation Forest
    *   Local Outlier Factor (LOF)
4.  **Intelligent Partitioning**:
    *   **Gender**: Uses Harris-Boyd method (Z-test) to determine if separate intervals are needed for males and females.
    *   **Age**: Implements an iterative algorithm to group ages based on minimum sample size and effect size (Cohen's d).
5.  **Reference Interval Calculation**: Uses the robust non-parametric percentile method (2.5th and 97.5th percentiles) with Bootstrap confidence intervals (90% CI).
6.  **Visualization**: Generates comprehensive plots:
    *   Age vs. Result scatter plots with GAM (Generalized Additive Model) trends.
    *   Kernel Density Estimation (KDE) distributions.
    *   Forest plots for comparing reference intervals across groups.
7.  **Internal Validation**: Performs a second pass of analysis after removing detected outliers to refine the intervals.

## Usage

1.  Place your data files (`.xlsx`) in the directory where the script is run. The files should contain at least `sex` (or `gender`), `age`, and `result` columns.
2.  Run the analysis script.
3.  Results will be saved in a new `results_YYYYMMDD_HHMMSS` directory, containing:
    *   `analysis_summary.xlsx`: Detailed statistical results.
    *   `reference_intervals.xlsx`: Calculated reference intervals.
    *   `*.png`: Visualization plots.

## Requirements

*   pandas
*   numpy
*   scipy
*   matplotlib
*   seaborn
*   scikit-learn
*   pygam
*   openpyxl
