"""
Exploratory Data Analysis (EDA) Module
Analyzes student performance distributions, correlations, missing values,
and saves diagnostic visualizations to reports/figures/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure aesthetic matplotlib styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.autolayout": True
})

def perform_eda(csv_path: Path, output_dir: Path) -> dict:
    """
    Executes comprehensive EDA and generates visualization charts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # 1. Basic Health & Profiling
    total_records = len(df)
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {k: round(v / total_records * 100, 2) for k, v in missing_counts.items() if v > 0}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "student_id" in categorical_cols:
        categorical_cols.remove("student_id")
    if "performance_tier" in categorical_cols:
        categorical_cols.remove("performance_tier")
        
    stats_dict = df[numeric_cols].describe().round(2).to_dict()
    
    # Check Outliers via IQR
    outlier_counts = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
        
    # Correlations with target
    correlations = df[numeric_cols].corr()["final_exam_marks"].sort_values(ascending=False).round(4).to_dict()
    
    # ==========================================
    # Visualizations
    # ==========================================
    
    # Plot 1: Target Variable Distribution & Performance Tiers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df["final_exam_marks"].dropna(), kde=True, color="#2b5c8f", ax=axes[0], bins=25)
    axes[0].axvline(50, color="#d9534f", linestyle="--", linewidth=1.5, label="Pass Cutoff (50)")
    axes[0].axvline(80, color="#5cb85c", linestyle="--", linewidth=1.5, label="Distinction (80)")
    axes[0].set_title("Distribution of Final Exam Marks")
    axes[0].set_xlabel("Final Exam Marks (0 - 100)")
    axes[0].set_ylabel("Student Count")
    axes[0].legend(loc="upper left")
    
    tier_order = ["At-Risk", "Pass", "Good", "Distinction"]
    sns.barplot(x=tier_counts.index, y=tier_counts.values, hue=tier_counts.index, palette="Blues_d", legend=False, ax=axes[1])
    axes[1].set_title("Distribution of Performance Tiers")
    axes[1].set_xlabel("Academic Performance Standing")
    axes[1].set_ylabel("Number of Students")
    for i, v in enumerate(tier_counts.values):
        axes[1].text(i, v + 8, f"{v} ({v/total_records*100:.1f}%)", ha="center", fontweight="semibold")
        
    plt.tight_layout()
    fig1_path = figures_dir / "eda_1_target_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.8}, vmin=-0.2, vmax=1.0)
    plt.title("Correlation Heatmap of Academic & Behavioural Factors")
    plt.tight_layout()
    fig2_path = figures_dir / "eda_2_correlation_matrix.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    # Plot 3: Key Scatter Relationships with Regression Trends
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    key_features = [
        ("previous_exam_marks", "Previous Exam Marks (Prior Aptitude)"),
        ("attendance_rate", "Class Attendance Rate (%)"),
        ("study_hours_weekly", "Weekly Independent Study Hours"),
        ("assignment_avg_score", "Assignment Average Score")
    ]
    
    for ax, (feat, label) in zip(axes.flatten(), key_features):
        sns.regplot(
            data=df,
            x=feat,
            y="final_exam_marks",
            ax=ax,
            scatter_kws={"alpha": 0.45, "color": "#2c7bb6", "s": 25},
            line_kws={"color": "#d7191c", "linewidth": 2}
        )
        r_val = correlations.get(feat, 0)
        ax.set_title(f"{label} vs Final Marks (r = {r_val:.2f})")
        ax.set_xlabel(label)
        ax.set_ylabel("Final Exam Marks")
        
    plt.tight_layout()
    fig3_path = figures_dir / "eda_3_key_relationships.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # Plot 4: Categorical Factor Influences
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cat_specs = [
        ("parent_education_level", "Parent Education Level", ["High School", "Diploma", "Bachelor", "Postgraduate"]),
        ("stress_level", "Perceived Stress Level", ["Low", "Medium", "High"]),
        ("tutoring_support", "Access to Tutoring Support", ["No", "Yes"]),
        ("internet_access", "Home Internet Access", ["No", "Yes"])
    ]
    
    for ax, (cat_col, title, order) in zip(axes.flatten(), cat_specs):
        sns.boxplot(data=df, x=cat_col, y="final_exam_marks", order=order, hue=cat_col, palette="Set2", legend=False, ax=ax)
        sns.stripplot(data=df, x=cat_col, y="final_exam_marks", order=order, color="black", alpha=0.15, jitter=0.2, size=3, ax=ax)
        ax.set_title(f"Impact of {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Final Exam Marks")
        
    plt.tight_layout()
    fig4_path = figures_dir / "eda_4_categorical_comparisons.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    
    # Plot 5: Sleep Hours vs Academic Performance (Cognitive Threshold Effect)
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x=np.round(df["sleep_hours_daily"]), y="final_exam_marks", marker="o", color="#e66101", linewidth=2.5, errorbar="ci")
    plt.title("Effect of Daily Sleep Duration on Student Performance")
    plt.xlabel("Average Daily Sleep Hours (Binned)")
    plt.ylabel("Mean Final Exam Marks (with 95% CI)")
    plt.axvspan(6.5, 8.5, color="green", alpha=0.1, label="Optimal Cognitive Rest Range (7-8 hrs)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig5_path = figures_dir / "eda_5_sleep_vs_performance.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    
    summary = {
        "total_records": total_records,
        "features_count": df.shape[1],
        "missing_values": missing_pct,
        "outlier_counts": outlier_counts,
        "correlations_with_target": correlations,
        "figures_generated": [
            str(fig1_path.name),
            str(fig2_path.name),
            str(fig3_path.name),
            str(fig4_path.name),
            str(fig5_path.name)
        ]
    }
    
    summary_path = output_dir / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "data" / "student_performance.csv"
    reports_dir = base_dir / "reports"
    
    results = perform_eda(data_file, reports_dir)
    print("EDA Execution Complete!")
    print(f"Missing Values detected: {results['missing_values']}")
    print(f"Top 5 Correlations with Final Marks:")
    for feat, corr in list(results['correlations_with_target'].items())[:6]:
        print(f"  - {feat}: {corr}")
    print(f"Charts generated in reports/figures/: {len(results['figures_generated'])}")
