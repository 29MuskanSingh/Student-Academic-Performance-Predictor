

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Local imports
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from preprocess import build_preprocessor, get_feature_names

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "figure.autolayout": True})

def print_section(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()} ")
    print("=" * 75)

def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "student_performance.csv"
    figures_dir = base_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. LOAD DATASET & BASIC INSPECTION
    # =========================================================================
    print_section("1. Loading Dataset & Basic Inspection")
    df = pd.read_csv(data_path)
    print(f"Total Rows (Students): {df.shape[0]}")
    print(f"Total Columns (Features): {df.shape[1]}")
    
    print("\n--- df.head(5) ---")
    print(df.head(5).to_string())

    print("\n--- df.tail(5) ---")
    print(df.tail(5).to_string())

    print("\n--- df.info() Summary ---")
    df.info()

    # =========================================================================
    # 2. MISSING VALUES & CLEANING
    # =========================================================================
    print_section("2. Data Quality & Missing Value Inspection")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        print("Missing values detected:")
        for col, count in missing_data.items():
            pct = (count / len(df)) * 100
            print(f"  * {col:<28}: {count} missing ({pct:.1f}%) -> Handled via Median/Mode imputation in pipeline")
    else:
        print("No missing values detected.")

    # =========================================================================
    # 3. STATISTICAL SUMMARY (df.describe())
    # =========================================================================
    print_section("3. Statistical Summary: df.describe()")
    print(df.describe().round(2).to_string())

    # =========================================================================
    # 4. EXPLORATORY DATA VISUALIZATIONS
    # =========================================================================
    print_section("4. Generating Exploratory Data Analysis (EDA) Graphs")
    print("Opening and saving graphs:")

    # Graph 1: Distribution of Final Marks & Performance Tier
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["final_exam_marks"].dropna(), kde=True, color="#1f77b4", ax=axes[0], bins=25)
    axes[0].axvline(50, color="red", linestyle="--", label="Pass Cutoff (50)")
    axes[0].axvline(80, color="green", linestyle="--", label="Distinction (80)")
    axes[0].set_title("Distribution of Final Exam Marks")
    axes[0].set_xlabel("Marks (0-100)")
    axes[0].legend()

    tier_order = ["At-Risk", "Pass", "Good", "Distinction"]
    tier_counts = df["performance_tier"].value_counts().reindex(tier_order)
    sns.barplot(x=tier_counts.index, y=tier_counts.values, hue=tier_counts.index, palette="Blues_d", legend=False, ax=axes[1])
    axes[1].set_title("Student Count by Academic Performance Tier")
    axes[1].set_ylabel("Number of Students")
    for i, v in enumerate(tier_counts.values):
        axes[1].text(i, v + 8, f"{v}", ha="center", fontweight="bold")
    plt.savefig(figures_dir / "plot1_target_distribution.png", dpi=150)
    print("  [1/6] Saved & Displayed: Distribution Plot (Histogram & Performance Tier)")
    plt.show(block=False)
    plt.pause(1)

    # Graph 2: Correlation Heatmap
    plt.figure(figsize=(9, 7))
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.8}, vmin=-0.2, vmax=1.0)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.savefig(figures_dir / "plot2_correlation_heatmap.png", dpi=150)
    print("  [2/6] Saved & Displayed: Correlation Heatmap")
    plt.show(block=False)
    plt.pause(1)

    # Graph 3: Scatter Plots with Regression Trendlines
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.regplot(data=df, x="study_hours_weekly", y="final_exam_marks", ax=axes[0],
                scatter_kws={"alpha": 0.45, "color": "#2c7bb6"}, line_kws={"color": "red"})
    axes[0].set_title("Scatter Plot: Weekly Study Hours vs Final Marks")
    axes[0].set_xlabel("Weekly Study Hours")
    axes[0].set_ylabel("Final Exam Marks")

    sns.regplot(data=df, x="previous_exam_marks", y="final_exam_marks", ax=axes[1],
                scatter_kws={"alpha": 0.45, "color": "#2ca02c"}, line_kws={"color": "darkgreen"})
    axes[1].set_title("Scatter Plot: Previous Exam Marks vs Final Marks")
    axes[1].set_xlabel("Previous Exam Marks")
    axes[1].set_ylabel("Final Exam Marks")
    plt.savefig(figures_dir / "plot3_scatterplots.png", dpi=150)
    print("  [3/6] Saved & Displayed: Scatter Plots (Study Hours & Previous Marks vs Final Marks)")
    plt.show(block=False)
    plt.pause(1)

    # Graph 4: Box Plots for Categorical Features
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stress_order = ["Low", "Medium", "High"]
    sns.boxplot(data=df, x="stress_level", y="final_exam_marks", order=stress_order, hue="stress_level", palette="Set2", legend=False, ax=axes[0])
    sns.stripplot(data=df, x="stress_level", y="final_exam_marks", order=stress_order, color="black", alpha=0.15, jitter=0.2, ax=axes[0])
    axes[0].set_title("Box Plot: Perceived Stress Level vs Final Marks")
    axes[0].set_xlabel("Stress Level")
    axes[0].set_ylabel("Final Marks")

    parent_order = ["High School", "Diploma", "Bachelor", "Postgraduate"]
    sns.boxplot(data=df, x="parent_education_level", y="final_exam_marks", order=parent_order, hue="parent_education_level", palette="Pastel1", legend=False, ax=axes[1])
    axes[1].set_title("Box Plot: Parent Education Level vs Final Marks")
    axes[1].set_xlabel("Parent Education Level")
    axes[1].set_ylabel("Final Marks")
    plt.savefig(figures_dir / "plot4_boxplots.png", dpi=150)
    print("  [4/6] Saved & Displayed: Box Plots (Stress Level & Parent Education vs Final Marks)")
    plt.show(block=False)
    plt.pause(1)

    # =========================================================================
    # 5. TRAIN-TEST SPLIT & MODEL TRAINING
    # =========================================================================
    print_section("5. Train-Test Split (Zero Data Leakage)")
    X = df.drop(columns=["student_id", "final_exam_marks", "performance_tier"], errors="ignore")
    y = df["final_exam_marks"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training Features Shape (X_train): {X_train.shape} (80% of data)")
    print(f"Testing Features Shape (X_test)  : {X_test.shape}  (20% of data)")
    print(f"Training Labels Shape (y_train)  : {y_train.shape}")
    print(f"Testing Labels Shape (y_test)    : {y_test.shape}")

    # Build Pipeline & Fit Models
    from sklearn.pipeline import Pipeline
    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }

    print_section("6. Model Training & 5-Fold Cross-Validation Benchmark")
    print(f"{'Model Name':<22} | {'CV R² (Mean ± Std)':<20} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<8}")
    print("-" * 75)

    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fitted_pipelines = {}
    best_model_name = None
    best_r2 = -float("inf")

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model)
        ])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_kfold, scoring="r2", n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{name:<22} | {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}         | {mae:<10.3f} | {rmse:<10.3f} | {r2:<8.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print("-" * 75)
    print(f"--> BEST MODEL: {best_model_name} (Test R² = {best_r2:.4f}, Test MAE = {mean_absolute_error(y_test, fitted_pipelines[best_model_name].predict(X_test)):.2f} marks)")

    best_pipe = fitted_pipelines[best_model_name]
    y_test_pred = best_pipe.predict(X_test)

    # Graph 5: Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, color="#1f77b4", edgecolors="black", linewidths=0.5, label="Test Predictions")
    min_v, max_v = min(y_test.min(), y_test_pred.min()) - 2, max(y_test.max(), y_test_pred.max()) + 2
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Ideal Fit (100% Accuracy Line)")
    plt.title(f"Actual vs Predicted Marks: {best_model_name} (R² = {best_r2:.3f})")
    plt.xlabel("Actual Exam Marks")
    plt.ylabel("Predicted Exam Marks")
    plt.legend()
    plt.savefig(figures_dir / "plot5_actual_vs_predicted.png", dpi=150)
    print("  [5/6] Saved & Displayed: Actual vs Predicted Scatter Plot")
    plt.show(block=False)
    plt.pause(1)

    # Graph 6: Permutation Feature Importance
    print_section("7. Permutation Feature Importance (Top Predictors)")
    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=15, random_state=42, n_jobs=1)
    perm_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": perm.importances_mean
    }).sort_values(by="Importance", ascending=False)

    print(perm_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    sns.barplot(data=perm_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False)
    plt.title("Feature Importance (Permutation Drop in R² on Unseen Test Data)")
    plt.xlabel("Permutation Importance")
    plt.savefig(figures_dir / "plot6_feature_importance.png", dpi=150)
    print("  [6/6] Saved & Displayed: Feature Importance Horizontal Bar Chart")
    plt.show(block=False)
    plt.pause(1)

    # =========================================================================
    # 8. SAMPLE PREDICTION DEMO
    # =========================================================================
    print_section("8. Live Sample Prediction Demo (Test Set Student)")
    sample_idx = 0
    sample_features = X_test.iloc[[sample_idx]]
    actual_score = y_test.iloc[sample_idx]
    predicted_score = best_pipe.predict(sample_features)[0]

    print("Student Attributes:")
    for col, val in sample_features.iloc[0].items():
        print(f"  * {col:<26}: {val}")
    print("-" * 50)
    print(f"Actual Final Exam Mark    : {actual_score:.1f} / 100")
    print(f"Predicted Final Exam Mark : {predicted_score:.1f} / 100")
    print(f"Absolute Difference (Error) : {abs(actual_score - predicted_score):.1f} marks")
    print("=" * 75)
    print("Analysis Completed Successfully! All 6 plots saved in: reports/figures/")
    print("=" * 75)
    
    plt.show()

if __name__ == "__main__":
    main()
