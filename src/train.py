"""
Model Training, Cross-Validation, Evaluation, and Diagnostic Interpretability Module.
Follows rigorous ML best practices:
- Preprocessing fit strictly on training set.
- Comparison against naive baseline.
- 5-Fold Cross-Validation.
- Full residual analysis and feature importance extraction.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

from preprocess import build_preprocessor, get_feature_names, load_and_split_data

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.autolayout": True
})

def evaluate_models(csv_path: Path, models_dir: Path, figures_dir: Path) -> dict:
    """
    Trains, validates, tests, and interprets multiple ML models.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Strict Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = load_and_split_data(str(csv_path), test_size=0.20, random_state=42)
    print(f"Data Split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")
    
    # Define candidate models
    candidate_algorithms = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted_pipelines = {}
    
    print("\n" + "=" * 65)
    print(f"{'Model':<22} | {'CV R² (mean±std)':<18} | {'Test MAE':<10} | {'Test R²':<8}")
    print("=" * 65)
    
    best_model_name = None
    best_test_r2 = -float("inf")
    
    for name, estimator in candidate_algorithms.items():
        # Pipeline: Preprocessor -> Estimator
        preprocessor = build_preprocessor()
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator)
        ])
        
        # 5-Fold Cross Validation on Train Set
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_kfold, scoring="r2", n_jobs=-1)
        
        # Fit on entire Training Set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Predict on Test Set
        y_pred = pipe.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        results[name] = {
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_r2": float(r2),
            "test_mape": float(mape)
        }
        
        print(f"{name:<22} | {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}       | {mae:<10.3f} | {r2:<8.3f}")
        
        if r2 > best_test_r2:
            best_test_r2 = r2
            best_model_name = name
            
    print("=" * 65)
    print(f"Top Performing Model: {best_model_name} (Test R² = {best_test_r2:.4f})")
    
    best_pipe = fitted_pipelines[best_model_name]
    
    # Save best pipeline
    model_save_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipe, model_save_path)
    print(f"Serialized best model pipeline saved to: {model_save_path}")
    
    # Save metrics JSON
    metrics_path = models_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "models_comparison": results
        }, f, indent=2)
        
    # =========================================================
    # Visualizations
    # =========================================================
    
    # 1. Model Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = list(results.keys())
    mae_vals = [results[m]["test_mae"] for m in model_names]
    r2_vals = [results[m]["test_r2"] for m in model_names]
    
    sns.barplot(x=mae_vals, y=model_names, hue=model_names, palette="rocket_r", legend=False, ax=axes[0])
    axes[0].set_title("Test Mean Absolute Error (MAE - Lower is Better)")
    axes[0].set_xlabel("MAE (Marks)")
    for i, v in enumerate(mae_vals):
        axes[0].text(v + 0.15, i, f"{v:.2f}", va="center", fontweight="semibold")
        
    sns.barplot(x=r2_vals, y=model_names, hue=model_names, palette="viridis", legend=False, ax=axes[1])
    axes[1].set_title("Test R² Score (Variance Explained - Higher is Better)")
    axes[1].set_xlabel("R² Score")
    for i, v in enumerate(r2_vals):
        axes[1].text(max(0.02, v + 0.02), i, f"{v:.3f}", va="center", fontweight="semibold")
        
    plt.tight_layout()
    fig1_path = figures_dir / "eval_1_model_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # 2. Actual vs Predicted Plot for Best Model
    y_test_pred = best_pipe.predict(X_test)
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_test_pred, alpha=0.6, color="#1f77b4", edgecolors="w", s=50, label="Test Predictions")
    min_val = min(y_test.min(), y_test_pred.min()) - 2
    max_val = max(y_test.max(), y_test_pred.max()) + 2
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (1:1 Line)")
    plt.title(f"Actual vs Predicted Marks: {best_model_name} (R² = {best_test_r2:.3f})")
    plt.xlabel("Actual Exam Marks")
    plt.ylabel("Predicted Exam Marks")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig2_path = figures_dir / "eval_2_actual_vs_predicted.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    # 3. Residual Diagnostics
    residuals = y_test - y_test_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(residuals, kde=True, color="#2b5c8f", ax=axes[0], bins=20)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Residuals Distribution (Mean = {residuals.mean():.2f})")
    axes[0].set_xlabel("Residual Error (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    
    axes[1].scatter(y_test_pred, residuals, alpha=0.55, color="#e6550d", edgecolors="k", linewidth=0.5)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted Values (Homoscedasticity Check)")
    axes[1].set_xlabel("Predicted Exam Marks")
    axes[1].set_ylabel("Residuals")
    
    plt.tight_layout()
    fig3_path = figures_dir / "eval_3_residual_analysis.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # 4. Feature Importance Extraction
    transformed_feature_names = get_feature_names(best_pipe.named_steps["preprocessor"])
    
    # Permutation Importance on Test Set (Model-agnostic and unbiased)
    perm_importance = permutation_importance(best_pipe, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
    raw_features = X_test.columns.tolist()
    perm_mean = perm_importance.importances_mean
    perm_df = pd.DataFrame({
        "feature": raw_features,
        "importance": perm_mean
    }).sort_values(by="importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=perm_df, x="importance", y="feature", hue="feature", palette="Blues_r", legend=False)
    plt.title(f"Feature Importance (Permutation Importance on Test Set)")
    plt.xlabel("Permutation Importance Score (Drop in R² upon shuffling)")
    plt.ylabel("Student Attribute")
    for i, row in enumerate(perm_df.itertuples()):
        plt.text(row.importance + 0.005, i, f"{row.importance:.3f}", va="center", fontweight="semibold")
        
    plt.tight_layout()
    fig4_path = figures_dir / "eval_4_feature_importance.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    
    # Linear Regression standardized coefficients for directionality
    lr_pipe = fitted_pipelines["Linear Regression"]
    lr_coefs = lr_pipe.named_steps["regressor"].coef_
    coef_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "coefficient": lr_coefs
    }).sort_values(by="coefficient", key=abs, ascending=False)
    
    # Save feature importance metadata
    feat_meta_path = models_dir / "feature_metadata.json"
    with open(feat_meta_path, "w") as f:
        json.dump({
            "permutation_importance": perm_df.to_dict(orient="records"),
            "linear_coefficients": coef_df.to_dict(orient="records"),
            "transformed_feature_names": transformed_feature_names
        }, f, indent=2)
        
    print(f"Feature metadata saved to: {feat_meta_path}")
    print(f"Evaluation figures saved to: {figures_dir}")
    
    return {
        "best_model": best_model_name,
        "metrics": results[best_model_name],
        "top_features": perm_df.head(5).to_dict(orient="records")
    }

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    csv_file = base_dir / "data" / "student_performance.csv"
    models_path = base_dir / "models"
    figures_path = base_dir / "reports" / "figures"
    
    eval_summary = evaluate_models(csv_file, models_path, figures_path)
    print("\nTraining & Evaluation Pipeline Finished Successfully!")
