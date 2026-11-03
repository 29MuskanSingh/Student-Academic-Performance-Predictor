"""
Preprocessing and Featurization Pipeline Module
Builds Scikit-Learn transformers following strict ML best practices:
Fit exclusively on training data to prevent data leakage.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Explicit feature definitions
NUMERIC_FEATURES = [
    "attendance_rate",
    "study_hours_weekly",
    "previous_exam_marks",
    "assignment_completion_rate",
    "assignment_avg_score",
    "class_participation_score",
    "sleep_hours_daily"
]

ORDINAL_FEATURES = [
    "parent_education_level",
    "stress_level"
]

ORDINAL_CATEGORIES = [
    ["High School", "Diploma", "Bachelor", "Postgraduate"],  # parent_education_level
    ["Low", "Medium", "High"]                                 # stress_level
]

NOMINAL_FEATURES = [
    "extracurricular_activities",
    "internet_access",
    "tutoring_support"
]

TARGET_COL = "final_exam_marks"
TIER_COL = "performance_tier"
ID_COL = "student_id"

def build_preprocessor() -> ColumnTransformer:
    """
    Creates an end-to-end ColumnTransformer that handles:
    1. Numerical imputation (median) and standardization (Z-score).
    2. Ordinal imputation (mode) and ranked integer mapping with scaling.
    3. Nominal categorical imputation and one-hot encoding (drop='first').
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ORDINAL_CATEGORIES)),
        ("scaler", StandardScaler())
    ])

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("nom", nominal_transformer, NOMINAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts the names of all transformed features after fitting.
    """
    feature_names = []
    
    # Numeric
    feature_names.extend(NUMERIC_FEATURES)
    
    # Ordinal
    feature_names.extend(ORDINAL_FEATURES)
    
    # Nominal
    nom_encoder = preprocessor.named_transformers_["nom"].named_steps["encoder"]
    encoded_nom = nom_encoder.get_feature_names_out(NOMINAL_FEATURES).tolist()
    feature_names.extend(encoded_nom)
    
    return feature_names

def load_and_split_data(csv_path: str, test_size: float = 0.20, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw dataset and splits into Train and Test partitions BEFORE featurization.
    Ensures strict separation to avoid data leakage.
    """
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[ID_COL, TARGET_COL, TIER_COL], errors="ignore")
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
