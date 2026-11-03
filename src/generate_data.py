"""
Synthetic Student Performance Dataset Generator
Generates realistic academic, behavioural, and socio-educational student data
grounded in empirical educational literature.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_student_dataset(n_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic student academic performance dataset.
    
    Features include:
      - attendance_rate: % of classes attended (50% - 100%)
      - study_hours_weekly: Study hours outside classroom (2 - 35 hrs/wk)
      - previous_exam_marks: Prior semester/midterm marks (0 - 100)
      - assignment_completion_rate: % of assignments submitted (40% - 100%)
      - assignment_avg_score: Average marks achieved on assignments (0 - 100)
      - class_participation_score: Engagement rating (1 - 10)
      - sleep_hours_daily: Average sleep hours per night (4 - 10 hrs)
      - extracurricular_activities: 'Yes' / 'No'
      - internet_access: 'Yes' / 'No'
      - parent_education_level: 'High School', 'Diploma', 'Bachelor', 'Postgraduate'
      - stress_level: 'Low', 'Medium', 'High'
      - tutoring_support: 'Yes' / 'No'
      - final_exam_marks: Target variable (0 - 100 continuous score)
      - performance_tier: Categorical performance tier ('At-Risk', 'Pass', 'Good', 'Distinction')
    """
    np.random.seed(random_seed)
    
    # 1. Base academic & behavioural distributions
    attendance_rate = np.clip(np.random.beta(a=7, b=2, size=n_samples) * 100, 45.0, 100.0)
    
    study_hours_weekly = np.clip(np.random.gamma(shape=3.5, scale=3.2, size=n_samples), 2.0, 36.0)
    
    # Prior marks with moderate positive correlation to study habits
    base_ability = np.random.normal(loc=65.0, scale=12.0, size=n_samples)
    previous_exam_marks = np.clip(
        0.55 * base_ability + 0.35 * (attendance_rate * 0.7) + 0.5 * (study_hours_weekly * 1.5) + np.random.normal(0, 7, n_samples),
        25.0, 99.0
    )
    
    assignment_completion_rate = np.clip(
        0.7 * attendance_rate + 0.3 * (study_hours_weekly * 3) + np.random.normal(0, 8, n_samples),
        40.0, 100.0
    )
    
    assignment_avg_score = np.clip(
        0.5 * previous_exam_marks + 0.3 * assignment_completion_rate + 0.2 * (study_hours_weekly * 2) + np.random.normal(0, 6, n_samples),
        20.0, 100.0
    )
    
    class_participation_score = np.clip(
        np.round(0.06 * attendance_rate + 0.04 * (study_hours_weekly) + np.random.normal(0, 1.2, n_samples)),
        1, 10
    ).astype(int)
    
    sleep_hours_daily = np.clip(np.random.normal(loc=7.1, scale=1.1, size=n_samples), 4.0, 10.0)
    
    # Categorical and socio-contextual factors
    extracurricular_activities = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.55, 0.45])
    internet_access = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.88, 0.12])
    tutoring_support = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.32, 0.68])
    
    parent_education_choices = ['High School', 'Diploma', 'Bachelor', 'Postgraduate']
    parent_education_level = np.random.choice(parent_education_choices, size=n_samples, p=[0.28, 0.22, 0.35, 0.15])
    
    stress_choices = ['Low', 'Medium', 'High']
    stress_level = np.random.choice(stress_choices, size=n_samples, p=[0.25, 0.50, 0.25])
    
    # 2. Latent Target Calculation (Real-world non-linear & synergistic effects)
    # Factor weights based on educational performance research
    w_prev = 0.32
    w_assign = 0.22
    w_attend = 0.18
    w_study = 0.16
    w_part = 0.07
    
    excess_study = np.maximum(0.0, study_hours_weekly - 18.0)
    effective_study_hours = np.where(study_hours_weekly <= 18, study_hours_weekly, 18.0 + 4.0 * np.log1p(excess_study))
    study_norm = (effective_study_hours / 24.0) * 100.0
    
    # Sleep penalty: under 6 hours impairs performance, >9 slight drowsiness
    sleep_effect = np.where(sleep_hours_daily < 6.0, -2.5 * (6.0 - sleep_hours_daily), 
                   np.where(sleep_hours_daily > 8.5, -0.8 * (sleep_hours_daily - 8.5), 1.5))
    
    # Socio-environmental bonus/penalties
    parent_bonus = pd.Series(parent_education_level).map({
        'High School': 0.0,
        'Diploma': 1.0,
        'Bachelor': 2.5,
        'Postgraduate': 3.5
    }).to_numpy()
    
    internet_effect = np.where(internet_access == 'Yes', 1.8, -3.0)
    tutoring_effect = np.where(tutoring_support == 'Yes', 2.2, 0.0)
    stress_effect = pd.Series(stress_level).map({'Low': 1.0, 'Medium': 0.5, 'High': -3.2}).to_numpy()
    extra_effect = np.where(extracurricular_activities == 'Yes', 0.8, 0.0)
    
    # Core score formulation
    raw_final_score = (
        w_prev * previous_exam_marks +
        w_assign * assignment_avg_score +
        w_attend * attendance_rate +
        w_study * study_norm +
        w_part * (class_participation_score * 10.0) +
        sleep_effect +
        parent_bonus +
        internet_effect +
        tutoring_effect +
        stress_effect +
        extra_effect +
        np.random.normal(loc=0.0, scale=4.0, size=n_samples) # Natural variance/unobserved factors
    )
    
    final_exam_marks = np.clip(np.round(raw_final_score, 1), 20.0, 100.0)
    
    # Performance categorization
    # <50: At-Risk, 50-64: Pass, 65-79: Good, >=80: Distinction
    performance_tiers = []
    for score in final_exam_marks:
        if score < 50.0:
            performance_tiers.append('At-Risk')
        elif score < 65.0:
            performance_tiers.append('Pass')
        elif score < 80.0:
            performance_tiers.append('Good')
        else:
            performance_tiers.append('Distinction')
            
    df = pd.DataFrame({
        'student_id': [f'STU{1000 + i}' for i in range(n_samples)],
        'attendance_rate': np.round(attendance_rate, 1),
        'study_hours_weekly': np.round(study_hours_weekly, 1),
        'previous_exam_marks': np.round(previous_exam_marks, 1),
        'assignment_completion_rate': np.round(assignment_completion_rate, 1),
        'assignment_avg_score': np.round(assignment_avg_score, 1),
        'class_participation_score': class_participation_score,
        'sleep_hours_daily': np.round(sleep_hours_daily, 1),
        'extracurricular_activities': extracurricular_activities,
        'internet_access': internet_access,
        'tutoring_support': tutoring_support,
        'parent_education_level': parent_education_level,
        'stress_level': stress_level,
        'final_exam_marks': final_exam_marks,
        'performance_tier': performance_tiers
    })
    
    # Introduce controlled realistic missing values (~2-3%) for data cleaning practice
    missing_indices_study = np.random.choice(n_samples, size=int(n_samples * 0.025), replace=False)
    missing_indices_assign = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    missing_indices_part = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    
    df.loc[missing_indices_study, 'study_hours_weekly'] = np.nan
    df.loc[missing_indices_assign, 'assignment_avg_score'] = np.nan
    df.loc[missing_indices_part, 'class_participation_score'] = np.nan
    
    return df

if __name__ == '__main__':
    output_dir = Path(__file__).resolve().parent.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'student_performance.csv'
    
    dataset = generate_student_dataset(n_samples=1200, random_seed=42)
    dataset.to_csv(csv_path, index=False)
    print(f"Dataset generated successfully at: {csv_path}")
    print(f"Total records: {len(dataset)}, Total features: {dataset.shape[1]}")
    print(dataset.head(3))
