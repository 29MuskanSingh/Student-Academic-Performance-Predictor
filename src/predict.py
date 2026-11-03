"""
Student Performance Prediction & Recommendation Engine
Provides single-sample and batch prediction capabilities using the trained pipeline.
Generates performance tier classifications and actionable educational recommendations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd

def get_performance_tier(score: float) -> Dict[str, str]:
    """
    Maps continuous mark (0-100) to academic standing and badge color.
    """
    if score >= 80.0:
        return {"tier": "Distinction", "status": "Excellent Proficiency", "badge": "success"}
    elif score >= 65.0:
        return {"tier": "Good", "status": "Above Average", "badge": "primary"}
    elif score >= 50.0:
        return {"tier": "Pass", "status": "Satisfactory Baseline", "badge": "warning"}
    else:
        return {"tier": "At-Risk", "status": "Requires Academic Support", "badge": "danger"}

def generate_recommendations(student_data: Dict[str, Any], predicted_score: float) -> list:
    """
    Generates personalized, evidence-grounded recommendations for academic improvement.
    """
    recommendations = []
    
    # 1. Attendance check
    att = float(student_data.get("attendance_rate", 100))
    if att < 75.0:
        recommendations.append(
            f"Attendance Alert: Current attendance ({att:.1f}%) is below institutional threshold (75%). "
            f"Improving attendance to 85%+ can directly improve performance and lecture engagement."
        )
    elif att < 85.0:
        recommendations.append(
            f"Attendance: Raising attendance from {att:.1f}% to 90%+ is projected to yield ~2-3 additional marks."
        )
        
    # 2. Study hours check
    study = float(student_data.get("study_hours_weekly", 10))
    if study < 10.0:
        recommendations.append(
            f"Study Habits: Weekly study time ({study:.1f} hrs) is in the lower quartile. "
            f"Increasing independent study to 14-16 hrs/week provides the highest positive marginal score boost."
        )
    elif study > 26.0:
        recommendations.append(
            f"Diminishing Returns: High study load ({study:.1f} hrs/week). "
            f"Focus on active recall and spaced repetition rather than increasing hours to prevent burnout."
        )
        
    # 3. Sleep check
    sleep = float(student_data.get("sleep_hours_daily", 7))
    if sleep < 6.0:
        recommendations.append(
            f"Cognitive Rest Warning: Sleep duration ({sleep:.1f} hrs/day) indicates chronic sleep deficit. "
            f"Targeting 7-8 hours improves memory retention and cognitive exam focus."
        )
        
    # 4. Assignment performance
    assign_score = float(student_data.get("assignment_avg_score", 70))
    if assign_score < 60.0:
        recommendations.append(
            f"Coursework Intervention: Assignment average ({assign_score:.1f}) is lagging. "
            f"Utilize instructor office hours and collaborative study groups on upcoming homework."
        )
        
    # 5. Stress Level & Tutoring
    stress = str(student_data.get("stress_level", "Medium"))
    tutoring = str(student_data.get("tutoring_support", "No"))
    if stress == "High" and tutoring == "No":
        recommendations.append(
            "Academic Well-being: Student reports High Stress without tutoring support. "
            "Enrolling in peer tutoring or study workshops can reduce exam anxiety and boost confidence."
        )
        
    if not recommendations:
        recommendations.append("Performance Profile: Excellent academic habits! Maintain current study routine and rest schedule.")
        
    return recommendations

class StudentPredictor:
    def __init__(self, model_path: Union[str, Path] = None):
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_path = base_dir / "models" / "best_model.joblib"
            
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please run train.py first.")
            
        self.pipeline = joblib.load(self.model_path)
        
        # Load metadata if present
        meta_path = self.model_path.parent / "feature_metadata.json"
        self.metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
                
    def predict_student(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts performance for a single student profile.
        """
        df_input = pd.DataFrame([student_dict])
        pred_val = float(self.pipeline.predict(df_input)[0])
        pred_val = float(np.clip(pred_val, 0.0, 100.0))
        
        tier_info = get_performance_tier(pred_val)
        
        # Approximate 95% Prediction Interval (based on Test RMSE ~ 4.1 marks)
        rmse_est = 4.11
        ci_lower = max(0.0, round(pred_val - 1.96 * rmse_est, 1))
        ci_upper = min(100.0, round(pred_val + 1.96 * rmse_est, 1))
        
        recs = generate_recommendations(student_dict, pred_val)
        
        return {
            "predicted_score": round(pred_val, 1),
            "prediction_interval_95": [ci_lower, ci_upper],
            "tier": tier_info["tier"],
            "status": tier_info["status"],
            "badge": tier_info["badge"],
            "recommendations": recs
        }

    def predict_batch(self, csv_input_path: Union[str, Path]) -> pd.DataFrame:
        """
        Performs batch inference on a CSV file and appends prediction columns.
        """
        df = pd.read_csv(csv_input_path)
        preds = self.pipeline.predict(df)
        preds = np.clip(preds, 0.0, 100.0)
        
        df["predicted_final_marks"] = np.round(preds, 1)
        df["predicted_tier"] = [get_performance_tier(p)["tier"] for p in preds]
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Student Academic Performance")
    parser.add_argument("--attendance", type=float, default=85.0, help="Attendance Rate (50-100)")
    parser.add_argument("--study_hours", type=float, default=14.0, help="Weekly Study Hours (2-35)")
    parser.add_argument("--prev_marks", type=float, default=70.0, help="Previous Exam Marks (0-100)")
    parser.add_argument("--assign_comp", type=float, default=90.0, help="Assignment Completion Rate %")
    parser.add_argument("--assign_score", type=float, default=75.0, help="Assignment Average Score (0-100)")
    parser.add_argument("--participation", type=int, default=7, help="Class Participation Score (1-10)")
    parser.add_argument("--sleep", type=float, default=7.5, help="Daily Sleep Hours (4-10)")
    parser.add_argument("--parent_edu", type=str, default="Bachelor", choices=["High School", "Diploma", "Bachelor", "Postgraduate"])
    parser.add_argument("--stress", type=str, default="Medium", choices=["Low", "Medium", "High"])
    parser.add_argument("--extracurricular", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--internet", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--tutoring", type=str, default="No", choices=["Yes", "No"])
    
    args = parser.parse_args()
    
    sample_input = {
        "attendance_rate": args.attendance,
        "study_hours_weekly": args.study_hours,
        "previous_exam_marks": args.prev_marks,
        "assignment_completion_rate": args.assign_comp,
        "assignment_avg_score": args.assign_score,
        "class_participation_score": args.participation,
        "sleep_hours_daily": args.sleep,
        "parent_education_level": args.parent_edu,
        "stress_level": args.stress,
        "extracurricular_activities": args.extracurricular,
        "internet_access": args.internet,
        "tutoring_support": args.tutoring
    }
    
    predictor = StudentPredictor()
    result = predictor.predict_student(sample_input)
    
    print("\n" + "=" * 55)
    print("        STUDENT PERFORMANCE PREDICTION SYSTEM        ")
    print("=" * 55)
    print(f"Predicted Final Marks : {result['predicted_score']} / 100")
    print(f"95% Prediction Range  : {result['prediction_interval_95'][0]} - {result['prediction_interval_95'][1]}")
    print(f"Performance Tier      : {result['tier']} ({result['status']})")
    print("-" * 55)
    print("Personalized Actionable Recommendations:")
    for idx, rec in enumerate(result['recommendations'], 1):
        print(f" {idx}. {rec}")
    print("=" * 55 + "\n")
