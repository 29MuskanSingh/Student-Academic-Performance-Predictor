"""
Interactive User-Input Student Performance Predictor
Asks student attributes via terminal inputs and outputs expected marks & advice.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict import StudentPredictor

def get_input(prompt: str, cast_type=float, default=None):
    while True:
        try:
            val = input(f"{prompt} : ").strip()
            if not val and default is not None:
                return default
            return cast_type(val)
        except ValueError:
            print("  [!] Please enter a valid value.")

def main():
    print("\n" + "=" * 70)
    print("         STUDENT ACADEMIC PERFORMANCE PREDICTOR (LIVE INPUT)         ")
    print("=" * 70)
    print("Please enter the student's details below:\n")

    attendance = get_input("Enter Attendance Rate % (50 - 100)", float)
    study_hours = get_input("Enter Weekly Study Hours (2 - 35)", float)
    prev_marks = get_input("Enter Previous Exam Marks (25 - 100)", float)
    assign_score = get_input("Enter Assignment Average Marks (20 - 100)", float)
    assign_comp = get_input("Enter Assignment Completion % (40 - 100)", float)
    participation = get_input("Enter Class Participation Rating (1 - 10)", int)
    sleep = get_input("Enter Daily Sleep Hours (4 - 10)", float)

    # Categorical Inputs
    print("\nSelect Stress Level [1: Low, 2: Medium, 3: High]")
    stress_choice = get_input("Enter choice (1/2/3)", int, default=2)
    stress_map = {1: "Low", 2: "Medium", 3: "High"}
    stress = stress_map.get(stress_choice, "Medium")

    print("\nSelect Parent Education [1: High School, 2: Diploma, 3: Bachelor, 4: Postgraduate]")
    edu_choice = get_input("Enter choice (1/2/3/4)", int, default=3)
    edu_map = {1: "High School", 2: "Diploma", 3: "Bachelor", 4: "Postgraduate"}
    parent_edu = edu_map.get(edu_choice, "Bachelor")

    internet = "Yes" if get_input("Home Internet Access? (1: Yes, 2: No)", int, default=1) == 1 else "No"
    tutoring = "Yes" if get_input("Tutoring / Mentorship Support? (1: Yes, 2: No)", int, default=2) == 1 else "No"
    extra = "Yes" if get_input("Extracurricular Activities? (1: Yes, 2: No)", int, default=1) == 1 else "No"

    student_data = {
        "attendance_rate": attendance,
        "study_hours_weekly": study_hours,
        "previous_exam_marks": prev_marks,
        "assignment_completion_rate": assign_comp,
        "assignment_avg_score": assign_score,
        "class_participation_score": participation,
        "sleep_hours_daily": sleep,
        "parent_education_level": parent_edu,
        "stress_level": stress,
        "extracurricular_activities": extra,
        "internet_access": internet,
        "tutoring_support": tutoring
    }

    predictor = StudentPredictor()
    result = predictor.predict_student(student_data)

    print("\n" + "=" * 70)
    print("                         PREDICTION RESULT                           ")
    print("=" * 70)
    print(f"🎓 Predicted Final Exam Marks : {result['predicted_score']} / 100")
    print(f"📊 Academic Standing          : {result['tier']} ({result['status']})")
    print(f"🎯 95% Confidence Interval    : {result['prediction_interval_95'][0]} - {result['prediction_interval_95'][1]} marks")
    print("-" * 70)
    print("💡 Actionable Recommendations for this Student:")
    for idx, rec in enumerate(result['recommendations'], 1):
        print(f"  {idx}. {rec}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
