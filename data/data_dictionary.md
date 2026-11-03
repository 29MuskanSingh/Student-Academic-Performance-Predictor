# Student Academic Performance Data Dictionary

This dataset is designed to model and predict student final academic marks and performance classifications based on multi-dimensional academic, behavioural, cognitive, and socio-contextual attributes.

## Dataset Summary
- **Total Records**: 1,200
- **Total Features**: 15 (1 identifier, 11 predictor features, 2 target representations)
- **File Format**: CSV (`data/student_performance.csv`)

---

## Variable Definitions

| Variable Name | Type | Range / Categories | Description | Educational Significance |
| :--- | :--- | :--- | :--- | :--- |
| `student_id` | Identifier | `STU1000` - `STU2199` | Unique anonymized identifier for each student | Excluded from modeling to avoid identity bias |
| `attendance_rate` | Numeric (Float) | 45.0% - 100.0% | Percentage of mandatory classes attended | Direct proxy for exposure to instruction and discipline |
| `study_hours_weekly` | Numeric (Float) | 2.0 - 36.0 hrs | Self-directed independent study hours per week | Quantitative measure of academic effort |
| `previous_exam_marks` | Numeric (Float) | 25.0 - 99.0 marks | Cumulative marks from prior semester / midterm exam | Baseline indicator of academic aptitude and subject mastery |
| `assignment_completion_rate`| Numeric (Float) | 40.0% - 100.0% | Percentage of scheduled assignments submitted | Behavioral measure of consistency and time management |
| `assignment_avg_score` | Numeric (Float) | 20.0 - 100.0 marks | Average score obtained on submitted coursework | Continuous formative assessment indicator |
| `class_participation_score`| Numeric (Integer)| 1 - 10 | Instructor-assessed in-class engagement and discussions | Active learning metric |
| `sleep_hours_daily` | Numeric (Float) | 4.0 - 10.0 hrs | Average nightly sleep duration | Cognitive restorative metric (non-linear threshold impact) |
| `extracurricular_activities`| Categorical | `Yes`, `No` | Active involvement in athletics, clubs, or arts | Holistic student development indicator |
| `internet_access` | Categorical | `Yes`, `No` | Reliable high-speed internet connectivity at home | Socio-digital resource equity factor |
| `tutoring_support` | Categorical | `Yes`, `No` | Access to remedial or supplementary academic tutoring | Targeted academic intervention factor |
| `parent_education_level` | Categorical (Ordinal)| `High School`, `Diploma`, `Bachelor`, `Postgraduate` | Highest educational level attained by parents | Socioeconomic and home academic support proxy |
| `stress_level` | Categorical (Ordinal)| `Low`, `Medium`, `High` | Self-reported perceived psychological stress | Affective well-being factor affecting exam performance |
| **`final_exam_marks`** | **Target (Continuous)**| **20.0 - 100.0** | **End-of-term comprehensive examination marks** | **Primary target variable for Regression models** |
| **`performance_tier`** | **Target (Categorical)**| **`At-Risk`, `Pass`, `Good`, `Distinction`** | **Academic performance standing band** | **Target variable for Multi-class / Risk categorization** |

---

## Grading Scheme for `performance_tier`
- **At-Risk**: Marks `< 50.0` (Requires immediate academic intervention)
- **Pass**: Marks `50.0 - 64.9` (Satisfactory baseline competency)
- **Good**: Marks `65.0 - 79.9` (Above-average proficiency)
- **Distinction**: Marks `>= 80.0` (High academic achievement)
