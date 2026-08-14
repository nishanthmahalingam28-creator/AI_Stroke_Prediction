# Dataset

## Source
**Stroke Prediction Dataset** by fedesoriano, hosted on Kaggle:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

File used: `healthcare-dataset-stroke-data.csv`

This is **real, anonymized patient data** (not synthetic). It is one of the
most widely used and cited stroke-risk datasets for educational ML projects.

## Basic facts
- **Rows:** 5,110
- **Columns:** 12 (11 features + 1 target)
- **Target column:** `stroke` (0 = no stroke, 1 = had a stroke)
- **Class balance:** 4,861 negative (95.1%) vs 249 positive (4.9%) — imbalanced
- **Missing values:** only `bmi` (201 rows, 3.9%) — not used in this project anyway

## Columns in the raw file

| Column | Type | Values |
|---|---|---|
| id | int | unique row id (dropped, not a feature) |
| gender | categorical | Male, Female, Other |
| age | numeric | 0.08–82 |
| hypertension | binary | 0/1 |
| heart_disease | binary | 0/1 |
| ever_married | categorical | Yes, No |
| work_type | categorical | Private, Self-employed, Govt_job, children, Never_worked |
| Residence_type | categorical | Urban, Rural |
| avg_glucose_level | numeric | lab measurement — **not used** (project requires no lab tests) |
| bmi | numeric | lab measurement — **not used** (project requires no lab tests) |
| smoking_status | categorical | formerly smoked, never smoked, smokes, Unknown |
| stroke | binary | **target** |

## Features actually used in this project

`age`, `gender`, `hypertension`, `heart_disease`, `smoking_status`,
`ever_married`, `work_type`, `Residence_type`

`avg_glucose_level` and `bmi` are excluded because they require lab
measurements/medical equipment, which this project's user-friendly
questionnaire explicitly avoids.

## Known limitations
- Small positive class (only 249 stroke cases) — model recall will be
  imperfect; this is disclosed to the end user as an educational estimate,
  not a diagnosis.
- `gender = "Other"` has only 1 row — too sparse to model reliably; handled
  with an explicit fallback in preprocessing.
- Dataset does not include diabetes, alcohol use, physical activity, diet,
  stress level, or family history — so the questionnaire does not ask about
  these, to avoid implying a model relationship that doesn't exist.
- This dataset reflects the population it was collected from and may not
  generalize to all demographics.

## License / usage
Distributed on Kaggle for educational and research use. See the Kaggle
dataset page for the author's full terms before any non-educational use.
