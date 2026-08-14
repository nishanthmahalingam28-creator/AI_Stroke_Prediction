# AI-Based User-Friendly Stroke Risk Prediction System

An educational machine-learning project that estimates stroke risk
from information a typical user already knows about themselves —
**no glucose test, no BMI measurement, and no blood pressure reading
required.**

> **This is an educational risk-screening prototype. It is NOT a
> medical diagnosis and must not be used as one.**

---

## Features

- Step-by-step, plain-language questionnaire (8 questions) that a
  non-technical user can complete in under a minute
- No lab tests or equipment needed — no BMI, glucose, or blood
  pressure reading is ever requested
- Logistic Regression model trained with SMOTENC to address the
  dataset's severe class imbalance (stroke cases are rare)
- REST API (Flask) with clean, consistent JSON responses and no
  leaked stack traces
- Client-side and server-side input validation, with clear
  human-readable error messages
- Animated risk gauge plus a LOW / MODERATE / HIGH display category
- Automated test suite covering the prediction layer and the API layer

## Technologies

- **Python** — core language for training, preprocessing, and the API
- **Pandas** / **NumPy** — data loading and manipulation
- **Scikit-learn** — Logistic Regression model and preprocessing pipeline
- **Imbalanced-learn** — SMOTENC oversampling (training only)
- **Flask** / **Flask-CORS** — REST API and local cross-origin support
- **HTML / CSS / JavaScript** — static, dependency-free frontend

## Architecture

```
Frontend (HTML/CSS/JS)
      ↓
Flask REST API
      ↓
Input validation
      ↓
Saved preprocessor (encoding/scaling)
      ↓
Trained Logistic Regression model
      ↓
Stroke risk probability
      ↓
LOW / MODERATE / HIGH display category
      ↓
Frontend result + gauge visualization
```

No database, authentication, or prediction history is used — each
request is stateless and independent.

---

## Project status

| Step | Description | Status |
|------|--------------|--------|
| 1 | Dataset | ✅ Complete |
| 2 | Data cleaning | ✅ Complete |
| 3 | Preprocessing + train/test split | ✅ Complete |
| 4 | Class imbalance handling (SMOTENC) | ✅ Complete |
| 5 | Model training + cross-validation | ✅ Complete |
| 6 | Detailed model evaluation | ✅ Complete |
| 7 | Final model + saved prediction pipeline | ✅ Complete |
| 8 | Flask REST API | ✅ Complete |
| 9 | Backend application layer | ✅ Complete |
| 10 | User-friendly frontend | ✅ Complete |
| 11 | Prediction-layer unit tests | ✅ Complete (this update) |

Steps 1–7 (dataset, features, model, evaluation metrics, saved model
artifacts) were **not modified** in this update. Step 8 only adds a
Flask API on top of the already-saved, already-tested model.

---

## The 8 production features

The model intentionally uses only information a person can self-report,
with no lab tests or equipment:

1. `age`
2. `gender`
3. `hypertension` (ever told by a doctor you have high blood pressure)
4. `heart_disease` (ever diagnosed with heart disease)
5. `ever_married`
6. `work_type`
7. `Residence_type`
8. `smoking_status`

**Not used:** `id`, `bmi`, `avg_glucose_level` — deliberately excluded
so the app never requires a medical test to get a risk estimate.

---

## Final model (Step 7)

- **Model:** Logistic Regression
- **Imbalance handling:** SMOTENC (`sampling_strategy=0.5`), applied
  **only** to the training data — never at prediction time
- **Decision threshold:** 0.50

### Step 6 independent test results (unchanged)

| Metric | Value |
|---|---|
| Accuracy | 83.46% |
| Precision | 18.85% |
| Recall | 72.00% |
| F1-score | 29.88% |
| Specificity | 84.05% |
| ROC-AUC | 83.93% |
| Average Precision | 23.05% |

Confusion matrix: TN = 817, FP = 155, FN = 14, TP = 36.

### Display risk categories (software-only, not clinical)

- `LOW` — probability < 0.30
- `MODERATE` — 0.30 ≤ probability < 0.50
- `HIGH` — probability ≥ 0.50

The binary prediction always uses the fixed threshold of 0.50,
regardless of the display category.

---

## Step 8 — Flask REST API

Step 8 wraps the saved Step 7 artifacts (`model/stroke_risk_model.pkl`,
`model/preprocessor.pkl`, `model/model_config.json`) in a small Flask
API. The model is loaded **once** at application startup — it is never
retrained and SMOTENC is never applied during prediction.

### Installation

```bash
pip install -r requirements.txt
```

### Running the API

```bash
python api/app.py
```

The server starts at `http://127.0.0.1:5000`.

### Health endpoint

```
GET /api/health
```

```json
{
    "status": "ok",
    "service": "stroke-risk-prediction-api",
    "model_loaded": true
}
```

### Prediction endpoint

```
POST /api/predict
Content-Type: application/json
```

#### Example request

```json
{
    "age": 55,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "formerly smoked"
}
```

#### Example response

The values below are one previously verified example test result for
the example request above — not a fixed or hard-coded output. The
live API returns different values for different input.

```json
{
    "success": true,
    "data": {
        "risk_score": 0.4744,
        "risk_percentage": 47.44,
        "prediction": 0,
        "predicted_class": "Lower risk",
        "risk_category": "MODERATE",
        "threshold": 0.5
    },
    "disclaimer": "This is an educational stroke-risk screening prediction and not a medical diagnosis."
}
```

#### Validation errors — `400 Bad Request`

Missing fields, out-of-range age, or any unrecognized categorical
value (gender, work type, smoking status, etc.) all return:

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Missing required field: 'age'"
    }
}
```

Full field-by-field validation rules and error/status-code reference:
see [`docs/API.md`](docs/API.md).

### Running the tests

```bash
python -m pytest tests/test_api.py -v
```

### Project structure (Step 8 additions in bold)

```
data/
dataset/
docs/
    step6_evaluation_report.md
    **API.md**
model/
    stroke_risk_model.pkl
    preprocessor.pkl
    model_config.json
    model_info.json
prediction/
    __init__.py
    predict.py
results/
training/
    train_final_model.py
    test_saved_model.py
    ... (Steps 1-6 scripts)
**api/**
    **__init__.py**
    **app.py**
    **routes.py**
    **schemas.py**
**tests/**
    **test_api.py**
**requirements.txt**
**README.md**
```

---

## Limitations

- **Class imbalance:** strokes are rare in the source dataset (only
  ~5% of records), so even after SMOTENC-based resampling during
  training, precision is modest (18.85%) — most positive flags will
  turn out to be false positives. Recall (72.00%) and ROC-AUC
  (83.93%) are the more informative metrics for this kind of
  screening use case.
- **Small positive test set:** the test set contains only 50 stroke
  cases, so metrics carry meaningful sampling uncertainty.
- **No clinical measurements:** the model deliberately excludes BMI,
  glucose, and blood-pressure readings, trading some predictive power
  for a form anyone can fill out without lab equipment.
- **No independent clinical validation:** this project has not been
  evaluated on an external clinical population and is not a
  substitute for medical screening.

## Medical disclaimer

> This system provides an estimated stroke-risk score based on the
> information entered by the user. It is an educational
> machine-learning screening prototype and is not a medical diagnosis.
> It should not replace professional medical advice, clinical
> assessment, or medical testing.

---

## Step 10 — User-friendly frontend

A plain HTML/CSS/JavaScript frontend that lets a non-technical user get
a risk estimate by answering 8 simple questions — no BMI, glucose, or
blood pressure measurement required. It talks to the existing,
**unmodified** Step 8 Flask API; no model, preprocessing, training, or
API-contract code was changed for this step.

### Location

```
frontend/
├── index.html      # step-by-step questionnaire + result screen
├── css/
│   └── style.css
└── js/
    └── app.js       # validation, API call, result rendering
```

### How to run it

1. Start the Flask API (from the project root):
   ```bash
   .\venv\Scripts\python.exe -m api.app
   ```
   or, with an activated venv:
   ```bash
   python -m api.app
   ```
   It serves on `http://127.0.0.1:5000`.

2. Open `frontend/index.html` directly in a browser (double-click it,
   or serve the `frontend/` folder with any static file server). The
   page is static and makes cross-origin requests to the API, which
   already has CORS enabled for `/api/*`.

3. Answer the 8 questions and select **Check My Stroke Risk**.

If you serve the frontend from a different host/port than
`127.0.0.1:5000`, update the single `CONFIG.API_BASE_URL` constant at
the top of `frontend/js/app.js` — the base URL is never duplicated
elsewhere in the file.

### API endpoint used

`POST http://127.0.0.1:5000/api/predict` — the same Step 8 endpoint,
same request/response contract. See [`docs/API.md`](docs/API.md).

### The 8 user-friendly questions

| Question shown to the user | Sent to the API as |
|---|---|
| What is your age? | `age` (number, 0–120) |
| What is your gender? | `gender`: `Female` / `Male` / `Other` |
| Ever told you have high blood pressure? | `hypertension`: `1` / `0` |
| Ever diagnosed with heart disease? | `heart_disease`: `1` / `0` |
| Have you ever been married? | `ever_married`: `Yes` / `No` |
| What best describes your work? | `work_type`: `Private` / `Self-employed` / `Govt_job` / `children` / `Never_worked` |
| Where do you currently live? | `Residence_type`: `Urban` / `Rural` |
| Smoking history? | `smoking_status`: `never smoked` / `formerly smoked` / `smokes` / `Unknown` |

### Result display

The result card shows only whatever the live API returns for that
request — `risk_percentage`, `risk_category`, and `predicted_class` —
rendered as a percentage, a category badge, and a plain-language
label, plus an animated gauge. Nothing is hard-coded; a different
input produces a different displayed result.

### Validation & error handling

- Each step is validated before the user can continue (required age
  in range, a choice selected for each question).
- API validation errors (`success: false`) are shown as a plain-
  language message instead of the raw JSON.
- Network failures (e.g. the Flask server isn't running) show:
  *"We couldn't reach the prediction server. Please make sure it's
  running and try again."*
- No stack traces or technical error text are ever shown to the user.

### Disclaimer

The footer disclaimer is always visible:

> This tool is an educational stroke-risk screening prototype and is
> not a medical diagnosis. It should not replace advice from a
> qualified healthcare professional.

---

## Step 11 — Prediction-layer unit tests

The project's own documented `tests/` structure (from the Step 8/9
handoff) always specified `tests/test_prediction.py` alongside
`tests/test_api.py`, but that file had never actually been created —
`prediction/predict.py` was previously only exercised indirectly,
through the Flask HTTP layer.

Step 11 adds `tests/test_prediction.py`: 11 unit tests that call
`validate_user_data()` and `predict_stroke_risk()` directly, with no
Flask app or HTTP client involved. They cover valid input, missing
fields, out-of-range age, boundary ages (0 and 120), invalid category
values, non-binary hypertension/heart_disease values, the absence of
any BMI/glucose/blood-pressure requirement, determinism of a repeated
prediction, and that different inputs produce different scores (a
guard against an accidentally hard-coded return value).

No model, preprocessing, API, or frontend code was changed for this
step — verified by comparing file checksums before and after.

```bash
python -m pytest tests -v
```
now runs **24 tests** total: the original 13 API-level tests plus the
11 new prediction-layer tests.
