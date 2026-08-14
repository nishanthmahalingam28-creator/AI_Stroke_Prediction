"""
prediction/predict.py
------------------------
STEP 7 - Reusable production prediction function.

Loads the saved preprocessor + trained Logistic Regression model and
model_config.json, validates new user input, and returns a structured
stroke-RISK-SCORE result.

IMPORTANT:
- SMOTENC is NEVER applied here. It is a training-only technique.
- This does not diagnose stroke. It returns a model-estimated risk
  score and a display risk category.

Usage:
    from prediction.predict import predict_stroke_risk

    result = predict_stroke_risk({
        "age": 45,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "never smoked",
    })
"""

import os
import json
import joblib
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "model")

_MODEL_PATH = os.path.join(_MODEL_DIR, "stroke_risk_model.pkl")
_PREPROCESSOR_PATH = os.path.join(_MODEL_DIR, "preprocessor.pkl")
_CONFIG_PATH = os.path.join(_MODEL_DIR, "model_config.json")

# Valid category values, taken from the actual training dataset's
# unique values (Step 2 exploration) -- used for input validation only.
VALID_CATEGORIES = {
    "gender": {"Male", "Female", "Other"},
    "ever_married": {"Yes", "No"},
    "work_type": {"Private", "Self-employed", "Govt_job", "children", "Never_worked"},
    "Residence_type": {"Urban", "Rural"},
    "smoking_status": {"formerly smoked", "never smoked", "smokes", "Unknown"},
}
VALID_BINARY = {0, 1}
MIN_AGE = 0
MAX_AGE = 120

_model = None
_preprocessor = None
_config = None


def _load_artifacts():
    """Loads (and caches) the saved model, preprocessor, and config."""
    global _model, _preprocessor, _config
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {_MODEL_PATH}. "
                "Run training/train_final_model.py first."
            )
        _model = joblib.load(_MODEL_PATH)
    if _preprocessor is None:
        if not os.path.exists(_PREPROCESSOR_PATH):
            raise FileNotFoundError(
                f"Preprocessor file not found at {_PREPROCESSOR_PATH}. "
                "Run training/train_final_model.py first."
            )
        _preprocessor = joblib.load(_PREPROCESSOR_PATH)
    if _config is None:
        if not os.path.exists(_CONFIG_PATH):
            raise FileNotFoundError(
                f"Config file not found at {_CONFIG_PATH}. "
                "Run training/train_final_model.py first."
            )
        with open(_CONFIG_PATH) as f:
            _config = json.load(f)
    return _model, _preprocessor, _config


def validate_user_data(user_data):
    """
    Validates raw user input against the required Model B fields.
    Returns a list of error strings (empty list means valid).
    Never raises -- callers check the returned list.
    """
    errors = []
    _, _, config = _load_artifacts()
    required_fields = config["features"]

    # 1. Missing fields
    for field in required_fields:
        if field not in user_data or user_data[field] in (None, ""):
            errors.append(f"Missing required field: '{field}'")

    if errors:
        # Don't bother checking types/ranges if fields are missing outright
        return errors

    # 2. age
    age = user_data.get("age")
    try:
        age_val = float(age)
        if age_val < MIN_AGE or age_val > MAX_AGE:
            errors.append(f"Invalid age: {age}. Must be between {MIN_AGE} and {MAX_AGE}.")
    except (TypeError, ValueError):
        errors.append(f"Invalid age: {age}. Must be a number.")

    # 3. hypertension / heart_disease (binary 0/1)
    for field in ["hypertension", "heart_disease"]:
        val = user_data.get(field)
        try:
            int_val = int(val)
            if int_val not in VALID_BINARY:
                errors.append(f"Invalid {field}: {val}. Must be 0 or 1.")
        except (TypeError, ValueError):
            errors.append(f"Invalid {field}: {val}. Must be 0 or 1.")

    # 4. categorical fields
    for field, valid_values in VALID_CATEGORIES.items():
        val = user_data.get(field)
        if val not in valid_values:
            errors.append(
                f"Invalid {field}: '{val}'. Must be one of: {sorted(valid_values)}"
            )

    return errors


def _risk_category(probability, config):
    """
    Categorizes a probability into a display-only risk band.

    The 0.30 / 0.50 cut points below must stay in sync with the
    human-readable values documented in model_config.json's
    "risk_categories" field (low: "<0.30", moderate: "0.30-<0.50",
    high: ">=0.50"). They are kept as plain numeric literals here
    (rather than parsed from that string) for clarity and to avoid a
    fragile string parser, but any change to the risk bands must be
    made in both places.
    """
    if probability < 0.30:
        return "LOW"
    elif probability < 0.50:
        return "MODERATE"
    else:
        return "HIGH"


def predict_stroke_risk(user_data):
    """
    Predicts stroke risk for a single user.

    Parameters
    ----------
    user_data : dict
        Must contain exactly the 8 Model B features:
        age, gender, hypertension, heart_disease, ever_married,
        work_type, Residence_type, smoking_status

    Returns
    -------
    dict with keys:
        risk_score      -- model-estimated probability (0.0-1.0)
        prediction      -- 0 or 1, using the project's default threshold
        risk_category   -- "LOW" / "MODERATE" / "HIGH" (display only)
        threshold_used  -- the threshold applied
        valid           -- True if input passed validation
        errors          -- list of validation error strings (empty if valid)

    IMPORTANT: risk_score is a model output, not a medically validated
    probability of an individual's future stroke. This function never
    applies SMOTENC -- SMOTENC is a training-only technique.
    """
    errors = validate_user_data(user_data)
    if errors:
        return {
            "risk_score": None,
            "prediction": None,
            "risk_category": None,
            "threshold_used": None,
            "valid": False,
            "errors": errors,
        }

    model, preprocessor, config = _load_artifacts()
    features = config["features"]
    threshold = config["threshold"]

    # Build a single-row DataFrame in the exact verified feature order
    row = {}
    for field in features:
        if field in ("hypertension", "heart_disease"):
            row[field] = int(user_data[field])
        elif field == "age":
            row[field] = float(user_data[field])
        else:
            row[field] = user_data[field]
    input_df = pd.DataFrame([row], columns=features)

    # Preprocess (no SMOTENC -- inference only) and predict
    input_processed = preprocessor.transform(input_df)
    probability = float(model.predict_proba(input_processed)[0, 1])
    prediction = int(probability >= threshold)
    category = _risk_category(probability, config)

    return {
        "risk_score": round(probability, 4),
        "prediction": prediction,
        "risk_category": category,
        "threshold_used": threshold,
        "valid": True,
        "errors": [],
    }


if __name__ == "__main__":
    # Small smoke test when run directly (not used by the Flask API).
    example = {
        "age": 45,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "never smoked",
    }
    print(predict_stroke_risk(example))
