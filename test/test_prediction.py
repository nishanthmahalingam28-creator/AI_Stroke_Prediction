"""
tests/test_prediction.py
------------------------
STEP 11 - Direct unit tests for prediction/predict.py.

test_api.py already exercises this module indirectly through Flask's
HTTP layer (api/routes.py -> api/schemas.py -> prediction/predict.py).
These tests call validate_user_data() and predict_stroke_risk()
directly, with no Flask app / HTTP client involved, so the prediction
layer is verified in isolation.

Does NOT retrain anything, does NOT modify the saved model or
preprocessor, and does NOT apply SMOTENC (training-only technique).

Run from the project root:

    python -m pytest tests/test_prediction.py -v
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prediction.predict import (  # noqa: E402
    predict_stroke_risk,
    validate_user_data,
)

VALID_INPUT = {
    "age": 55,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "formerly smoked",
}


# 1. A valid input produces a valid, well-formed prediction.
def test_valid_input_returns_valid_prediction():
    result = predict_stroke_risk(VALID_INPUT)
    assert result["valid"] is True
    assert result["errors"] == []
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["risk_category"] in ("LOW", "MODERATE", "HIGH")
    assert result["threshold_used"] == 0.5


# 2. validate_user_data() accepts a fully valid payload (no errors).
def test_validate_user_data_accepts_valid_payload():
    errors = validate_user_data(VALID_INPUT)
    assert errors == []


# 3. Missing required field is caught before prediction runs.
def test_missing_field_is_rejected():
    bad = {k: v for k, v in VALID_INPUT.items() if k != "smoking_status"}
    result = predict_stroke_risk(bad)
    assert result["valid"] is False
    assert result["risk_score"] is None
    assert any("smoking_status" in e for e in result["errors"])


# 4. Negative age is rejected.
def test_negative_age_is_rejected():
    bad = {**VALID_INPUT, "age": -1}
    result = predict_stroke_risk(bad)
    assert result["valid"] is False
    assert any("age" in e.lower() for e in result["errors"])


# 5. Age above the supported range is rejected.
def test_age_above_range_is_rejected():
    bad = {**VALID_INPUT, "age": 121}
    result = predict_stroke_risk(bad)
    assert result["valid"] is False


# 6. Boundary ages (0 and 120) are accepted.
def test_boundary_ages_are_accepted():
    low = predict_stroke_risk({**VALID_INPUT, "age": 0})
    high = predict_stroke_risk({**VALID_INPUT, "age": 120})
    assert low["valid"] is True
    assert high["valid"] is True


# 7. Unrecognized category value is rejected.
def test_invalid_category_value_is_rejected():
    bad = {**VALID_INPUT, "work_type": "Freelancer"}
    result = predict_stroke_risk(bad)
    assert result["valid"] is False
    assert any("work_type" in e for e in result["errors"])


# 8. Non-binary hypertension/heart_disease values are rejected.
def test_non_binary_hypertension_is_rejected():
    bad = {**VALID_INPUT, "hypertension": 2}
    result = predict_stroke_risk(bad)
    assert result["valid"] is False


# 9. The model never receives BMI, glucose, or blood pressure --
#    those keys are not part of the required feature set at all.
def test_no_disallowed_features_required():
    disallowed = {"bmi", "avg_glucose_level", "blood_pressure", "id"}
    errors_without_extra_fields = validate_user_data(VALID_INPUT)
    assert errors_without_extra_fields == []
    assert disallowed.isdisjoint(VALID_INPUT.keys())


# 10. Same valid input always produces the same result (deterministic,
#     no retraining or randomness at prediction time).
def test_prediction_is_deterministic():
    first = predict_stroke_risk(VALID_INPUT)
    second = predict_stroke_risk(VALID_INPUT)
    assert first["risk_score"] == second["risk_score"]
    assert first["prediction"] == second["prediction"]
    assert first["risk_category"] == second["risk_category"]


# 11. Different inputs are not silently mapped to the same score
#     (i.e. the result genuinely depends on the input -- guards
#     against an accidental hard-coded / constant return value).
def test_different_inputs_can_produce_different_scores():
    lower_risk = predict_stroke_risk(
        {
            "age": 20,
            "gender": "Female",
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "No",
            "work_type": "children",
            "Residence_type": "Rural",
            "smoking_status": "never smoked",
        }
    )
    higher_risk = predict_stroke_risk(
        {
            "age": 79,
            "gender": "Male",
            "hypertension": 1,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Govt_job",
            "Residence_type": "Urban",
            "smoking_status": "smokes",
        }
    )
    assert lower_risk["valid"] is True
    assert higher_risk["valid"] is True
    assert lower_risk["risk_score"] != higher_risk["risk_score"]
