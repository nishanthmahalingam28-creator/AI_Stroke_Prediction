"""
api/schemas.py
------------------------
STEP 8 - Request validation helpers for the Flask API.

This module does NOT duplicate the model's validation rules. It reuses
the exact same validation function already used and tested in Step 7
(prediction.predict.validate_user_data), so the API rejects input using
the same rules that were already verified against the saved model.

It also does light request-shape checking (e.g. "is this valid JSON",
"is this a JSON object") before handing the payload to the model's own
validator.
"""

from prediction.predict import validate_user_data


def validate_request_shape(payload):
    """
    Checks that the parsed JSON body is a non-null object (dict).
    Returns a list of error strings (empty means shape is OK).
    """
    if payload is None:
        return ["Request body must be valid JSON."]
    if not isinstance(payload, dict):
        return ["Request body must be a JSON object."]
    return []


def validate_predict_payload(payload):
    """
    Full validation for POST /api/predict.

    Returns a list of error strings. Empty list means the payload is
    valid and safe to pass to prediction.predict.predict_stroke_risk.
    """
    shape_errors = validate_request_shape(payload)
    if shape_errors:
        return shape_errors

    # Reuse the exact same, already-tested validation logic from Step 7.
    return validate_user_data(payload)
