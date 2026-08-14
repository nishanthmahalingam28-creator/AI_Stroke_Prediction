"""
api/routes.py
------------------------
STEP 8 - Flask routes for the Stroke Risk Prediction API.

Endpoints:
    GET  /api/health
    POST /api/predict

IMPORTANT:
- No retraining happens here.
- SMOTENC is never applied here (it is training-only).
- The saved model/preprocessor are loaded once at app startup
  (see api/app.py) and reused for every request.
"""

from flask import Blueprint, jsonify, request, current_app

from api.schemas import validate_predict_payload
from prediction.predict import predict_stroke_risk

api_bp = Blueprint("api", __name__, url_prefix="/api")

DISCLAIMER = (
    "This is an educational stroke-risk screening prediction and not a "
    "medical diagnosis."
)


def _error_response(code, message, status_code):
    return (
        jsonify(
            {
                "success": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            }
        ),
        status_code,
    )


@api_bp.route("/health", methods=["GET"])
def health():
    model_loaded = bool(current_app.config.get("MODEL_LOADED", False))
    return jsonify(
        {
            "status": "ok",
            "service": "stroke-risk-prediction-api",
            "model_loaded": model_loaded,
        }
    )


@api_bp.route("/predict", methods=["POST"])
def predict():
    # Parse JSON safely -- do not let a bad body raise an uncaught 500.
    payload = request.get_json(silent=True)

    errors = validate_predict_payload(payload)
    if errors:
        # Surface the first validation error as the primary message,
        # but include the full list for completeness.
        return _error_response(
            "VALIDATION_ERROR",
            errors[0],
            400,
        )

    result = predict_stroke_risk(payload)

    if not result["valid"]:
        # Defensive: predict_stroke_risk re-validates internally.
        # This should not normally trigger since schemas.py already
        # validated, but is kept as a safety net.
        message = result["errors"][0] if result["errors"] else "Invalid input."
        return _error_response("VALIDATION_ERROR", message, 400)

    risk_score = result["risk_score"]
    prediction = result["prediction"]
    risk_category = result["risk_category"]
    threshold = result["threshold_used"]

    return jsonify(
        {
            "success": True,
            "data": {
                "risk_score": risk_score,
                "risk_percentage": round(risk_score * 100, 2),
                "prediction": prediction,
                "predicted_class": "Higher risk" if prediction == 1 else "Lower risk",
                "risk_category": risk_category,
                "threshold": threshold,
            },
            "disclaimer": DISCLAIMER,
        }
    )
