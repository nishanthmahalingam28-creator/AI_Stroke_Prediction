"""
tests/test_api.py
------------------------
STEP 8 - Automated tests for the Flask REST API.

Uses Flask's built-in test client (no live server / network needed).
Run from the project root:

    python -m pytest tests/test_api.py -v
"""

import os
import sys
import json

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.app import create_app  # noqa: E402


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


VALID_PAYLOAD = {
    "age": 55,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "formerly smoked",
}


# 1. GET /api/health
def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "stroke-risk-prediction-api"
    assert data["model_loaded"] is True


# 2. Valid POST /api/predict
def test_valid_predict(client):
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert 0.0 <= data["data"]["risk_score"] <= 1.0
    assert data["data"]["prediction"] in (0, 1)
    assert data["data"]["risk_category"] in ("LOW", "MODERATE", "HIGH")
    assert data["data"]["threshold"] == 0.5
    assert "disclaimer" in data


# 3. Missing field
def test_missing_field(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"


# 4. Invalid age (negative)
def test_invalid_age_negative(client):
    bad = {**VALID_PAYLOAD, "age": -5}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


# 5. Age > 120
def test_age_too_high(client):
    bad = {**VALID_PAYLOAD, "age": 150}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


# 6. Invalid gender
def test_invalid_gender(client):
    bad = {**VALID_PAYLOAD, "gender": "Unknown123"}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


# 7. Invalid work type
def test_invalid_work_type(client):
    bad = {**VALID_PAYLOAD, "work_type": "Freelancer"}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


# 8. Invalid smoking status
def test_invalid_smoking_status(client):
    bad = {**VALID_PAYLOAD, "smoking_status": "vapes"}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


# 9. Invalid hypertension
def test_invalid_hypertension(client):
    bad = {**VALID_PAYLOAD, "hypertension": 5}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


# 10. Invalid heart disease
def test_invalid_heart_disease(client):
    bad = {**VALID_PAYLOAD, "heart_disease": "yes"}
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


# 11. Invalid JSON body
def test_invalid_json(client):
    resp = client.post(
        "/api/predict",
        data="{not valid json",
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"


# 12. Wrong HTTP method
def test_wrong_http_method(client):
    resp = client.get("/api/predict")
    assert resp.status_code == 405
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "METHOD_NOT_ALLOWED"


# Bonus: 404 handling
def test_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"
