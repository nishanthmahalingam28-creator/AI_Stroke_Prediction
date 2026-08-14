# Stroke Risk Prediction API — Step 8

Base URL (development): `http://127.0.0.1:5000`

This API exposes the Step 7 saved model (Logistic Regression trained with
SMOTENC applied only during training) as a REST service. It does not
retrain, refit, or resample anything at request time.

> **Disclaimer:** This system provides an estimated stroke-risk score
> based on the information entered by the user. It is an educational
> machine-learning screening prototype and is not a medical diagnosis.
> It should not replace professional medical advice, clinical
> assessment, or medical testing.

---

## GET /api/health

Checks that the service is running and that the saved model artifacts
loaded successfully at startup.

- **Method:** `GET`
- **URL:** `/api/health`
- **Content-Type:** none required (no body)

### Example request
```
GET /api/health
```

### Example response — `200 OK`
```json
{
    "status": "ok",
    "service": "stroke-risk-prediction-api",
    "model_loaded": true
}
```

If the saved model/preprocessor failed to load at startup,
`model_loaded` will be `false`, but the endpoint itself still returns
`200` (the service is "up"; the model is simply not ready).

---

## POST /api/predict

Returns a model-estimated stroke-risk score for one user, based on the
8 production features only (no BMI, no glucose, no blood pressure
measurement required).

- **Method:** `POST`
- **URL:** `/api/predict`
- **Content-Type:** `application/json`

### Required fields

| Field             | Type   | Allowed values |
|-------------------|--------|-----------------|
| `age`             | number | `0`–`120` |
| `gender`          | string | `Male`, `Female`, `Other` |
| `hypertension`    | int    | `0`, `1` |
| `heart_disease`   | int    | `0`, `1` |
| `ever_married`    | string | `Yes`, `No` |
| `work_type`       | string | `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked` |
| `Residence_type`  | string | `Urban`, `Rural` |
| `smoking_status`  | string | `formerly smoked`, `never smoked`, `smokes`, `Unknown` |

All 8 fields are required on every request.

### Example request
```json
POST /api/predict
Content-Type: application/json

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

### Successful response — `200 OK`
```json
{
    "success": true,
    "data": {
        "risk_score": 0.6833,
        "risk_percentage": 68.33,
        "prediction": 1,
        "predicted_class": "Higher risk",
        "risk_category": "HIGH",
        "threshold": 0.5
    },
    "disclaimer": "This is an educational stroke-risk screening prediction and not a medical diagnosis."
}
```

`risk_category` is a **software display category only**, not a
medically validated clinical risk category:

- `LOW` — probability < 0.30
- `MODERATE` — 0.30 ≤ probability < 0.50
- `HIGH` — probability ≥ 0.50

`prediction` (0 or 1) always uses the fixed decision threshold of
**0.50**.

### Validation error response — `400 Bad Request`
```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Missing required field: 'age'"
    }
}
```

Triggers for `400 VALIDATION_ERROR`:
- Any missing required field
- `age` < 0 or `age` > 120, or non-numeric age
- Invalid/unrecognized `gender`
- Invalid `hypertension` or `heart_disease` (must be `0` or `1`)
- Invalid `ever_married`
- Invalid `work_type`
- Invalid `Residence_type`
- Invalid `smoking_status`
- Malformed / non-JSON request body

### Wrong HTTP method — `405 Method Not Allowed`
```json
{
    "success": false,
    "error": {
        "code": "METHOD_NOT_ALLOWED",
        "message": "This HTTP method is not allowed for this endpoint."
    }
}
```

### Unknown route — `404 Not Found`
```json
{
    "success": false,
    "error": {
        "code": "NOT_FOUND",
        "message": "The requested resource was not found."
    }
}
```

### Unexpected server error — `500 Internal Server Error`
```json
{
    "success": false,
    "error": {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected server error occurred."
    }
}
```
No Python stack traces or internal file paths are ever included in any
API response.

---

## Status code summary

| Code | Meaning |
|------|---------|
| 200  | Success (health check, or a valid prediction) |
| 400  | Validation error or malformed JSON |
| 404  | Unknown route |
| 405  | Wrong HTTP method for the route |
| 500  | Unexpected server-side error |
