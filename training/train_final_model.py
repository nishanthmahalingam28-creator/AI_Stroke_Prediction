"""
train_final_model.py
-----------------------
STEP 7 - Train and save the FINAL production pipeline for the selected
candidate: Logistic Regression + SMOTENC.

This script does NOT compare models again and does NOT touch the test
set. It trains on the FULL TRAINING DATA only (the same training split
established in Step 3), using the exact preprocessing and SMOTENC
settings verified in Steps 5-6, and saves the resulting artifacts.

IMPORTANT: the preprocessor saved here is fit on the SMOTENC-resampled
training data, matching exactly how Step 6's evaluation candidate was
built (see evaluate_final_candidate.py: fit_candidate()). This means
model/preprocessor.pkl is INTENTIONALLY overwritten by this script --
the original Step 3 preprocessor.pkl was fit on non-resampled training
data and is superseded by this verified, Step-6-consistent version.

Run from the project root:
    python training/train_final_model.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTENC

# ------------------------------------------------------------------
# CONFIG -- must match Steps 3-6 exactly (verified, not re-derived)
# ------------------------------------------------------------------
CLEANED_DATA_PATH = os.path.join("data", "cleaned_stroke_data.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "stroke"

# VERIFIED feature order from Steps 3-6 (preserved exactly, per Step 7
# instructions: "If the actual Step 3-6 pipeline uses a different
# verified order, preserve that exact verified order instead.")
MODEL_FEATURES = [
    "age", "gender", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "smoking_status"
]
NUMERIC_COLS = ["age"]
CATEGORICAL_COLS = ["gender", "ever_married", "work_type",
                     "Residence_type", "smoking_status"]
BINARY_COLS = ["hypertension", "heart_disease"]
SMOTENC_CATEGORICAL_COLS = CATEGORICAL_COLS + BINARY_COLS

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50  # project default operating threshold (Step 6)

RISK_CATEGORIES = {
    "low": "<0.30",
    "moderate": "0.30-<0.50",
    "high": ">=0.50",
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])


def main():
    # ----------------------------------------------------------------
    # 1. LOAD DATA / REPRODUCE THE VERIFIED STEP 3 SPLIT
    # ----------------------------------------------------------------
    section("1. LOADING DATA AND REPRODUCING VERIFIED TRAIN/TEST SPLIT")
    df = pd.read_csv(CLEANED_DATA_PATH)
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Same split as Steps 3-6 (random_state=42, test_size=0.20, stratify=y).")
    print("Training rows:", len(X_train_raw), " | Test rows (NOT used below):", len(X_test_raw))
    print("\nNOTE: X_test_raw/y_test are loaded here only to prove they exist")
    print("and are never referenced again in this script. The final model")
    print("is trained on X_train_raw/y_train exclusively.")

    # ----------------------------------------------------------------
    # 2. SMOTENC ON TRAINING DATA ONLY
    # ----------------------------------------------------------------
    section("2. APPLYING SMOTENC (TRAINING DATA ONLY)")
    categorical_indices = [X_train_raw.columns.get_loc(c) for c in SMOTENC_CATEGORICAL_COLS]
    smotenc = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=0.5,          # same ratio verified in Steps 4-6
        random_state=RANDOM_STATE
    )
    X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train_raw, y_train)
    print("Before SMOTENC:", y_train.value_counts().to_dict())
    print("After SMOTENC: ", y_train_resampled.value_counts().to_dict())
    print("SMOTENC applied ONLY to training data -- test data was never passed in.")

    # ----------------------------------------------------------------
    # 3. FIT PREPROCESSING ON RESAMPLED TRAINING DATA
    # ----------------------------------------------------------------
    section("3. FITTING PREPROCESSING PIPELINE")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_resampled)
    X_train_processed = preprocessor.transform(X_train_resampled)
    feature_names_out = list(preprocessor.get_feature_names_out())
    print("Preprocessor fit on SMOTENC-resampled training data (verified Step 6 methodology).")
    print("Transformed feature count:", len(feature_names_out))

    # ----------------------------------------------------------------
    # 4. TRAIN FINAL LOGISTIC REGRESSION
    # ----------------------------------------------------------------
    section("4. TRAINING FINAL LOGISTIC REGRESSION MODEL")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train_resampled)
    print("Final model trained on full (resampled) training data.")
    print("Model type:", type(model).__name__)

    # ----------------------------------------------------------------
    # 5. SAVE ARTIFACTS
    # ----------------------------------------------------------------
    section("5. SAVING PRODUCTION ARTIFACTS")

    model_path = os.path.join(MODEL_DIR, "stroke_risk_model.pkl")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print("Saved:", model_path)
    print("Saved:", preprocessor_path, "(overwrites the Step 3 preprocessor --")
    print("  see module docstring for why this is intentional)")

    model_config = {
        "model_name": "Logistic Regression",
        "imbalance_strategy": "SMOTENC",
        "smotenc_sampling_strategy": 0.5,
        "threshold": THRESHOLD,
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "binary_features": BINARY_COLS,
        "target": TARGET,
        "risk_categories": RISK_CATEGORIES,
        "random_state": RANDOM_STATE,
    }
    config_path = os.path.join(MODEL_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print("Saved:", config_path)

    model_info = {
        "project_name": "AI-Based User-Friendly Stroke Risk Prediction System Using Machine Learning",
        "model_name": "Logistic Regression",
        "imbalance_strategy": "SMOTENC (sampling_strategy=0.5, applied to training data only)",
        "features": MODEL_FEATURES,
        "excluded_features": ["id", "bmi", "avg_glucose_level"],
        "target": TARGET,
        "threshold": THRESHOLD,
        "risk_category_rules": RISK_CATEGORIES,
        "random_state": RANDOM_STATE,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print("Saved:", info_path)

    # ----------------------------------------------------------------
    # 6. FINAL MODEL SUMMARY (using Step 6's already-completed test results)
    # ----------------------------------------------------------------
    section("6. SAVING FINAL MODEL SUMMARY (Step 6 test results -- not re-run)")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    summary = pd.DataFrame([{
        "model": "Logistic Regression",
        "imbalance_strategy": "SMOTENC",
        "threshold": THRESHOLD,
        "test_accuracy": 0.8346,
        "test_precision": 0.1885,
        "test_recall": 0.7200,
        "test_f1": 0.2988,
        "test_specificity": 0.8405,
        "test_roc_auc": 0.8393,
        "test_average_precision": 0.2305,
    }])
    summary_path = os.path.join(results_dir, "final_model_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    print("\nThese are the already-completed Step 6 independent test results,")
    print("copied here for reference. No test evaluation was re-run.")

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    section("TRAINING SUMMARY")
    print(f"Model:              Logistic Regression")
    print(f"Imbalance strategy: SMOTENC (training data only, sampling_strategy=0.5)")
    print(f"Features ({len(MODEL_FEATURES)}):     {MODEL_FEATURES}")
    print(f"Target:             {TARGET}")
    print(f"Threshold:          {THRESHOLD}")
    print(f"Training rows used: {len(X_train_resampled)} (post-SMOTENC, from {len(X_train_raw)} original training rows)")
    print(f"Test rows:          {len(X_test_raw)} (NOT used in this script)")

    print("\nSTEP 7 FINAL MODEL TRAINING COMPLETE")


if __name__ == "__main__":
    main()
