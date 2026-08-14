"""
preprocess_data.py
--------------------
STEP 3 - Data preprocessing and train/test splitting for MODEL B
(the user-friendly, no-lab-test stroke risk model).

This script does NOT train any ML model. It only:
  - selects Model B's 8 features
  - splits into train/test (stratified, before any balancing)
  - builds and fits a preprocessing pipeline (encoding/scaling)
  - saves the fitted preprocessor and the processed arrays

Run from the project root:
    python training/preprocess_data.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
CLEANED_DATA_PATH = os.path.join("data", "cleaned_stroke_data.csv")
MODEL_DIR = "model"
DATA_DIR = "data"

TARGET = "stroke"

# Model B feature set -- fixed by project design, nothing added/removed here
MODEL_B_FEATURES = [
    "age", "gender", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "smoking_status"
]

NUMERICAL_FEATURES = ["age"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type",
                         "Residence_type", "smoking_status"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # ----------------------------------------------------------------
    # 1. LOAD CLEANED DATASET (from Step 2 -- never touch the raw file)
    # ----------------------------------------------------------------
    section("1. LOADING CLEANED DATASET")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print("Loaded:", CLEANED_DATA_PATH)
    print("Shape:", df.shape)

    original_rows = df.shape[0]

    # ----------------------------------------------------------------
    # 2. SELECT MODEL B FEATURES + TARGET
    # ----------------------------------------------------------------
    section("2. SELECTING MODEL B FEATURES")
    X = df[MODEL_B_FEATURES].copy()
    y = df[TARGET].copy()

    print("X columns (Model B inputs):", list(X.columns))
    print("y column (target):", TARGET)
    print("Confirmed excluded from X: id, avg_glucose_level, bmi")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ----------------------------------------------------------------
    # 3. VERIFY DATA TYPES
    # ----------------------------------------------------------------
    section("3. FEATURE DATA TYPES")
    type_table = pd.DataFrame({
        "dtype": X.dtypes.astype(str),
        "example_value": [X[col].iloc[0] for col in X.columns]
    })
    print(type_table)

    print("\nNumerical features:", NUMERICAL_FEATURES)
    print("  -> age is a continuous number, so it can be scaled.")
    print("Binary features:", BINARY_FEATURES)
    print("  -> already 0/1, no encoding needed, passed through as-is.")
    print("Categorical features:", CATEGORICAL_FEATURES)
    print("  -> text categories with no natural order, so they need")
    print("     one-hot encoding rather than being treated as numbers.")

    # ----------------------------------------------------------------
    # 4. TRAIN/TEST SPLIT (BEFORE any balancing)
    # ----------------------------------------------------------------
    section("4. TRAIN/TEST SPLIT")
    print("Splitting with test_size=0.20, random_state=42, stratify=y")
    print("\nWhy stratify=y matters: stroke cases are only ~4.9% of the")
    print("data. Without stratification, a random split could easily put")
    print("too few (or too many) stroke cases in the test set purely by")
    print("chance, making evaluation unreliable. Stratification forces")
    print("both the train and test sets to keep the same ~95/5 ratio as")
    print("the full dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # ----------------------------------------------------------------
    # 5. VERIFY CLASS DISTRIBUTION AFTER SPLIT
    # ----------------------------------------------------------------
    section("5. CLASS DISTRIBUTION AFTER SPLIT")

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    train_no = train_counts.get(0, 0)
    train_yes = train_counts.get(1, 0)
    test_no = test_counts.get(0, 0)
    test_yes = test_counts.get(1, 0)

    print("Training set:")
    print(f"  No Stroke: {train_no}  ({train_no/len(y_train)*100:.2f}%)")
    print(f"  Stroke:    {train_yes}  ({train_yes/len(y_train)*100:.2f}%)")

    print("\nTest set:")
    print(f"  No Stroke: {test_no}  ({test_no/len(y_test)*100:.2f}%)")
    print(f"  Stroke:    {test_yes}  ({test_yes/len(y_test)*100:.2f}%)")

    print("\nBoth sets preserve the original ~95/5 imbalance -- the test")
    print("set is NOT artificially balanced to 50/50, because it must")
    print("reflect real-world conditions for a fair evaluation later.")

    # ----------------------------------------------------------------
    # 6. NOTE: NO SMOTE / NO RESAMPLING HERE
    # ----------------------------------------------------------------
    section("6. CLASS IMBALANCE HANDLING")
    print("No SMOTE, oversampling, or undersampling is applied in this")
    print("script. That is intentionally left for the model-training")
    print("step, where it will be applied ONLY to X_train/y_train,")
    print("never to the test set.")

    # ----------------------------------------------------------------
    # 7. BUILD PREPROCESSING PIPELINE
    # ----------------------------------------------------------------
    section("7. BUILDING PREPROCESSING PIPELINE")

    print("Numerical pipeline: StandardScaler() on 'age'")
    print("  -> age ranges from ~0 to 82. Some models (e.g. Logistic")
    print("     Regression) are sensitive to feature scale, so scaling")
    print("     keeps age from dominating just because its raw numbers")
    print("     are bigger than the 0/1 binary columns. Tree-based")
    print("     models don't strictly need this, but scaling doesn't")
    print("     hurt them either, so one shared pipeline is used for all")
    print("     candidate models tested in Step 4.")

    print("\nCategorical pipeline: OneHotEncoder(handle_unknown='ignore')")
    print("  -> gender, ever_married, work_type, Residence_type, and")
    print("     smoking_status have no natural ranking (e.g. 'Private'")
    print("     work is not 'greater than' 'Govt_job'). LabelEncoder")
    print("     would wrongly imply an order by turning categories into")
    print("     0,1,2,3..., so OneHotEncoder is used instead, which")
    print("     creates a separate 0/1 column per category.")
    print("     handle_unknown='ignore' protects against a category")
    print("     appearing in production user input that the training")
    print("     data never saw (e.g. someone selects an option that")
    print("     happened to have zero rows in training).")

    print("\nBinary pass-through: hypertension, heart_disease")
    print("  -> already clean 0/1 values, so they are passed through")
    print("     unchanged rather than re-encoded.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )

    # ----------------------------------------------------------------
    # 8. FIT ON TRAINING DATA ONLY -- AVOID DATA LEAKAGE
    # ----------------------------------------------------------------
    section("8. FITTING PREPROCESSOR (TRAIN DATA ONLY)")
    print("Data leakage, in simple terms: it happens when information")
    print("from the test set 'leaks' into how the model or preprocessing")
    print("is built. For example, if we computed the age mean/std using")
    print("ALL the data (train + test) before scaling, the model would")
    print("indirectly know something about the test set before it was")
    print("ever evaluated on it -- making the evaluation overly optimistic")
    print("and not trustworthy. To avoid this, the preprocessor below is")
    print("fit ONLY on X_train. X_test is only ever transformed using")
    print("parameters already learned from X_train, never re-fit.")

    preprocessor.fit(X_train)
    print("\nPreprocessor fitted on X_train only. X_test was NOT used.")

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # ----------------------------------------------------------------
    # 9. FEATURE NAMES AFTER TRANSFORMATION
    # ----------------------------------------------------------------
    section("9. TRANSFORMED FEATURE NAMES")
    feature_names = preprocessor.get_feature_names_out()
    print("Total transformed features:", len(feature_names))
    for name in feature_names:
        print(" -", name)

    # ----------------------------------------------------------------
    # 10. SAVE PREPROCESSOR (NOT A TRAINED MODEL)
    # ----------------------------------------------------------------
    section("10. SAVING PREPROCESSOR")
    os.makedirs(MODEL_DIR, exist_ok=True)
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print("Saved fitted preprocessor to:", preprocessor_path)
    print("(This is only the preprocessing pipeline -- no ML model has")
    print(" been trained or saved. There is no stroke_model.pkl yet.)")
    print("\nIn the final web application, this same preprocessor.pkl")
    print("will be loaded and used to transform new user questionnaire")
    print("answers into the exact format the trained model expects,")
    print("before the model makes a prediction.")

    # ----------------------------------------------------------------
    # 11. SAVE PROCESSED DATA
    # ----------------------------------------------------------------
    section("11. SAVING PROCESSED DATASETS")
    os.makedirs(DATA_DIR, exist_ok=True)

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    X_train_path = os.path.join(DATA_DIR, "X_train_processed.csv")
    X_test_path = os.path.join(DATA_DIR, "X_test_processed.csv")
    y_train_path = os.path.join(DATA_DIR, "y_train.csv")
    y_test_path = os.path.join(DATA_DIR, "y_test.csv")

    X_train_df.to_csv(X_train_path, index=False)
    X_test_df.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print("Saved:", X_train_path)
    print("Saved:", X_test_path)
    print("Saved:", y_train_path)
    print("Saved:", y_test_path)

    # ----------------------------------------------------------------
    # 12. SUMMARY
    # ----------------------------------------------------------------
    section("STEP 3 PREPROCESSING SUMMARY")
    print(f"Original rows: {original_rows}")
    print()
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print()
    print(f"Original features: {len(MODEL_B_FEATURES)}")
    print(f"Final transformed features: {len(feature_names)}")
    print()
    print(f"Training stroke cases: {train_yes}")
    print(f"Training non-stroke cases: {train_no}")
    print()
    print(f"Testing stroke cases: {test_yes}")
    print(f"Testing non-stroke cases: {test_no}")
    print()
    print("Preprocessor saved:")
    print(f"  {preprocessor_path}")

    # ----------------------------------------------------------------
    # 13. DATA LEAKAGE CHECKLIST
    # ----------------------------------------------------------------
    section("DATA LEAKAGE CHECK")
    print("[OK] Test data was not used to fit the preprocessor")
    print("[OK] No SMOTE was applied to test data")
    print("[OK] No manual oversampling was applied")
    print("[OK] No rows were deleted to balance classes")
    print("[OK] id is excluded from features")
    print("[OK] avg_glucose_level is excluded from Model B")
    print("[OK] bmi is excluded from Model B")
    print("[OK] target stroke is separated from X")
    print("[OK] stratified split was used")

    print("\nSTEP 3 PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
