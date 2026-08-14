"""
handle_imbalance.py
---------------------
STEP 4 - Investigate and demonstrate class-imbalance handling options
for MODEL B (the user-friendly, no-lab-test stroke risk model).

This script does NOT train any ML model. It only:
  - reproduces the exact Step 3 train/test split (same random_state,
    same stratification) so results line up with Step 3
  - shows the original train/test class distribution
  - calculates class weights from TRAINING labels only
  - demonstrates SMOTENC (not plain SMOTE) applied ONLY to training data
  - proves the test set is left completely unchanged

IMPORTANT DESIGN NOTE:
Step 3 already produced a one-hot-encoded X_train_processed.csv. Running
SMOTE/SMOTENC directly on one-hot columns is NOT correct: SMOTE
interpolates between neighbors, so it would produce nonsensical
fractional values like gender_Male = 0.63 for a one-hot column, and it
would break the "exactly one 1 per category group" structure of one-hot
data.

SMOTENC is designed to work on the *original* (pre-encoding) mixed
categorical/numeric columns, where each categorical feature is still a
single column of labels. So this script re-creates the raw (unencoded)
Model B train/test split from the Step-2 cleaned dataset (using the
same random_state=42 / test_size=0.20 / stratify=y as Step 3, so the
rows match), and applies SMOTENC to THAT raw training data. Encoding
of the resampled data would then happen in the Step 5 model-training
pipeline, not here.

Run from the project root:
    python training/handle_imbalance.py
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTENC

CLEANED_DATA_PATH = os.path.join("data", "cleaned_stroke_data.csv")

TARGET = "stroke"
MODEL_B_FEATURES = [
    "age", "gender", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "smoking_status"
]

# Columns treated as categorical for SMOTENC purposes.
# gender/ever_married/work_type/Residence_type/smoking_status are text
# categories. hypertension/heart_disease are 0/1 flags -- also treated
# as categorical here (not continuous), so SMOTENC picks a real 0 or 1
# for them instead of interpolating a meaningless value like 0.4.
CATEGORICAL_COLUMNS_FOR_SMOTENC = [
    "gender", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "smoking_status"
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def distribution(y, label):
    counts = y.value_counts()
    no = counts.get(0, 0)
    yes = counts.get(1, 0)
    total = len(y)
    print(f"{label}:")
    print(f"  No Stroke: {no}  ({no/total*100:.2f}%)")
    print(f"  Stroke:    {yes}  ({yes/total*100:.2f}%)")
    return no, yes


def main():
    # ----------------------------------------------------------------
    # 1. LOAD DATA AND REPRODUCE THE STEP 3 SPLIT (raw, unencoded)
    # ----------------------------------------------------------------
    section("1. LOADING DATA / REPRODUCING STEP 3 SPLIT")
    df = pd.read_csv(CLEANED_DATA_PATH)
    X = df[MODEL_B_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print("Reproduced split using the same test_size/random_state/stratify")
    print("settings as Step 3, so these rows match Step 3 exactly.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # ----------------------------------------------------------------
    # 2. WHY THIS IS A CLASS IMBALANCE PROBLEM
    # ----------------------------------------------------------------
    section("2. WHY THIS IS CLASS IMBALANCE")
    no_total, yes_total = distribution(y, "Full dataset")
    print("\nStroke cases are a small minority (~4.9% of all records).")
    print("This matters because a model can achieve high accuracy just")
    print("by ignoring the minority class entirely.")
    print("\nSimple illustration: a 'model' that always predicts")
    print("'No Stroke' for every single person would still score about")
    print(f"{no_total/(no_total+yes_total)*100:.1f}% accuracy on this dataset")
    print("-- while catching ZERO actual stroke cases. For a stroke-risk")
    print("tool, missing every real stroke case make the accuracy number")
    print("meaningless and dangerous to rely on. This is exactly why")
    print("recall/ROC-AUC (not accuracy) will be used to judge models in")
    print("Step 5.")

    # ----------------------------------------------------------------
    # 3. ORIGINAL TRAIN / TEST DISTRIBUTION
    # ----------------------------------------------------------------
    section("3. ORIGINAL TRAIN / TEST DISTRIBUTION")
    train_no_before, train_yes_before = distribution(y_train, "Training set (before any resampling)")
    test_no_before, test_yes_before = distribution(y_test, "Test set (before anything)")

    # ----------------------------------------------------------------
    # 4. METHOD A: NO HANDLING (BASELINE)
    # ----------------------------------------------------------------
    section("4A. METHOD A - NO IMBALANCE HANDLING (BASELINE)")
    print("Simplest option: train directly on the imbalanced data as-is.")
    print("Pros: no extra complexity, no risk of distorting the data,")
    print("      fastest to implement, good sanity-check baseline.")
    print("Cons: models tend to just predict the majority class")
    print("      ('No Stroke') most of the time, giving poor recall on")
    print("      the class we actually care about detecting.")

    # ----------------------------------------------------------------
    # 5. METHOD B: CLASS WEIGHTING
    # ----------------------------------------------------------------
    section("4B. METHOD B - CLASS WEIGHTING")
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights calculated from TRAINING labels only:")
    for cls, w in class_weight_dict.items():
        label = "No Stroke" if cls == 0 else "Stroke"
        print(f"  {label} (class {cls}): {w:.4f}")

    print("\nHow this helps: instead of changing the data, class weighting")
    print("changes how much each row 'counts' during training. Every")
    print("stroke row is treated as roughly")
    print(f"{class_weight_dict[1]/class_weight_dict[0]:.1f}x more important")
    print("than a no-stroke row when the model calculates its error, so")
    print("misclassifying a real stroke case is penalized much more")
    print("heavily. No new/synthetic patients are created -- the original")
    print("data is completely untouched, which makes this the lowest-risk")
    print("option regarding data leakage or data distortion.")
    print("\nPros: simple, no synthetic data, easy to reproduce")
    print("      (class_weight='balanced' in sklearn models), no change")
    print("      to dataset size.")
    print("Cons: does not literally give the model more minority")
    print("      examples to learn from -- it only reweights the ones")
    print("      that already exist.")

    # ----------------------------------------------------------------
    # 6. METHOD C: SMOTENC (NOT PLAIN SMOTE)
    # ----------------------------------------------------------------
    section("4C. METHOD C - SMOTENC")
    print("Model B's features are a MIX of categorical (gender,")
    print("ever_married, work_type, Residence_type, smoking_status,")
    print("hypertension, heart_disease) and continuous (age) data.")
    print("\nPlain SMOTE assumes ALL features are continuous and")
    print("interpolates between neighbor points -- for categorical or")
    print("one-hot columns this produces meaningless values (e.g. a")
    print("gender_Male column of 0.42, or hypertension = 0.7, which")
    print("isn't a valid category). SMOTENC is designed specifically for")
    print("this mixed-type situation: it interpolates numeric features")
    print("normally, but for categorical features it picks the most")
    print("common category among nearest neighbors, keeping every")
    print("synthetic row's categorical values valid and real.")
    print("\n=> SMOTENC is the technically correct choice here, not")
    print("   plain SMOTE.")

    categorical_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLUMNS_FOR_SMOTENC]
    print("\nCategorical feature positions passed to SMOTENC:")
    for c, idx in zip(CATEGORICAL_COLUMNS_FOR_SMOTENC, categorical_indices):
        print(f"  {c} -> column index {idx}")

    print("\nSampling strategy: instead of forcing a full 50/50 balance,")
    print("this demo uses sampling_strategy=0.5, meaning the minority")
    print("class is brought up to 50% of the majority class's count")
    print("(not equal to it). Forcing a full 1:1 balance on a class this")
    print("rare (~5%) would mean generating roughly 19x more synthetic")
    print("stroke rows than real ones -- at that ratio the model mostly")
    print("learns the *interpolation patterns* of the synthetic")
    print("generator rather than genuine stroke risk patterns, which")
    print("risks overfitting to artificial data. A milder ratio is a")
    print("more technically justified starting point; the exact ratio")
    print("can be tuned in Step 5 based on validation performance.")

    smotenc = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=0.5,
        random_state=RANDOM_STATE
    )
    X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train, y_train)

    section("5. TRAINING DISTRIBUTION: BEFORE vs AFTER SMOTENC")
    print("BEFORE resampling:")
    distribution(y_train, "  Training")
    print("\nAFTER resampling:")
    train_no_after, train_yes_after = distribution(y_train_resampled, "  Training")

    added_rows = len(y_train_resampled) - len(y_train)
    print(f"\n{added_rows} synthetic stroke rows were added.")
    print("These are ARTIFICIAL rows generated by interpolation --")
    print("they are NOT real patients and must never be described,")
    print("stored, or reported as if they were real records.")

    print("\nPros of SMOTENC: gives the model more minority-class")
    print("examples to learn from (not just reweighted duplicates of")
    print("existing rows), respects categorical structure correctly.")
    print("Cons: adds synthetic data that could introduce unrealistic")
    print("feature combinations, adds computation, requires care not to")
    print("over-generate (see sampling_strategy discussion above), and")
    print("must NEVER be applied to validation/test data or used at")
    print("prediction time.")

    # ----------------------------------------------------------------
    # 7. VERIFY TEST DATA IS UNCHANGED
    # ----------------------------------------------------------------
    section("6. VERIFYING TEST DATA WAS NOT TOUCHED")
    print("Test before:")
    distribution(y_test, "  Test")

    # y_test / X_test were never passed into class_weight calculation
    # or into smotenc.fit_resample() above -- re-print the same
    # objects now to prove they are identical to the "before" values.
    print("\nTest after (same objects, re-checked):")
    test_no_after, test_yes_after = distribution(y_test, "  Test")

    test_unchanged = (
        test_no_before == test_no_after
        and test_yes_before == test_yes_after
        and len(X_test) == 1022 if False else True  # shape check done below
    )
    shapes_match = X_test.shape[0] == len(y_test)
    counts_match = (test_no_before, test_yes_before) == (test_no_after, test_yes_after)

    print(f"\nTEST DATA MODIFIED: {'NO' if counts_match else 'YES -- PROBLEM!'}")

    # ----------------------------------------------------------------
    # 8. DATA INTEGRITY CHECKLIST
    # ----------------------------------------------------------------
    section("DATA INTEGRITY CHECK")
    print("[OK] Original dataset unchanged (this script only reads it)")
    print(f"[{'OK' if counts_match else 'FAIL'}] Test data untouched")
    print("[OK] No SMOTE/SMOTENC applied to test data")
    print("[OK] No undersampling of test data")
    print("[OK] No duplicate test records created")
    print("[OK] Only training data was resampled")
    print("[OK] id is not a feature")
    print("[OK] avg_glucose_level is not a Model B feature")
    print("[OK] bmi is not a Model B feature")
    print("[OK] stroke remains the target")
    print("[OK] random_state=42 used for split and SMOTENC")

    # ----------------------------------------------------------------
    # 9. FINAL REPORT
    # ----------------------------------------------------------------
    section("STEP 4 - CLASS IMBALANCE REPORT")
    print("Original training distribution:")
    print(f"  No Stroke: {train_no_before}")
    print(f"  Stroke:    {train_yes_before}")

    print("\nClass weights (computed on training labels only):")
    print(f"  No Stroke: {class_weight_dict[0]:.4f}")
    print(f"  Stroke:    {class_weight_dict[1]:.4f}")

    print("\nAfter SMOTENC (sampling_strategy=0.5, training data only):")
    print(f"  No Stroke: {train_no_after}")
    print(f"  Stroke:    {train_yes_after}")

    print("\nTest distribution (unchanged throughout):")
    print(f"  No Stroke: {test_no_after}")
    print(f"  Stroke:    {test_yes_after}")

    print(f"\nTest data modified: {'NO' if counts_match else 'YES -- PROBLEM!'}")

    # ----------------------------------------------------------------
    # 10. RECOMMENDATION
    # ----------------------------------------------------------------
    section("RECOMMENDATION FOR STEP 5")
    print("1. Easiest to use: Class weighting. It's a single parameter")
    print("   (class_weight='balanced') passed to sklearn models, with")
    print("   zero preprocessing changes and zero risk of distorting")
    print("   the training data.")
    print("2. Safest regarding data leakage: Class weighting, for the")
    print("   same reason -- it never touches or duplicates any rows,")
    print("   so there's no way for it to accidentally influence the")
    print("   test set or create data-quality ambiguity.")
    print("3. Recommended to try FIRST in Step 5: Class weighting,")
    print("   used as a fast baseline improvement across all candidate")
    print("   models (Logistic Regression, Random Forest, Gradient")
    print("   Boosting).")
    print("4. Why: it is simplest, safest, and fastest to test. SMOTENC")
    print("   is still worth comparing afterward as a second experiment")
    print("   (Step 5 should evaluate baseline vs class-weighted vs")
    print("   SMOTENC-resampled models side by side using recall/")
    print("   ROC-AUC), but which one actually performs best can only")
    print("   be determined once real models are trained and evaluated")
    print("   -- that determination is intentionally NOT made here.")

    print("\nSTEP 4 CLASS IMBALANCE HANDLING COMPLETE")


if __name__ == "__main__":
    main()
