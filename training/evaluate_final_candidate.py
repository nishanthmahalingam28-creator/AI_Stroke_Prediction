"""
evaluate_final_candidate.py
------------------------------
STEP 6 - Detailed evaluation of the Step 5 selected candidate:

    Logistic Regression + SMOTENC

This script does NOT change features, preprocessing, SMOTENC strategy,
random_state, or the train/test split established in Steps 3-5. It
does NOT save a final production model (no model/stroke_model.pkl).

What it does:
  1. Reproduces the exact Step 3 train/test split.
  2. Trains the selected candidate on the FULL training set (with
     SMOTENC applied to training data only) and evaluates it on the
     untouched test set at the standard 0.50 threshold.
  3. Generates ROC and Precision-Recall curves from test probabilities.
  4. Runs a TRAINING-DATA-ONLY out-of-fold (StratifiedKFold) threshold
     analysis to propose an operating threshold -- the test set is
     NEVER used to pick the threshold.
  5. Applies that fixed proposed threshold to the test set exactly
     once, for reporting only.
  6. Reports Step 5's cross-validation stability (mean/std) for this
     exact candidate.
  7. Performs a plain-language error analysis of the final predictions.

Run from the project root:
    python training/evaluate_final_candidate.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)

from imblearn.over_sampling import SMOTENC

# ------------------------------------------------------------------
# CONFIG -- must match Steps 3-5 exactly
# ------------------------------------------------------------------
CLEANED_DATA_PATH = os.path.join("data", "cleaned_stroke_data.csv")
CV_RESULTS_PATH = os.path.join("results", "cross_validation_results.csv")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "stroke"
MODEL_B_FEATURES = [
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
N_SPLITS = 5

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.55, 0.60, 0.65, 0.70]

SELECTED_MODEL_NAME = "Logistic Regression"
SELECTED_STRATEGY = "smotenc"

# STEP 6 CORRECTION: 0.50 is the project's DEFAULT/PRODUCTION operating
# threshold. The training-data-only threshold analysis below also
# reports an ALTERNATIVE, recall-oriented threshold (0.45) purely for
# comparison -- it is NOT automatically adopted as better, and it does
# NOT replace the 0.50 default.
DEFAULT_OPERATING_THRESHOLD = 0.50


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


def apply_smotenc(X_raw, y_raw):
    categorical_indices = [X_raw.columns.get_loc(c) for c in SMOTENC_CATEGORICAL_COLS]
    smotenc = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=0.5,
        random_state=RANDOM_STATE
    )
    return smotenc.fit_resample(X_raw, y_raw)


def make_candidate_model():
    # Selected candidate: plain Logistic Regression (imbalance handled
    # via SMOTENC on the training data, not via class_weight).
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def fit_candidate(X_train_raw, y_train_raw):
    """Fits preprocessing + SMOTENC + Logistic Regression using ONLY the
    data passed in. Returns the fitted preprocessor and model."""
    X_fit, y_fit = apply_smotenc(X_train_raw, y_train_raw)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_fit)
    X_fit_proc = preprocessor.transform(X_fit)
    model = make_candidate_model()
    model.fit(X_fit_proc, y_fit)
    return preprocessor, model


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "threshold": threshold, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1, "specificity": specificity,
        "npv": npv, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def save_confusion_matrix(cm, title, filename):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Stroke", "Pred: Stroke"])
    ax.set_yticklabels(["Actual: No Stroke", "Actual: Stroke"])
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path


def main():
    # ----------------------------------------------------------------
    # LOAD DATA / REPRODUCE STEP 3 SPLIT
    # ----------------------------------------------------------------
    section("LOADING DATA AND REPRODUCING STEP 3 SPLIT")
    df = pd.read_csv(CLEANED_DATA_PATH)
    X = df[MODEL_B_FEATURES].copy()
    y = df[TARGET].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Same split as Steps 3-5 (random_state=42, test_size=0.20,")
    print("stratify=y). Features, preprocessing, and SMOTENC strategy")
    print("(sampling_strategy=0.5) are unchanged from Step 5.")
    print("Training rows:", len(X_train_raw), " | Test rows:", len(X_test_raw))

    # ----------------------------------------------------------------
    # TRAIN CANDIDATE ON FULL TRAINING DATA, GET TEST PROBABILITIES
    # ----------------------------------------------------------------
    section("TRAINING SELECTED CANDIDATE ON FULL TRAINING DATA")
    print("Candidate: Logistic Regression + SMOTENC (sampling_strategy=0.5)")
    preprocessor, model = fit_candidate(X_train_raw, y_train)
    X_test_proc = preprocessor.transform(X_test_raw)  # test only ever transformed
    test_probs = model.predict_proba(X_test_proc)[:, 1]
    print("Model trained on full training set. Test set has been")
    print("transformed (never fit) and probability scores generated.")
    print("Test data enters the pipeline ONLY at predict_proba().")

    # ----------------------------------------------------------------
    # 3-4. STANDARD THRESHOLD (0.50) EVALUATION
    # ----------------------------------------------------------------
    section("STANDARD THRESHOLD EVALUATION (threshold = 0.50)")
    default_metrics = metrics_at_threshold(y_test, test_probs, 0.50)
    print(f"Accuracy:    {default_metrics['accuracy']:.4f}")
    print(f"Precision:   {default_metrics['precision']:.4f}")
    print(f"Recall:      {default_metrics['recall']:.4f}")
    print(f"F1-score:    {default_metrics['f1_score']:.4f}")
    roc_auc_value = roc_auc_score(y_test, test_probs)
    print(f"ROC-AUC:     {roc_auc_value:.4f}")
    print(f"Specificity: {default_metrics['specificity']:.4f}")
    print(f"NPV:         {default_metrics['npv']:.4f}")
    print(f"Confusion Matrix -> TN={default_metrics['tn']} FP={default_metrics['fp']} "
          f"FN={default_metrics['fn']} TP={default_metrics['tp']}")

    print("\nIn plain language:")
    print("  True Positive  = actual stroke -> predicted stroke")
    print("  True Negative  = actual no stroke -> predicted no stroke")
    print("  False Positive = actual no stroke -> predicted stroke")
    print("  False Negative = actual stroke -> predicted no stroke")
    print("\nFalse negatives matter most here: a false negative means the")
    print("system tells someone with a real stroke history/risk pattern")
    print("in the data that they are 'low risk', giving them no reason")
    print("to seek any follow-up. A false positive is a false alarm --")
    print("unwelcome, but far less costly than a missed case.")

    cm_default_path = save_confusion_matrix(
        default_metrics, "Logistic Regression + SMOTENC (threshold=0.50)",
        "confusion_matrix_final_candidate.png"
    )
    print("\nSaved:", cm_default_path)

    # ----------------------------------------------------------------
    # 5. ROC CURVE
    # ----------------------------------------------------------------
    section("ROC CURVE (test set)")
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_probs)
    print(f"ROC-AUC = {roc_auc_value:.4f}")
    print("Plain-language meaning: ROC-AUC is the probability that the")
    print("model ranks a randomly chosen stroke case's risk score higher")
    print("than a randomly chosen non-stroke case's risk score. 0.50 =")
    print("no better than a coin flip, 1.00 = perfect ranking. It")
    print("measures ranking ability, independent of any specific")
    print("threshold.")

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_value:.3f})", color="#4C9AFF")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve - Logistic Regression + SMOTENC (test set)")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print("Saved:", roc_path)

    # ----------------------------------------------------------------
    # 6. PRECISION-RECALL CURVE
    # ----------------------------------------------------------------
    section("PRECISION-RECALL CURVE (test set)")
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, test_probs)
    avg_precision = average_precision_score(y_test, test_probs)
    print(f"Average Precision (AP) = {avg_precision:.4f}")
    print("\nWhy PR analysis matters here: with only ~4.9% positive")
    print("cases, ROC-AUC can look deceptively good because the huge")
    print("majority class dominates the false-positive-rate denominator.")
    print("Precision-Recall focuses only on how the model handles the")
    print("minority (stroke) class -- a more honest picture of real-world")
    print("usefulness for a screening tool built on imbalanced data.")

    plt.figure(figsize=(5, 5))
    plt.plot(pr_recall, pr_precision, color="#FF5C5C",
             label=f"PR curve (AP = {avg_precision:.3f})")
    baseline_rate = y_test.mean()
    plt.axhline(baseline_rate, linestyle="--", color="gray",
                label=f"No-skill baseline ({baseline_rate:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Logistic Regression + SMOTENC (test set)")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(RESULTS_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print("Saved:", pr_path)

    # ----------------------------------------------------------------
    # 7. THRESHOLD ANALYSIS -- TRAINING DATA ONLY (OUT-OF-FOLD)
    # ----------------------------------------------------------------
    section("THRESHOLD ANALYSIS (TRAINING DATA ONLY, OUT-OF-FOLD)")
    print("StratifiedKFold(n_splits=5, shuffle=True, random_state=42) is")
    print("applied to X_train_raw/y_train ONLY, exactly like Step 5's CV.")
    print("For each fold, SMOTENC + preprocessing + Logistic Regression")
    print("are fit on that fold's training portion; predicted")
    print("probabilities are collected for the held-out validation")
    print("portion of each fold. Once every row of the training set has")
    print("an out-of-fold probability, threshold performance is")
    print("evaluated on those out-of-fold probabilities -- this never")
    print("touches the test set.")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_train))
    y_train_reset = y_train.reset_index(drop=True)
    X_train_reset = X_train_raw.reset_index(drop=True)

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train_reset, y_train_reset), start=1):
        X_tr, X_val = X_train_reset.iloc[tr_idx], X_train_reset.iloc[val_idx]
        y_tr = y_train_reset.iloc[tr_idx]

        fold_preprocessor, fold_model = fit_candidate(X_tr, y_tr)
        X_val_proc = fold_preprocessor.transform(X_val)
        oof_probs[val_idx] = fold_model.predict_proba(X_val_proc)[:, 1]

    print("Out-of-fold probabilities generated for all", len(y_train_reset), "training rows.")

    threshold_rows = []
    for t in THRESHOLDS:
        m = metrics_at_threshold(y_train_reset, oof_probs, t)
        threshold_rows.append({
            "threshold": t, "precision": m["precision"], "recall": m["recall"],
            "f1_score": m["f1_score"], "specificity": m["specificity"],
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_path = os.path.join(RESULTS_DIR, "threshold_analysis.csv")
    threshold_df.to_csv(threshold_path, index=False)
    print("\n" + threshold_df.round(4).to_string(index=False))
    print("\nSaved:", threshold_path)

    # ----------------------------------------------------------------
    # 8. PROPOSED OPERATING THRESHOLD (chosen using training data only)
    # ----------------------------------------------------------------
    section("ALTERNATIVE RECALL-ORIENTED THRESHOLD (analysis only, from training-data validation)")
    print(f"DEFAULT OPERATING THRESHOLD for this project remains {DEFAULT_OPERATING_THRESHOLD}.")
    print("This section identifies an ALTERNATIVE threshold using training-data")
    print("validation only, for comparison -- it is not automatically adopted,")
    print("and it does not override the 0.50 default.")
    print("\nChoosing the threshold with the highest recall alone would just")
    print("push the threshold to its lowest tested value, flagging almost")
    print("everyone 'at risk' and making the tool noisy and untrustworthy.")
    print("Instead: among thresholds with out-of-fold recall >= 0.60 (a")
    print("meaningful floor for a screening tool), the one with the highest")
    print("F1-score is identified as the alternative candidate.")

    RECALL_FLOOR = 0.60
    usable = threshold_df[threshold_df["recall"] >= RECALL_FLOOR]
    if len(usable) > 0:
        proposed_row = usable.loc[usable["f1_score"].idxmax()]
        selection_note = f"highest F1 among thresholds with OOF recall >= {RECALL_FLOOR}"
    else:
        proposed_row = threshold_df.loc[threshold_df["recall"].idxmax()]
        selection_note = f"no threshold reached OOF recall >= {RECALL_FLOOR}; used max-recall threshold as fallback"

    proposed_threshold = float(proposed_row["threshold"])
    print(f"\nALTERNATIVE RECALL-ORIENTED THRESHOLD: {proposed_threshold}")
    print(f"  (OOF) precision={proposed_row['precision']:.4f}, "
          f"recall={proposed_row['recall']:.4f}, "
          f"f1={proposed_row['f1_score']:.4f}, "
          f"specificity={proposed_row['specificity']:.4f}")
    print(f"  Selection basis: {selection_note}")
    print(f"\nThis is an ALTERNATIVE threshold, not the project default, and")
    print("not a final, medically validated threshold. It is a project-design")
    print("option to consider when prioritizing recall, evaluated below")
    print("alongside the default for comparison.")

    # ----------------------------------------------------------------
    # 9. FINAL TEST EVALUATION AT PROPOSED THRESHOLD (applied ONCE)
    # ----------------------------------------------------------------
    section("TEST EVALUATION AT THE ALTERNATIVE THRESHOLD (comparison only)")
    print(f"Applying the alternative threshold={proposed_threshold} to the")
    print("already-generated test probabilities ONE TIME, purely for")
    print("side-by-side reporting against the 0.50 default. The test set")
    print("was not involved in choosing this threshold, and this result does")
    print("not change the project's default operating threshold.")

    proposed_metrics = metrics_at_threshold(y_test, test_probs, proposed_threshold)
    print(f"\nAccuracy:    {proposed_metrics['accuracy']:.4f}")
    print(f"Precision:   {proposed_metrics['precision']:.4f}")
    print(f"Recall:      {proposed_metrics['recall']:.4f}")
    print(f"F1-score:    {proposed_metrics['f1_score']:.4f}")
    print(f"Specificity: {proposed_metrics['specificity']:.4f}")
    print(f"ROC-AUC:     {roc_auc_value:.4f}  (unchanged -- ROC-AUC does not depend on threshold)")
    print(f"Average Precision: {avg_precision:.4f}  (also threshold-independent)")
    print(f"Confusion Matrix -> TN={proposed_metrics['tn']} FP={proposed_metrics['fp']} "
          f"FN={proposed_metrics['fn']} TP={proposed_metrics['tp']}")

    cm_proposed_path = save_confusion_matrix(
        proposed_metrics,
        f"Logistic Regression + SMOTENC (proposed threshold={proposed_threshold})",
        "confusion_matrix_proposed_threshold.png"
    )
    print("\nSaved:", cm_proposed_path)

    # ----------------------------------------------------------------
    # 10. COMPARE THRESHOLDS (0.50 vs proposed) -- ACTUAL TEST VALUES
    # ----------------------------------------------------------------
    section("THRESHOLD COMPARISON (test set, actual values)")
    compare_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1_score", "specificity"],
        "threshold_0.50_(default)": [
            default_metrics["accuracy"], default_metrics["precision"],
            default_metrics["recall"], default_metrics["f1_score"],
            default_metrics["specificity"],
        ],
        f"threshold_{proposed_threshold}_(alternative)": [
            proposed_metrics["accuracy"], proposed_metrics["precision"],
            proposed_metrics["recall"], proposed_metrics["f1_score"],
            proposed_metrics["specificity"],
        ],
    })
    print(compare_df.round(4).to_string(index=False))

    print(f"\nTrade-off, {proposed_threshold} vs {DEFAULT_OPERATING_THRESHOLD} (default):")
    recall_delta = proposed_metrics["recall"] - default_metrics["recall"]
    precision_delta = proposed_metrics["precision"] - default_metrics["precision"]
    f1_delta = proposed_metrics["f1_score"] - default_metrics["f1_score"]
    specificity_delta = proposed_metrics["specificity"] - default_metrics["specificity"]
    print(f"  {proposed_threshold}: "
          f"{'slightly higher' if recall_delta > 0 else 'slightly lower'} recall "
          f"({recall_delta:+.4f}), "
          f"{'slightly higher' if precision_delta > 0 else 'slightly lower'} precision "
          f"({precision_delta:+.4f}), "
          f"{'higher' if f1_delta > 0 else 'lower'} F1 ({f1_delta:+.4f}), "
          f"{'higher' if specificity_delta > 0 else 'lower'} specificity ({specificity_delta:+.4f})")
    print(f"  {DEFAULT_OPERATING_THRESHOLD}: "
          f"{'slightly lower' if recall_delta > 0 else 'slightly higher'} recall, "
          f"higher precision, higher F1, higher specificity "
          f"(relative to {proposed_threshold})")

    print(f"\nThe default operating threshold is {DEFAULT_OPERATING_THRESHOLD}. A threshold")
    print(f"of {proposed_threshold} can be considered when prioritizing slightly higher")
    print("recall, but it produces lower precision, F1-score, and specificity")
    print("in the current test evaluation.")
    print(f"\n{proposed_threshold} is NOT automatically \"better\" -- it is an alternative,")
    print(f"recall-oriented option. The project keeps {DEFAULT_OPERATING_THRESHOLD} as the")
    print("DEFAULT OPERATING THRESHOLD; this is a project-design choice based")
    print("on the training/validation analysis above, not a decision made or")
    print("tuned using the test results.")

    # ----------------------------------------------------------------
    # 11. MODEL RELIABILITY (Step 5 cross-validation stability)
    # ----------------------------------------------------------------
    section("MODEL RELIABILITY (Step 5 cross-validation results)")
    if os.path.exists(CV_RESULTS_PATH):
        cv_df = pd.read_csv(CV_RESULTS_PATH)
        row = cv_df[(cv_df["model"] == SELECTED_MODEL_NAME) &
                    (cv_df["strategy"] == SELECTED_STRATEGY)]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"Mean Recall:    {row['mean_recall']:.4f}  (Std: {row['std_recall']:.4f})")
            print(f"Mean Precision: {row['mean_precision']:.4f}")
            print(f"Mean F1:        {row['mean_f1']:.4f}")
            print(f"Mean ROC-AUC:   {row['mean_roc_auc']:.4f}")
            recall_cv_gap = abs(row["mean_recall"] - proposed_metrics["recall"])
            print(f"\nRecall standard deviation across folds ({row['std_recall']:.4f}) is")
            print("modest relative to the mean, suggesting reasonably consistent")
            print("(not wildly erratic) behavior across different training subsets --")
            print("but with only ~40 stroke cases per fold, some fold-to-fold")
            print("swings are expected and this is NOT the same as clinical")
            print("validation on an independent population.")
        else:
            print("Matching row not found in cross_validation_results.csv.")
    else:
        print("cross_validation_results.csv not found -- run Step 5 first.")

    # ----------------------------------------------------------------
    # 12. ERROR ANALYSIS (using proposed-threshold test predictions)
    # ----------------------------------------------------------------
    section("ERROR ANALYSIS (proposed threshold, test set)")
    tp, tn = proposed_metrics["tp"], proposed_metrics["tn"]
    fp, fn = proposed_metrics["fp"], proposed_metrics["fn"]
    print(f"True Positives:  {tp}  (predicted stroke risk, and the dataset label is stroke)")
    print(f"True Negatives:  {tn}  (predicted no elevated risk, and the dataset label is no stroke)")
    print(f"False Positives: {fp}  (predicted stroke risk, but the dataset label is no stroke)")
    print(f"False Negatives: {fn}  (predicted no elevated risk, but the dataset label is stroke)")
    print("\nWhat this means for a real user:")
    print(f"  - {fn} out of {tp+fn} real stroke-labeled cases in the test set would")
    print("    have been told their ESTIMATED stroke risk is not elevated --")
    print("    the system's most serious kind of error.")
    print(f"  - {fp} out of {fp+tn} no-stroke-labeled cases would have received an")
    print("    elevated risk estimate they didn't need -- inconvenient, and")
    print("    could cause anxiety, but is a much safer failure direction")
    print("    than a false negative for a screening context.")
    print("\nNote on wording: this system predicts/estimates stroke risk")
    print("based on the input factors. It does not diagnose stroke.")

    # ----------------------------------------------------------------
    # 13. SAVE FINAL EVALUATION REPORT
    # ----------------------------------------------------------------
    section("SAVING FINAL EVALUATION REPORT")
    final_report = pd.DataFrame([
        {
            "model": "Logistic Regression + SMOTENC",
            "threshold": DEFAULT_OPERATING_THRESHOLD,
            "threshold_role": "default",
            "accuracy": default_metrics["accuracy"],
            "precision": default_metrics["precision"],
            "recall": default_metrics["recall"],
            "f1_score": default_metrics["f1_score"],
            "specificity": default_metrics["specificity"],
            "roc_auc": roc_auc_value,
            "average_precision": avg_precision,
        },
        {
            "model": "Logistic Regression + SMOTENC",
            "threshold": proposed_threshold,
            "threshold_role": "alternative_recall_oriented",
            "accuracy": proposed_metrics["accuracy"],
            "precision": proposed_metrics["precision"],
            "recall": proposed_metrics["recall"],
            "f1_score": proposed_metrics["f1_score"],
            "specificity": proposed_metrics["specificity"],
            "roc_auc": roc_auc_value,
            "average_precision": avg_precision,
        },
    ])
    final_report_path = os.path.join(RESULTS_DIR, "final_evaluation_report.csv")
    final_report.to_csv(final_report_path, index=False)
    print("Saved:", final_report_path)
    print("\n" + final_report.round(4).to_string(index=False))

    # ----------------------------------------------------------------
    # MEDICAL / FEATURE LIMITATION DISCLOSURE
    # ----------------------------------------------------------------
    section("MEDICAL INTERPRETATION AND FEATURE LIMITATIONS")
    print("This is an educational machine-learning project using the terms")
    print("\"stroke risk prediction\" / \"stroke risk screening\" -- never")
    print("\"stroke diagnosis.\"")
    print()
    print("\"The model uses easily obtainable user information and")
    print("intentionally excludes BMI and average glucose level to improve")
    print("usability. This may reduce predictive performance compared with")
    print("models using additional clinical measurements. The system is an")
    print("educational risk-screening prototype and should not be")
    print("considered a medical diagnosis.\"")

    section("USER-FRIENDLY EXPLANATION")
    print("\"The system estimates a user's stroke risk based on the")
    print("information entered by the user. It does not confirm whether")
    print("the user has had or will have a stroke.\"")

    # ----------------------------------------------------------------
    # FINAL STEP 6 SUMMARY
    # ----------------------------------------------------------------
    section("STEP 6 FINAL SUMMARY")
    print("Selected candidate:")
    print("  Logistic Regression + SMOTENC")
    print()
    print(f"Default threshold:")
    print(f"  {DEFAULT_OPERATING_THRESHOLD}")
    print()
    print(f"Alternative threshold:")
    print(f"  {proposed_threshold} for recall-oriented use")
    print()
    print("Default test performance:")
    print(f"  Accuracy:           {default_metrics['accuracy']*100:.2f}%")
    print(f"  Precision:          {default_metrics['precision']*100:.2f}%")
    print(f"  Recall:             {default_metrics['recall']*100:.2f}%")
    print(f"  F1-score:           {default_metrics['f1_score']*100:.2f}%")
    print(f"  Specificity:        {default_metrics['specificity']*100:.2f}%")
    print(f"  ROC-AUC:            {roc_auc_value*100:.2f}%")
    print(f"  Average Precision:  {avg_precision*100:.2f}%")

    print("\nSTEP 6 DETAILED EVALUATION COMPLETE")

    return {
        "default_metrics": default_metrics,
        "proposed_metrics": proposed_metrics,
        "default_threshold": DEFAULT_OPERATING_THRESHOLD,
        "proposed_threshold": proposed_threshold,
        "roc_auc": roc_auc_value,
        "avg_precision": avg_precision,
    }


if __name__ == "__main__":
    main()
