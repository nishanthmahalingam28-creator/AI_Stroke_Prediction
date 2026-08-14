"""
train_models.py
------------------
STEP 5 (CORRECTED) - Train and compare multiple ML models x
imbalance-handling strategies for MODEL B (the user-friendly,
no-lab-test stroke risk model).

METHODOLOGY FIX vs the original Step 5:
Model/strategy SELECTION now happens using ONLY stratified k-fold
cross-validation on the TRAINING data. The test set is never touched
during selection. Only after a single "selected candidate" is chosen
from cross-validation results is that candidate trained once on the
full training set and evaluated once on the untouched test set, purely
for independent reporting -- that test result is never fed back into
the selection decision.

This script does NOT save a final production model (no
model/stroke_model.pkl). It only compares candidates and reports one
selected candidate for Step 6 to evaluate further.

Run from the project root:
    python training/train_models.py
"""

import os
import warnings
warnings.filterwarnings("ignore")  # keep console output readable; does not affect results

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from imblearn.over_sampling import SMOTENC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
CLEANED_DATA_PATH = os.path.join("data", "cleaned_stroke_data.csv")
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
RECALL_FLOOR = 0.50  # a screening candidate that misses over half of
                      # real stroke cases isn't seriously usable

EXPERIMENTS = [
    ("Logistic Regression", "baseline"),
    ("Logistic Regression", "class_weight"),
    ("Logistic Regression", "smotenc"),
    ("Random Forest", "baseline"),
    ("Random Forest", "class_weight"),
    ("Random Forest", "smotenc"),
    ("Gradient Boosting", "baseline"),
    ("Gradient Boosting", "sample_weight"),
]
if XGBOOST_AVAILABLE:
    EXPERIMENTS += [
        ("XGBoost", "baseline"),
        ("XGBoost", "scale_pos_weight"),
    ]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_preprocessor():
    """A fresh, unfit preprocessing pipeline. Built fresh every time it's
    used so it only ever learns from whatever training data it's given
    (a CV fold's training portion, or later the full training set) --
    never from a validation fold or the test set."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])


def apply_smotenc(X_train_raw, y_train_raw):
    categorical_indices = [X_train_raw.columns.get_loc(c) for c in SMOTENC_CATEGORICAL_COLS]
    smotenc = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=0.5,
        random_state=RANDOM_STATE
    )
    return smotenc.fit_resample(X_train_raw, y_train_raw)


def make_classifier(model_name, strategy, y_for_weight=None):
    if model_name == "Logistic Regression":
        if strategy == "class_weight":
            return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                       class_weight="balanced")
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    if model_name == "Random Forest":
        if strategy == "class_weight":
            return RandomForestClassifier(random_state=RANDOM_STATE,
                                           class_weight="balanced")
        return RandomForestClassifier(random_state=RANDOM_STATE)

    if model_name == "Gradient Boosting":
        # sklearn's GradientBoostingClassifier has no class_weight param;
        # "sample_weight" strategy is applied at fit() time instead.
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

    if model_name == "XGBoost":
        if strategy == "scale_pos_weight":
            neg = (y_for_weight == 0).sum()
            pos = (y_for_weight == 1).sum()
            spw = neg / pos
            return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                                  scale_pos_weight=spw)
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")

    raise ValueError(f"Unknown model: {model_name}")


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Model has neither predict_proba nor decision_function")


def fit_one_fold(model_name, strategy, X_tr, y_tr):
    """Fits preprocessing (+ SMOTENC if applicable) + classifier using
    ONLY the data handed to this function (a CV training fold, or later
    the full training set). Returns the fitted preprocessor and model."""
    if strategy == "smotenc":
        X_tr_fit, y_tr_fit = apply_smotenc(X_tr, y_tr)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    preprocessor = build_preprocessor()
    preprocessor.fit(X_tr_fit)
    X_tr_proc = preprocessor.transform(X_tr_fit)

    model = make_classifier(model_name, strategy, y_for_weight=y_tr_fit)

    if strategy == "sample_weight":
        sw = compute_sample_weight(class_weight="balanced", y=y_tr_fit)
        model.fit(X_tr_proc, y_tr_fit, sample_weight=sw)
    else:
        model.fit(X_tr_proc, y_tr_fit)

    return preprocessor, model


def cross_validate_experiment(model_name, strategy, X, y):
    """Runs 5-fold StratifiedKFold CV entirely on TRAINING data.
    SMOTENC/preprocessing/class-weighting are all fit fresh inside each
    fold's training portion only -- the held-out validation fold within
    each split is only ever transformed and predicted on, never used to
    fit anything. This never touches the outer test set at all."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    recalls, precisions, f1s, roc_aucs = [], [], [], []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        preprocessor, model = fit_one_fold(model_name, strategy, X_tr, y_tr)

        X_val_proc = preprocessor.transform(X_val)  # val fold never fit on
        y_pred = model.predict(X_val_proc)
        y_score = get_scores(model, X_val_proc)

        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        roc_aucs.append(roc_auc_score(y_val, y_score))

    result = {
        "model": model_name,
        "strategy": strategy,
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "mean_precision": float(np.mean(precisions)),
        "mean_f1": float(np.mean(f1s)),
        "mean_roc_auc": float(np.mean(roc_aucs)),
    }
    print(f"[CV] {model_name:20s} | {strategy:16s} "
          f"recall={result['mean_recall']:.3f}(+/-{result['std_recall']:.3f}) "
          f"precision={result['mean_precision']:.3f} "
          f"f1={result['mean_f1']:.3f} roc_auc={result['mean_roc_auc']:.3f}")
    return result


def evaluate_on_test(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "specificity": specificity,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
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
    # LOAD DATA / REPRODUCE STEP 3 SPLIT (raw, unencoded features)
    # ----------------------------------------------------------------
    section("LOADING DATA")
    df = pd.read_csv(CLEANED_DATA_PATH)
    X = df[MODEL_B_FEATURES].copy()
    y = df[TARGET].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Reproduced the same split as Steps 3 & 4 (random_state=42,")
    print("test_size=0.20, stratify=y). This split, and only this split,")
    print("is used everywhere below -- nothing in Steps 1-4 was changed.")
    print("Training rows:", len(X_train_raw), " | Test rows:", len(X_test_raw))

    # ----------------------------------------------------------------
    # CROSS-VALIDATION ON TRAINING DATA ONLY (MODEL/STRATEGY SELECTION)
    # ----------------------------------------------------------------
    section("CROSS-VALIDATION ON TRAINING DATA (5-FOLD, STRATIFIED)")
    print("StratifiedKFold(n_splits=5, shuffle=True, random_state=42) is")
    print("applied to X_train_raw/y_train ONLY. The test set is not used")
    print("anywhere in this section. For every fold, preprocessing and")
    print("(where applicable) SMOTENC are fit fresh on that fold's")
    print("training portion alone; the held-out validation portion of")
    print("the fold is only ever transformed/predicted on -- never fit.")
    print("This means SMOTENC-created synthetic rows can never appear in")
    print("a validation fold.\n")

    cv_results = []
    for model_name, strategy in EXPERIMENTS:
        cv_results.append(cross_validate_experiment(model_name, strategy, X_train_raw, y_train))

    cv_df = pd.DataFrame(cv_results)
    cv_path = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
    cv_df.to_csv(cv_path, index=False)

    section("CROSS-VALIDATION RESULTS TABLE (TRAINING DATA ONLY)")
    print(cv_df.round(4).to_string(index=False))
    print("\nSaved to:", cv_path)

    # ----------------------------------------------------------------
    # SELECT ONE CANDIDATE FROM CV RESULTS (NO TEST DATA INVOLVED)
    # ----------------------------------------------------------------
    section("CANDIDATE SELECTION (BASED ON CROSS-VALIDATION ONLY)")
    print("Selection priority for this stroke-risk screening project:")
    print("  1. Recall  -- missing a real stroke case (false negative) is")
    print("                the most costly kind of error for a screening")
    print("                tool; a case that's never flagged gets zero")
    print("                chance of early follow-up.")
    print("  2. F1-score -- but recall alone is not enough. A model can")
    print("                trivially get perfect recall by flagging")
    print("                EVERYONE as 'at risk', which would overwhelm")
    print("                users with false alarms and erode trust in the")
    print("                tool. F1 balances recall against precision, so")
    print("                it penalizes that kind of degenerate strategy.")
    print("  3. ROC-AUC  -- used as a tie-breaker measure of how well the")
    print("                model separates the two classes overall.")
    print(f"\nMethod: only strategies with mean_recall >= {RECALL_FLOOR} are")
    print("considered 'usable' for screening purposes. Among usable")
    print("candidates, the highest mean F1 is selected (ROC-AUC used to")
    print("break exact ties). This avoids both failure modes: models")
    print("that never flag anyone (high accuracy, near-zero recall) AND")
    print("models that flag almost everyone (high recall, near-zero")
    print("precision/F1).")

    usable = cv_df[cv_df["mean_recall"] >= RECALL_FLOOR]
    if len(usable) > 0:
        selected = usable.loc[usable["mean_f1"].idxmax()]
        selection_note = (f"selected from CV rows with mean_recall >= {RECALL_FLOOR}, "
                           f"ranked by mean F1")
    else:
        selected = cv_df.loc[cv_df["mean_recall"].idxmax()]
        selection_note = (f"no strategy reached mean_recall >= {RECALL_FLOOR} in CV, "
                           f"so the highest-recall row was used as a fallback")

    print(f"\nSELECTED CANDIDATE: {selected['model']} | {selected['strategy']}")
    print(f"  CV mean_recall:    {selected['mean_recall']:.4f} (+/- {selected['std_recall']:.4f})")
    print(f"  CV mean_precision: {selected['mean_precision']:.4f}")
    print(f"  CV mean_f1:        {selected['mean_f1']:.4f}")
    print(f"  CV mean_roc_auc:   {selected['mean_roc_auc']:.4f}")
    print(f"  ({selection_note})")
    print("\nThis is called a SELECTED CANDIDATE, not a final model.")
    print("Selection was based entirely on cross-validated TRAINING data")
    print("performance -- the test set has not been touched yet.")

    # ----------------------------------------------------------------
    # TRAIN SELECTED CANDIDATE ON FULL TRAINING SET, EVALUATE ONCE ON TEST
    # ----------------------------------------------------------------
    section("FINAL (ONE-TIME) TEST-SET EVALUATION OF SELECTED CANDIDATE")
    print("The selected candidate is now trained on the FULL training set")
    print("(all folds combined) and evaluated EXACTLY ONCE on the")
    print("untouched test set. This test result is for independent")
    print("reporting only -- it does not feed back into or change which")
    print("candidate was selected above. Threshold is the default 0.50;")
    print("no threshold tuning is performed using the test set.")

    sel_model_name = selected["model"]
    sel_strategy = selected["strategy"]

    preprocessor, model = fit_one_fold(sel_model_name, sel_strategy, X_train_raw, y_train)
    X_test_proc = preprocessor.transform(X_test_raw)  # test only ever transformed
    y_pred = model.predict(X_test_proc)
    y_score = get_scores(model, X_test_proc)

    test_metrics = evaluate_on_test(y_test, y_pred, y_score)

    print(f"\nTest-set results for {sel_model_name} | {sel_strategy}:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall:      {test_metrics['recall']:.4f}")
    print(f"  F1-score:    {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Confusion Matrix: TN={test_metrics['tn']} FP={test_metrics['fp']} "
          f"FN={test_metrics['fn']} TP={test_metrics['tp']}")

    safe_name = f"{sel_model_name}_{sel_strategy}".lower().replace(" ", "_")
    cm_path = save_confusion_matrix(
        test_metrics,
        f"SELECTED CANDIDATE (test set): {sel_model_name} - {sel_strategy}",
        f"confusion_matrix_selected_candidate_{safe_name}.png"
    )
    print("\nSaved confusion matrix to:", cm_path)

    # Save a small one-row summary of the candidate's test performance
    # (reporting artifact only -- NOT a trained model file).
    candidate_summary = pd.DataFrame([{
        "model": sel_model_name,
        "strategy": sel_strategy,
        "cv_mean_recall": selected["mean_recall"],
        "cv_mean_precision": selected["mean_precision"],
        "cv_mean_f1": selected["mean_f1"],
        "cv_mean_roc_auc": selected["mean_roc_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_specificity": test_metrics["specificity"],
    }])
    summary_path = os.path.join(RESULTS_DIR, "selected_candidate_summary.csv")
    candidate_summary.to_csv(summary_path, index=False)
    print("Saved candidate summary to:", summary_path)

    # ----------------------------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------------------------
    section("STEP 5 (CORRECTED) - MODEL TRAINING REPORT")
    print("Cross-validation performed on: TRAINING data only (5-fold,")
    print("stratified, random_state=42)")
    print(f"Models compared: {', '.join(sorted(set(m for m, s in EXPERIMENTS)))}")
    print(f"Strategies compared: {', '.join(sorted(set(s for m, s in EXPERIMENTS)))}")
    print()
    print(f"Selected candidate: {sel_model_name} | {sel_strategy}")
    print(f"  ({selection_note})")
    print()
    print("Independent test-set performance (evaluated once, not used")
    print("for selection):")
    print(f"  Accuracy={test_metrics['accuracy']:.4f}  "
          f"Precision={test_metrics['precision']:.4f}  "
          f"Recall={test_metrics['recall']:.4f}  "
          f"F1={test_metrics['f1_score']:.4f}  "
          f"ROC-AUC={test_metrics['roc_auc']:.4f}")
    print()
    print("This remains a SELECTED CANDIDATE, not a final production")
    print("model. No stroke_model.pkl was created. Step 6 will perform")
    print("more detailed evaluation before any final model is chosen and")
    print("saved.")

    print("\nSTEP 5 (CORRECTED) MODEL TRAINING AND COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
