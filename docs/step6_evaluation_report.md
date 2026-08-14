# STEP 6 — Detailed Model Evaluation Report

**Project:** AI-Based User-Friendly Stroke Risk Prediction System Using Machine Learning

---

## 1. Purpose of Step 6

Step 5 used training-data cross-validation to select one candidate for further study:

**Selected candidate: Logistic Regression + SMOTENC**

Step 6 exists to check that candidate carefully before it becomes the production model in Step 7. A model that looked promising during cross-validation still needs to be checked on data it has never seen at all, examined from several different angles (confusion matrix, ROC-AUC, precision-recall, threshold behavior), and checked for consistency across folds. Skipping this step would risk shipping a model whose real-world behavior is not properly understood.

---

## 2. Model Used

**Algorithm:** Logistic Regression
**Imbalance-handling method:** SMOTENC (Synthetic Minority Over-sampling for mixed categorical/numeric data), applied only to training data

**Model B — user-friendly feature set (8 inputs):**

| # | Feature |
|---|---|
| 1 | age |
| 2 | gender |
| 3 | hypertension |
| 4 | heart_disease |
| 5 | ever_married |
| 6 | work_type |
| 7 | Residence_type |
| 8 | smoking_status |

**Target:** `stroke`

**Explicitly NOT used:** `id`, `bmi`, `avg_glucose_level`

These three columns are excluded on purpose. `id` carries no medical meaning. `bmi` and `avg_glucose_level` both require lab measurements or medical equipment, which this project's design goal avoids — the questionnaire is meant to be answerable by an ordinary person with no clinical tools.

---

## 3. Data Split

- **Training data** was used for all model development: preprocessing, SMOTENC resampling, cross-validation, and model fitting.
- **Test data** was kept completely independent and was never used for training.
- **Test data was never used to choose the classification threshold.**

Keeping the test set untouched matters because it is the only honest way to estimate how the model will behave on data it has never influenced in any way. If the test set were used during training or threshold selection, the reported performance would be optimistic and misleading — a form of information leakage from test to training.

---

## 4. Evaluation Metrics (Plain-Language Explanations)

| Metric | Meaning |
|---|---|
| **Accuracy** | Percentage of all predictions (stroke and no-stroke) that were correct. Misleading on this dataset because of severe class imbalance. |
| **Precision** | Of everyone the model flagged as "at risk," what fraction actually had the stroke label? |
| **Recall** | Of everyone who actually has the stroke label in the dataset, what fraction did the model correctly flag? |
| **F1-score** | A balance between precision and recall in a single number. |
| **Specificity** | Of everyone who does NOT have the stroke label, what fraction did the model correctly leave unflagged? |
| **ROC-AUC** | How well the model ranks stroke cases above non-stroke cases, across all possible thresholds at once. |
| **Average Precision** | Summarizes the precision-recall curve into one number; more informative than ROC-AUC on imbalanced data. |

**Recall receives special weight in this project** because it is a stroke-risk *screening* tool. Recall answers:

> "Of the people who actually belong to the stroke class in the dataset, how many did the model identify?"

Two related error types matter here:

- **False Negative** — actual stroke (in the dataset) → predicted no stroke. False negatives are particularly important in a screening-oriented system because a potentially higher-risk case may not be flagged.
- **False Positive** — actual no stroke (in the dataset) → predicted stroke. An unwanted false alarm, but far less costly than a missed case.

---

## 5. Test Results at Threshold 0.50

| Metric | Value |
|---|---|
| Accuracy | 83.46% |
| Precision | 18.85% |
| Recall | 72.00% |
| F1-score | 29.88% |
| Specificity | 84.05% |
| ROC-AUC | 83.93% |
| Average Precision | 23.05% |

**What these mean in plain language:**

- **Accuracy (83.46%)** — 83 out of every 100 predictions, across both classes, matched the dataset label. On its own this number is not very informative here, because simply predicting "no stroke" for everyone would already score about 95% — accuracy rewards the majority class too easily.
- **Precision (18.85%)** — of everyone the model flagged as "at risk," fewer than 1 in 5 actually carried the stroke label in the dataset. Most flags are false alarms.
- **Recall (72.00%)** — the model correctly identified 72 out of every 100 real stroke-labeled cases. This is the headline number for a screening tool.
- **F1-score (29.88%)** — reflects the tension between low precision and moderate-to-good recall.
- **Specificity (84.05%)** — of everyone without the stroke label, about 84 out of 100 were correctly left unflagged.
- **Average Precision (23.05%)** — summarizes precision across all recall levels; a low number here reflects how hard it is to be precise on a ~5%-positive dataset.

---

## 6. Confusion Matrix (Threshold 0.50)

| | Predicted: No Stroke | Predicted: Stroke |
|---|---|---|
| **Actual: No Stroke** | TN = 817 | FP = 155 |
| **Actual: Stroke** | FN = 14 | TP = 36 |

*(Verified from the saved evaluation results; specificity 84.05% = 817 / (817+155).)*

**In plain terms:**

- The model correctly identified **36 of the 50 stroke-labeled cases** in the test set (True Positives).
- It **missed 14 stroke-labeled cases** (False Negatives) — these were predicted "no stroke" despite carrying the stroke label.
- It correctly left **817 no-stroke-labeled cases** unflagged (True Negatives).
- It incorrectly flagged **155 no-stroke-labeled cases** as at-risk (False Positives).

These numbers describe how the model's predictions line up against the **dataset's labels** — they are not real-world medical diagnoses, and the people in this test set were not clinically re-evaluated.

---

## 7. ROC-AUC

**ROC-AUC = 83.93%**

This does **not** mean "the model is 83.93% accurate." ROC-AUC measures something different: the model's ability to correctly rank a randomly chosen stroke case above a randomly chosen non-stroke case, averaged across every possible classification threshold — not just 0.50. A value of 50% would mean no better than random guessing; 100% would mean perfect separation between the two classes. 83.93% indicates the model has learned a genuinely useful, non-random signal for telling the two classes apart.

---

## 8. Precision-Recall Analysis

Because stroke cases make up only about 4.9% of the dataset, ROC-AUC alone can look more favorable than the model's real usefulness for the minority class, since it is partly driven by how well the model handles the (very large) majority class. Precision-Recall analysis focuses specifically on the stroke class.

**Average Precision = 23.05%**

This number summarizes precision across all recall levels into a single score. It should not be overstated: 23.05% is a modest number, and it reflects the real difficulty of this problem — a small number of positive cases, and no lab-based features. It is meaningfully above what a "no-skill" baseline would achieve (roughly equal to the stroke rate, ~4.9%), but it is not high in an absolute sense.

---

## 9. Threshold Analysis

The classification threshold is the probability cutoff above which a prediction is labeled "stroke risk." Changing it changes the precision/recall balance without retraining the model.

**Default threshold: 0.50**
**Alternative threshold investigated: 0.45**

| Metric | Threshold 0.50 | Threshold 0.45 |
|---|---|---|
| Recall | 72.00% | 74.00% |
| Precision | 18.85% | 16.82% |
| F1-score | 29.88% | 27.41% |
| Specificity | 84.05% | 81.17% |

**Trade-off:** Lowering the threshold to 0.45 gives a small recall gain (+2.00 points) but costs precision (−2.03 points), F1-score (−2.47 points), and specificity (−2.88 points). **0.45 is not better overall** — it is a different balance point, not a strict improvement.

**Decision: DEFAULT OPERATING THRESHOLD = 0.50.** Threshold 0.45 is documented as an alternative, recall-oriented option that a future deployment could consider, but it is not adopted as the default.

---

## 10. Threshold-Selection Leakage Prevention

The threshold analysis above was performed using **out-of-fold predictions generated entirely from training data** (5-fold stratified cross-validation, with SMOTENC and preprocessing refit inside each fold). The test set was never used to choose either threshold.

This matters because if the test set had been used to pick the threshold, the test-set performance numbers reported afterward would no longer be an independent, trustworthy estimate — the threshold would have been implicitly "tuned to the test answers," inflating the reported results. Selecting the threshold from training data only, and then applying it to the test set exactly once, keeps the final test evaluation honest.

---

## 11. Cross-Validation Stability (from Step 5)

For **Logistic Regression + SMOTENC**, the Step 5 cross-validation results report:

| Metric | Mean | Std. Dev. |
|---|---|---|
| Recall | 53.78% | 3.89% |
| Precision | 14.54% | Not reported in the current Step 6 results |
| F1-score | 22.86% | Not reported in the current Step 6 results |
| ROC-AUC | 81.97% | Not reported in the current Step 6 results |

**What this means:** the standard deviation of recall across the 5 folds (3.89 percentage points) is fairly small relative to its mean (53.78%), suggesting the model's recall behavior does not swing wildly from one training subset to another. This is a reassuring but limited signal — with roughly 40 stroke-labeled cases per fold, some fold-to-fold variation is expected simply from the small sample size, and this is not equivalent to validation on an independent clinical population.

*(Note: the single-run test recall of 72.00% is higher than the 5-fold CV mean of 53.78%. This kind of gap is expected with only 50 positive cases in the test set — a single evaluation split can land noticeably above or below the cross-validated average purely by chance.)*

---

## 12. Error Analysis

**False Negatives: 14 cases (at threshold 0.50)**

False negatives are particularly important in a screening-oriented system because a potentially higher-risk case may not be flagged. Each one represents a person whose dataset record carries a stroke label, but whom the model predicted as "not at elevated risk."

**False Positives: 155 cases (at threshold 0.50)**

These represent no-stroke-labeled records that the model flagged as elevated risk anyway. This means some users could be told their estimated risk is higher than their actual (dataset-labeled) outcome — an unwanted false alarm, but a much safer direction of error than a false negative.

This trade-off — accepting more false positives in exchange for better recall — is expected and intentional when handling a highly imbalanced dataset like this one. Without SMOTENC or another imbalance-handling method, the model would default toward almost never predicting the minority "stroke" class at all (as seen in the Step 5 baseline results), which is far worse for a screening tool.

---

## 13. User-Friendly Design

The model is deliberately built around information an ordinary person can provide without any medical equipment:

- Age
- Gender
- Hypertension (high blood pressure): Have you ever been told by a doctor or healthcare professional that you have high blood pressure?
- Heart disease (doctor-diagnosed heart condition)
- Ever married
- Work type
- Residence type (urban/rural)
- Smoking status

**BMI and average glucose level were intentionally excluded** to keep the questionnaire usable without lab tests or medical devices — a person can answer every question from memory.

---

## 14. Important Limitation

The model uses easily obtainable user information and intentionally excludes BMI and average glucose level to improve usability. This may reduce predictive performance compared with models using additional clinical measurements.

**The system is an educational risk-screening prototype. It is NOT a medical diagnosis. It should NOT replace a doctor, clinical assessment, or medical examination.**

---

## 15. Overall Interpretation

The model shows a genuinely useful ability to distinguish between the stroke and no-stroke classes (ROC-AUC ≈ 84%), and — thanks to SMOTENC — it detects a majority of stroke-labeled cases in the test set (recall = 72%). This is a meaningful improvement over a baseline model, which effectively fails to detect stroke cases at all.

At the same time, precision is low (≈19%), meaning most "at risk" flags are false alarms, and average precision (≈23%) confirms this is a modest, not a strong, level of overall performance on the minority class.

Given this balance, the model should be treated as a **risk-screening / educational prototype**, not a clinical-grade predictor. Further validation with larger, more representative, and ideally clinically-collected data would be required before any real-world medical use could be considered.

---

## 16. Step 6 Conclusion

**Selected candidate:** Logistic Regression + SMOTENC

**Default threshold:** 0.50

**Test performance:**

| Metric | Value |
|---|---|
| Accuracy | 83.46% |
| Precision | 18.85% |
| Recall | 72.00% |
| F1-score | 29.88% |
| Specificity | 84.05% |
| ROC-AUC | 83.93% |
| Average Precision | 23.05% |

**Confusion matrix:**

| | TP = 36 | FN = 14 |
|---|---|---|
| | FP = 155 | TN = 817 |

The model is **technically suitable to proceed to Step 7** for final production-model preparation. This conclusion comes with clearly documented limitations: modest precision, a feature set that intentionally excludes clinical measurements, and evaluation on a relatively small, imbalanced test set (50 positive cases) rather than an independently validated clinical population.

---

## 17. Step 7 Outcome

Step 7 has since been completed. Using the settings validated in Steps
5–6, it performed:

- Final model preparation
- Final preprocessing pipeline
- Final model training (on the full training data)
- Saving the trained model (`model/stroke_risk_model.pkl`)
- Saving the preprocessing pipeline (`model/preprocessor.pkl`)
- Saving the selected operating threshold (0.50, as the documented default)
- Preparing the model for integration with the Flask API

The saved artifacts were verified against this report's metrics (see
`training/test_saved_model.py` and `results/final_evaluation_report.csv`)
and match the values documented above.

---

**STEP 6 EVALUATION: COMPLETED. STEP 7 (FINAL MODEL + SAVED PREDICTION
PIPELINE): COMPLETED.**
