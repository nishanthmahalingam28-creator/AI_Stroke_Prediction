"""
clean_dataset.py
-----------------
STEP 2 - Cleaning for the Stroke Risk Prediction project.

This script produces a CLEANED COPY of the dataset. It never touches
the original file.

What "cleaning" means here, based on explore_dataset.py findings:
  - Drop the 'id' column (identifier, not predictive, would leak
    row order/index information if accidentally used as a feature)
  - Keep every other column, INCLUDING bmi and avg_glucose_level,
    even though Model B will not use them. We keep the full
    information here so the cleaned file stays reusable and honest;
    Model B simply selects a subset of columns later during training.
  - Do NOT impute bmi. Model B does not use it, and inventing values
    for a column we're not even going to model would be fabricating
    data for no reason. If a future Model A (that uses labs) is ever
    built, imputation should happen inside that model's own pipeline
    fold-by-fold, not here.
  - Do NOT touch 'Unknown' in smoking_status. It is a valid survey
    answer (the person didn't know/report their smoking status), not
    a missing-value placeholder, so it stays as its own category.
  - No duplicate rows were found, so no deduplication is needed.
  - No invalid values (out-of-range age/bmi/glucose, or hypertension/
    heart_disease/stroke values outside {0,1}) were found, so no
    value-correction step is needed.
  - The single 'Other' gender row is kept in the cleaned dataset.
    It is NOT dropped here -- that is a modeling-time decision for
    Step 3, not a cleaning decision. Removing real records is a
    training choice, not a cleaning task.

Run from the project root:
    python training/clean_dataset.py
"""

import pandas as pd
import os

RAW_PATH = os.path.join("dataset", "healthcare-dataset-stroke-data.csv")
CLEANED_PATH = os.path.join("data", "cleaned_stroke_data.csv")


def main():
    print("Loading raw dataset from:", RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)

    original_columns = set(df.columns)

    # ------------------------------------------------------------
    # Drop 'id' -- not a feature, just a row identifier
    # ------------------------------------------------------------
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        print("Dropped column: id")

    # ------------------------------------------------------------
    # Confirm no duplicate rows (re-check, don't assume)
    # ------------------------------------------------------------
    dup_count = df.duplicated().sum()
    print("Duplicate rows found:", dup_count)
    if dup_count > 0:
        df = df.drop_duplicates()
        print("Dropped", dup_count, "duplicate rows")
    else:
        print("No duplicates to drop.")

    # ------------------------------------------------------------
    # We intentionally do NOT impute bmi or avg_glucose_level here.
    # We intentionally do NOT touch smoking_status 'Unknown'.
    # We intentionally do NOT drop the 'Other' gender row.
    # These are all left as-is and documented above.
    # ------------------------------------------------------------
    print("bmi missing values kept as-is (not imputed):", df["bmi"].isnull().sum())
    print("smoking_status 'Unknown' rows kept as-is:",
          (df["smoking_status"] == "Unknown").sum())
    print("gender 'Other' rows kept as-is:", (df["gender"] == "Other").sum())

    # ------------------------------------------------------------
    # Save cleaned dataset (original file is never modified)
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    print("\nCleaned dataset saved to:", CLEANED_PATH)
    print("Cleaned shape:", df.shape)
    print("Columns kept:", list(df.columns))

    removed_cols = original_columns - set(df.columns)
    print("Columns removed:", removed_cols if removed_cols else "None (only 'id' dropped)")

    print("\nOriginal file left untouched at:", RAW_PATH)
    print("STEP 2 CLEANING COMPLETE")


if __name__ == "__main__":
    main()
