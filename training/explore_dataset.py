"""
explore_dataset.py
-------------------
STEP 2 - Data exploration for the Stroke Risk Prediction project.

This script ONLY looks at the data. It does not clean anything,
does not train any model, and does not modify the original CSV.

Run from the project root:
    python training/explore_dataset.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it works without a display (saves images instead of popping up windows)
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_PATH = os.path.join("dataset", "healthcare-dataset-stroke-data.csv")
OUTPUT_DIR = os.path.join("training", "eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns that Model B (the user-friendly, no-lab-test model) will use.
MODEL_B_FEATURES = [
    "age", "gender", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "smoking_status"
]
TARGET = "stroke"


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # ----------------------------------------------------------------
    # 1. LOAD THE DATASET
    # ----------------------------------------------------------------
    section("1. LOADING DATASET")
    df = pd.read_csv(DATA_PATH)
    print("File loaded successfully:", DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("Column names:", list(df.columns))

    # ----------------------------------------------------------------
    # 2. BASIC INFORMATION
    # ----------------------------------------------------------------
    section("2. HEAD (first 5 rows)")
    print(df.head())

    section("2. TAIL (last 5 rows)")
    print(df.tail())

    section("2. INFO")
    df.info()

    section("2. DESCRIBE (all columns)")
    print(df.describe(include="all"))

    # ----------------------------------------------------------------
    # 3. DUPLICATE RECORDS
    # ----------------------------------------------------------------
    section("3. DUPLICATE CHECK")
    full_row_duplicates = df.duplicated().sum()
    id_duplicates = df["id"].duplicated().sum()
    print("Fully duplicated rows:", full_row_duplicates)
    print("Duplicated id values:", id_duplicates)

    # ----------------------------------------------------------------
    # 4. MISSING VALUES
    # ----------------------------------------------------------------
    section("4. MISSING VALUES")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_table = pd.DataFrame({
        "Missing": missing_count,
        "Percentage": missing_pct.round(2)
    })
    print(missing_table)

    # ----------------------------------------------------------------
    # 5. UNIQUE VALUES FOR CATEGORICAL / DISCRETE COLUMNS
    # ----------------------------------------------------------------
    section("5. UNIQUE VALUE COUNTS")
    categorical_like = [
        "gender", "hypertension", "heart_disease", "ever_married",
        "work_type", "Residence_type", "smoking_status", "stroke"
    ]
    for col in categorical_like:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))

    # ----------------------------------------------------------------
    # 6. INVALID / SUSPICIOUS VALUES
    # ----------------------------------------------------------------
    section("6. INVALID VALUE CHECKS")
    problems = []

    if (df["age"] <= 0).any():
        n = (df["age"] <= 0).sum()
        problems.append(f"age <= 0 found in {n} rows")
    print("age <= 0 rows:", (df["age"] <= 0).sum())
    print("age min/max:", df["age"].min(), "/", df["age"].max())

    if (df["bmi"] < 0).any():
        n = (df["bmi"] < 0).sum()
        problems.append(f"negative bmi found in {n} rows")
    print("Negative bmi rows:", (df["bmi"] < 0).sum())

    if (df["avg_glucose_level"] < 0).any():
        n = (df["avg_glucose_level"] < 0).sum()
        problems.append(f"negative avg_glucose_level found in {n} rows")
    print("Negative avg_glucose_level rows:", (df["avg_glucose_level"] < 0).sum())

    bad_hyp = df[~df["hypertension"].isin([0, 1])]
    if len(bad_hyp) > 0:
        problems.append(f"hypertension has {len(bad_hyp)} values outside {{0,1}}")
    print("hypertension invalid rows:", len(bad_hyp))

    bad_heart = df[~df["heart_disease"].isin([0, 1])]
    if len(bad_heart) > 0:
        problems.append(f"heart_disease has {len(bad_heart)} values outside {{0,1}}")
    print("heart_disease invalid rows:", len(bad_heart))

    bad_stroke = df[~df[TARGET].isin([0, 1])]
    if len(bad_stroke) > 0:
        problems.append(f"stroke has {len(bad_stroke)} values outside {{0,1}}")
    print("stroke invalid rows:", len(bad_stroke))

    print("\nSummary of problems found:")
    if problems:
        for p in problems:
            print(" -", p)
    else:
        print(" - No invalid/impossible values detected in the checked columns.")

    # ----------------------------------------------------------------
    # 7. TARGET DISTRIBUTION
    # ----------------------------------------------------------------
    section("7. TARGET DISTRIBUTION (stroke)")
    counts = df[TARGET].value_counts()
    total = len(df)
    no_stroke = counts.get(0, 0)
    stroke = counts.get(1, 0)
    print(f"No stroke (0): {no_stroke}  ({no_stroke/total*100:.2f}%)")
    print(f"Stroke (1):    {stroke}  ({stroke/total*100:.2f}%)")
    print(f"Total: {total}")

    plt.figure(figsize=(5, 4))
    counts.sort_index().plot(kind="bar", color=["#4C9AFF", "#FF5C5C"])
    plt.xticks([0, 1], ["No Stroke", "Stroke"], rotation=0)
    plt.ylabel("Number of records")
    plt.title("Target Distribution: No Stroke vs Stroke")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "target_distribution.png")
    plt.savefig(chart_path)
    plt.close()
    print("Saved chart to:", chart_path)

    # ----------------------------------------------------------------
    # 8. FEATURE DISTRIBUTIONS (numeric columns)
    # ----------------------------------------------------------------
    section("8. NUMERIC FEATURE DISTRIBUTIONS")
    for col in ["age", "avg_glucose_level", "bmi"]:
        print(f"\n{col}:")
        print(df[col].describe())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["age", "avg_glucose_level", "bmi"]):
        df[col].dropna().hist(bins=30, ax=ax, color="#4C9AFF", edgecolor="white")
        ax.set_title(col)
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "numeric_distributions.png")
    plt.savefig(hist_path)
    plt.close()
    print("Saved chart to:", hist_path)

    # ----------------------------------------------------------------
    # 9. MODEL B FEATURE ANALYSIS
    # ----------------------------------------------------------------
    section("9. MODEL B FEATURE ANALYSIS")
    numeric_features = []
    categorical_features = []
    binary_features = []

    for col in MODEL_B_FEATURES:
        n_unique = df[col].nunique()
        dtype = df[col].dtype
        if dtype in [np.float64, np.int64] and n_unique > 2:
            numeric_features.append(col)
        elif n_unique == 2:
            binary_features.append(col)
        else:
            categorical_features.append(col)

    print("Numerical features:", numeric_features)
    print("Binary features:", binary_features)
    print("Categorical (multi-class) features:", categorical_features)

    print("\nMissing values within Model B feature set:")
    print(df[MODEL_B_FEATURES].isnull().sum())

    # ----------------------------------------------------------------
    # DATA QUALITY REPORT
    # ----------------------------------------------------------------
    section("DATA QUALITY REPORT")
    print("DATASET SUMMARY")
    print("-" * 40)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Duplicates:", full_row_duplicates)
    print("Total missing cells:", int(df.isnull().sum().sum()))
    print("Invalid value issues:", len(problems))

    print("\nTARGET DISTRIBUTION")
    print("-" * 40)
    print("No Stroke:", no_stroke)
    print("Stroke:", stroke)
    print(f"Stroke percentage: {stroke/total*100:.2f}%")

    print("\nMODEL B FEATURES")
    print("-" * 40)
    print("Numerical:", numeric_features)
    print("Categorical:", categorical_features)
    print("Binary:", binary_features)

    print("\nPOTENTIAL PROBLEMS")
    print("-" * 40)
    if problems:
        for p in problems:
            print(" -", p)
    else:
        print(" - No invalid values found.")
    print(" - bmi has", int(df['bmi'].isnull().sum()), "missing values (not used by Model B)")
    print(" - stroke target is heavily imbalanced (~5% positive class)")
    print(" - gender has a category ('Other') with only 1 record")

    print("\nRECOMMENDED CLEANING")
    print("-" * 40)
    print(" - Drop 'id' (identifier, not predictive)")
    print(" - Keep 'bmi' and 'avg_glucose_level' in the cleaned CSV for completeness,")
    print("   but Model B will not use them, so their missing values do not need imputation")
    print("   for this project's purposes")
    print(" - Model B categorical columns (gender, ever_married, work_type,")
    print("   Residence_type, smoking_status) have zero missing values, so no")
    print("   imputation is required there")
    print(" - 'Unknown' in smoking_status is a real, valid category (not a missing")
    print("   value marker) and should be kept as its own category, not imputed away")
    print(" - Class imbalance in 'stroke' should be handled at TRAIN time only")
    print("   (e.g. class_weight or SMOTE on the training split) -- this is a Step 3")
    print("   modeling decision, not a Step 2 cleaning decision")

    print("\nSTEP 2 EXPLORATION COMPLETE")


if __name__ == "__main__":
    main()
