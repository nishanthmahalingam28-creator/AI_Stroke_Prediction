"""
test_saved_model.py
----------------------
STEP 7 - Loads the SAVED production artifacts from disk (does NOT
retrain anything) and runs example predictions to prove the saved
model can actually be reused -- exactly what the future Flask API
will need to do.

Run from the project root:
    python training/test_saved_model.py
"""

import sys
import os

# Allow importing prediction/predict.py when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.predict import predict_stroke_risk, validate_user_data


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_result(label, user_input, result):
    print(f"\n{label}")
    print("Input:")
    for k, v in user_input.items():
        print(f"  {k}: {v}")
    if result["valid"]:
        print(f"Probability:    {result['risk_score']:.4f}  ({result['risk_score']*100:.2f}%)")
        print(f"Prediction:     {result['prediction']}  "
              f"({'Elevated risk flagged' if result['prediction'] == 1 else 'Not flagged'})")
        print(f"Risk category:  {result['risk_category']}  (display category only, not clinical)")
        print(f"Threshold used: {result['threshold_used']}")
    else:
        print("Prediction FAILED -- validation errors:")
        for e in result["errors"]:
            print("  -", e)


def main():
    section("CRITICAL SAVED-MODEL TEST")
    print("This script loads the model, preprocessor, and config that were")
    print("saved to disk by train_final_model.py. Nothing is retrained here.")
    print("This proves the saved artifacts can be reloaded and reused --")
    print("exactly what the future Flask API will do.")

    # ----------------------------------------------------------------
    # THREE EXAMPLE USERS (8 production features only)
    # ----------------------------------------------------------------
    section("EXAMPLE 1: LOWER-RISK PROFILE")
    example_1 = {
        "age": 28,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "never smoked",
    }
    result_1 = predict_stroke_risk(example_1)
    print_result("Example 1 (lower-risk)", example_1, result_1)

    section("EXAMPLE 2: MODERATE-RISK PROFILE")
    example_2 = {
        "age": 58,
        "gender": "Male",
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Self-employed",
        "Residence_type": "Rural",
        "smoking_status": "formerly smoked",
    }
    result_2 = predict_stroke_risk(example_2)
    print_result("Example 2 (moderate-risk)", example_2, result_2)

    section("EXAMPLE 3: HIGHER-RISK PROFILE")
    example_3 = {
        "age": 76,
        "gender": "Male",
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Govt_job",
        "Residence_type": "Urban",
        "smoking_status": "smokes",
    }
    result_3 = predict_stroke_risk(example_3)
    print_result("Example 3 (higher-risk)", example_3, result_3)

    # ----------------------------------------------------------------
    # INPUT VALIDATION TESTS (must fail cleanly, not crash)
    # ----------------------------------------------------------------
    section("INPUT VALIDATION TESTS")

    invalid_cases = {
        "age < 0": {**example_1, "age": -5},
        "age > 120": {**example_1, "age": 150},
        "missing required field (smoking_status)": {
            k: v for k, v in example_1.items() if k != "smoking_status"
        },
        "invalid gender": {**example_1, "gender": "Unknown123"},
        "invalid work_type": {**example_1, "work_type": "Freelancer"},
        "invalid smoking_status": {**example_1, "smoking_status": "vapes"},
        "invalid hypertension value": {**example_1, "hypertension": 5},
        "invalid heart_disease value": {**example_1, "heart_disease": "yes"},
    }

    all_correctly_rejected = True
    for label, bad_input in invalid_cases.items():
        result = predict_stroke_risk(bad_input)
        status = "REJECTED (correct)" if not result["valid"] else "ACCEPTED (PROBLEM!)"
        if result["valid"]:
            all_correctly_rejected = False
        print(f"\n[{label}] -> {status}")
        if not result["valid"]:
            for e in result["errors"]:
                print("  -", e)

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    section("SAVED MODEL TEST SUMMARY")
    all_valid_predictions_worked = all(r["valid"] for r in [result_1, result_2, result_3])
    print(f"Valid example predictions succeeded: {all_valid_predictions_worked}")
    print(f"All invalid inputs correctly rejected: {all_correctly_rejected}")

    if all_valid_predictions_worked and all_correctly_rejected:
        print("\nSAVED MODEL TEST: PASSED")
        print("The saved model, preprocessor, and config can be loaded from")
        print("disk and used to generate predictions and reject bad input --")
        print("this is exactly what the Flask API in Step 8 will rely on.")
    else:
        print("\nSAVED MODEL TEST: ISSUES FOUND -- see output above.")

    print("\nSTEP 7 SAVED-MODEL TEST COMPLETE")


if __name__ == "__main__":
    main()
