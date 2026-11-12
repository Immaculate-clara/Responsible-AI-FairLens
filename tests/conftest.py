"""Pytest configuration and purely in-memory synthetic fixtures for FairLens testing.

CRITICAL RULE: ZERO sample CSV/XLSX files are stored on disk.
All test datasets are generated purely in RAM at test execution time.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_biased_dataset() -> pd.DataFrame:
    """
    Generate an in-memory biased credit decision dataset with known disparities.
    Demographic features: 'gender' (Male, Female) and 'age_group' (Young, Middle, Senior).
    Target: 'approved' (1 = Approved, 0 = Denied).
    """
    np.random.seed(42)
    n = 300

    # Demographic attributes
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age_group = np.random.choice(["Young", "Middle", "Senior"], size=n, p=[0.3, 0.5, 0.2])

    # Continuous financial features
    income = np.random.normal(50000, 15000, size=n).clip(15000, 120000)
    credit_score = np.random.normal(650, 80, size=n).clip(350, 850)
    loan_amount = np.random.normal(20000, 8000, size=n).clip(2000, 50000)

    # Induce synthetic historical bias:
    # Male applicants receive a systematic 0.70 logit boost in historical approval rates
    logits = (
        (income / 50000.0) * 1.2
        + (credit_score / 650.0) * 1.5
        - (loan_amount / 20000.0) * 0.8
        + (gender == "Male") * 0.90
        - 1.8
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    approved = (np.random.rand(n) < probs).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"APP_{i:04d}" for i in range(n)],
        "gender": gender,
        "age_group": age_group,
        "income": np.round(income, 2),
        "credit_score": np.round(credit_score, 0),
        "loan_amount": np.round(loan_amount, 2),
        "approved": approved,
    })
    return df


@pytest.fixture
def hand_calculated_fairness_fixture():
    """
    Pre-calculated tiny synthetic scenario for exact hand verification.
    
    Subgroup A (4 samples):
      Ground truth: [1, 1, 0, 0]
      Predictions:  [1, 1, 1, 0] -> 3 positives out of 4 -> Selection Rate = 0.75
      TP = 2, FP = 1, TN = 1, FN = 0
      TPR (Recall) = 2 / 2 = 1.0
      FPR = 1 / 2 = 0.5
      Accuracy = 3 / 4 = 0.75
      
    Subgroup B (4 samples):
      Ground truth: [1, 1, 0, 0]
      Predictions:  [1, 0, 0, 0] -> 1 positive out of 4 -> Selection Rate = 0.25
      TP = 1, FP = 0, TN = 2, FN = 1
      TPR (Recall) = 1 / 2 = 0.5
      FPR = 0 / 2 = 0.0
      Accuracy = 3 / 4 = 0.75

    Hand calculated expectations:
      - Demographic Parity Difference = |0.75 - 0.25| = 0.50
      - Disparate Impact Ratio = 0.25 / 0.75 = 1/3 ~ 0.333333
      - Equal Opportunity Difference = |1.0 - 0.5| = 0.50
      - Equalized Odds Difference = max(|1.0 - 0.5|, |0.5 - 0.0|) = 0.50
    """
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    sens = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "sensitive": sens,
        "expected_sr_A": 0.75,
        "expected_sr_B": 0.25,
        "expected_dpd": 0.50,
        "expected_dir": 1.0 / 3.0,
        "expected_eod": 0.50,
        "expected_eq_odds": 0.50,
        "expected_tpr_A": 1.0,
        "expected_tpr_B": 0.5,
    }

