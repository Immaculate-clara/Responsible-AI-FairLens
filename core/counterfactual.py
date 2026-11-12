"""Counterfactual perturbation and interactive What-If simulation engine for FairLens."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from core.fairness_metrics import compute_fairness_metrics


def predict_single_instance(
    model: Any,
    preprocessor: Any,
    input_data: Dict[str, Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Generate model prediction and probability for a single user-specified instance.
    """
    df_instance = pd.DataFrame([input_data])

    try:
        X_proc = preprocessor.transform(df_instance)
    except Exception as e:
        return {
            "error": f"Preprocessing failed: {str(e)}",
            "prediction": 0,
            "probability": 0.0,
        }

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_proc)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X_proc)[0])
        prob = 1.0 / (1.0 + np.exp(-score))
    else:
        prob = float(model.predict(X_proc)[0])

    decision = 1 if prob >= threshold else 0

    return {
        "probability": prob,
        "decision": decision,
        "threshold": threshold,
    }


def run_counterfactual_flip_test(
    model: Any,
    preprocessor: Any,
    base_instance: Dict[str, Any],
    sensitive_col: str,
    all_groups: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Counterfactual Fairness Test:
    Keep all non-sensitive attributes strictly constant and toggle ONLY the sensitive attribute
    across all available demographic groups. Check if the decision flips.
    """
    original_group = str(base_instance.get(sensitive_col, ""))
    base_result = predict_single_instance(model, preprocessor, base_instance, threshold=threshold)
    base_decision = base_result["decision"]
    base_prob = base_result["probability"]

    variations = []
    has_decision_flip = False

    for group in all_groups:
        perturbed_instance = dict(base_instance)
        perturbed_instance[sensitive_col] = group

        res = predict_single_instance(model, preprocessor, perturbed_instance, threshold=threshold)
        decision_flipped = (res["decision"] != base_decision)
        if decision_flipped:
            has_decision_flip = True

        variations.append({
            "group": group,
            "probability": res["probability"],
            "decision": res["decision"],
            "is_original": (str(group) == original_group),
            "flipped": decision_flipped,
            "prob_delta": res["probability"] - base_prob,
        })

    return {
        "original_group": original_group,
        "base_decision": base_decision,
        "base_probability": base_prob,
        "has_decision_flip": has_decision_flip,
        "variations": variations,
    }


def compute_live_threshold_metrics(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    sensitive_series: pd.Series,
    threshold: float,
    attribute_name: str,
) -> Dict[str, Any]:
    """
    Dynamically recalculate fairness and accuracy metrics as the decision threshold changes.
    """
    preds = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_test, preds))
    fairness = compute_fairness_metrics(y_test, preds, sensitive_series, attribute_name=attribute_name)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "fairness_score": fairness["fairness_score"],
        "disparate_impact_ratio": fairness["disparate_impact_ratio"],
        "demographic_parity_diff": fairness["demographic_parity_difference"],
        "equal_opportunity_diff": fairness["equal_opportunity_difference"],
        "fairness_summary": fairness["summary_text"],
        "selection_rates": fairness["selection_rates"],
    }

