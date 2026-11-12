"""Mathematical definitions and calculators for AI fairness and equity metrics.

All metrics are computed strictly from real predictions, ground truth labels, and sensitive attributes.
ZERO sample data, zero fake output.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_group_confusion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate accuracy, TPR, FPR, TNR, FNR, and selection rate for a single subgroup.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    total = len(y_true)

    if total == 0:
        return {
            "count": 0,
            "accuracy": 0.0,
            "selection_rate": 0.0,
            "tpr": 0.0,
            "fpr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    selection_rate = float(np.mean(y_pred == 1))
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion Matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "count": int(total),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": accuracy,
        "selection_rate": selection_rate,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_fairness_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    sensitive_features: Union[pd.Series, np.ndarray],
    attribute_name: str = "Demographic Attribute",
) -> Dict[str, Any]:
    """
    Compute standard algorithmic fairness metrics across groups for a given sensitive attribute.
    
    Metrics:
    - Group Selection Rates: P(Y_hat=1 | A=g)
    - Demographic Parity Difference (DPD): max(SR) - min(SR)
    - Disparate Impact Ratio (DIR): min(SR) / max(SR)
    - Equal Opportunity Difference (EOD): max(TPR) - min(TPR)
    - Equalized Odds Difference: max(max(TPR)-min(TPR), max(FPR)-min(FPR))
    - Fairness Score (0-100)
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    sens_arr = np.asarray(sensitive_features).astype(str)

    unique_groups = np.unique(sens_arr)
    if len(unique_groups) < 2:
        return {
            "status": "error",
            "message": f"Sensitive feature '{attribute_name}' contains fewer than 2 groups in test split.",
            "groups": list(unique_groups),
            "fairness_score": 0.0,
        }

    group_metrics: Dict[str, Dict[str, float]] = {}
    selection_rates: Dict[str, float] = {}
    tprs: Dict[str, float] = {}
    fprs: Dict[str, float] = {}

    for g in unique_groups:
        mask = (sens_arr == g)
        metrics = compute_group_confusion_metrics(y_true_arr[mask], y_pred_arr[mask])
        group_metrics[g] = metrics
        selection_rates[g] = metrics["selection_rate"]
        tprs[g] = metrics["tpr"]
        fprs[g] = metrics["fpr"]

    sr_vals = list(selection_rates.values())
    max_sr = max(sr_vals)
    min_sr = min(sr_vals)
    
    # 1. Demographic Parity Difference (DPD)
    demographic_parity_diff = float(max_sr - min_sr)

    # 2. Disparate Impact Ratio (DIR) (4/5ths Rule)
    if max_sr > 0:
        disparate_impact_ratio = float(min_sr / max_sr)
    else:
        # If no group receives positive outcomes, selection rate is identical (0.0 == 0.0)
        disparate_impact_ratio = 1.0

    # 3. Equal Opportunity Difference (EOD)
    tpr_vals = list(tprs.values())
    equal_opportunity_diff = float(max(tpr_vals) - min(tpr_vals))

    # 4. Equalized Odds Difference (Max of TPR diff and FPR diff)
    fpr_vals = list(fprs.values())
    fpr_diff = float(max(fpr_vals) - min(fpr_vals))
    equalized_odds_diff = float(max(equal_opportunity_diff, fpr_diff))

    # 5. Composite Fairness Score (0-100)
    # DIR component: normalized against 0.80 benchmark (capped at 1.0)
    dir_component = min(1.0, disparate_impact_ratio / 0.80) if disparate_impact_ratio >= 0 else 0.0
    dpd_component = max(0.0, 1.0 - demographic_parity_diff)
    eod_component = max(0.0, 1.0 - equal_opportunity_diff)

    # Weighted: DIR (40%), DPD (35%), EOD (25%)
    fairness_score = 100.0 * (0.40 * dir_component + 0.35 * dpd_component + 0.25 * eod_component)
    fairness_score = float(np.clip(fairness_score, 0.0, 100.0))

    # 6. Overall Performance
    overall_metrics = compute_group_confusion_metrics(y_true_arr, y_pred_arr)

    # 7. Generate Plain-English Narrative Summary
    # Identify privileged & underprivileged groups based on selection rate
    best_group = max(selection_rates, key=selection_rates.get)
    worst_group = min(selection_rates, key=selection_rates.get)
    dir_pct = (disparate_impact_ratio * 100)
    
    if disparate_impact_ratio >= 0.80:
        dir_narrative = (
            f"The disparate impact ratio stands at {disparate_impact_ratio:.3f} ({dir_pct:.1f}%), "
            "which complies with the standard EEOC 80% (four-fifths) rule."
        )
    else:
        dir_narrative = (
            f"The disparate impact ratio is {disparate_impact_ratio:.3f} ({dir_pct:.1f}%), "
            f"falling below the 80% threshold. Group '{worst_group}' receives favorable decisions "
            f"at only {disparate_impact_ratio:.1%} the rate of group '{best_group}'."
        )

    summary_text = (
        f"For demographic attribute '{attribute_name}', FairLens audited {len(y_true_arr):,} evaluation instances "
        f"across {len(unique_groups)} groups. {dir_narrative} "
        f"The Demographic Parity Difference is {demographic_parity_diff:.1%}, and the Equal Opportunity Difference is "
        f"{equal_opportunity_diff:.1%}. The resulting composite Fairness Score is {round(fairness_score):d}/100."
    )

    return {
        "status": "success",
        "attribute_name": attribute_name,
        "unique_groups": list(unique_groups),
        "group_metrics": group_metrics,
        "selection_rates": selection_rates,
        "tprs": tprs,
        "fprs": fprs,
        "demographic_parity_difference": demographic_parity_diff,
        "disparate_impact_ratio": disparate_impact_ratio,
        "equal_opportunity_difference": equal_opportunity_diff,
        "equalized_odds_difference": equalized_odds_diff,
        "fairness_score": fairness_score,
        "overall_metrics": overall_metrics,
        "best_group": best_group,
        "worst_group": worst_group,
        "summary_text": summary_text,
    }


def compute_intersectional_fairness(
    df: pd.DataFrame,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    attr1: str,
    attr2: str,
) -> pd.DataFrame:
    """
    Compute selection rates across 2D intersectional demographic combinations (e.g. Gender x Age).
    Returns a pivot table of selection rates.
    """
    df_eval = pd.DataFrame({
        "attr1": df[attr1].astype(str),
        "attr2": df[attr2].astype(str),
        "y_pred": np.asarray(y_pred).astype(int),
        "y_true": np.asarray(y_true).astype(int),
    })

    # Group by both attributes and calculate selection rate
    grouped = df_eval.groupby(["attr1", "attr2"])["y_pred"].agg(
        selection_rate="mean",
        sample_count="count",
    ).reset_index()

    pivot_rates = grouped.pivot(index="attr1", columns="attr2", values="selection_rate").fillna(0.0)
    pivot_counts = grouped.pivot(index="attr1", columns="attr2", values="sample_count").fillna(0)

    return pivot_rates, pivot_counts

