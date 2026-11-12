"""Bias Mitigation Algorithms for FairLens: Pre-, In-, and Post-processing techniques.

Supported algorithms:
1. Pre-processing: Sample Reweighing (fairness-aware sample weights)
2. In-processing: Fairlearn ExponentiatedGradient reduction
3. Post-processing: Fairlearn ThresholdOptimizer

All reports use the precise terminology: 'bias reduced' (never 'bias removed').
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
import plotly.graph_objects as go
from core.fairness_metrics import compute_fairness_metrics
from core.utils import apply_theme_to_figure, COLOR_FAIR, COLOR_PRIMARY, COLOR_BIASED, COLOR_CAUTION


def compute_reweighing_weights(
    y: np.ndarray,
    sensitive_features: np.ndarray,
) -> np.ndarray:
    """
    Calculate Kamiran & Calders (2012) sample weights for pre-processing mitigation.
    W(A=a, Y=y) = [P(A=a) * P(Y=y)] / P(A=a, Y=y)
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(sensitive_features).astype(str)
    n = len(y)

    weights = np.ones(n, dtype=float)
    unique_groups = np.unique(s)
    unique_labels = np.unique(y)

    # Probabilities
    p_a = {g: np.mean(s == g) for g in unique_groups}
    p_y = {label: np.mean(y == label) for label in unique_labels}

    for g in unique_groups:
        for label in unique_labels:
            mask = (s == g) & (y == label)
            p_joint = np.mean(mask)
            if p_joint > 0:
                expected = p_a[g] * p_y[label]
                weight_val = expected / p_joint
                weights[mask] = weight_val

    # Normalize weights so sum equals n
    weights = weights * (n / np.sum(weights))
    return weights


def run_reweighing_mitigation(
    prepared_data: Dict[str, Any],
    sensitive_col: str,
    base_model_type: str = "Logistic Regression",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train estimator with fairness-aware sample reweighing."""
    X_train = prepared_data["X_train_proc"]
    X_test = prepared_data["X_test_proc"]
    y_train = prepared_data["y_train"]
    y_test = prepared_data["y_test"]

    s_train = prepared_data["X_train_df"][sensitive_col].astype(str).values
    s_test = prepared_data["sensitive_test_df"][sensitive_col].astype(str).values

    # Compute weights
    weights = compute_reweighing_weights(y_train, s_train)

    if base_model_type == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)

    clf.fit(X_train, y_train, sample_weight=weights)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

    fairness = compute_fairness_metrics(y_test, y_pred, s_test, attribute_name=sensitive_col)
    acc = float(accuracy_score(y_test, y_pred))

    return {
        "method": "Pre-processing: Sample Reweighing",
        "description": "Adjusted historical observation weights during training to balance demographic outcome frequencies.",
        "model": clf,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "fairness": fairness,
    }


def run_exponentiated_gradient_mitigation(
    prepared_data: Dict[str, Any],
    sensitive_col: str,
    constraint_type: str = "DemographicParity",
    random_state: int = 42,
) -> Dict[str, Any]:
    """In-processing reduction using Fairlearn ExponentiatedGradient."""
    X_train = prepared_data["X_train_proc"]
    X_test = prepared_data["X_test_proc"]
    y_train = prepared_data["y_train"]
    y_test = prepared_data["y_test"]

    s_train = prepared_data["X_train_df"][sensitive_col].astype(str).values
    s_test = prepared_data["sensitive_test_df"][sensitive_col].astype(str).values

    constraint = DemographicParity() if constraint_type == "DemographicParity" else EqualizedOdds()
    base_estimator = LogisticRegression(max_iter=1000, random_state=random_state)

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.01,
        max_iter=50,
    )

    mitigator.fit(X_train, y_train, sensitive_features=s_train)
    y_pred = mitigator.predict(X_test)

    fairness = compute_fairness_metrics(y_test, y_pred, s_test, attribute_name=sensitive_col)
    acc = float(accuracy_score(y_test, y_pred))

    return {
        "method": f"In-processing: Exponentiated Gradient ({constraint_type})",
        "description": f"Solved a min-max game with {constraint_type} fairness constraint directly during optimization.",
        "model": mitigator,
        "y_pred": y_pred,
        "y_prob": y_pred.astype(float),
        "accuracy": acc,
        "fairness": fairness,
    }


def run_threshold_optimizer_mitigation(
    base_model: Any,
    prepared_data: Dict[str, Any],
    sensitive_col: str,
    constraint_type: str = "demographic_parity",
) -> Dict[str, Any]:
    """Post-processing threshold tuning using Fairlearn ThresholdOptimizer."""
    X_train = prepared_data["X_train_proc"]
    X_test = prepared_data["X_test_proc"]
    y_train = prepared_data["y_train"]
    y_test = prepared_data["y_test"]

    s_train = prepared_data["X_train_df"][sensitive_col].astype(str).values
    s_test = prepared_data["sensitive_test_df"][sensitive_col].astype(str).values

    c_name = "demographic_parity" if "demographic" in constraint_type.lower() else "equalized_odds"

    optimizer = ThresholdOptimizer(
        estimator=base_model,
        constraints=c_name,
        prefit=True,
        predict_method="predict_proba" if hasattr(base_model, "predict_proba") else "predict",
    )

    optimizer.fit(X_train, y_train, sensitive_features=s_train)
    y_pred = optimizer.predict(X_test, sensitive_features=s_test)

    fairness = compute_fairness_metrics(y_test, y_pred, s_test, attribute_name=sensitive_col)
    acc = float(accuracy_score(y_test, y_pred))

    return {
        "method": f"Post-processing: Threshold Optimizer ({c_name.replace('_', ' ').title()})",
        "description": f"Optimized separate decision cutoffs per demographic group to satisfy {c_name} post-hoc.",
        "model": optimizer,
        "y_pred": y_pred,
        "y_prob": y_pred.astype(float),
        "accuracy": acc,
        "fairness": fairness,
    }


def create_tradeoff_chart(runs: List[Dict[str, Any]]) -> go.Figure:
    """Create interactive 2D scatter plot of Accuracy vs Fairness Score across models."""
    names = [r["name"] for r in runs]
    accuracies = [r["accuracy"] * 100 for r in runs]
    scores = [r["fairness_score"] for r in runs]
    colors = [COLOR_FAIR if s >= 80 else (COLOR_CAUTION if s >= 60 else COLOR_BIASED) for s in scores]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=accuracies,
                y=scores,
                mode="markers+text",
                text=names,
                textposition="top center",
                marker=dict(size=14, color=colors, line=dict(width=1.5, color="#ffffff")),
                hovertext=[
                    f"<b>{n}</b><br>Accuracy: {a:.2f}%<br>Fairness Score: {s:.1f}/100"
                    for n, a, s in zip(names, accuracies, scores)
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="<b>Fairness vs. Accuracy Pareto Trade-off Frontier</b>",
        xaxis_title="Overall Accuracy (%)",
        yaxis_title="Fairness Score (0 - 100)",
        yaxis_range=[0, 105],
        height=380,
    )
    return apply_theme_to_figure(fig)

