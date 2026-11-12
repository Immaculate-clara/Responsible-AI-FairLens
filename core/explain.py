"""Explainability and Proxy Feature Detection for FairLens using SHAP and statistical associations."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import plotly.graph_objects as go
from core.utils import apply_theme_to_figure, COLOR_PRIMARY, COLOR_BIASED, COLOR_CAUTION, COLOR_CYAN


def compute_shap_explanations(
    model: Any,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: List[str],
    sensitive_test_series: pd.Series,
    max_eval_samples: int = 150,
) -> Dict[str, Any]:
    """
    Compute SHAP explanations and per-group feature attribution comparisons.
    Handles both tree-based models and linear models efficiently.
    """
    import shap

    # Subsample if test set is large for swift interactive rendering
    n_samples = min(len(X_test_proc), max_eval_samples)
    eval_indices = np.random.RandomState(42).choice(len(X_test_proc), n_samples, replace=False)
    X_sub = X_test_proc[eval_indices]
    sens_sub = sensitive_test_series.iloc[eval_indices].astype(str).values

    try:
        # Determine appropriate SHAP explainer
        model_class_name = model.__class__.__name__.lower()
        if "forest" in model_class_name or "tree" in model_class_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sub)
        else:
            # Linear or general model: use background sample
            background = shap.sample(X_train_proc, min(50, len(X_train_proc)), random_state=42)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_sub)
    except Exception:
        # Fallback to general KernelExplainer on small sample
        background = shap.sample(X_train_proc, min(20, len(X_train_proc)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, background)
        shap_values = explainer.shap_values(X_sub[:30])
        X_sub = X_sub[:30]
        sens_sub = sens_sub[:30]

    # Handle multi-class / binary list output from TreeExplainer
    if isinstance(shap_values, list):
        # In binary classification, index 1 corresponds to positive outcome
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif len(shap_values.shape) == 3:
        # shape: (n_samples, n_features, n_classes)
        shap_matrix = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
    else:
        shap_matrix = shap_values

    # 1. Global Mean Absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    
    # Match feature names length
    n_feats = min(len(feature_names), shap_matrix.shape[1])
    global_importance = [
        {"feature": feature_names[i], "importance": float(mean_abs_shap[i])}
        for i in range(n_feats)
    ]
    global_importance.sort(key=lambda x: x["importance"], reverse=True)

    # 2. Per-group Mean Absolute SHAP values
    unique_groups = np.unique(sens_sub)
    group_attributions: Dict[str, List[Dict[str, Any]]] = {}

    for g in unique_groups:
        mask = (sens_sub == g)
        if np.sum(mask) > 0:
            g_mean_abs = np.mean(np.abs(shap_matrix[mask]), axis=0)
            g_list = [
                {"feature": feature_names[i], "importance": float(g_mean_abs[i])}
                for i in range(n_feats)
            ]
            g_list.sort(key=lambda x: x["importance"], reverse=True)
            group_attributions[g] = g_list

    return {
        "global_importance": global_importance,
        "group_attributions": group_attributions,
        "unique_groups": list(unique_groups),
        "evaluated_samples": len(X_sub),
        "top_features": [item["feature"] for item in global_importance[:10]],
    }


def detect_proxy_features(
    df: pd.DataFrame,
    sensitive_col: str,
    feature_cols: List[str],
    threshold_caution: float = 0.25,
    threshold_high: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Detect non-sensitive features that act as statistical proxies for the sensitive attribute.
    Calculates normalized mutual information / correlation.
    """
    sens_series = df[sensitive_col].fillna("Unknown").astype(str)
    # Encode sensitive labels as integers
    sens_encoded, _ = pd.factorize(sens_series)

    proxy_candidates = []

    for col in feature_cols:
        if col == sensitive_col:
            continue

        feat_series = df[col].copy()
        try:
            if pd.api.types.is_numeric_dtype(feat_series):
                feat_clean = feat_series.fillna(feat_series.median())
                # Mutual information for continuous feature vs discrete sensitive
                mi = mutual_info_classif(feat_clean.values.reshape(-1, 1), sens_encoded, random_state=42)[0]
                # Normalize approx to [0, 1] range using entropy
                norm_score = float(min(1.0, mi / 0.693))  # log(2) normalization
            else:
                cat_encoded, _ = pd.factorize(feat_series.fillna("Unknown").astype(str))
                mi = mutual_info_classif(cat_encoded.reshape(-1, 1), sens_encoded, discrete_features=[True], random_state=42)[0]
                norm_score = float(min(1.0, mi / 0.693))
        except Exception:
            norm_score = 0.0

        if norm_score >= threshold_high:
            risk = "HIGH PROXY RISK"
            status = "biased"
        elif norm_score >= threshold_caution:
            risk = "MODERATE PROXY"
            status = "caution"
        else:
            risk = "LOW CORRELATION"
            status = "fair"

        proxy_candidates.append({
            "feature": col,
            "correlation_score": norm_score,
            "risk": risk,
            "status": status,
            "dtype": "Numeric" if pd.api.types.is_numeric_dtype(feat_series) else "Categorical",
        })

    proxy_candidates.sort(key=lambda x: x["correlation_score"], reverse=True)
    return proxy_candidates


def create_shap_summary_chart(global_importance: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """Create interactive horizontal bar chart for top global SHAP features."""
    top_items = global_importance[:top_n][::-1]  # Reverse for bottom-to-top rendering
    features = [item["feature"] for item in top_items]
    importances = [item["importance"] for item in top_items]

    fig = go.Figure(
        go.Bar(
            x=importances,
            y=features,
            orientation="h",
            marker=dict(
                color=importances,
                colorscale="Viridis",
            ),
            text=[f"{v:.4f}" for v in importances],
            textposition="auto",
            hovertext=[f"Feature: <b>{f}</b><br>Mean |SHAP|: {v:.4f}" for f, v in zip(features, importances)],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=f"Top {len(top_items)} Global Drivers of Model Predictions (Mean |SHAP Value|)",
        xaxis_title="Average Impact on Model Decision Magnitude |SHAP|",
        yaxis_title="Feature",
        height=380,
    )
    return apply_theme_to_figure(fig)

