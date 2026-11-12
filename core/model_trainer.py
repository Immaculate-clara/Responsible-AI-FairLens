"""Model training, evaluation, and pipeline orchestration for FairLens.

Trains Logistic Regression and Random Forest side by side with proper
preprocessing and disaggregated per-group performance tracking.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from core.fairness_metrics import compute_group_confusion_metrics


def prepare_data_pipeline(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Any,
    sensitive_cols: List[str],
    exclude_cols: Optional[List[str]] = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Split data into train and test sets and configure preprocessor.
    Preserves raw sensitive columns in test set for disaggregated fairness evaluation.
    """
    exclude = set(exclude_cols or [])
    exclude.add(target_col)

    # Convert target to binary {0, 1}
    y_raw = df[target_col].astype(str)
    pos_str = str(positive_label)
    y = (y_raw == pos_str).astype(int).values

    # Feature columns (all columns not in exclude list)
    feature_cols = [c for c in df.columns if c not in exclude]
    X_df = df[feature_cols].copy()

    # Identify numeric and categorical columns
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    # Preprocessor pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # Train-test split (stratified by target if possible)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Fit preprocessor on train only
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)

    # Retrieve transformed feature names
    feature_names: List[str] = []
    if numeric_features:
        feature_names.extend(numeric_features)
    if categorical_features:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
        feature_names.extend(encoded_cat_names)

    # Keep sensitive attributes for test evaluation
    sensitive_test_df = df.loc[X_test_df.index, sensitive_cols].copy()

    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_test_df": sensitive_test_df,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def train_baseline_models(
    prepared_data: Dict[str, Any],
    threshold: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train Logistic Regression and Random Forest models side-by-side.
    Computes overall and per-group performance metrics.
    """
    X_train = prepared_data["X_train_proc"]
    X_test = prepared_data["X_test_proc"]
    y_train = prepared_data["y_train"]
    y_test = prepared_data["y_test"]
    sensitive_test_df = prepared_data["sensitive_test_df"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state),
    }

    trained_models = {}
    model_predictions = {}
    model_probabilities = {}
    performance_results = {}

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
            probs = 1 / (1 + np.exp(-probs))

        preds = (probs >= threshold).astype(int)

        model_probabilities[name] = probs
        model_predictions[name] = preds

        # Overall Metrics
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        overall = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "test_sample_size": len(y_test),
        }

        # Per-Group Disaggregated Metrics
        per_group = {}
        for scol in sensitive_test_df.columns:
            scol_vals = sensitive_test_df[scol].astype(str).values
            unique_groups = np.unique(scol_vals)
            group_breakdown = {}
            for g in unique_groups:
                mask = (scol_vals == g)
                group_breakdown[g] = compute_group_confusion_metrics(y_test[mask], preds[mask])
            per_group[scol] = group_breakdown

        performance_results[name] = {
            "overall": overall,
            "per_group": per_group,
        }

    return {
        "trained_models": trained_models,
        "model_predictions": model_predictions,
        "model_probabilities": model_probabilities,
        "performance_results": performance_results,
    }

