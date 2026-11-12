"""End-to-end integration tests for the full FairLens audit, training, explain, and mitigation pipeline."""

import pytest
import io
import pandas as pd
import numpy as np
from core.data_loader import (
    load_uploaded_file, detect_column_types, validate_setup_configuration,
    DataValidationError
)
from core.audit import compute_data_audit
from core.model_trainer import prepare_data_pipeline, train_baseline_models
from core.explain import compute_shap_explanations, detect_proxy_features
from core.mitigation import (
    run_reweighing_mitigation, run_exponentiated_gradient_mitigation,
    run_threshold_optimizer_mitigation
)
from core.counterfactual import predict_single_instance, run_counterfactual_flip_test
from core.report import generate_pdf_report, generate_markdown_model_card


def test_validation_edge_cases(synthetic_biased_dataset):
    """Test friendly validation for edge cases: single-class target, missing target, invalid format."""
    df = synthetic_biased_dataset.copy()

    # 1. Single-class target
    df_single_class = df.copy()
    df_single_class["approved"] = 1
    is_valid, msg = validate_setup_configuration(
        df_single_class, target_col="approved", positive_label="1", sensitive_cols=["gender"]
    )
    assert not is_valid
    assert "single class" in msg

    # 2. Missing target column
    is_valid, msg = validate_setup_configuration(
        df, target_col="non_existent_column", positive_label="1", sensitive_cols=["gender"]
    )
    assert not is_valid
    assert "not found" in msg

    # 3. Target same as sensitive
    is_valid, msg = validate_setup_configuration(
        df, target_col="gender", positive_label="Male", sensitive_cols=["gender"]
    )
    assert not is_valid
    assert "cannot be identical" in msg

    # 4. Unsupported file extension
    with pytest.raises(DataValidationError):
        load_uploaded_file(io.BytesIO(b"dummy"), "data.pdf")


def test_full_audit_train_mitigate_pipeline(synthetic_biased_dataset):
    """Test the complete end-to-end workflow on the synthetic dataset."""
    df = synthetic_biased_dataset

    # 1. Ingestion & Audit
    audit = compute_data_audit(
        df=df,
        target_col="approved",
        positive_label="1",
        sensitive_cols=["gender"],
        small_group_threshold=30,
    )
    assert audit["total_rows"] == len(df)
    assert "gender" in audit["group_audits"]

    # 2. Model Preparation & Training
    prep = prepare_data_pipeline(
        df=df,
        target_col="approved",
        positive_label="1",
        sensitive_cols=["gender"],
        exclude_cols=["applicant_id"],
        test_size=0.30,
        random_state=42,
    )
    assert prep["X_train_proc"].shape[0] > 0
    assert prep["X_test_proc"].shape[0] > 0

    training_res = train_baseline_models(prep, threshold=0.5, random_state=42)
    models = training_res["trained_models"]
    assert "Logistic Regression" in models
    assert "Random Forest" in models

    rf_model = models["Random Forest"]
    rf_perf = training_res["performance_results"]["Random Forest"]["overall"]
    assert 0.0 <= rf_perf["accuracy"] <= 1.0

    # 3. Explainability (SHAP & Proxy Detection)
    shap_res = compute_shap_explanations(
        model=rf_model,
        X_train_proc=prep["X_train_proc"],
        X_test_proc=prep["X_test_proc"],
        feature_names=prep["feature_names"],
        sensitive_test_series=prep["sensitive_test_df"]["gender"],
        max_eval_samples=50,
    )
    assert len(shap_res["global_importance"]) > 0

    proxies = detect_proxy_features(
        df=df,
        sensitive_col="gender",
        feature_cols=["income", "credit_score", "loan_amount"],
    )
    assert len(proxies) == 3

    # 4. Mitigation Algorithms
    # Pre-processing: Reweighing
    reweigh_res = run_reweighing_mitigation(
        prepared_data=prep,
        sensitive_col="gender",
        base_model_type="Random Forest",
    )
    assert reweigh_res["accuracy"] > 0
    assert "fairness_score" in reweigh_res["fairness"]

    # In-processing: Exponentiated Gradient
    eg_res = run_exponentiated_gradient_mitigation(
        prepared_data=prep,
        sensitive_col="gender",
        constraint_type="DemographicParity",
    )
    assert eg_res["accuracy"] > 0

    # Post-processing: Threshold Optimizer
    to_res = run_threshold_optimizer_mitigation(
        base_model=rf_model,
        prepared_data=prep,
        sensitive_col="gender",
        constraint_type="demographic_parity",
    )
    assert to_res["accuracy"] > 0

    # 5. Counterfactual Flip Test
    sample_record = {
        "gender": "Male",
        "age_group": "Middle",
        "income": 50000.0,
        "credit_score": 650.0,
        "loan_amount": 20000.0,
    }
    cf_res = run_counterfactual_flip_test(
        model=rf_model,
        preprocessor=prep["preprocessor"],
        base_instance=sample_record,
        sensitive_col="gender",
        all_groups=["Male", "Female"],
        threshold=0.5,
    )
    assert "has_decision_flip" in cf_res
    assert len(cf_res["variations"]) == 2

    # 6. Report Generation
    session_mock = {
        "dataset_name": "Synthetic_Credit_Data",
        "target_col": "approved",
        "sensitive_cols": ["gender"],
        "active_model_name": "Random Forest",
        "fairness_results": reweigh_res["fairness"],
        "mitigation_results": [reweigh_res],
        "model_performance_results": training_res["performance_results"],
    }
    pdf_bytes = generate_pdf_report(session_mock)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000  # Non-trivial PDF generated

    md_card = generate_markdown_model_card(session_mock)
    assert "Model Card" in md_card
    assert "Disparate Impact Ratio" in md_card

