"""Tests for Edge Cases in FairLens.

Validates that edge cases produce user-friendly, descriptive errors or proper warnings:
1. File with missing values (handled with imputation)
2. Single-class target (friendly error message, not stack trace)
3. Tiny group (demographic subgroup N < 30 flagged with warning)
4. Wrong file format (friendly error message)
5. Empty file (friendly error message)
"""

import io
import pytest
import numpy as np
import pandas as pd
from core.data_loader import (
    load_uploaded_file,
    detect_column_types,
    validate_setup_configuration,
    DataValidationError,
)
from core.audit import compute_data_audit
from core.model_trainer import prepare_data_pipeline


def test_edge_case_missing_values():
    """Verify missing values in features are successfully handled and imputed without crashing."""
    df = pd.DataFrame({
        "income": [50000, np.nan, 60000, 75000, np.nan, 80000, 45000, 90000, 62000, 54000],
        "category": ["A", "B", None, "A", "B", "A", None, "B", "A", "B"],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
        "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })

    prep = prepare_data_pipeline(
        df=df,
        target_col="target",
        positive_label=1,
        sensitive_cols=["gender"],
        exclude_cols=[],
        test_size=0.3,
        random_state=42,
    )
    # Confirm processed arrays contain zero NaN values
    assert not np.isnan(prep["X_train_proc"]).any()
    assert not np.isnan(prep["X_test_proc"]).any()


def test_edge_case_single_class_target():
    """Verify single-class target column raises a clear, user-friendly validation error."""
    df = pd.DataFrame({
        "gender": ["Male", "Female", "Male", "Female"],
        "income": [50000, 60000, 70000, 80000],
        "outcome": [1, 1, 1, 1],  # Only 1 class!
    })

    is_valid, err_msg = validate_setup_configuration(
        df=df,
        target_col="outcome",
        positive_label="1",
        sensitive_cols=["gender"],
    )
    assert not is_valid
    assert "single class" in err_msg.lower() or "at least two distinct classes" in err_msg.lower()


def test_edge_case_tiny_group_flag():
    """Verify subgroups with N < 30 are accurately flagged with a small group warning."""
    # 50 Males, 10 Females (Female group < 30)
    genders = ["Male"] * 50 + ["Female"] * 10
    approved = [1] * 35 + [0] * 15 + [1] * 4 + [0] * 6
    df = pd.DataFrame({
        "gender": genders,
        "approved": approved,
    })

    audit = compute_data_audit(
        df=df,
        target_col="approved",
        positive_label=1,
        sensitive_cols=["gender"],
    )

    warnings = audit.get("small_group_warnings", [])
    assert len(warnings) > 0
    assert any(w["group"] == "Female" and w["count"] < 30 for w in warnings)
    female_item = next(x for x in audit["group_audits"]["gender"]["breakdown"] if x["group"] == "Female")
    male_item = next(x for x in audit["group_audits"]["gender"]["breakdown"] if x["group"] == "Male")
    assert female_item["is_small_group"] == True
    assert male_item["is_small_group"] == False


def test_edge_case_wrong_file_type():
    """Verify wrong file extension (e.g. .pdf, .txt, .docx) produces a friendly DataValidationError."""
    fake_pdf = io.BytesIO(b"%PDF-1.4 ... fake pdf bytes")
    with pytest.raises(DataValidationError) as exc_info:
        load_uploaded_file(fake_pdf, "document.pdf")
    assert "Unsupported file format" in str(exc_info.value)
    assert ".pdf" in str(exc_info.value)


def test_edge_case_empty_file():
    """Verify uploading an empty CSV buffer raises a friendly DataValidationError."""
    empty_csv = io.BytesIO(b"")
    with pytest.raises(DataValidationError) as exc_info:
        load_uploaded_file(empty_csv, "empty.csv")
    assert "empty file" in str(exc_info.value) or "contains no data" in str(exc_info.value)
