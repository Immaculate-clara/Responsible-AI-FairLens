"""Unit tests for FairLens fairness metrics using hand-calculated mathematical ground truth."""

import pytest
import numpy as np
from core.fairness_metrics import (
    compute_fairness_metrics, compute_group_confusion_metrics
)
from core.utils import get_metric_status


def test_hand_calculated_fairness_metrics(hand_calculated_fairness_fixture):
    """
    Verify that compute_fairness_metrics returns mathematically exact values
    matching independent hand calculations.
    """
    f = hand_calculated_fairness_fixture
    results = compute_fairness_metrics(
        y_true=f["y_true"],
        y_pred=f["y_pred"],
        sensitive_features=f["sensitive"],
        attribute_name="TestAttr",
    )

    assert results["status"] == "success"
    assert pytest.approx(results["selection_rates"]["A"], rel=1e-5) == f["expected_sr_A"]
    assert pytest.approx(results["selection_rates"]["B"], rel=1e-5) == f["expected_sr_B"]
    assert pytest.approx(results["demographic_parity_difference"], rel=1e-5) == f["expected_dpd"]
    assert pytest.approx(results["disparate_impact_ratio"], rel=1e-5) == f["expected_dir"]
    assert pytest.approx(results["equal_opportunity_difference"], rel=1e-5) == f["expected_eod"]
    assert pytest.approx(results["equalized_odds_difference"], rel=1e-5) == f["expected_eq_odds"]
    assert pytest.approx(results["tprs"]["A"], rel=1e-5) == f["expected_tpr_A"]
    assert pytest.approx(results["tprs"]["B"], rel=1e-5) == f["expected_tpr_B"]


def test_perfect_equity_metrics():
    """Verify that identical performance across subgroups yields 100% Fairness Score and DIR=1.0."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sens = np.array(["M", "M", "M", "M", "F", "F", "F", "F"])

    results = compute_fairness_metrics(y_true, y_pred, sens, attribute_name="Gender")

    assert results["disparate_impact_ratio"] == 1.0
    assert results["demographic_parity_difference"] == 0.0
    assert results["equal_opportunity_difference"] == 0.0
    assert results["equalized_odds_difference"] == 0.0
    assert pytest.approx(results["fairness_score"], rel=1e-3) == 100.0


def test_fairness_score_bounds():
    """Ensure fairness score is always clamped within [0.0, 100.0]."""
    # Extreme bias scenario: Group A receives 100% approvals, Group B receives 0%
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    sens = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    results = compute_fairness_metrics(y_true, y_pred, sens)
    assert 0.0 <= results["fairness_score"] <= 100.0
    assert results["disparate_impact_ratio"] == 0.0
    assert results["demographic_parity_difference"] == 1.0


def test_metric_status_thresholds():
    """Verify regulatory threshold classifications (80% rule, parity diffs)."""
    # Disparate Impact
    assert get_metric_status("disparate_impact", 0.85)[0] == "fair"
    assert get_metric_status("disparate_impact", 0.75)[0] == "caution"
    assert get_metric_status("disparate_impact", 0.50)[0] == "biased"

    # Demographic Parity Diff
    assert get_metric_status("demographic_parity", 0.05)[0] == "fair"
    assert get_metric_status("demographic_parity", 0.15)[0] == "caution"
    assert get_metric_status("demographic_parity", 0.30)[0] == "biased"


def test_single_group_error_handling():
    """Verify graceful handling when only a single demographic group is present."""
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 0])
    sens = np.array(["OnlyOneGroup", "OnlyOneGroup", "OnlyOneGroup"])

    results = compute_fairness_metrics(y_true, y_pred, sens)
    assert results["status"] == "error"
    assert "fewer than 2 groups" in results["message"]

