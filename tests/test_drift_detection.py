"""Unit and integration tests for the drift detection framework."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from drift_detection import chi_squared_test, ks_test, psi
from monitor import ModelMonitor
from alerting import SlackAlerter, SnowflakeAuditor
from synth_data import generate_current, generate_reference


def test_ks_no_drift_reports_ok():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=5_000)
    cur = rng.normal(0, 1, size=5_000)
    result = ks_test(ref, cur, feature="x")
    assert result.severity == "OK"
    assert result.p_value is not None and result.p_value > 0.05


def test_ks_mean_shift_reports_critical():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=5_000)
    cur = rng.normal(0.5, 1, size=5_000)
    result = ks_test(ref, cur, feature="x")
    assert result.severity == "CRITICAL"


def test_ks_handles_nan():
    ref = np.array([1.0, 2.0, np.nan, 3.0] * 100)
    cur = np.array([1.0, 2.0, np.nan, 3.0] * 100)
    result = ks_test(ref, cur, feature="x")
    assert result.severity == "OK"


def test_ks_insufficient_samples_warns():
    result = ks_test([1.0, 2.0], [1.0, 2.0], feature="x")
    assert result.severity == "WARNING"
    assert result.p_value is None


def test_psi_no_drift_under_warning_threshold():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, size=10_000)
    cur = rng.normal(0, 1, size=10_000)
    result = psi(ref, cur, feature="x")
    assert result.severity == "OK"
    assert result.statistic < 0.1


def test_psi_large_shift_reports_critical():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, size=10_000)
    cur = rng.normal(2, 1, size=10_000)
    result = psi(ref, cur, feature="x")
    assert result.severity == "CRITICAL"
    assert result.statistic > 0.25


def test_chi2_same_categorical_distribution_is_ok():
    rng = np.random.default_rng(4)
    cats = ["a", "b", "c"]
    ref = rng.choice(cats, p=[0.5, 0.3, 0.2], size=5_000)
    cur = rng.choice(cats, p=[0.5, 0.3, 0.2], size=5_000)
    result = chi_squared_test(ref, cur, feature="cat")
    assert result.severity == "OK"


def test_chi2_shifted_categorical_distribution_is_critical():
    rng = np.random.default_rng(5)
    cats = ["a", "b", "c"]
    ref = rng.choice(cats, p=[0.5, 0.3, 0.2], size=5_000)
    cur = rng.choice(cats, p=[0.2, 0.3, 0.5], size=5_000)
    result = chi_squared_test(ref, cur, feature="cat")
    assert result.severity == "CRITICAL"


def test_monitor_clean_run_returns_ok_or_warning():
    ref = generate_reference(20_000)
    cur = generate_current(8_000, drift_scenario="none", seed=11)
    monitor = ModelMonitor(
        model_name="churn_v1",
        reference=ref,
        numeric_features=["monthly_charges", "tenure_months", "support_tickets_30d", "bandwidth_gb"],
        categorical_features=["plan_tier", "payment_method"],
        score_column="churn_score",
    )
    report = monitor.run(cur)
    assert report.aggregate_severity in {"OK", "WARNING"}


def test_monitor_price_hike_detected():
    ref = generate_reference(20_000)
    cur = generate_current(8_000, drift_scenario="price_hike", seed=12)
    monitor = ModelMonitor(
        model_name="churn_v1",
        reference=ref,
        numeric_features=["monthly_charges", "tenure_months", "support_tickets_30d", "bandwidth_gb"],
        categorical_features=["plan_tier", "payment_method"],
        score_column="churn_score",
    )
    report = monitor.run(cur)
    assert report.aggregate_severity == "CRITICAL"
    assert "monthly_charges" in report.critical_features()


def test_monitor_plan_mix_drift_detected():
    ref = generate_reference(20_000)
    cur = generate_current(8_000, drift_scenario="plan_mix", seed=13)
    monitor = ModelMonitor(
        model_name="churn_v1",
        reference=ref,
        numeric_features=["monthly_charges", "tenure_months"],
        categorical_features=["plan_tier", "payment_method"],
        score_column="churn_score",
    )
    report = monitor.run(cur)
    assert report.aggregate_severity == "CRITICAL"
    assert "plan_tier" in report.critical_features()


def test_monitor_missing_feature_raises():
    ref = generate_reference(1_000)
    with pytest.raises(ValueError):
        ModelMonitor(model_name="bad", reference=ref, numeric_features=["does_not_exist"])


def test_slack_alerter_dry_run_returns_payload():
    ref = generate_reference(20_000)
    cur = generate_current(8_000, drift_scenario="price_hike", seed=14)
    monitor = ModelMonitor(
        model_name="churn_v1",
        reference=ref,
        numeric_features=["monthly_charges", "tenure_months"],
        score_column="churn_score",
    )
    report = monitor.run(cur)
    alerter = SlackAlerter(webhook_url="https://example.invalid", dry_run=True)
    payload = alerter.post(report)
    assert "attachments" in payload
    assert report.model_name in payload["text"]


def test_snowflake_auditor_dry_run_returns_row():
    ref = generate_reference(20_000)
    cur = generate_current(8_000, drift_scenario="none", seed=15)
    monitor = ModelMonitor(
        model_name="churn_v1",
        reference=ref,
        numeric_features=["monthly_charges", "tenure_months"],
    )
    report = monitor.run(cur)
    auditor = SnowflakeAuditor(dry_run=True)
    row = auditor.write(report)
    assert row["model_name"] == "churn_v1"
    assert "payload" in row
