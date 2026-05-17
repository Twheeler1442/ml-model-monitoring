"""
drift_detection.py
==================

Statistical drift detection for production ML models.

Three core tests are exposed, mirroring industry-standard approaches:

- Kolmogorov-Smirnov (KS) test  -> continuous features
- Population Stability Index (PSI) -> continuous and binned features
- Chi-squared test               -> categorical features

Each function returns a `DriftResult` dataclass with the test statistic,
p-value (where applicable), severity tier (OK / WARNING / CRITICAL), and
a human-readable message suitable for Slack delivery.

Severity thresholds are configurable but ship with sensible defaults
based on common practice in credit-risk and telecom monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Sequence
import numpy as np
import pandas as pd
from scipy import stats


Severity = Literal["OK", "WARNING", "CRITICAL"]


@dataclass
class DriftResult:
    """Standardized output for any drift test."""
    feature: str
    test: str
    statistic: float
    p_value: float | None
    severity: Severity
    message: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def ks_test(
    reference: Sequence[float],
    current: Sequence[float],
    feature: str,
    warning_p: float = 0.05,
    critical_p: float = 0.01,
) -> DriftResult:
    """Two-sample KS test for continuous distributions."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < 30 or len(cur) < 30:
        return DriftResult(
            feature=feature,
            test="KS",
            statistic=float("nan"),
            p_value=None,
            severity="WARNING",
            message=f"{feature}: insufficient samples for KS test "
                    f"(ref={len(ref)}, cur={len(cur)})",
        )

    stat, p = stats.ks_2samp(ref, cur)

    if p < critical_p:
        severity: Severity = "CRITICAL"
    elif p < warning_p:
        severity = "WARNING"
    else:
        severity = "OK"

    return DriftResult(
        feature=feature,
        test="KS",
        statistic=float(stat),
        p_value=float(p),
        severity=severity,
        message=f"{feature}: KS={stat:.4f}, p={p:.4g} [{severity}]",
        metadata={"n_ref": len(ref), "n_cur": len(cur)},
    )


def psi(
    reference: Sequence[float],
    current: Sequence[float],
    feature: str,
    bins: int = 10,
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.25,
) -> DriftResult:
    """Population Stability Index with reference-quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < bins * 5 or len(cur) < bins * 5:
        return DriftResult(
            feature=feature,
            test="PSI",
            statistic=float("nan"),
            p_value=None,
            severity="WARNING",
            message=f"{feature}: insufficient samples for PSI "
                    f"(need {bins * 5} per side)",
        )

    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return DriftResult(
            feature=feature,
            test="PSI",
            statistic=float("nan"),
            p_value=None,
            severity="WARNING",
            message=f"{feature}: cannot bin (low cardinality)",
        )
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = (ref_hist + 1) / (ref_hist.sum() + len(ref_hist))
    cur_pct = (cur_hist + 1) / (cur_hist.sum() + len(cur_hist))

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi_value > critical_threshold:
        severity: Severity = "CRITICAL"
    elif psi_value > warning_threshold:
        severity = "WARNING"
    else:
        severity = "OK"

    return DriftResult(
        feature=feature,
        test="PSI",
        statistic=psi_value,
        p_value=None,
        severity=severity,
        message=f"{feature}: PSI={psi_value:.4f} [{severity}]",
        metadata={"bins": len(ref_hist), "n_ref": len(ref), "n_cur": len(cur)},
    )


def chi_squared_test(
    reference: Sequence,
    current: Sequence,
    feature: str,
    warning_p: float = 0.05,
    critical_p: float = 0.01,
) -> DriftResult:
    """Chi-squared test of independence for categorical features."""
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).values
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).values

    table = np.vstack([ref_counts, cur_counts])
    nonzero = table.sum(axis=0) > 0
    table = table[:, nonzero]

    if table.shape[1] < 2:
        return DriftResult(
            feature=feature,
            test="CHI2",
            statistic=float("nan"),
            p_value=None,
            severity="OK",
            message=f"{feature}: insufficient categories for chi-squared",
        )

    chi2, p, _, _ = stats.chi2_contingency(table)

    if p < critical_p:
        severity: Severity = "CRITICAL"
    elif p < warning_p:
        severity = "WARNING"
    else:
        severity = "OK"

    return DriftResult(
        feature=feature,
        test="CHI2",
        statistic=float(chi2),
        p_value=float(p),
        severity=severity,
        message=f"{feature}: chi2={chi2:.4f}, p={p:.4g} [{severity}]",
        metadata={"n_categories": int(table.shape[1])},
    )
