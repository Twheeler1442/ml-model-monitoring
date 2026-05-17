"""Model monitoring orchestration for nightly drift detection runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
import pandas as pd

from drift_detection import DriftResult, Severity, chi_squared_test, ks_test, psi

MONITOR_VERSION = "1.0.0"


@dataclass
class MonitoringReport:
    model_name: str
    run_timestamp: str
    aggregate_severity: Severity
    feature_results: list[DriftResult]
    score_result: DriftResult | None = None
    metadata: dict = field(default_factory=dict)

    def critical_features(self) -> list[str]:
        seen, out = set(), []
        for r in self.feature_results:
            if r.severity == "CRITICAL" and r.feature not in seen:
                seen.add(r.feature)
                out.append(r.feature)
        return out

    def warning_features(self) -> list[str]:
        crit = set(self.critical_features())
        seen, out = set(), []
        for r in self.feature_results:
            if r.severity == "WARNING" and r.feature not in seen and r.feature not in crit:
                seen.add(r.feature)
                out.append(r.feature)
        return out

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "run_timestamp": self.run_timestamp,
            "aggregate_severity": self.aggregate_severity,
            "feature_results": [r.to_dict() for r in self.feature_results],
            "score_result": self.score_result.to_dict() if self.score_result else None,
            "metadata": self.metadata,
        }


class ModelMonitor:
    """Drift monitor for a single registered model."""

    def __init__(
        self,
        model_name: str,
        reference: pd.DataFrame,
        numeric_features: Iterable[str],
        categorical_features: Iterable[str] = (),
        score_column: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.reference = reference.copy()
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.score_column = score_column

        missing = [
            f for f in self.numeric_features + self.categorical_features
            if f not in reference.columns
        ]
        if missing:
            raise ValueError(f"Reference frame missing features: {missing}")

    def run(self, current: pd.DataFrame) -> MonitoringReport:
        """Run all drift checks against the current sample."""
        results: list[DriftResult] = []

        for col in self.numeric_features:
            results.append(ks_test(self.reference[col], current[col], col))
            results.append(psi(self.reference[col], current[col], col))

        for col in self.categorical_features:
            results.append(chi_squared_test(self.reference[col], current[col], col))

        score_result = None
        if self.score_column and self.score_column in current.columns:
            score_result = ks_test(
                self.reference.get(self.score_column, pd.Series(dtype=float)),
                current[self.score_column],
                feature=self.score_column,
            )

        aggregate = self._aggregate([*results, *([score_result] if score_result else [])])

        return MonitoringReport(
            model_name=self.model_name,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            aggregate_severity=aggregate,
            feature_results=results,
            score_result=score_result,
            metadata={
                "n_reference": len(self.reference),
                "n_current": len(current),
                "monitor_version": MONITOR_VERSION,
            },
        )

    @staticmethod
    def _aggregate(results: list[DriftResult]) -> Severity:
        levels = {r.severity for r in results}
        if "CRITICAL" in levels:
            return "CRITICAL"
        if "WARNING" in levels:
            return "WARNING"
        return "OK"
