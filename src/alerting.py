"""Slack alerting and Snowflake-style audit logging for monitoring reports."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from monitor import MonitoringReport

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {"OK": ":white_check_mark:", "WARNING": ":warning:", "CRITICAL": ":rotating_light:"}
SEVERITY_COLOR = {"OK": "#36a64f", "WARNING": "#f2c744", "CRITICAL": "#e01e5a"}


@dataclass
class SlackAlerter:
    """Posts a monitoring summary to a Slack incoming webhook."""
    webhook_url: str
    dry_run: bool = False

    def post(self, report: MonitoringReport) -> dict[str, Any]:
        payload = self._build_payload(report)

        if self.dry_run:
            return payload

        try:
            import requests
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            logger.exception("Slack delivery failed: %s", exc)

        return payload

    @staticmethod
    def _build_payload(report: MonitoringReport) -> dict[str, Any]:
        emoji = SEVERITY_EMOJI.get(report.aggregate_severity, "")
        color = SEVERITY_COLOR.get(report.aggregate_severity, "#888888")

        crit = report.critical_features()
        warn = report.warning_features()

        lines = [
            f"*Model:* `{report.model_name}`",
            f"*Status:* {report.aggregate_severity}",
            f"*Reference rows:* {report.metadata.get('n_reference')}",
            f"*Current rows:* {report.metadata.get('n_current')}",
        ]
        if crit:
            lines.append(f"*Critical features:* {', '.join(crit)}")
        if warn:
            lines.append(f"*Warning features:* {', '.join(warn)}")
        if report.score_result:
            lines.append(f"*Score drift:* {report.score_result.message}")

        return {
            "text": f"{emoji} Monitoring run — {report.model_name} — {report.aggregate_severity}",
            "attachments": [{
                "color": color,
                "text": "\n".join(lines),
                "footer": f"monitor v{report.metadata.get('monitor_version')}",
                "ts": int(__import__('time').time()),
            }],
        }


@dataclass
class SnowflakeAuditor:
    """Writes the full monitoring report as one audit row."""
    connection: Any | None = None
    table: str = "model_monitoring_audit"
    dry_run: bool = True

    def write(self, report: MonitoringReport) -> dict[str, Any]:
        row = {
            "run_timestamp": report.run_timestamp,
            "model_name": report.model_name,
            "severity": report.aggregate_severity,
            "payload": json.dumps(report.to_dict()),
        }

        if self.dry_run or self.connection is None:
            return row

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"INSERT INTO {self.table} "
                "(run_timestamp, model_name, severity, payload) "
                "SELECT %s, %s, %s, PARSE_JSON(%s)",
                (row["run_timestamp"], row["model_name"], row["severity"], row["payload"]),
            )
            cursor.close()
        except Exception as exc:
            logger.exception("Snowflake audit write failed: %s", exc)

        return row
