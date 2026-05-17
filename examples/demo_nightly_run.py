"""End-to-end example: simulate a nightly monitoring job."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synth_data import generate_reference, generate_current
from monitor import ModelMonitor
from alerting import SlackAlerter, SnowflakeAuditor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["none", "price_hike", "ticket_surge", "plan_mix", "score_drift"],
        default="none",
    )
    parser.add_argument("--n-ref", type=int, default=50_000)
    parser.add_argument("--n-cur", type=int, default=20_000)
    args = parser.parse_args()

    print(f"\n=== Nightly monitoring run — scenario: {args.scenario} ===\n")

    reference = generate_reference(args.n_ref)
    current = generate_current(args.n_cur, drift_scenario=args.scenario, seed=2024)

    monitor = ModelMonitor(
        model_name="churn_classifier_v2",
        reference=reference,
        numeric_features=[
            "monthly_charges",
            "tenure_months",
            "support_tickets_30d",
            "bandwidth_gb",
        ],
        categorical_features=["plan_tier", "payment_method"],
        score_column="churn_score",
    )

    report = monitor.run(current)

    print(f"Aggregate severity: {report.aggregate_severity}")
    print(f"Reference rows:     {report.metadata['n_reference']:,}")
    print(f"Current rows:       {report.metadata['n_current']:,}\n")

    print("Per-feature results:")
    for result in report.feature_results:
        print(f"  [{result.severity:8}] {result.message}")

    if report.score_result:
        print(f"\nScore drift: {report.score_result.message}")

    print("\n--- Slack payload (dry run) ---")
    slack = SlackAlerter(webhook_url="https://hooks.slack.com/services/REDACTED", dry_run=True)
    payload = slack.post(report)
    print(json.dumps(payload, indent=2, default=str))

    print("\n--- Snowflake audit row (dry run) ---")
    auditor = SnowflakeAuditor(dry_run=True)
    row = auditor.write(report)
    preview = dict(row)
    preview["payload"] = preview["payload"][:200] + "..."
    print(json.dumps(preview, indent=2, default=str))


if __name__ == "__main__":
    main()
