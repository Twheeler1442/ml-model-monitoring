# ML Model Monitoring & Data Quality Framework

A lightweight, sklearn-compatible drift detection framework for production classifiers. Runs nightly distribution checks, prediction-score drift, and tiered Slack/Snowflake alerting.

> **Note on data:** This is a portfolio reimplementation of a production system. All datasets here are synthetically generated; no real customer data or proprietary schemas are reproduced. The architecture and statistical approach mirror what's deployed at scale in industry settings.

## What it does

Catches three classes of drift before they affect model performance in production:

- **Feature drift** — KS test and PSI on numeric features, chi-squared on categoricals
- **Prediction drift** — KS test on the model's output score distribution
- **Sample integrity** — minimum-sample guards, NaN handling, low-cardinality detection

Each run produces a single `MonitoringReport` with per-feature results and a worst-case aggregate severity. Alerts are tiered as `WARNING` (investigate) or `CRITICAL` (act).

## Quickstart

```bash
git clone https://github.com/Twheeler1442/ml-model-monitoring.git
cd ml-model-monitoring
pip install -r requirements.txt
python examples/demo_nightly_run.py --scenario price_hike
```

## Usage

```python
from synth_data import generate_reference, generate_current
from monitor import ModelMonitor

reference = generate_reference(n=50_000)
current = generate_current(n=20_000, drift_scenario="price_hike")

monitor = ModelMonitor(
    model_name="churn_classifier_v2",
    reference=reference,
    numeric_features=["monthly_charges", "tenure_months", "bandwidth_gb"],
    categorical_features=["plan_tier", "payment_method"],
    score_column="churn_score",
)

report = monitor.run(current)
print(report.aggregate_severity)
print(report.critical_features())
```

## Drift tests

| Test | Applies to | Severity bands |
|------|------------|----------------|
| Kolmogorov-Smirnov two-sample | continuous | p < 0.01 critical, p < 0.05 warning |
| Population Stability Index | continuous / binned | PSI > 0.25 critical, PSI > 0.10 warning |
| Chi-squared independence | categorical | p < 0.01 critical, p < 0.05 warning |

## Repository layout

```text
ml-model-monitoring/
├── src/
│   ├── drift_detection.py
│   ├── monitor.py
│   ├── alerting.py
│   └── synth_data.py
├── tests/
│   └── test_drift_detection.py
├── notebooks/
│   └── 01_walkthrough.ipynb
├── examples/
│   └── demo_nightly_run.py
├── configs/
│   └── example_monitor.yaml
└── docs/
    └── architecture.md
```

## Tests

```bash
pytest tests/ -v
```

## Stack

Python, NumPy, SciPy, pandas, pyarrow. Slack webhook and Snowflake connector are optional lazy imports.

## License

MIT. See `LICENSE`.
