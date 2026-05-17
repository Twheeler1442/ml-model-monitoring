# Architecture

## System overview

```text
                          +------------------+
   training data  ------> | reference store  |
                          +--------+---------+
                                   |
                                   v
   production scores  --->  +------+-------+        +-----------------+
   feature snapshots  --->  | ModelMonitor | -----> | Slack webhook   |
                            +------+-------+        +-----------------+
                                   |                +-----------------+
                                   +--------------> | Snowflake audit |
                                                    +-----------------+
```

The monitor is intentionally stateless between runs. State lives in two places:

1. The reference frame, which is set once per model version and only changes when the model is retrained.
2. The Snowflake audit table, which stores every run for historical analysis and dashboarding.

## Design choices

**Statelessness over fancy storage.** A drift monitor that has its own state store becomes another thing to operate. Pushing all run history into Snowflake means the existing data platform owns durability, backups, retention, and access control.

**Sklearn-compatible only in spirit.** The monitor does not implement `fit` or `predict` because those verbs do not match what it does. It does keep the same idiomatic surface: pandas in, dataclass out, no hidden global state.

**Tiered severity, not single threshold.** A single drift threshold either fires constantly during normal-but-noisy weeks or misses real shifts. Two tiers, `WARNING` and `CRITICAL`, let teams separate investigation from immediate action.

**Reference quantiles for PSI bins.** PSI is sensitive to bin choice; using equal-frequency bins from the reference makes PSI more stable across feature scales. Edges are forced to `-inf` and `inf` to handle current-period outliers without crashing.

**Laplace smoothing.** Both PSI and chi-squared apply small additive smoothing so empty bins do not produce `log(0)` or zero-cell warnings.

## Extending

| Need | Where to extend |
|------|-----------------|
| New drift test, such as Wasserstein | Add a function in `drift_detection.py` returning `DriftResult`; call from `ModelMonitor.run` |
| New alert channel | Add a dataclass alongside `SlackAlerter` following the `dry_run` pattern |
| Per-feature threshold overrides | Pass a threshold mapping to `ModelMonitor.__init__` and route per-feature in `run` |
| Multi-model batch runs | Wrap `ModelMonitor` instances in a registry; the audit table already partitions by model name |
