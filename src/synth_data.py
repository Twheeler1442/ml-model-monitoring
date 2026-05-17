"""Synthetic telecom-style subscriber data for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

PLAN_TIERS = np.array(["basic", "standard", "premium"])
PAYMENT_METHODS = np.array(["auto_card", "auto_bank", "manual"])


def generate_reference(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _generate(n, rng, drift_scenario="none")


def generate_current(
    n: int = 20_000,
    drift_scenario: str = "none",
    seed: int = 99,
) -> pd.DataFrame:
    """
    drift_scenario:
        none         -> no drift, sampled from same population
        price_hike   -> monthly_charges shifted up
        ticket_surge -> support_tickets_30d distribution shifted right
        plan_mix     -> categorical mix shifted toward basic tier
        score_drift  -> output churn_score distribution drifts upward
    """
    rng = np.random.default_rng(seed)
    return _generate(n, rng, drift_scenario)


def _generate(n: int, rng: np.random.Generator, drift_scenario: str) -> pd.DataFrame:
    monthly_charges = rng.normal(loc=72.0, scale=18.0, size=n).clip(20, 200)
    tenure_months = rng.integers(low=1, high=72, size=n)
    support_tickets_30d = rng.poisson(lam=0.6, size=n)
    bandwidth_gb = rng.gamma(shape=2.0, scale=180.0, size=n).clip(5, 4000)

    plan_tier = rng.choice(PLAN_TIERS, size=n, p=np.array([0.35, 0.45, 0.20]))
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=np.array([0.55, 0.30, 0.15]))

    z = (
        -3.5
        + 0.02 * (monthly_charges - 70)
        - 0.04 * (tenure_months - 24)
        + 0.6 * support_tickets_30d
        + 0.0005 * (bandwidth_gb - 300)
        + np.where(plan_tier == "basic", 0.4, 0)
        + np.where(payment_method == "manual", 0.5, 0)
    )
    churn_score = 1.0 / (1.0 + np.exp(-z))
    churn_score = (churn_score + rng.normal(0, 0.03, size=n)).clip(0.001, 0.999)

    if drift_scenario == "price_hike":
        monthly_charges = monthly_charges + 12.0
    elif drift_scenario == "ticket_surge":
        support_tickets_30d = support_tickets_30d + rng.poisson(0.8, size=n)
    elif drift_scenario == "plan_mix":
        plan_tier = rng.choice(PLAN_TIERS, size=n, p=[0.65, 0.25, 0.10])
    elif drift_scenario == "score_drift":
        churn_score = (churn_score + 0.15).clip(0.001, 0.999)

    return pd.DataFrame({
        "monthly_charges": monthly_charges,
        "tenure_months": tenure_months,
        "support_tickets_30d": support_tickets_30d,
        "bandwidth_gb": bandwidth_gb,
        "plan_tier": plan_tier,
        "payment_method": payment_method,
        "churn_score": churn_score,
    })


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out, exist_ok=True)
    generate_reference(50_000).to_parquet(os.path.join(out, "reference.parquet"))
    for scenario in ["none", "price_hike", "ticket_surge", "plan_mix", "score_drift"]:
        generate_current(20_000, drift_scenario=scenario, seed=hash(scenario) & 0xFFFF).to_parquet(
            os.path.join(out, f"current_{scenario}.parquet")
        )
    print("Wrote synthetic datasets to data/")
