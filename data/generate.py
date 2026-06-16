"""
Generate a synthetic but *internally consistent* customer-twin dataset.

The key design choice: churn, conversion and LTV are driven by a hidden,
deterministic process over behavioural features. That means a model trained
on this data learns a real signal (not noise), and the campaign simulator
can compute genuine counterfactual deltas when an intervention shifts a
feature. No external APIs, no network.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

FIRST = ["Aarav", "Diya", "Vivaan", "Ananya", "Aditya", "Ishaan", "Saanvi",
         "Kabir", "Myra", "Reyansh", "Aadhya", "Arjun", "Anika", "Vihaan",
         "Pari", "Sai", "Navya", "Krishna", "Riya", "Dhruv", "Tara", "Veer",
         "Mira", "Rohan", "Kiara", "Yash", "Zoya", "Aryan", "Nisha", "Kunal"]
LAST = ["Sharma", "Iyer", "Reddy", "Nair", "Mehta", "Gupta", "Rao", "Bose",
        "Khan", "Patel", "Das", "Menon", "Verma", "Joshi", "Pillai"]
TIERS = ["Base", "Silver", "Gold", "Platinum"]
CHANNELS = ["Email", "App Push", "SMS", "WhatsApp"]
CITIES = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_customers(n=900):
    rows = []
    for i in range(n):
        tier = RNG.choice(TIERS, p=[0.45, 0.30, 0.18, 0.07])
        tier_idx = TIERS.index(tier)

        # behavioural features
        recency_days = int(np.clip(RNG.exponential(28 + tier_idx * 6), 1, 240))
        frequency_90d = int(np.clip(RNG.poisson(3 + tier_idx * 3), 0, 60))
        avg_order_value = float(np.round(RNG.normal(900 + tier_idx * 1400, 500), 0))
        avg_order_value = max(avg_order_value, 120)
        tenure_months = int(np.clip(RNG.normal(14 + tier_idx * 8, 9), 1, 96))
        sessions_30d = int(np.clip(RNG.poisson(5 + tier_idx * 2), 0, 80))
        # browse-to-redeem gap: high browsing, low redemption = intent leaking
        browse_events_30d = int(np.clip(RNG.poisson(10 + tier_idx * 4), 0, 120))
        rewards_unredeemed = int(np.clip(RNG.poisson(2 + tier_idx), 0, 14))
        support_tickets_90d = int(np.clip(RNG.poisson(0.6), 0, 8))
        nps = int(np.clip(RNG.normal(7 - support_tickets_90d * 0.7 + tier_idx * 0.4, 2), 0, 10))
        discount_sensitivity = float(np.round(_sigmoid(RNG.normal(0 - tier_idx * 0.5, 1)), 3))

        # hidden latent churn driver (this is the "truth" the model recovers)
        z = (
            0.030 * recency_days
            - 0.090 * frequency_90d
            - 0.020 * sessions_30d
            + 0.060 * support_tickets_90d
            - 0.110 * nps
            - 0.015 * tenure_months
            + 0.080 * rewards_unredeemed          # unredeemed value = disengagement
            - 0.010 * browse_events_30d
            - 0.30 * tier_idx
            + RNG.normal(0, 0.32)          # real-world noise: behaviour isn't deterministic
        )
        churn_prob_true = float(_sigmoid(z))
        churned = int(RNG.random() < churn_prob_true)
        if RNG.random() < 0.02:            # 6% label noise (data is never clean)
            churned = 1 - churned

        # conversion propensity for a "next best offer" (latent, separate signal)
        c = (
            0.6 * discount_sensitivity
            + 0.04 * browse_events_30d
            + 0.10 * rewards_unredeemed
            - 0.02 * recency_days
            + 0.15 * tier_idx
            + RNG.normal(0, 1.0)
        )
        converted = int(RNG.random() < _sigmoid(c - 1.0))
        if RNG.random() < 0.05:
            converted = 1 - converted

        ltv = float(np.round(avg_order_value * (frequency_90d * 4 + 1) *
                             (1 + tier_idx * 0.4) * (1 - churn_prob_true * 0.6), 0))

        rows.append(dict(
            customer_id=f"CX{i:04d}",
            name=f"{RNG.choice(FIRST)} {RNG.choice(LAST)}",
            city=RNG.choice(CITIES),
            tier=tier,
            tier_idx=tier_idx,
            recency_days=recency_days,
            frequency_90d=frequency_90d,
            avg_order_value=avg_order_value,
            tenure_months=tenure_months,
            sessions_30d=sessions_30d,
            browse_events_30d=browse_events_30d,
            rewards_unredeemed=rewards_unredeemed,
            support_tickets_90d=support_tickets_90d,
            nps=nps,
            discount_sensitivity=discount_sensitivity,
            preferred_channel=RNG.choice(CHANNELS, p=[0.3, 0.35, 0.15, 0.2]),
            ltv=ltv,
            churned=churned,
            converted=converted,
        ))
    return pd.DataFrame(rows)


FEATURES = [
    "recency_days", "frequency_90d", "avg_order_value", "tenure_months",
    "sessions_30d", "browse_events_30d", "rewards_unredeemed",
    "support_tickets_90d", "nps", "discount_sensitivity", "tier_idx",
]

if __name__ == "__main__":
    df = make_customers()
    df.to_parquet("data/customers.parquet", index=False)
    df.to_csv("data/customers.csv", index=False)
    print(f"wrote {len(df)} customer twins")
    print(df[["tier", "churned", "converted"]].groupby("tier").mean().round(2))
