"""
Prediction layer. Pure scikit-learn, trains in <2s, no external calls.

Three trained models recover the latent processes baked into the data:
  - churn  (GradientBoostingClassifier -> probability)
  - convert(GradientBoostingClassifier -> probability of accepting an offer)
  - ltv    (GradientBoostingRegressor)

The simulator calls `score_frame` on a *modified* copy of the customer
features (after an intervention shifts them) and diffs against baseline.
That diff is a genuine model-driven counterfactual, not an LLM guess.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate import FEATURES


class TwinModels:
    def __init__(self):
        self.churn = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
        self.convert = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
        self.ltv = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
        self.churn_auc = None
        self.convert_auc = None

    def fit(self, df: pd.DataFrame):
        from sklearn.model_selection import cross_val_score
        X = df[FEATURES]
        # honest, cross-validated metrics (these are what we report)
        self.churn_auc = round(float(cross_val_score(
            GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0),
            X, df["churned"], cv=5, scoring="roc_auc").mean()), 3)
        self.convert_auc = round(float(cross_val_score(
            GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0),
            X, df["converted"], cv=5, scoring="roc_auc").mean()), 3)
        # fit final models on all data for serving
        self.churn.fit(X, df["churned"])
        self.convert.fit(X, df["converted"])
        self.ltv.fit(X, df["ltv"])
        return self

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES]
        out = df.copy()
        out["churn_prob"] = self.churn.predict_proba(X)[:, 1]
        out["convert_prob"] = self.convert.predict_proba(X)[:, 1]
        out["pred_ltv"] = self.ltv.predict(X)
        out["revenue_at_risk"] = (out["churn_prob"] * out["pred_ltv"]).round(0)
        return out

    def feature_importance(self):
        return sorted(
            zip(FEATURES, self.churn.feature_importances_.round(3)),
            key=lambda kv: kv[1], reverse=True,
        )


# ---- interventions: how a campaign shifts a twin's behavioural features ----
# Each lever returns a *modified copy* of the feature frame. The model then
# re-scores it, so the predicted delta is the model's own response to the shift.

def apply_intervention(df: pd.DataFrame, lever: str, intensity: float = 1.0) -> pd.DataFrame:
    d = df.copy()
    if lever == "loyalty_credit":
        # redeeming rewards + a nudge: drops unredeemed balance, lifts sessions
        d["rewards_unredeemed"] = (d["rewards_unredeemed"] * (1 - 0.6 * intensity)).round()
        d["sessions_30d"] = (d["sessions_30d"] * (1 + 0.25 * intensity)).round()
        d["recency_days"] = (d["recency_days"] * (1 - 0.30 * intensity)).clip(lower=1).round()
    elif lever == "winback_offer":
        # personalised win-back: strong recency reset, frequency bump
        d["recency_days"] = (d["recency_days"] * (1 - 0.45 * intensity)).clip(lower=1).round()
        d["frequency_90d"] = (d["frequency_90d"] + 2 * intensity).round()
    elif lever == "concierge_support":
        # proactive support outreach: resolves tickets, lifts NPS
        d["support_tickets_90d"] = (d["support_tickets_90d"] * (1 - 0.7 * intensity)).round()
        d["nps"] = (d["nps"] + 2 * intensity).clip(upper=10).round()
    elif lever == "tier_nudge":
        # "you're close to the next tier" nudge: frequency + sessions
        d["frequency_90d"] = (d["frequency_90d"] + 3 * intensity).round()
        d["sessions_30d"] = (d["sessions_30d"] * (1 + 0.4 * intensity)).round()
    return d


LEVERS = {
    "loyalty_credit": "10% loyalty credit + redeem-now nudge",
    "winback_offer": "Personalised win-back offer",
    "concierge_support": "Proactive concierge support outreach",
    "tier_nudge": "Near-tier upgrade nudge",
}
