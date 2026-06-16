"""
Reasoning layer. ONE LLM API (Anthropic). The LLM never computes numbers;
it explains the model's output and drafts retention/exec narratives.

Critical demo-safety feature: if no ANTHROPIC_API_KEY is set, or the call
fails, every function degrades to a deterministic template built from the
SAME model outputs. The demo cannot die on stage because of WiFi.
"""
from __future__ import annotations
import os, json, httpx

MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"


def _llm(system: str, user: str, max_tokens: int = 700) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        r = httpx.post(
            API_URL,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=30,
        )
        r.raise_for_status()
        blocks = r.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    except Exception:
        return None


def explain_twin(twin: dict) -> str:
    """One-paragraph causal read of why this customer is at risk / valuable."""
    sys = ("You are a customer-intelligence analyst. In 2-3 sentences, explain in plain "
           "business British English why this customer behaves as the metrics show. "
           "Reference specific numbers. Do not invent data. Do not give a churn score "
           "yourself; the model already produced one.")
    out = _llm(sys, json.dumps(twin))
    if out:
        return out
    # deterministic fallback
    drivers = []
    if twin["recency_days"] > 45:
        drivers.append(f"hasn't transacted in {twin['recency_days']} days")
    if twin["rewards_unredeemed"] >= 4:
        drivers.append(f"{twin['rewards_unredeemed']} unredeemed rewards (a silent-accumulator signal)")
    if twin["nps"] <= 5:
        drivers.append(f"a low NPS of {twin['nps']}")
    if twin["support_tickets_90d"] >= 2:
        drivers.append(f"{twin['support_tickets_90d']} recent support tickets")
    reason = "; ".join(drivers) or "broadly healthy engagement"
    return (f"{twin['name']} ({twin['tier']} tier) shows {reason}. With a predicted "
            f"churn probability of {twin['churn_prob']:.0%} against an LTV of "
            f"\u20b9{twin['pred_ltv']:,.0f}, roughly \u20b9{twin['revenue_at_risk']:,.0f} of value is exposed.")


def simulate_narrative(segment_summary: dict) -> str:
    """Explain the counterfactual delta from a campaign simulation."""
    sys = ("You are a growth strategist. In 3-4 sentences of British English, explain what "
           "this simulated campaign does to churn, conversion and revenue, and which "
           "behavioural lever drives the change. Use the numbers given; invent nothing.")
    out = _llm(sys, json.dumps(segment_summary))
    if out:
        return out
    s = segment_summary
    return (f"Applying \u201c{s['lever_label']}\u201d to {s['n']} targeted twins is projected to cut "
            f"average churn from {s['churn_before']:.0%} to {s['churn_after']:.0%}. Of the "
            f"\u20b9{s['gross_value_saved']:,.0f} in modelled at-risk value this protects, a conservative "
            f"{s['realisation_rate']:.0%} realisation gives \u20b9{s['revenue_recovered']:,.0f} recovered "
            f"against a \u20b9{s['campaign_cost']:,.0f} cost \u2014 a {s['roi']:.1f}x return. The lever works "
            f"chiefly by resetting recency and re-engaging dormant value.")


def exec_brief(portfolio: dict) -> str:
    sys = ("You are briefing a CMO. In 4-5 sentences of British English, give the single most "
           "important insight from this loyalty portfolio and one concrete action. Be specific "
           "with numbers; invent nothing.")
    out = _llm(sys, json.dumps(portfolio))
    if out:
        return out
    p = portfolio
    return (f"Across {p['total_customers']} customers, \u20b9{p['total_revenue_at_risk']:,.0f} of lifetime "
            f"value sits in the high-churn band, concentrated in the {p['top_risk_tier']} tier. A "
            f"notable {p['silent_accumulators']} customers are 'silent accumulators' \u2014 actively "
            f"earning rewards but not redeeming, a leading churn indicator. Prioritise a redeem-now "
            f"campaign on this group: the simulator projects it recovers the largest share of at-risk "
            f"revenue at the lowest cost.")
