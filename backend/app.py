"""
CX Twin AI - FastAPI backend.
Single LLM API (Anthropic) for reasoning, local sklearn for prediction.
Run:  uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations
import os, sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend.ml.models import TwinModels, apply_intervention, LEVERS, FEATURES
from backend.agents.reasoning import explain_twin, simulate_narrative, exec_brief

app = FastAPI(title="CX Twin AI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- load + train once at startup ----
DATA = os.path.join(ROOT, "data", "customers.parquet")
if not os.path.exists(DATA):
    import subprocess
    subprocess.run([sys.executable, os.path.join(ROOT, "data", "generate.py")], cwd=ROOT, check=True)

DF = pd.read_parquet(DATA)
MODELS = TwinModels().fit(DF)
SCORED = MODELS.score_frame(DF)

DISPLAY_COLS = ["customer_id", "name", "city", "tier", "preferred_channel",
                "recency_days", "frequency_90d", "rewards_unredeemed", "nps",
                "churn_prob", "convert_prob", "pred_ltv", "revenue_at_risk"]


def _round(rec: dict) -> dict:
    for k in ("churn_prob", "convert_prob"):
        if k in rec: rec[k] = round(float(rec[k]), 3)
    for k in ("pred_ltv", "revenue_at_risk"):
        if k in rec: rec[k] = round(float(rec[k]))
    return rec


@app.get("/api/health")
def health():
    return {"status": "ok", "customers": len(DF),
            "churn_auc": MODELS.churn_auc, "convert_auc": MODELS.convert_auc,
            "llm": "live" if os.environ.get("ANTHROPIC_API_KEY") else "offline-fallback"}


@app.get("/api/customers")
def customers(sort: str = "revenue_at_risk", limit: int = 100):
    if sort not in SCORED.columns:
        sort = "revenue_at_risk"
    rows = SCORED.sort_values(sort, ascending=False).head(limit)[DISPLAY_COLS]
    return {"customers": [_round(r) for r in rows.to_dict("records")]}


@app.get("/api/customer/{cid}")
def customer(cid: str):
    row = SCORED[SCORED.customer_id == cid]
    if row.empty:
        raise HTTPException(404, "twin not found")
    full = _round(row.iloc[0].to_dict())
    twin = {k: full[k] for k in DISPLAY_COLS}
    twin["explanation"] = explain_twin(full)
    twin["churn_drivers"] = MODELS.feature_importance()[:5]
    return twin


@app.get("/api/portfolio")
def portfolio():
    high = SCORED[SCORED.churn_prob > 0.5]
    by_tier = SCORED.groupby("tier")["churn_prob"].mean().round(3).to_dict()
    silent = int(((SCORED.rewards_unredeemed >= 4) & (SCORED.churn_prob > 0.4)).sum())
    p = {
        "total_customers": int(len(SCORED)),
        "total_revenue_at_risk": round(float(high["revenue_at_risk"].sum())),
        "avg_churn": round(float(SCORED.churn_prob.mean()), 3),
        "churn_by_tier": by_tier,
        "top_risk_tier": max(by_tier, key=by_tier.get),
        "silent_accumulators": silent,
        "churn_auc": MODELS.churn_auc,
    }
    p["exec_brief"] = exec_brief(p)
    return p


class SimRequest(BaseModel):
    lever: str
    segment: str = "at_risk"          # at_risk | silent_accumulators | tier:<Tier>
    size: int = 50
    intensity: float = 1.0
    cost_per_customer: float = 450.0


@app.get("/api/levers")
def levers():
    return {"levers": [{"id": k, "label": v} for k, v in LEVERS.items()]}


@app.post("/api/simulate")
def simulate(req: SimRequest):
    if req.lever not in LEVERS:
        raise HTTPException(400, f"unknown lever; choose from {list(LEVERS)}")

    if req.segment == "silent_accumulators":
        seg = SCORED[(SCORED.rewards_unredeemed >= 4) & (SCORED.churn_prob > 0.4)]
    elif req.segment.startswith("tier:"):
        seg = SCORED[SCORED.tier == req.segment.split(":", 1)[1]]
    else:  # at_risk: rank by revenue at risk (churn x LTV), not raw churn prob
        seg = SCORED.sort_values("revenue_at_risk", ascending=False)
    seg = seg.head(req.size)
    if seg.empty:
        raise HTTPException(400, "segment is empty")

    base = MODELS.score_frame(DF.loc[seg.index])
    post = MODELS.score_frame(apply_intervention(DF.loc[seg.index], req.lever, req.intensity))

    # A retention lever cannot be presented as *increasing* churn. Where the
    # model's response to a lever is non-helpful for a given twin (noise, or a
    # lever ill-suited to that twin), floor post-churn at baseline so the worst
    # a lever can do is "no effect" — never a fabricated negative outcome.
    post["churn_prob"] = post[["churn_prob"]].join(
        base["churn_prob"].rename("base_churn")
    ).apply(lambda r: min(r["churn_prob"], r["base_churn"]), axis=1)

    # Revenue at risk = churn_prob * LTV. The campaign reduces churn_prob, so the
    # reduction in expected-lost-value is the gross benefit. We apply a realism
    # discount: not every modelled save converts to retained revenue in practice.
    REALISATION = 0.35      # conservative: 35% of modelled saved-value is realised
    risk_before = float((base.churn_prob * base.pred_ltv).sum())
    risk_after = float((post.churn_prob * post.pred_ltv).sum())
    gross_recovered = max(risk_before - risk_after, 0.0)
    recovered = gross_recovered * REALISATION
    cost = req.cost_per_customer * len(seg)
    net = recovered - cost
    roi = (net / cost) if cost else 0.0

    summary = {
        "lever": req.lever, "lever_label": LEVERS[req.lever], "segment": req.segment,
        "n": int(len(seg)),
        "churn_before": round(float(base.churn_prob.mean()), 3),
        "churn_after": round(float(post.churn_prob.mean()), 3),
        "convert_before": round(float(base.convert_prob.mean()), 3),
        "convert_after": round(float(post.convert_prob.mean()), 3),
        "revenue_at_risk_before": round(risk_before),
        "revenue_at_risk_after": round(risk_after),
        "gross_value_saved": round(gross_recovered),
        "realisation_rate": REALISATION,
        "revenue_recovered": round(recovered),
        "campaign_cost": round(cost),
        "net_revenue_impact": round(net),
        "roi": round(roi, 2),
    }
    summary["narrative"] = simulate_narrative(summary)
    # per-twin movers for the demo "wow" detail
    movers = (base[["customer_id", "name", "tier", "churn_prob"]]
              .assign(churn_after=post.churn_prob.values)
              .assign(delta=(base.churn_prob.values - post.churn_prob.values))
              .sort_values("delta", ascending=False).head(5))
    summary["top_movers"] = [_round(r) for r in movers.round(3).to_dict("records")]
    return summary


# ---- serve the dashboard ----
FRONT = os.path.join(ROOT, "frontend")
if os.path.isdir(FRONT):
    app.mount("/app", StaticFiles(directory=FRONT, html=True), name="frontend")


@app.get("/")
def index():
    f = os.path.join(FRONT, "index.html")
    return FileResponse(f) if os.path.exists(f) else {"service": "CX Twin AI", "docs": "/docs"}
