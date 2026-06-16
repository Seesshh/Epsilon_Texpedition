# CX Twin AI — How to run

**An AI digital twin for every customer.** Predict churn, simulate a retention
campaign on the twins, and see the revenue impact *before* you spend a rupee.

Pure Python + pip. No Docker. One optional LLM API key. Runs offline.

---

## 1. Requirements
- Python 3.10+ (tested on 3.12)
- pip

## 2. Install
```bash
cd cx-twin-ai
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## 3. Generate the data (one-off)
```bash
python3 data/generate.py
```
This writes `data/customers.parquet` (900 synthetic customer twins).

## 4. (Optional) enable live LLM reasoning
The product runs fully **without** a key — every explanation falls back to a
deterministic template built from the same model outputs, so the demo never
breaks on flaky WiFi. To enable live reasoning:
```bash
export ANTHROPIC_API_KEY=sk-ant-...      # macOS / Linux
# setx ANTHROPIC_API_KEY sk-ant-...      # Windows (new shell after)
```
The status bar shows `LLM live` vs `LLM offline-fallback`.

## 5. Run
```bash
uvicorn backend.app:app --port 8000
```
Open **http://localhost:8000/**

That's it. The backend trains the models on startup (≈3 s) and serves the
dashboard from the same port.

---

## What you're looking at
- **Portfolio** — total revenue at risk, silent-accumulator count, and a
  generated executive brief.
- **Campaign simulator** — pick a segment + intervention, hit *Simulate*. The
  before/after churn and revenue bars animate; ROI and a per-twin mover list
  appear. Drag the cost slider and re-run to watch ROI respond live.
- **Customer twins** — ranked by revenue at risk. Click any row for the twin's
  full profile and its plain-English "twin read".

## How the numbers are produced (the honest version)
- **Prediction is a trained model, not the LLM.** Churn, offer-acceptance and
  LTV come from scikit-learn gradient-boosting models. Churn AUC is **0.76,
  5-fold cross-validated** (the honest out-of-sample number).
- **The simulator is a real counterfactual.** An intervention shifts a twin's
  behavioural features (e.g. a loyalty credit lowers unredeemed-reward balance
  and recency); the *same trained model* re-scores the modified twin. The delta
  is the model's response, not a guess.
- **The LLM only reasons.** It explains *why* a twin behaves as it does and
  drafts the strategist/exec narratives. It never invents a number.
- **ROI is deliberately conservative.** Only 35% of modelled saved-value is
  counted as realised. Raise the cost slider to stress-test it live.

## API
Interactive docs at **http://localhost:8000/docs**. Key endpoints:
`GET /api/portfolio`, `GET /api/customers`, `GET /api/customer/{id}`,
`GET /api/levers`, `POST /api/simulate`.

## Project layout
```
cx-twin-ai/
├── data/generate.py        synthetic twin dataset + latent churn process
├── backend/
│   ├── app.py              FastAPI: endpoints + simulator economics
│   ├── ml/models.py        trained models + intervention levers
│   └── agents/reasoning.py single LLM API + offline fallback
├── frontend/index.html     the dashboard (vanilla, no build step)
└── docs/                   demo script, pitch deck, judge Q&A
```
