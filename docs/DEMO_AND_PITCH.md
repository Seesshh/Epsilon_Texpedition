# CX Twin AI — Demo script & pitch deck

---

## PART A · 5-minute demo script

**Roles:** one driver (laptop), one narrator. Have the app open at
`localhost:8000` *before* you start. Numbers below match the shipped build.

### 0:00 — The hook (don't open with the product)
> "Every CRM on earth can tell you a customer churned. Ours tells you who's
> *about* to — and lets you test the campaign to save them *before* you spend a
> rupee. We call it a digital twin for every customer."

### 0:30 — The problem, in one number
Point at the portfolio strip.
> "Nine hundred customers. **₹1.1 crore of lifetime value sits in the
> high-churn band.** A CRM shows you this *after* they've gone. Watch what we do
> instead."

### 1:00 — Show a twin (proof it's real, not a dashboard of averages)
Click the top row → drawer opens.
> "This is Aryan's twin. The model gives him an 87% churn probability — and the
> system explains *why*: 68 days silent, 8 unredeemed rewards, NPS of 4. That
> churn number isn't an LLM guessing. It's a gradient-boosting model,
> **0.76 AUC, cross-validated.** The AI writes the explanation; the maths is a
> real model."

### 1:45 — The wow: simulate before you spend
Close drawer. In the simulator: segment **Highest revenue at risk**,
intervention **10% loyalty credit**, hit **Simulate**.
> "Here's what nobody else in this room can do. I'm going to run this campaign
> on the twins — not on real customers, on their models — and watch."

Bars animate.
> "**Churn drops 69% → 48%.** Predicted revenue recovered: **₹6.6 lakh**, net
> ₹6.4 lakh after cost — a **40× return.** And the strategist explains the
> causal mechanism: it's resetting recency and re-engaging dormant value."

### 2:45 — Prove the model has judgement (this is what wins)
Switch intervention to **Near-tier upgrade nudge**, re-run.
> "Watch — same segment, different lever. ROI goes *negative*. The model knows
> you can't nudge someone toward a tier they'll never reach if they're already
> leaving. It's not pattern-matching; it's choosing."

Switch to **Concierge support**.
> "Support outreach? Barely moves it — 1.7×. Because this segment isn't leaving
> over support issues. The twin tells you which lever fits."

### 3:30 — Targeting intelligence
Switch segment to **Silent accumulators**, lever back to loyalty credit.
> "And it picks *who*. These 85 customers are earning rewards but never
> redeeming — a silent churn signal a human analyst would miss entirely. The
> simulator sizes the exact campaign to recover them."

### 4:00 — The cost stress-test (defuses the skeptic)
Drag the **cost-per-customer** slider up, re-run.
> "Think our ROI is too rosy? Push the cost up. It recalculates live. We
> already discount to 35% realisation — we're being conservative on purpose."

### 4:30 — Close
> "So: a digital twin that predicts, a simulator that tests interventions
> before you launch, and an AI that explains every decision in the language of
> a CMO. One LLM, one local model, no third-party sprawl. **From reacting to
> customers, to rehearsing their future.** That's CX Twin AI."

---

## PART B · 10-slide pitch deck

**Slide 1 — Title**
CX Twin AI · *A digital twin for every customer.*
Predict churn. Simulate the save. Measure the impact — before you spend.

**Slide 2 — Problem**
CRMs are rear-view mirrors. They record what customers did; they can't tell you
who's about to leave, which intervention will work, or what it'll return.
Marketers launch retention campaigns blind and measure success after the budget
is already gone. Result: wasted spend on the already-lost, and missed saves on
the quietly-slipping.

**Slide 3 — Market**
Customer-retention and CX software is a multi-billion-dollar category growing
double digits, riding the shift from acquisition to retention economics (a 5%
retention lift can lift profit 25%+). Every loyalty programme, D2C brand, bank,
and telco is a buyer. Wedge: loyalty-heavy retailers sitting on rich behavioural
data they can't act on.

**Slide 4 — Solution**
A living behavioural twin of each customer. Three moves a CRM can't make:
(1) predict churn / LTV / offer-acceptance per twin; (2) *simulate* a campaign
on the twins and see the counterfactual revenue impact before launch;
(3) explain every prediction and recommendation in plain business language.

**Slide 5 — Technology**
Prediction: scikit-learn gradient boosting, 0.76 cross-validated churn AUC,
trains in seconds, runs anywhere. Counterfactual engine: interventions shift a
twin's behavioural features; the same model re-scores them — a genuine
what-if, not an LLM guess. Reasoning: a single LLM API for explanation and
narrative. FastAPI backend, zero-build dashboard. No Docker, no multi-API
sprawl.

**Slide 6 — AI innovation**
The defensible idea is the *clean separation*: the model owns the numbers, the
LLM owns the meaning. That's why our churn scores survive scrutiny and our
explanations stay grounded. The simulator turns a predictive model into a
*decision* tool — you don't just see risk, you rehearse the response.

**Slide 7 — Business model**
SaaS, priced per active twin / per seat, with a usage tier on simulation
volume. Land with a retention-team pilot on historical data (instant ROI proof
on saved revenue), expand to campaign operations and executive reporting.

**Slide 8 — Competitive advantage**
Versus CRMs (Salesforce, HubSpot): they store and report; we predict and
simulate. Versus chatbots: they answer questions; we change outcomes. Versus
DIY data-science: we ship the counterfactual loop and the explanation layer
out of the box. The simulate-before-spend loop is the moat.

**Slide 9 — Roadmap**
Now: churn/LTV twins + campaign simulator (this build). Next: live event
ingestion, multi-touch journey twins, automated campaign execution on the
chosen channel. Later: reinforcement-learning policy that proposes the optimal
intervention per twin, and portfolio-level budget optimisation.

**Slide 10 — Close**
CRMs made customer data *visible*. CX Twin AI makes the customer's future
*testable*. Stop reacting. Start rehearsing.
*Demo: localhost:8000 · one LLM, one local model, runs offline.*
