# CX Twin AI — Judge Q&A

The questions that actually decide hackathons cluster around four fears:
*is the AI real, is the data real, does it matter to a business, and can it
scale?* Answers below are grounded in what the shipped build does. Don't
over-claim — the credibility *is* the pitch.

## On whether the AI is real

**1. Is the LLM just making up the churn scores?**
No. Churn, offer-acceptance and LTV come from trained scikit-learn
gradient-boosting models. Churn AUC is 0.76, five-fold cross-validated — the
honest out-of-sample figure. The LLM never produces a number; it only explains
the model's output.

**2. Why 0.76 and not 0.95?**
Because 0.95 on behavioural churn would mean we'd leaked the label or overfit,
and a sharp judge would catch it. 0.76 cross-validated is the realistic band
for behavioural churn, and it's the number that survives scrutiny. We optimised
for *defensible*, not *impressive*.

**3. Then what does the LLM actually add?**
Meaning. It reads a twin's feature vector plus the model output and writes the
causal explanation, the retention rationale, and the executive brief — in a
CMO's language. The model knows *what*; the LLM explains *why*. That separation
is the core design choice.

**4. The simulator — is that a real prediction or theatre?**
Real. An intervention shifts the twin's behavioural features (a loyalty credit
lowers unredeemed-reward balance and recency), then the *same trained model*
re-scores the modified twin. The before/after delta is the model's own
response. It's a genuine counterfactual, not a scripted animation.

**5. How do you know the intervention→feature mapping is right?**
We don't claim it's empirically calibrated yet — it's a behavioural prior
(redeeming rewards re-engages; nudging a far-off tier doesn't). In production
these elasticities are learned from past campaign outcomes via uplift modelling.
The architecture is built for that; the demo uses sensible priors.

**6. Why only one LLM API? Isn't more AI better?**
A single, well-placed LLM call beats a fragile chain of third-party services in
a 48-hour build and in production reliability. And the heavy lifting is the
local model, which costs nothing per call and runs offline. Constraint as
feature.

## On the data

**7. This is synthetic data.**
Yes, and deliberately so — real loyalty data isn't shareable in a hackathon.
But it's not random: churn, conversion and LTV are driven by a hidden,
consistent process over behavioural features, with realistic noise. That's why
the model recovers genuine signal and the counterfactuals move sensibly. Point
it at a real parquet export and the pipeline is unchanged.

**8. Would it work on our real data?**
The feature set (recency, frequency, AOV, tenure, unredeemed rewards, NPS,
support tickets, tier) is standard loyalty telemetry. Swap the data source,
retrain — same code path. We built it data-source-agnostic for exactly this.

**9. What about cold-start customers with no history?**
They fall back to tier- and segment-level priors until enough events accrue.
The twin gets sharper as behaviour accumulates; it degrades gracefully, it
doesn't fail.

## On business value

**10. What's the actual ROI claim?**
On the high-revenue-at-risk segment, the loyalty-credit lever models a 69%→48%
churn cut and ~₹6.6L recovered against ₹22.5k cost. We discount to 35%
realisation, so the headline is conservative. The point isn't the exact
multiple — it's that you can *see and tune it before launch*.

**11. 40× ROI sounds too good.**
It's high because retention spend on high-LTV customers genuinely is among the
highest-ROI marketing there is, and we're already discounting hard. If you
doubt it, drag the cost slider up live — it recalculates. We made the
skepticism testable on purpose.

**12. What stops a marketer doing this in a spreadsheet?**
A spreadsheet can't predict per-customer churn, can't re-score a counterfactual
population, and can't pick the right lever per segment. We showed the same lever
giving 40× on one segment and negative ROI on another — that judgement is the
product.

**13. Who's the buyer and what do they pay for?**
The retention/CRM team. They pay to stop wasting budget on the already-lost and
to catch the quietly-slipping (our "silent accumulators" — 85 customers earning
rewards but never redeeming, a signal humans miss). Priced per active twin plus
simulation usage.

**14. How is "success" measured — the theme asks for clear metrics.**
Three layers, all computed by the system: customer (predicted churn, NPS,
engagement), business (revenue at risk, recovered revenue, campaign ROI), and
model (cross-validated AUC, realisation-adjusted impact). Every claim on screen
traces to one of these.

## On scale and execution

**15. Does this scale past 900 rows?**
Prediction is vectorised and the model is lightweight; scoring hundreds of
thousands of twins is seconds, not minutes. The simulator scores only the
targeted segment. For millions, you batch-score offline and cache — the API
shape doesn't change.

**16. Real-time updates?**
The roadmap adds an event-ingestion path that updates twin features as
behaviour arrives. The model re-scores on a schedule or on threshold breach.
Nothing in the current architecture blocks it.

**17. What did you cut to ship in 48 hours?**
Live event streaming, automated campaign execution, and learned uplift
elasticities. We deliberately went deep on the one differentiating loop —
predict → simulate → explain — rather than shallow on twelve modules. Depth
wins demos.

**18. Biggest technical risk?**
Calibrating intervention elasticities to real outcomes. It's a known problem
with a known method (uplift / causal modelling) and it needs campaign-outcome
data we don't have in a hackathon. Everything else is production-shaped.

**19. Privacy / compliance?**
The twin stores behavioural aggregates, not raw PII for modelling, and runs
fully on-prem/offline if needed (no data leaves for prediction; the LLM call is
optional and sends only anonymised features). That matters for banking/telco
buyers.

**20. Why does this beat a chatbot project?**
A chatbot answers a question and the moment ends. CX Twin AI changes a business
outcome you can measure in rupees, and it does something a human + CRM
demonstrably cannot: rehearse a campaign on a population's behavioural models
before spending a budget. That's the difference between a feature and a company.
