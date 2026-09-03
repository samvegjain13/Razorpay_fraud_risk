# Demo video + build story

Two of the three things the judges said they actually look at live here: **a 5-minute video of it working**, and **what broke at 2 AM and how you got out**. (The third — a repo that runs — is the rest of this project.) This file is your script and your notes. Say it in your own words; don't read it robotically.

---

## Before you hit record

Do the real run first, so the video shows real numbers, not the synthetic smoke test:

1. Open `notebooks/walkthrough.ipynb` in Colab, run it top to bottom on the real IEEE-CIS data.
2. Copy the printed numbers into the README's **Honest results** section: headline model, test PR-AUC vs. base rate, and the three cost-per-1,000 figures (policy / allow-all / block-all), plus the chosen thresholds and the allow/review/block mix.
3. Save the three plots from `reports/` (`pr_curve.png`, `calibration.png`, `cost_curve.png`) — you'll show them.
4. Have one browser tab on the repo (README + BUILDLOG visible) and one on the finished Colab notebook.

Keep it to ~5 minutes. Screen recording + your voice is enough; no slides needed.

---

## The 5-minute script

**0:00–0:30 — What and why (no buzzwords).**
> "This is Sentinel, a payment-fraud risk engine for Track 02. A gateway sees a transaction and has to do one of three things: allow it, send it to a human for review, or block it. I didn't build a classifier that prints an accuracy score — I built the decision, and I tuned it to lose the least *money*, because that's the thing Razorpay actually cares about."

**0:30–1:15 — The framing that makes it real.**
[SHOW: README problem section.]
> "Two mistakes cost very different amounts. Miss a fraud and you eat the transaction plus a chargeback fee. Falsely decline a good customer and you lose the sale and some goodwill — a cost most fraud demos pretend doesn't exist. So my headline metric isn't accuracy, it's money lost per 1,000 transactions. All the cost assumptions are written down in `config.py` so anyone can challenge them."

**1:15–2:15 — Show it run.**
[SHOW: the Colab notebook, scroll to the pipeline output cell.]
> "Here it is on the real IEEE-CIS data. It does a time-ordered split — train on the past, test on the most recent window — because a random split leaks the future and every metric lies. It trains a baseline, a logistic model, and a gradient-boosted model, calibrates the scores, and picks the decision thresholds on the validation set. The test set is scored exactly once, at the end."
[SHOW: the cost lines.]
> "This line is the whole pitch: our policy loses **[X]** per 1,000 transactions, versus **[Y]** if you approve everything. That's the number."

**2:15–3:15 — Honesty: the metrics and the plots.**
[SHOW: `pr_curve.png` and `calibration.png`.]
> "Fraud is a couple percent of traffic, so I report PR-AUC, not accuracy — accuracy of 97% is just what you get by approving everyone. Here's precision-recall against the base rate. And this is calibration: a score of 0.9 really means about 90%, which is what lets a money threshold mean anything."

**3:15–4:00 — Explainable + graceful.**
[SHOW: the live-scoring cell output, including the broken input.]
> "Every decision comes with reasons. For the gradient-boosted model I use an occlusion method — swap each feature to its typical value and see how the risk moves — so no black box and no SHAP dependency that could silently break. And the last input here is deliberate garbage: instead of crashing, it routes to manual review. That's the safe default a payments system should have."

**4:00–5:00 — What broke (the honest part).**
[SHOW: BUILDLOG.md.]
> "Two things I want to be honest about. First, my cost optimizer's initial 'best' policy was to send *every* transaction to review — mathematically cheaper than eating fraud, operationally useless. I added a review-capacity cap so it has to actually separate risk. Second, when I reviewed my own code I caught that I was picking the best model using the test set — textbook leakage. I moved selection to validation and made the test set a one-time final check. Both are in the build log. I'd rather show you the bugs I caught than pretend there weren't any."

---

## The build story (say this truthfully)

The judges want the real thing, not a hero narrative. You genuinely have three honest moments from building this — use whichever feels most true to how it went for you:

- **The "review everything" trap.** The first cost-optimal policy routed 100% of transactions to manual review. It beat approving-everything on paper by 60%, and it was completely useless — no team can hand-check every payment. The fix was to model the real constraint: reviewers have finite capacity. Cap the review rate, and the optimizer has to earn its savings from the model instead of hiding behind an analyst army.
- **Catching my own leak.** I was selecting the headline model by its score on the test set. That's leakage — the test set is supposed to be untouched until the very end. I caught it re-reading my own pipeline, moved model selection to the validation window, and now the test set is scored once.
- **The explainer that would have lied.** I'd wired explanations through SHAP, then realized SHAP's tree explainer doesn't support the model I picked — it would have shown identical "reasons" for every transaction and looked fine. I dropped SHAP and wrote a simple occlusion explainer I could actually test.

Pick one, tell it plainly, and if you hit your *own* wall while running it (a Kaggle download that failed, a Colab timeout, a column that broke the merge) — tell that one instead. Yours is better than any of mine because it's yours.

---

## Recording tips

- Talk like you're explaining it to a teammate, not presenting to a panel. The judges explicitly said they don't want buzzwords.
- Show the terminal/notebook *actually running* at least once — a live run beats any slide.
- It's fine to say "I didn't get to X" — naming a limitation reads as competence, not weakness. The README already lists them; echo one.
- End on the money number and the graceful-failure demo. Those two land hardest.
