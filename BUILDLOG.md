# Build log

The point of this file: the judges explicitly want *"what broke at 2 AM, and how you
got out."* That story has to be real and yours. Log decisions and — especially —
things that broke and how you fixed them, as they happen. Raw and honest beats polished.

Format per entry: date, what happened, why it mattered, what you did.

---

## 2026-08-26 — Decisions locked

- **Track 02 (AI Risk Manager).** Chose fraud over returns/chargebacks because fraud is
  dead-centre a payment gateway's problem and has the cleanest honest-metrics story.
- **Problem framing:** not a bare classifier — a real-time *allow / review / block*
  decision engine, with thresholds chosen to minimise **money lost**.
- **Dataset:** IEEE-CIS Fraud Detection (Vesta). Closest public analog to gateway traffic
  (device, email, card, velocity signals). The competition test set has no public labels,
  so we carve a **time-ordered** held-out test set from the labelled training data.
- **Model choice:** scikit-learn `HistGradientBoostingClassifier` — strong on tabular data,
  handles missing values natively, no extra install. Logistic regression as the baseline.
- **Why time-based split:** fraud is temporally structured; a random split leaks the future
  and inflates metrics. Train early → validate next → test latest.

## 2026-08-26 — The cost optimizer wanted to review *everything*

- **Symptom:** first end-to-end run picked thresholds `0.02 / 0.26` and produced an action
  mix of **allow 0% / review 100% / block 0%**. On paper it "beat" allow-everything by 60%.
- **Why it mattered:** it's a degenerate optimum. My review cost is a flat $3 that's assumed
  to resolve the case, so when fraud losses per transaction exceed $3, the math says "just
  review every payment." That would sail through an offline cost metric and be *completely
  useless in production* — no fraud team has the headcount to manually review 100% of traffic.
  Shipping a model that quietly assumes infinite review capacity is exactly the kind of thing
  that looks great in a notebook and blows up in the real world.
- **Fix:** added a `MAX_REVIEW_RATE` capacity constraint (default 5% of traffic) to the cost
  model and made the threshold search reject any `(t_low, t_high)` pair that routes more than
  that to review. The optimizer now has to *earn* its savings by using the model to separate
  risk, not by hiding behind an analyst army. Post-fix on synthetic: allow 97% / review 3% /
  block 0%, and the weak synthetic model only saves ~4% — honest, because synthetic signal is
  barely above the base rate. The real lift shows up on real data with the gradient-boosted model.
- **Lesson:** an offline cost metric will happily exploit any assumption you forgot to bound.
  Constrain the *operational* reality (review capacity), not just the loss function.

## 2026-08-26 — Two leaks and a broken explainer, caught in a self-review

Ran a skeptical pass over the whole repo before trusting any number. Three things worth writing down:

- **I was selecting the model on the test set.** The training loop computed each model's
  PR-AUC on the *test* split and then picked the best one — which means the test set was
  influencing model selection. That's textbook leakage, and it quietly inflates the headline
  number. Fix: select the headline model on the **validation** PR-AUC, and touch the test set
  exactly once at the very end to report its honest score. (Calibration and thresholds were
  already validation-only — this was the one leak.) Lesson: "held-out" means held out from
  *selection* too, not just training.
- **SHAP can't explain the model I chose.** I'd wired per-transaction reasons through
  `shap.TreeExplainer`, but SHAP's TreeExplainer doesn't support
  `HistGradientBoostingClassifier`. It would have thrown, hit my fallback, and shown the
  *same* "reasons" for every transaction — worse than no explanation, because it looks real.
  It never surfaced locally because the offline run uses the NumPy logistic (whose exact-weight
  path works). Fix: dropped SHAP entirely and wrote a model-agnostic **occlusion** explainer —
  replace each feature with its typical value, measure how the fraud probability moves — plus
  permutation importance for the global view. Both are tested. Lesson: don't let a dependency
  you can't exercise sit on the critical path of your demo.
- **The cost curve plotted points the policy wasn't allowed to pick.** The plot minimised over
  thresholds that violated the review-capacity cap, so the chosen line looked suboptimal. Fixed
  the surface to only include capacity-eligible points.

Also hardened smaller things the review flagged: tie-safe time split (rows sharing a timestamp
can't straddle the boundary), canonical average-precision when scikit-learn is present, and the
data mode ("synthetic" vs "real") is now stamped into `summary.json` and every plot title so a
synthetic smoke-test can never be mistaken for a real result.

## Template — copy this when something breaks

```
## YYYY-MM-DD — <short title of what broke>
- Symptom: what I saw (error, weird metric, crash).
- Why it mattered: what it would have wrecked if shipped.
- Fix: what I actually changed.
- Lesson: the one line I'd tell past-me.
```
