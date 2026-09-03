# Sentinel — a cost-aware payment fraud risk engine

**One line:** score every incoming payment for fraud risk in real time, then make the call a payment gateway actually has to make — *allow, send to manual review, or block* — with the thresholds tuned to minimise **money lost**, not to maximise accuracy.

Built for the Razorpay Buildathon, **Track 02 — AI Risk Manager**. Strictly defence-only.

---

## The problem (and why it's Razorpay's problem)

A payment gateway sees a transaction and has milliseconds to decide what to do with it. Two mistakes cost very different amounts:

- **Miss a fraud** (allow it): you eat the transaction amount *plus* a chargeback/dispute fee.
- **Falsely decline a good customer** (block them): you lose a real sale and some goodwill — a cost most fraud demos ignore entirely.

So the honest question is never "what's our accuracy?" It's **"how much money does our decision policy lose per 1,000 transactions, and can we lose less?"** That framing — and measuring the false-positive cost out loud — is the whole point of this project.

## Approach

1. **Real data, gateway-shaped.** IEEE-CIS Fraud Detection (Vesta) — real online-transaction fraud carrying the device, email-domain, card, and velocity signals a gateway actually sees.
2. **A time-based, leakage-free split.** Fraud has strong temporal structure; a random split leaks the future into the past and inflates every metric. We train on the earliest window, validate on the next, and test on the most recent — the model faces "tomorrow" the way it will in production.
3. **An imbalance-aware model.** Fraud is a fraction of a percent of traffic, so we report **PR-AUC / average precision**, not accuracy or ROC-AUC alone, and compare against honest baselines (a majority-class rule and a logistic-regression baseline) so the strong model has to *earn* its complexity.
4. **Calibration.** We calibrate the scores so a "0.9" really means ~90% — otherwise a risk threshold is meaningless.
5. **A cost-based decision policy — with an operational guardrail.** We convert scores into *allow / review / block* using two thresholds chosen to **minimise expected money lost** on validation data, then report realised cost on the held-out test set. Crucially, the search is capped at a realistic **manual-review capacity** (default 5% of traffic): without that cap the optimizer "cheats" by sending everything to human review, which is free-looking on paper and impossible in production. (That failure actually happened — see `BUILDLOG.md`.)
6. **Explainable decisions + graceful failure.** Every decision comes with its top contributing features — exact weight contributions for the logistic model, and a model-agnostic *occlusion* method for the gradient-boosted one (swap each feature to its typical value, see how the fraud probability moves). No SHAP dependency, so nothing silently degrades. A transaction with missing/garbage fields is scored conservatively (routed to review) instead of crashing.

Model selection is done on the **validation** window only; the held-out test set is scored exactly once, at the end, to report the honest number.

## How each part of the judging bar is met

| The bar says | How this project answers it |
|---|---|
| "a working detector … for one class of loss" | Real-time payment-fraud risk engine |
| "measured precision and recall on a held-out test set" | Time-ordered held-out test set; precision/recall + PR-AUC reported |
| "honest metrics including false-positive cost" | Explicit money cost model; false-decline cost is a first-class number |
| "one failure handled gracefully" | Missing/garbage-input transactions score conservatively, with a logged fallback |
| "strictly defense-only" | The system only *scores and blocks* fraud; nothing offense-capable |

## Repo structure

```
razorpay-fraud-risk/
├── README.md            # you are here
├── requirements.txt
├── .gitignore
├── config.py            # columns + the money cost model & review-capacity cap (the heart)
├── run_pipeline.py      # one command: load → split → train → calibrate → cost → explain → demo
├── BUILDLOG.md          # running log of decisions + what broke (feeds the demo video)
├── src/
│   ├── data.py          # load IEEE-CIS or a synthetic stand-in; leakage-free time split
│   ├── features.py      # minimal, honest feature prep (time-of-day, log-amount, encodings)
│   ├── model.py         # majority baseline + logistic + gradient-boosted, all one interface
│   ├── evaluate.py      # PR-AUC, calibration, money cost curve, capacity-capped policy search
│   ├── explain.py       # explanations: exact weights (logistic), permutation (global), occlusion (local)
│   └── demo.py          # score one transaction end to end, graceful on bad input
├── tests/
│   └── test_core.py     # unit tests for the cost model, leakage guard, capacity cap, metrics
├── notebooks/
│   └── walkthrough.ipynb # Colab-ready end-to-end run on the real dataset
└── reports/             # generated plots + summary.json (created by run_pipeline.py)
```

## How to run

> The competition's own test set has **no public labels**, so we use *only* the labelled training files and carve our own time-ordered test set from them. That is the correct, honest choice — not a shortcut.

### Runs anywhere in ~10 seconds (no dataset, no internet)

The whole pipeline works on a built-in synthetic sample, so you can verify it end to end offline. This is a smoke test of the *plumbing*, not a results run — the synthetic signal is deliberately weak.

```bash
python run_pipeline.py --synthetic     # load → split → train → calibrate → cost → explain → demo
python tests/test_core.py              # unit tests: cost model, leakage guard, capacity cap, metrics
```

If scikit-learn isn't installed, the pipeline automatically falls back to a dependency-free NumPy logistic model so it still runs.

### Real results (Colab-first)

1. Open `notebooks/walkthrough.ipynb` in Google Colab.
2. Get the data from Kaggle (free account, accept the competition rules once):
   ```bash
   pip install kaggle
   # upload your kaggle.json token when prompted
   kaggle competitions download -c ieee-fraud-detection
   ```
3. Run the notebook top to bottom, or from a checkout with the data in `./data`:
   ```bash
   python run_pipeline.py --data data/
   ```
   It trains every model, calibrates, prints the money saved vs. baselines, writes the plots to `reports/`, and runs the live scoring demo (including a deliberately broken input).

## Honest results

_Filled in from the real IEEE-CIS run in Colab — PR-AUC, the cost-per-1,000 vs. baselines, the chosen thresholds, and the three plots (`reports/`) go here. No placeholder bragging until the held-out test set says so._

The offline synthetic smoke test (weak-by-design signal) currently produces: pipeline trains and calibrates cleanly, capacity-capped policy routes ~97% allow / ~3% review, graceful-failure path returns a conservative review on bad input, and all 8 unit tests pass.

## Limitations & honest notes

- The cost model's numbers (chargeback fee, false-decline cost, review cost, review-capacity cap) are **assumptions**, written down in `config.py` so they can be challenged and sensitivity-tested. The conclusions are only as good as those numbers, and the point is that they're explicit — not that they're perfect.
- Manual review is modelled as a flat cost that resolves the case correctly. Real review is noisier; the capacity cap is what keeps this assumption from being exploited.
- IEEE-CIS is US e-commerce card fraud, a close but not identical analog to Razorpay's India/UPI mix.
- Categorical fields are ordinal-encoded and fed to the tree as numbers, so category *order* is arbitrary. It's a reasonable, common baseline; native categorical splits (via HistGBM's `categorical_features`) are a deliberate next step, left out here rather than shipped untested.
- Synthetic mode is for pipeline testing only; all reported metrics come from the real dataset.
