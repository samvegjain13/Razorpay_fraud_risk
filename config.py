"""
config.py -- central knobs: columns and the money cost model.

The cost model is the heart of this project. A fraud system is not judged on
accuracy; it is judged on MONEY. Three outcomes cost very different amounts:

  - Missed fraud   (we ALLOW a fraudulent payment): lose the amount + a fixed
                    chargeback/dispute fee.
  - False decline  (we BLOCK a legitimate payment): lose a good customer's sale
                    plus goodwill -- a real, routinely-undercounted cost.
  - Manual review  (we send it to a human): costs analyst time, but resolves the
                    case correctly.

Every number below is an ASSUMPTION, written down so it can be challenged and
sensitivity-tested. That honesty is the point, not the exact figures.
"""

import numpy as np

# --- column names in the IEEE-CIS data ---------------------------------------
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"      # seconds offset from a reference point
AMOUNT_COL = "TransactionAmt"
TARGET = "isFraud"

# --- cost model (all amounts in the dataset's currency units) ----------------
CHARGEBACK_FEE = 15.0           # fixed dispute cost when a fraud is allowed
FALSE_DECLINE_FIXED = 5.0       # goodwill/lost-customer proxy per false decline
FALSE_DECLINE_RATE = 0.15       # + this fraction of the blocked amount (lost sale)
REVIEW_COST = 3.0               # analyst time per manually reviewed transaction

# Fraud operations cannot manually review every payment -- there is finite analyst
# headcount. Without this cap, a cheap flat review fee makes "review everything" look
# optimal on paper, which is operationally useless. This is the guardrail that forces
# the model to actually separate risk. Tune to your team's real capacity.
MAX_REVIEW_RATE = 0.05          # at most 5% of traffic may be routed to manual review


def decide(scores, t_low, t_high):
    """Turn risk scores into actions using two thresholds.

    score < t_low            -> "allow"
    t_low <= score < t_high  -> "review"
    score >= t_high          -> "block"
    """
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= t_high, "block",
                    np.where(scores >= t_low, "review", "allow"))


def transaction_costs(y_true, decisions, amounts):
    """Realised money cost of each decision, given the true labels.

    Returns a per-transaction cost array (same length as inputs).
    """
    y = np.asarray(y_true, dtype=int)
    d = np.asarray(decisions)
    a = np.asarray(amounts, dtype=float)
    cost = np.zeros(len(y), dtype=float)

    # ALLOW a fraud -> lose the money + chargeback fee
    m = (d == "allow") & (y == 1)
    cost[m] = a[m] + CHARGEBACK_FEE

    # BLOCK a legit customer -> false-decline cost
    m = (d == "block") & (y == 0)
    cost[m] = FALSE_DECLINE_FIXED + FALSE_DECLINE_RATE * a[m]

    # REVIEW anything -> analyst cost (assumed resolved correctly afterwards)
    m = (d == "review")
    cost[m] = REVIEW_COST

    # ALLOW a legit (0) and BLOCK a fraud (0) cost nothing here.
    return cost


def cost_summary(y_true, decisions, amounts):
    """Human-readable breakdown of total and per-1k cost + action mix."""
    costs = transaction_costs(y_true, decisions, amounts)
    d = np.asarray(decisions)
    n = len(d)
    return {
        "n": n,
        "total_cost": float(costs.sum()),
        "cost_per_1k": float(costs.sum() / n * 1000),
        "pct_allow": float((d == "allow").mean()),
        "pct_review": float((d == "review").mean()),
        "pct_block": float((d == "block").mean()),
    }
