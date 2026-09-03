"""
evaluate.py -- honest metrics, calibration check, and the money cost curve.

All metrics are pure NumPy so they can be unit-tested without a model. The
headline metric is PR-AUC (average precision): when fraud is a fraction of a
percent, accuracy and even ROC-AUC flatter you -- precision/recall tells the truth.
"""

from __future__ import annotations
import numpy as np

from config import decide, transaction_costs, MAX_REVIEW_RATE

try:                                            # canonical AP (correct tie handling) if available
    from sklearn.metrics import average_precision_score as _sk_ap
    _HAS_SK_AP = True
except Exception:
    _HAS_SK_AP = False


def pr_curve(y_true, scores):
    """Precision and recall arrays, thresholds swept high->low."""
    y = np.asarray(y_true, int)
    order = np.argsort(-np.asarray(scores, float))
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    return recall, precision


def average_precision(y_true, scores):
    """Area under the precision-recall curve (average precision).

    Uses scikit-learn's canonical implementation when available (correct handling of
    tied scores, so a constant predictor scores exactly the base rate). Falls back to a
    pure-NumPy step integration otherwise, so the metric is testable without sklearn.
    """
    y = np.asarray(y_true, int)
    if _HAS_SK_AP and len(np.unique(y)) > 1:
        return float(_sk_ap(y, np.asarray(scores, float)))
    recall, precision = pr_curve(y, scores)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def precision_recall_at(y_true, scores, threshold):
    y = np.asarray(y_true, int)
    pred = (np.asarray(scores, float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn,
    }


def calibration_curve(y_true, scores, n_bins=10):
    """Reliability table: mean predicted score vs. observed fraud rate per bin."""
    y = np.asarray(y_true, int)
    s = np.asarray(scores, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(s, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({
                "mean_score": float(s[m].mean()),
                "observed_rate": float(y[m].mean()),
                "count": int(m.sum()),
            })
    return rows


def search_cost_thresholds(y_true, scores, amounts, grid=None, max_review_rate=None):
    """Grid-search the (t_low, t_high) pair that minimises realised money cost,
    subject to a manual-review capacity cap.

    Thresholds should be searched on VALIDATION data, then applied to TEST.
    A pair is only eligible if it routes no more than `max_review_rate` of traffic
    to review (defaults to config.MAX_REVIEW_RATE) -- this is what stops the search
    from collapsing to the degenerate "review everything" optimum.
    Returns the best thresholds and the full cost surface (for plotting).
    """
    y = np.asarray(y_true, int)
    s = np.asarray(scores, float)
    a = np.asarray(amounts, float)
    if grid is None:
        grid = np.round(np.linspace(0.02, 0.98, 25), 3)
    if max_review_rate is None:
        max_review_rate = MAX_REVIEW_RATE

    best = None
    surface = []
    for t_low in grid:
        for t_high in grid:
            if t_high < t_low:
                continue
            d = decide(s, t_low, t_high)
            review_rate = float((d == "review").mean())
            if review_rate > max_review_rate:
                continue                       # violates review capacity -- ineligible
            total = float(transaction_costs(y, d, a).sum())
            surface.append((float(t_low), float(t_high), total))  # eligible points only
            if best is None or total < best["total_cost"]:
                best = {"t_low": float(t_low), "t_high": float(t_high),
                        "total_cost": total, "review_rate": review_rate}
    if best is None:                           # no pair satisfied the cap -> allow-all
        best = {"t_low": 1.01, "t_high": 1.01, "total_cost": float("inf"), "review_rate": 0.0}
    return best, surface
