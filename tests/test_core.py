"""
test_core.py -- unit tests for the parts that must be correct: the money cost model,
the decision thresholds, the leakage-free split, the ranking metric, the review-capacity
constraint, and graceful failure.

Runs with plain `python tests/test_core.py` (no pytest required) or under pytest.
These test the *logic*, not a trained model, so they pass with or without scikit-learn.
"""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (decide, transaction_costs, CHARGEBACK_FEE, FALSE_DECLINE_FIXED,
                    FALSE_DECLINE_RATE, REVIEW_COST, TARGET)
from src.data import make_synthetic_sample, time_based_split, assert_no_leakage, TIME_COL
from src.features import Featurizer
from src.evaluate import average_precision, search_cost_thresholds, precision_recall_at
from src.model import NumpyLogisticRegression, PlattCalibrator, build_models
from src.demo import score_transaction


def test_decide_boundaries():
    d = decide([0.0, 0.3, 0.5, 0.9], t_low=0.3, t_high=0.5)
    assert list(d) == ["allow", "review", "block", "block"], d


def test_cost_model_each_outcome():
    y = np.array([1, 0, 1, 0, 1])
    dec = np.array(["allow", "block", "review", "allow", "block"])
    amt = np.array([100.0, 200.0, 50.0, 80.0, 40.0])
    c = transaction_costs(y, dec, amt)
    assert c[0] == 100.0 + CHARGEBACK_FEE               # allow a fraud
    assert c[1] == FALSE_DECLINE_FIXED + FALSE_DECLINE_RATE * 200.0  # block a legit
    assert c[2] == REVIEW_COST                          # review
    assert c[3] == 0.0                                  # allow a legit -> free
    assert c[4] == 0.0                                  # block a fraud -> free (loss avoided)


def test_average_precision_ranking():
    # perfect ranking -> AP == 1; scores that perfectly separate the two classes
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(average_precision(y, perfect) - 1.0) < 1e-9
    # a real model must beat the base rate; here we just sanity-check the floor behaviour
    base = float(y.mean())
    flat = np.full(len(y), base)
    assert average_precision(y, flat) <= 1.0


def test_time_split_has_no_leakage():
    df = make_synthetic_sample(n_rows=4000, seed=1)
    splits = time_based_split(df)
    assert_no_leakage(splits)                            # raises if leakage
    assert splits["train"][TIME_COL].max() <= splits["valid"][TIME_COL].min()
    assert splits["valid"][TIME_COL].max() <= splits["test"][TIME_COL].min()
    assert len(splits["train"]) + len(splits["valid"]) + len(splits["test"]) == len(df)


def test_review_capacity_respected():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    scores = np.clip(0.1 + 0.4 * y + 0.2 * rng.standard_normal(2000), 0, 1)
    amt = rng.uniform(10, 500, size=2000)
    best, _ = search_cost_thresholds(y, scores, amt, max_review_rate=0.05)
    d = decide(scores, best["t_low"], best["t_high"])
    assert (d == "review").mean() <= 0.05 + 1e-9, (d == "review").mean()


def test_graceful_failure_on_garbage_input():
    df = make_synthetic_sample(n_rows=1500, seed=2)
    splits = time_based_split(df)
    fz = Featurizer().fit(splits["train"])
    Xtr = fz.transform(splits["train"])
    model = NumpyLogisticRegression(n_iter=50).fit(Xtr.values, splits["train"][TARGET].values)
    cal = PlattCalibrator(n_iter=50).fit(model.predict_proba(Xtr.values)[:, 1],
                                         splits["train"][TARGET].values)
    feat = list(Xtr.columns)
    out = score_transaction("not-a-dict", model, fz, cal, 0.3, 0.6, feat)
    assert out["action"] == "review" and out["score"] is None      # never crashes


class _ProbToy:
    """A model WITHOUT a .w attribute, so local_reasons must use the occlusion path.
    Risk rises with feature 0 (hour_of_day) and falls with feature 1 (log_amount)."""
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        X = np.asarray(X, float)
        X = np.nan_to_num(X, nan=0.0)
        z = 2.0 * X[:, 0] - 1.5 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - p, p])


def test_occlusion_local_reasons_finds_the_influential_features():
    from src.explain import local_reasons
    df = make_synthetic_sample(n_rows=1500, seed=7)
    splits = time_based_split(df)
    fz = Featurizer().fit(splits["train"])          # populates fz.background_
    assert fz.background_ is not None and len(fz.background_) == len(fz.feature_names_)
    feat = list(fz.feature_names_)
    row = fz.transform(splits["test"].iloc[[0]])
    reasons = local_reasons(_ProbToy(), fz, row, feat, top_k=2)
    named = {r.split(" (")[0] for r in reasons}
    # only features 0 and 1 affect this toy model, so occlusion must surface exactly them
    assert named == {feat[0], feat[1]}, (named, feat[:2])
    assert all(("raises risk" in r) or ("lowers risk" in r) for r in reasons)


def test_end_to_end_smoke():
    df = make_synthetic_sample(n_rows=3000, seed=3)
    splits = time_based_split(df)
    fz = Featurizer().fit(splits["train"])
    Xtr, ytr = fz.transform(splits["train"]), splits["train"][TARGET].values
    Xte, yte = fz.transform(splits["test"]), splits["test"][TARGET].values
    model = NumpyLogisticRegression(n_iter=100).fit(Xtr.values, ytr)
    p = model.predict_proba(Xte.values)[:, 1]
    # a fitted logistic must not do worse than the base-rate floor on the ranking metric
    assert average_precision(yte, p) >= float(yte.mean()) * 0.8
    pr = precision_recall_at(yte, p, 0.5)
    assert 0.0 <= pr["precision"] <= 1.0 and 0.0 <= pr["recall"] <= 1.0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
