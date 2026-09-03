"""
model.py -- an honest baseline plus a strong tabular model, with graceful fallback.

  * MajorityBaseline    : predicts the training fraud rate for everyone -- the PR-AUC
                          floor any real model must beat.
  * NumpyLogisticRegression : dependency-free logistic regression (median-imputes,
                          standardises, L2 + class-weighted gradient descent). Keeps
                          the pipeline runnable and testable even without scikit-learn.
  * scikit-learn (if installed): a LogisticRegression pipeline and a
                          HistGradientBoostingClassifier -- the headline model, strong
                          on tabular data and native to missing values.

PlattCalibrator turns raw scores into honest probabilities (fit on validation), so a
0.9 really means ~90% before we ever threshold on money.
"""

from __future__ import annotations
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except Exception:                              # pragma: no cover - depends on env
    SKLEARN = False


class MajorityBaseline:
    def __init__(self):
        self.rate_ = 0.0

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class NumpyLogisticRegression:
    """Transparent, dependency-free logistic regression (baseline + fallback)."""

    def __init__(self, l2=1.0, lr=0.5, n_iter=600, class_weight="balanced"):
        self.l2, self.lr, self.n_iter, self.class_weight = l2, lr, n_iter, class_weight

    def _prep(self, X, fit):
        X = np.asarray(X, float)
        if fit:
            med = np.nanmedian(X, axis=0)
            self.median_ = np.nan_to_num(med, nan=0.0)
        nan = np.isnan(X)
        if nan.any():
            X = X.copy()
            X[nan] = np.take(self.median_, np.where(nan)[1])
        if fit:
            self.mean_ = X.mean(0)
            std = X.std(0)
            std[std == 0] = 1.0
            self.std_ = std
        return (X - self.mean_) / self.std_

    def fit(self, X, y):
        y = np.asarray(y, float)
        Xs = self._prep(X, fit=True)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        if self.class_weight == "balanced":
            pos = max(y.mean(), 1e-6)
            w_pos, w_neg = 0.5 / pos, 0.5 / max(1 - pos, 1e-6)
        else:
            w_pos = w_neg = 1.0
        sw = np.where(y == 1, w_pos, w_neg)
        for _ in range(self.n_iter):
            z = np.clip(Xs @ self.w + self.b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            g = (p - y) * sw
            self.w -= self.lr * (Xs.T @ g / n + self.l2 * self.w / n)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        Xs = self._prep(X, fit=False)
        p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ self.w + self.b, -30, 30)))
        return np.column_stack([1 - p, p])


class PlattCalibrator:
    """1-D logistic calibration: raw score -> probability, fit on validation data."""

    def __init__(self, n_iter=1000, lr=0.5):
        self.n_iter, self.lr = n_iter, lr

    def fit(self, scores, y):
        s = np.asarray(scores, float)
        y = np.asarray(y, float)
        self.mu = s.mean()
        self.sd = s.std() or 1.0
        x = (s - self.mu) / self.sd
        self.a, self.b = 0.0, 0.0
        for _ in range(self.n_iter):
            z = np.clip(self.a * x + self.b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            self.a -= self.lr * ((p - y) * x).mean()
            self.b -= self.lr * (p - y).mean()
        return self

    def predict(self, scores):
        x = (np.asarray(scores, float) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-np.clip(self.a * x + self.b, -30, 30)))


def make_logistic():
    # keep_empty_features=True: don't drop all-NaN columns, so coefficients stay aligned
    # with the feature-name list (some IEEE-CIS columns are all-NaN within the train window).
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])


def make_gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )


def build_models():
    """Every model shares a .fit(X, y) / .predict_proba(X) interface.

    With scikit-learn present, the NumPy logistic model is redundant (and memory-heavy
    on large data), so we ship the majority baseline + sklearn logistic + gradient-boosted.
    Without scikit-learn, the NumPy logistic becomes the workhorse so the repo still runs.
    """
    models = {"baseline_majority": MajorityBaseline()}
    if SKLEARN:
        models["logistic_sklearn"] = make_logistic()
        models["gradient_boosted"] = make_gbm()
    else:
        models["logistic_numpy"] = NumpyLogisticRegression()
    return models
