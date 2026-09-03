"""
explain.py -- why did the model make THIS decision?

Two questions, two honest methods, no heavyweight dependencies:

  * GLOBAL ("what drives risk overall"):
      - transparent logistic  -> |weight|                         (exact)
      - linear pipeline tail   -> |coef|                          (exact)
      - any sklearn model+labels -> permutation importance         (model-agnostic; this
                                    is what covers HistGradientBoosting, which does NOT
                                    expose feature_importances_)

  * LOCAL ("why this transaction"):
      - transparent logistic  -> weight x standardised value      (exact contribution)
      - any other model        -> OCCLUSION: replace each feature with its "typical"
                                  (training-median) value and measure how the fraud
                                  probability moves. Faithful per-instance, works for the
                                  gradient-boosted model, and needs nothing but the model.

We deliberately avoid SHAP: its TreeExplainer does not support HistGradientBoostingClassifier,
and a silent fallback there would produce identical "reasons" for every transaction. The
occlusion method is simple, correct, and testable.
"""

from __future__ import annotations
import numpy as np


def global_importance(model, X, feature_names, y=None, sample=2000):
    """Return [(feature, importance)] sorted high->low. Never raises."""
    if hasattr(model, "w"):                        # NumpyLogisticRegression
        imp = np.abs(model.w)
    elif hasattr(model, "feature_importances_"):   # tree models that expose it
        imp = np.asarray(model.feature_importances_, float)
    elif hasattr(model, "named_steps"):            # sklearn pipeline w/ linear tail
        tail = list(model.named_steps.values())[-1]
        coef = getattr(tail, "coef_", None)
        imp = (np.abs(np.ravel(coef)) if coef is not None
               else _permutation_importance(model, X, feature_names, y, sample))
    else:                                          # opaque model (e.g. HistGBM)
        imp = _permutation_importance(model, X, feature_names, y, sample)
    imp = np.ravel(np.asarray(imp, float))[: len(feature_names)]
    if imp.shape[0] < len(feature_names):          # pad defensively
        imp = np.concatenate([imp, np.zeros(len(feature_names) - imp.shape[0])])
    order = np.argsort(-imp)
    return [(feature_names[i], float(imp[i])) for i in order]


def _permutation_importance(model, X, feature_names, y, sample):
    """sklearn permutation importance if we have labels; else zeros. Never raises."""
    if y is None:
        return np.zeros(len(feature_names))
    try:
        from sklearn.inspection import permutation_importance
        Xa = np.asarray(X)[:sample]
        ya = np.asarray(y)[:sample]
        r = permutation_importance(model, Xa, ya, n_repeats=5, random_state=42,
                                   scoring="average_precision")
        return np.asarray(r.importances_mean, float)
    except Exception:
        return np.zeros(len(feature_names))


def local_reasons(model, featurizer, X_row_df, feature_names, top_k=4):
    """Human-readable top drivers for a single (already-transformed) transaction."""
    # Exact contributions for the transparent logistic model.
    if hasattr(model, "w") and hasattr(model, "mean_"):
        x = np.asarray(X_row_df.values, float)[0]
        x = np.where(np.isnan(x), model.median_, x)
        xs = (x - model.mean_) / model.std_
        contrib = model.w * xs
        idx = np.argsort(-np.abs(contrib))[:top_k]
        return [f"{feature_names[i]} ({'raises' if contrib[i] > 0 else 'lowers'} risk)"
                for i in idx]

    # Model-agnostic OCCLUSION for everything else (incl. the gradient-boosted model):
    # swap each feature to its typical value; the bigger the drop in fraud probability,
    # the more that feature was raising this transaction's risk.
    try:
        bg = getattr(featurizer, "background_", None)
        if bg is None:
            raise ValueError("no background reference")
        x0 = np.asarray(X_row_df.values, float)[0]
        d = len(x0)
        base = float(model.predict_proba(x0.reshape(1, -1))[:, 1][0])
        variants = np.repeat(x0.reshape(1, -1), d, axis=0)
        for j in range(d):
            variants[j, j] = bg[j]                 # feature j -> typical value
        probs = np.asarray(model.predict_proba(variants))[:, 1]
        delta = base - probs                       # drop in risk when j becomes typical
        idx = np.argsort(-np.abs(delta))[:top_k]
        return [f"{feature_names[i]} ({'raises' if delta[i] > 0 else 'lowers'} risk)"
                for i in idx]
    except Exception:
        gi = global_importance(model, X_row_df, feature_names)
        return [f"{f} (top overall driver)" for f, _ in gi[:top_k]]
