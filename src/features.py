"""
features.py -- honest, minimal feature preparation.

Design choices worth defending out loud:
  * We DROP raw TransactionDT as a feature. It is just an increasing clock, so a
    model could exploit it to memorise the time-split boundary -- which would not
    generalise to tomorrow's traffic. Instead we derive an interpretable
    hour-of-day signal from it.
  * Object/string columns are ordinal-encoded using categories learned ONLY on the
    training split, so validation / test / live traffic cannot leak new information.
    Unseen categories at inference map to NaN, which the gradient-boosted model
    handles natively. (Honest caveat: ordinal codes are treated as numbers by the
    tree, so category *order* is arbitrary; native categorical splits are a noted
    future improvement -- see README.)
  * Everything is fit on train and merely applied elsewhere -- no peeking. `fit` also
    stores a per-feature "typical value" (`background_`), used by the explainer to
    compute per-transaction reasons.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import ID_COL, TIME_COL, AMOUNT_COL, TARGET


class Featurizer:
    def __init__(self):
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.categories_: dict[str, pd.Index] = {}
        self.feature_names_: list[str] = []
        self.background_: np.ndarray | None = None   # per-feature "typical" value (from train)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpretable features derived from raw columns."""
        out = pd.DataFrame(index=df.index)
        if TIME_COL in df.columns:
            out["hour_of_day"] = (pd.to_numeric(df[TIME_COL], errors="coerce") // 3600) % 24
        if AMOUNT_COL in df.columns:
            amt = pd.to_numeric(df[AMOUNT_COL], errors="coerce").clip(lower=0)
            out["log_amount"] = np.log1p(amt)
        return out

    def fit(self, df: pd.DataFrame) -> "Featurizer":
        drop = {ID_COL, TIME_COL, TARGET}
        base = [c for c in df.columns if c not in drop]
        self.numeric_cols = [c for c in base if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_cols = [c for c in base if c not in self.numeric_cols]
        self.categories_ = {
            c: pd.Index(pd.Series(df[c].astype("object")).dropna().unique())
            for c in self.categorical_cols
        }
        eng_cols = list(self._engineer(df.head(1)).columns)
        self.feature_names_ = eng_cols + self.numeric_cols + self.categorical_cols
        # "typical value" per feature, learned on train only, for the occlusion explainer
        Xtr = self.transform(df)
        bg = np.nanmedian(Xtr.values.astype(float), axis=0)
        self.background_ = np.nan_to_num(bg, nan=0.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        eng = self._engineer(df)

        if self.numeric_cols:
            num = df.reindex(columns=self.numeric_cols).apply(pd.to_numeric, errors="coerce")
        else:
            num = pd.DataFrame(index=df.index)

        cat = pd.DataFrame(index=df.index)
        for c in self.categorical_cols:
            col = df[c].astype("object") if c in df.columns else pd.Series(np.nan, index=df.index)
            codes = pd.Categorical(col, categories=self.categories_[c]).codes.astype("float")
            codes[codes < 0] = np.nan          # unknown / missing -> NaN (model handles it)
            cat[c] = codes

        X = pd.concat([eng, num, cat], axis=1).reindex(columns=self.feature_names_)
        return X
