"""
data.py -- dataset loading, a synthetic stand-in, and a leakage-free time split.

Two data paths:
  1. REAL: IEEE-CIS Fraud Detection (Kaggle `ieee-fraud-detection`). We use ONLY the
     labelled training files (train_transaction.csv [+ train_identity.csv]); the
     competition test set has no public labels, so we carve our own time-ordered
     held-out test set from the labelled data (the honest choice).
  2. SYNTHETIC: make_synthetic_sample() builds a small IEEE-CIS-shaped frame with
     injected signal + noise, so the whole pipeline runs offline during development.

Splitting is TIME-BASED (sorted by TransactionDT), never random: fraud has strong
temporal structure, and a random split leaks future information into the past.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import ID_COL, TIME_COL, AMOUNT_COL, TARGET


# ---------------------------------------------------------------------------
# Synthetic sample (offline development / smoke tests)
# ---------------------------------------------------------------------------
def make_synthetic_sample(n_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """A small IEEE-CIS-shaped dataset with genuine-but-noisy fraud signal.

    Not for reporting metrics -- only to exercise the pipeline without the real
    files. The signal is deliberately noisy so the task is non-trivial.
    """
    rng = np.random.default_rng(seed)

    dt = np.sort(rng.integers(0, 60 * 24 * 3600, size=n_rows))          # ~60 days, sorted
    amt = np.round(np.exp(rng.normal(3.6, 1.0, n_rows)), 2)             # lognormal amounts
    product = rng.choice(["W", "C", "R", "H", "S"], n_rows, p=[.60, .15, .10, .10, .05])
    card4 = rng.choice(["visa", "mastercard", "amex", "discover"], n_rows, p=[.55, .35, .06, .04])
    card6 = rng.choice(["debit", "credit"], n_rows, p=[.60, .40])
    email = rng.choice(
        ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "protonmail.com", "anonymous.com", "other"],
        n_rows, p=[.40, .15, .10, .10, .05, .05, .15],
    )
    device = rng.choice(["desktop", "mobile"], n_rows, p=[.55, .45])
    C1 = rng.poisson(2.0, n_rows)                                       # velocity-style counts
    C2 = rng.poisson(1.5, n_rows)
    D1 = rng.integers(0, 400, n_rows)                                   # days since prior event
    V1 = rng.normal(0, 1, n_rows)
    V2 = rng.normal(0, 1, n_rows)
    hour = (dt // 3600) % 24

    # Latent fraud log-odds: a few real drivers + heavy noise -> non-separable.
    logit = (
        -3.2
        + 0.9 * (email == "anonymous.com")
        + 0.5 * (email == "protonmail.com")
        + 0.6 * (product == "C")
        + 0.4 * (card6 == "credit")
        + 0.03 * C1 + 0.05 * C2
        + 0.5 * (hour < 6)                 # night-time risk
        + 0.0007 * amt                     # larger amounts slightly riskier
        + 0.30 * V1                         # latent signal
        + rng.normal(0, 1.0, n_rows)        # noise
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = (rng.random(n_rows) < prob).astype(int)

    return pd.DataFrame({
        ID_COL: np.arange(1, n_rows + 1),
        TIME_COL: dt,
        AMOUNT_COL: amt,
        "ProductCD": product,
        "card4": card4,
        "card6": card6,
        "P_emaildomain": email,
        "DeviceType": device,
        "C1": C1, "C2": C2, "D1": D1, "V1": V1, "V2": V2,
        TARGET: is_fraud,
    })


# ---------------------------------------------------------------------------
# Real IEEE-CIS loader
# ---------------------------------------------------------------------------
def load_ieee_cis(transaction_path: str, identity_path: str | None = None) -> pd.DataFrame:
    """Load labelled IEEE-CIS training data, optionally merged with identity."""
    df = pd.read_csv(transaction_path)
    if identity_path:
        idf = pd.read_csv(identity_path)
        df = df.merge(idf, on=ID_COL, how="left")
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' not found -- load the *train* files (they carry labels).")
    return df


# ---------------------------------------------------------------------------
# Leakage-free, time-based split
# ---------------------------------------------------------------------------
def time_based_split(df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15) -> dict:
    """Sort by time and cut into train / valid / test by position.

    Earliest rows train, next validate, most-recent test -- so evaluation always
    faces 'future' transactions, exactly like production. Split boundaries are snapped
    forward to a change in timestamp, so rows sharing the same TransactionDT (common in
    IEEE-CIS, which is second-resolution) never straddle two splits -- otherwise the
    same moment would leak across the train/test line.
    """
    if not 0 < train_frac < 1 or not 0 < valid_frac < 1 or train_frac + valid_frac >= 1:
        raise ValueError("Need train_frac + valid_frac < 1, each in (0, 1).")

    d = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(d)
    t = d[TIME_COL].values
    i_train = int(n * train_frac)
    i_valid = int(n * (train_frac + valid_frac))

    def _snap(i):                               # advance to the next timestamp boundary
        while 0 < i < n and t[i] == t[i - 1]:
            i += 1
        return i

    i_train, i_valid = _snap(i_train), _snap(i_valid)
    if not (0 < i_train < i_valid < n):
        raise ValueError("timestamp ties collapsed a split to empty; adjust fractions or data.")
    return {
        "train": d.iloc[:i_train].copy(),
        "valid": d.iloc[i_train:i_valid].copy(),
        "test": d.iloc[i_valid:].copy(),
    }


def assert_no_leakage(splits: dict) -> None:
    """Guard rail: splits must be strictly time-ordered (no shared boundary timestamp)
    and all carry the target. Uses explicit raises (survives `python -O`, unlike assert)."""
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    if not (len(tr) and len(va) and len(te)):
        raise ValueError("a split is empty -- check fractions / timestamp ties")
    if tr[TIME_COL].max() >= va[TIME_COL].min():
        raise ValueError("train/valid overlap or share a boundary timestamp!")
    if va[TIME_COL].max() >= te[TIME_COL].min():
        raise ValueError("valid/test overlap or share a boundary timestamp!")
    for name, part in splits.items():
        if TARGET not in part.columns:
            raise ValueError(f"{name} is missing the target column")


def fraud_rates(splits: dict) -> dict:
    """Fraud base rate per split -- sanity check that all three see some fraud."""
    return {name: round(float(part[TARGET].mean()), 5) for name, part in splits.items()}
