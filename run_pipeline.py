"""
run_pipeline.py -- end to end:
  load -> time split -> featurise -> train + calibrate every model
  -> evaluate (PR-AUC, calibration, money cost curve) -> explain -> demo.

Usage:
  python run_pipeline.py --synthetic            # offline smoke test (runs anywhere)
  python run_pipeline.py --data data/           # real IEEE-CIS: expects train_transaction.csv
                                                #   (and optionally train_identity.csv)

If scikit-learn is installed, the gradient-boosted model is the headline; otherwise
the pipeline falls back to the NumPy logistic model so it always runs.
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TARGET, AMOUNT_COL, decide, cost_summary
from src.data import (make_synthetic_sample, load_ieee_cis, time_based_split,
                      assert_no_leakage, fraud_rates)
from src.features import Featurizer
from src.model import build_models, PlattCalibrator, SKLEARN
from src.evaluate import (average_precision, precision_recall_at, pr_curve,
                          calibration_curve, search_cost_thresholds)
from src.explain import global_importance
from src.demo import score_transaction


def _amounts(df):
    return df[AMOUNT_COL].values if AMOUNT_COL in df.columns else np.ones(len(df))


def _plot_pr(y, s, base_rate, name, outdir, mode="real"):
    recall, precision = pr_curve(y, s)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=name)
    plt.axhline(base_rate, ls="--", color="grey", label=f"base rate {base_rate:.3f}")
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title(f"Precision-Recall (held-out test) [{mode} data]"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=120); plt.close()


def _plot_calibration(y, s, outdir, mode="real"):
    rows = calibration_curve(y, s)
    xs = [r["mean_score"] for r in rows]; ys = [r["observed_rate"] for r in rows]
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], ls="--", color="grey", label="perfect")
    plt.plot(xs, ys, marker="o", label="model")
    plt.xlabel("mean predicted score"); plt.ylabel("observed fraud rate")
    plt.title(f"Calibration (held-out test) [{mode} data]"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration.png"), dpi=120); plt.close()


def _plot_cost(surface, t_low, t_high, outdir, mode="real"):
    # surface already contains only review-capacity-eligible points, so the plotted
    # minimum lines up with the chosen threshold (no misleading lower-but-ineligible dip).
    by_high = {}
    for tl, th, tot in surface:
        by_high[th] = min(tot, by_high.get(th, float("inf")))
    xs = sorted(by_high)
    ys = [by_high[x] for x in xs]
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker=".")
    plt.axvline(t_high, ls="--", color="red", label=f"chosen block thr {t_high:.2f}")
    plt.xlabel("block threshold"); plt.ylabel("min total cost (capacity-eligible)")
    plt.title(f"Money cost vs. threshold (validation) [{mode} data]"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cost_curve.png"), dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use built-in synthetic sample")
    ap.add_argument("--data", default=None, help="dir containing train_transaction.csv")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    if args.data:
        tpath = os.path.join(args.data, "train_transaction.csv")
        ipath = os.path.join(args.data, "train_identity.csv")
        df = load_ieee_cis(tpath, ipath if os.path.exists(ipath) else None)
        print(f"[data] loaded IEEE-CIS: {df.shape[0]:,} rows x {df.shape[1]} cols")
    else:
        df = make_synthetic_sample()
        print("[data] SYNTHETIC sample (pipeline test only -- not for reporting)")

    splits = time_based_split(df)
    assert_no_leakage(splits)
    print("[data] split sizes:", {k: len(v) for k, v in splits.items()},
          "| fraud rates:", fraud_rates(splits))

    # 2. features -----------------------------------------------------------
    fz = Featurizer().fit(splits["train"])
    Xtr, ytr = fz.transform(splits["train"]), splits["train"][TARGET].values
    Xva, yva = fz.transform(splits["valid"]), splits["valid"][TARGET].values
    Xte, yte = fz.transform(splits["test"]),  splits["test"][TARGET].values
    feat_names = list(Xtr.columns)
    print(f"[features] {len(feat_names)} features")

    # 3. train + calibrate + evaluate every model --------------------------
    # NOTE: model SELECTION uses validation PR-AUC only. The test set is touched
    # exactly once, at the very end, to report the chosen model's honest number.
    results = {}
    for name, est in build_models().items():
        est.fit(Xtr.values, ytr)
        cal = PlattCalibrator().fit(est.predict_proba(Xva.values)[:, 1], yva)
        p_va = cal.predict(est.predict_proba(Xva.values)[:, 1])
        p_te = cal.predict(est.predict_proba(Xte.values)[:, 1])
        ap_va = average_precision(yva, p_va)
        ap_te = average_precision(yte, p_te)
        results[name] = {"est": est, "cal": cal, "p_va": p_va, "p_te": p_te,
                         "ap_va": ap_va, "ap_te": ap_te}
        print(f"[model] {name:<18} valid PR-AUC = {ap_va:.4f} | test PR-AUC = {ap_te:.4f}")

    base_rate = float(yte.mean())
    best_name = max(results, key=lambda k: results[k]["ap_va"])   # SELECT on validation
    best = results[best_name]
    print(f"[ref] base fraud rate (PR-AUC floor) = {base_rate:.4f}")
    print(f"[select] headline model = {best_name} "
          f"(valid PR-AUC {best['ap_va']:.4f} -> test PR-AUC {best['ap_te']:.4f})")

    # 4. money-optimal thresholds on VALIDATION, applied to TEST -----------
    va_amt, te_amt = _amounts(splits["valid"]), _amounts(splits["test"])
    bestthr, surface = search_cost_thresholds(yva, best["p_va"], va_amt)
    t_low, t_high = bestthr["t_low"], bestthr["t_high"]
    d_te = decide(best["p_te"], t_low, t_high)
    summ = cost_summary(yte, d_te, te_amt)
    allow_all = cost_summary(yte, np.array(["allow"] * len(yte)), te_amt)["cost_per_1k"]
    block_all = cost_summary(yte, np.array(["block"] * len(yte)), te_amt)["cost_per_1k"]
    pr = precision_recall_at(yte, best["p_te"], t_high)

    print("\n==== HELD-OUT TEST: money cost per 1,000 transactions ====")
    print(f"  allow everything : {allow_all:10.2f}")
    print(f"  block everything : {block_all:10.2f}")
    print(f"  OUR POLICY       : {summ['cost_per_1k']:10.2f}   (thresholds {t_low:.2f} / {t_high:.2f})")
    print(f"  vs allow-all     : {100*(1-summ['cost_per_1k']/max(allow_all,1e-9)):.1f}% less money lost")
    print(f"  action mix       : allow {summ['pct_allow']:.1%} | review {summ['pct_review']:.1%} | block {summ['pct_block']:.1%}")
    print(f"  block precision  : {pr['precision']:.3f} | block recall {pr['recall']:.3f}")

    # 5. plots --------------------------------------------------------------
    mode = "real" if args.data else "synthetic"
    _plot_pr(yte, best["p_te"], base_rate, best_name, args.outdir, mode)
    _plot_calibration(yte, best["p_te"], args.outdir, mode)
    _plot_cost(surface, t_low, t_high, args.outdir, mode)
    print(f"\n[plots] pr_curve.png, calibration.png, cost_curve.png -> {args.outdir}/")

    # 6. explanations -------------------------------------------------------
    try:
        gi = global_importance(best["est"], Xva, feat_names, y=yva)
        print("[explain] top global drivers:", ", ".join(f for f, _ in gi[:5]))
    except Exception as e:
        print("[explain] skipped:", e)

    # 7. demo, including a deliberately broken input -----------------------
    print("\n==== DEMO: live scoring ====")
    for i in range(min(3, len(splits["test"]))):
        tx = splits["test"].iloc[i].drop(labels=[TARGET]).to_dict()
        out = score_transaction(tx, best["est"], fz, best["cal"], t_low, t_high, feat_names)
        print(f"  tx#{i}: action={out['action']:<6} score={out['score']} | {', '.join(out['reasons'][:2])}")
    broken = score_transaction("not-a-transaction", best["est"], fz, best["cal"], t_low, t_high, feat_names)
    print(f"  broken input -> action={broken['action']} (graceful): {broken.get('reason','')}")

    # 8. persist summary ----------------------------------------------------
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({
            "mode": mode,
            "headline_model": best_name,
            "pr_auc_valid": best["ap_va"],
            "pr_auc_test": best["ap_te"],
            "base_rate": base_rate,
            "thresholds": {"t_low": t_low, "t_high": t_high},
            "cost_per_1k": {"policy": summ["cost_per_1k"], "allow_all": allow_all, "block_all": block_all},
            "sklearn_available": SKLEARN,
        }, f, indent=2)
    print("\n[done] wrote summary.json | mode =", mode, "| sklearn_available =", SKLEARN)


if __name__ == "__main__":
    main()
