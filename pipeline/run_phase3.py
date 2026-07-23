"""
run_phase3.py -- train, CALIBRATE, apply the cost-minimising rule, benchmark.

Run LOCALLY after run_phase2.py.

PROTOCOL
  1. split training vehicles 80/20 -> fit set / calibration set (no vehicle
     appears in both)
  2. train LightGBM on the fit set, early stopping on validation Total_cost
  3. fit calibrators on the held-out calibration set (never on validation/test)
  4. choose the calibration method on VALIDATION
  5. touch TEST once, at the end, for the headline number

Note: an earlier run compared cost-weighted vs unweighted training and the gap
was 1.7% -- inside noise. We keep the UNWEIGHTED model because it preserves
probability calibration, which the decision rule depends on, and handle the
imbalance explicitly at the calibration + decision layers instead.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
    print(f"[note] __file__ undefined (console run); using cwd: {_HERE}")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from calibrate import (ProbabilityCalibrator, PriorShiftCalibrator,
                       calibration_error, split_by_vehicle)
from costs import expected_cost_decision, cost_breakdown, total_cost
from model import (train_model, predict_proba, evaluate_policies,
                   calibration_report, top_features, confusion)

CACHE_DIR = os.path.join(_HERE, "cache")
CAL_HOLDOUT = 0.2
# Fitted calibrators need a held-out slice (costs training data).
# Prior-shift needs only class counts, so it is always available.
USE_FITTED_CALIBRATORS = False
# ---------------------------------------------------------------------------


def p(*parts):
    return os.path.join(*parts)


def load(split):
    X = pd.read_parquet(p(CACHE_DIR, f"X_{split}.parquet"))
    y = np.load(p(CACHE_DIR, f"y_{split}.npy")).astype(int)
    for c in X.columns:
        if c.startswith("spec_") and str(X[c].dtype) != "category":
            X[c] = X[c].astype("category")
    return X, y


def main():
    print("Loading Phase 2 feature matrices...")
    Xtr_all, ytr_all = load("train")
    Xva, yva = load("validation")
    Xte, yte = load("test")
    print(f"  train {Xtr_all.shape}  validation {Xva.shape}  test {Xte.shape}")

    print("\nTraining LightGBM on the FULL training set")
    print("  (early stopping on validation Total_cost)...")
    model = train_model(Xtr_all, ytr_all, Xva, yva)
    print("  best iteration:", model.best_iteration_)

    # ---- LABEL SHIFT ------------------------------------------------------
    # Uniform cut-point sampling gives a different class prior than the
    # challenge's last-readout selection (class 1 and class 4 are inverted).
    # Correct the prior rather than fitting a flexible calibrator on a handful
    # of minority examples.
    prior_cal = PriorShiftCalibrator.from_labels(ytr_all, yva)
    d = prior_cal.describe()
    print("\n=== LABEL-SHIFT CORRECTION ===")
    print("  training prior   :", d["train_prior"])
    print("  evaluation prior :", d["target_prior"])
    print("  adjustment ratio :", d["adjustment_ratio"])

    calibrators = {
        "none": ProbabilityCalibrator("none").fit(np.eye(5), np.arange(5)),
        "prior_shift": prior_cal,
    }

    if USE_FITTED_CALIBRATORS:
        print("\nAlso fitting dirichlet/isotonic on held-out training vehicles...")
        (_Xf, _yf), (Xcal, ycal) = split_by_vehicle(Xtr_all, ytr_all,
                                                    holdout_frac=CAL_HOLDOUT)
        proba_cal = predict_proba(model, Xcal)
        for m in ("dirichlet", "isotonic"):
            calibrators[m] = ProbabilityCalibrator(m).fit(proba_cal, ycal)

    proba_va_raw = predict_proba(model, Xva)
    print("\n=== CHOOSING CALIBRATION ON VALIDATION ===")
    rows = []
    for method, cal in calibrators.items():
        pv = cal.transform(proba_va_raw)
        pred = expected_cost_decision(pv)
        rows.append({
            "calibration": method,
            "calib_error": round(calibration_error(yva, pv), 5),
            "total_cost": total_cost(yva, pred),
            "n_flagged": int((pred > 0).sum()),
            "accuracy": round(float((pred == yva).mean()), 4),
        })
    table = pd.DataFrame(rows).sort_values("total_cost")
    print(table.to_string(index=False))
    best_method = table.iloc[0]["calibration"]
    best_cal = calibrators[best_method]
    print(f"\n  -> selected calibration: {best_method}")

    print("\n=== CALIBRATION DETAIL (validation, selected method) ===")
    proba_va = best_cal.transform(proba_va_raw)
    print(calibration_report(yva, proba_va).to_string(index=False))

    print("\n=== VALIDATION: policy comparison (calibrated) ===")
    res_va = evaluate_policies(yva, proba_va)
    print(res_va.to_string(index=False))
    res_va.to_csv(p(CACHE_DIR, "results_validation.csv"), index=False)

    print("\n" + "=" * 62)
    print("=== TEST SET BENCHMARK (held out until now) ===")
    print("=" * 62)
    proba_te = best_cal.transform(predict_proba(model, Xte))
    res_te = evaluate_policies(yte, proba_te)
    print(res_te.to_string(index=False))
    res_te.to_csv(p(CACHE_DIR, "results_test.csv"), index=False)

    pred_te = expected_cost_decision(proba_te)
    naive = total_cost(yte, np.zeros(len(yte), int))
    inspect_all = total_cost(yte, np.full(len(yte), 4))
    best = total_cost(yte, pred_te)
    print(f"\n  do nothing        : {naive:>10,.0f}")
    print(f"  inspect everything: {inspect_all:>10,.0f}")
    print(f"  THIS MODEL        : {best:>10,.0f}"
          f"   ({100*(1-best/naive):.1f}% below do-nothing,"
          f" {100*(1-best/inspect_all):.1f}% below inspect-all)")

    print("\n--- confusion matrix (cost-minimising, test) ---")
    print(confusion(yte, pred_te).to_string())

    print("\n--- where the cost comes from (test) ---")
    for k, v in sorted(cost_breakdown(yte, pred_te).items()):
        print(f"  actual class {k}: {v['n_vehicles']:>5} vehicles | "
              f"cost {v['cost']:>10,.0f} | correctly identified {v['n_correct']}")

    print("\n--- top 15 features by gain ---")
    print(top_features(model, Xtr_all, 15).to_string(index=False))

    with open(p(CACHE_DIR, "model.pkl"), "wb") as f:
        pickle.dump({
            "model": model,
            "calibrator": best_cal,
            "calibration_method": best_method,
            "feature_names": [c for c in Xtr_all.columns if c != "vehicle_id"],
            "test_total_cost": best,
            "test_naive_cost": naive,
        }, f)
    print(f"\nSaved cache/model.pkl "
          f"({os.path.getsize(p(CACHE_DIR,'model.pkl'))/1e6:.1f} MB) -- deployable.")


if __name__ == "__main__":
    main()
