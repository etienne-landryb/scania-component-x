"""
export_demo.py -- build the small bundle that gets deployed.

Run LOCALLY after run_phase3.py.

The training files are ~1.2 GB and Streamlit's free tier has ~1 GB of RAM, so
the app can never touch them. This script extracts everything the app needs
into a `demo/` folder of a few hundred kilobytes:

    demo/vehicles.parquet   sample vehicles: identifiers, calibrated class
                            probabilities, true class, and a few readable
                            summary fields for the detail view
    demo/meta.json          benchmark numbers, class priors, feature ranking

DESIGN NOTE: probabilities are precomputed here rather than running the model
inside the app. Two reasons. First, robustness -- shipping a pickled model ties
the app to exact library versions, and a portfolio demo that breaks on a
dependency bump is worse than no demo. Second, honesty about where the
interesting part is: the model produces probabilities, but the DECISION layer
is this project's contribution, and that layer stays fully live and interactive
in the app (cost assumptions, capacity, ranking all recompute in real time).
"""

from __future__ import annotations

import json
import os
import pickle
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from costs import COST_MATRIX, total_cost, expected_cost_decision
from model import predict_proba, top_features

CACHE_DIR = os.path.join(_HERE, "cache")
DEMO_DIR = os.path.join(_HERE, "demo")
N_DEMO_VEHICLES = 600      # keep the bundle small but the fleet view meaningful
SEED = 7
# ---------------------------------------------------------------------------


def p(*parts):
    return os.path.join(*parts)


def main():
    os.makedirs(DEMO_DIR, exist_ok=True)

    print("Loading trained model + calibrator...")
    with open(p(CACHE_DIR, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    calibrator = bundle["calibrator"]

    print("Loading test features...")
    X = pd.read_parquet(p(CACHE_DIR, "X_test.parquet"))
    for c in X.columns:
        if c.startswith("spec_") and str(X[c].dtype) != "category":
            X[c] = X[c].astype("category")
    y = np.load(p(CACHE_DIR, "y_test.npy")).astype(int)

    print("Scoring the full test fleet...")
    proba_all = calibrator.transform(predict_proba(model, X))
    pred_all = expected_cost_decision(proba_all)
    full_cost = total_cost(y, pred_all)
    naive_cost = total_cost(y, np.zeros(len(y), int))
    inspect_all_cost = total_cost(y, np.full(len(y), 4))

    # ---- sample a demo fleet -------------------------------------------------
    # Stratified so every risk class is represented; a purely random 600 would
    # contain almost no class-3 or class-4 trucks and the demo would look empty.
    rng = np.random.default_rng(SEED)
    idx = []
    for c in range(5):
        pool = np.flatnonzero(y == c)
        take = len(pool) if c > 0 else N_DEMO_VEHICLES - int((y > 0).sum())
        take = min(take, len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    idx = np.sort(np.concatenate(idx))
    print(f"  demo fleet: {len(idx)} vehicles, "
          f"class counts {pd.Series(y[idx]).value_counts().sort_index().to_dict()}")

    demo = pd.DataFrame({
        "vehicle_id": X["vehicle_id"].to_numpy()[idx],
        "true_class": y[idx],
    })
    for c in range(5):
        demo[f"p{c}"] = proba_all[idx, c]

    # A few human-readable operational fields for the detail view.
    readable = {
        "age_at_readout": "meta_age_at_cut",
        "readouts_recorded": "meta_n_readouts",
        "history_span": "meta_history_span",
    }
    for nice, col in readable.items():
        if col in X.columns:
            demo[nice] = X[col].to_numpy()[idx]
    for col in [c for c in X.columns if c.startswith("spec_")][:3]:
        demo[col.replace("spec_", "spec_")] = X[col].to_numpy()[idx].astype(str)

    demo.to_parquet(p(DEMO_DIR, "vehicles.parquet"), index=False)

    # ---- metadata ------------------------------------------------------------
    feats = top_features(model, X, 12)
    meta = {
        "benchmark": {
            "test_vehicles": int(len(y)),
            "model_total_cost": float(full_cost),
            "naive_total_cost": float(naive_cost),
            "inspect_all_total_cost": float(inspect_all_cost),
            "pct_below_naive": round(100 * (1 - full_cost / naive_cost), 1),
            "pct_below_inspect_all": round(100 * (1 - full_cost / inspect_all_cost), 1),
            "class4_recall": round(float(((pred_all == 4) & (y == 4)).sum() / max((y == 4).sum(), 1)), 3),
            "class4_missed_as_healthy": int(((pred_all == 0) & (y == 4)).sum()),
        },
        "cost_matrix": COST_MATRIX.tolist(),
        "top_features": feats.to_dict(orient="records"),
        "calibration_method": bundle.get("calibration_method", "unknown"),
    }
    with open(p(DEMO_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    v_kb = os.path.getsize(p(DEMO_DIR, "vehicles.parquet")) / 1024
    m_kb = os.path.getsize(p(DEMO_DIR, "meta.json")) / 1024
    print(f"\nWrote demo/vehicles.parquet ({v_kb:.0f} KB)")
    print(f"Wrote demo/meta.json ({m_kb:.0f} KB)")
    print(f"\nBenchmark carried into the app: {full_cost:,.0f} vs naive {naive_cost:,.0f} "
          f"({meta['benchmark']['pct_below_naive']}% lower)")
    print("Deploy the demo/ folder together with app.py -- nothing else is needed.")


if __name__ == "__main__":
    main()
