"""
run_phase2.py -- build model-ready feature matrices from the Phase 1 cache.

Run LOCALLY, after run_phase1.py. Reads the Parquet caches, builds:

    train  : N_CUTS sampled cut points per vehicle (mirrors test conditions)
    val/test: one row per vehicle at its final available readout

and writes them to cache/ for Phase 3 (modelling).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from ingest import (read_operational, read_specifications, read_provided_labels,
                    VEHICLE_COL, TIME_COL, memory_report)
from labels import CLASS_COL, feature_columns
from features import build_examples, attach_specifications

# --- same DATA_DIR as run_phase1.py -----------------------------------------
DATA_DIR = r"C:\Users\dell\Documents\00 TECH JOBS - GERMANY - 2026\02 VEHICLE FAILURE DETECTION\DATA\0 RAW DATA\2024-34-2\data"
CACHE_DIR = os.path.join(_HERE, "cache")

# Cut points sampled per training vehicle. More = more data but slower.
N_CUTS = 8
SEED = 42
# ---------------------------------------------------------------------------


def p(*parts):
    return os.path.join(*parts)


def build_split(op_df, op_cols, spec_df, n_cuts, has_labels, seed=SEED):
    X, y, vids = build_examples(op_df, op_cols, n_cuts=n_cuts,
                                seed=seed, has_labels=has_labels)
    X = attach_specifications(X, vids, spec_df)
    X.insert(0, VEHICLE_COL, vids)
    return X, y


def main():
    print("Loading Phase 1 labelled training cache...")
    train = pd.read_parquet(p(CACHE_DIR, "train_labelled.parquet"))
    print("  ", memory_report(train))

    op_cols = [c for c in feature_columns(train)]
    print(f"  operational feature columns: {len(op_cols)}")

    train_spec = read_specifications(p(DATA_DIR, "train_specifications.csv"))

    print(f"\nBuilding TRAIN examples ({N_CUTS} cut points per vehicle)...")
    Xtr, ytr = build_split(train, op_cols, train_spec, n_cuts=N_CUTS, has_labels=True)
    print("  ", Xtr.shape, "class counts:",
          pd.Series(ytr).value_counts().sort_index().to_dict())
    Xtr.to_parquet(p(CACHE_DIR, "X_train.parquet"), index=False)
    np.save(p(CACHE_DIR, "y_train.npy"), ytr)
    del train

    for split in ("validation", "test"):
        print(f"\nBuilding {split.upper()} examples (final readout per vehicle)...")
        op = read_operational(
            p(DATA_DIR, f"{split}_operational_readouts.csv"),
            parquet_cache=p(CACHE_DIR, f"{split}_operational.parquet"),
        )
        spec = read_specifications(p(DATA_DIR, f"{split}_specifications.csv"))
        X, _ = build_split(op, op_cols, spec, n_cuts=None, has_labels=False)

        lbl = read_provided_labels(p(DATA_DIR, f"{split}_labels.csv"))
        label_col = [c for c in lbl.columns if c != VEHICLE_COL][0]
        y = X[[VEHICLE_COL]].merge(lbl, on=VEHICLE_COL, how="left")[label_col].to_numpy()

        print("  ", X.shape, "class counts:",
              pd.Series(y).value_counts().sort_index().to_dict())
        if np.isnan(y.astype(float)).any():
            print("  WARNING: some vehicles have no label after the join.")

        X.to_parquet(p(CACHE_DIR, f"X_{split}.parquet"), index=False)
        np.save(p(CACHE_DIR, f"y_{split}.npy"), y)
        del op

    print("\nDone. Wrote X_train / X_validation / X_test (+ y_*.npy) to cache/.")


if __name__ == "__main__":
    main()


