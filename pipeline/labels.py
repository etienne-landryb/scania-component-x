"""
labels.py -- Phase 1 label construction (the crux) for SCANIA Component X.

The training operational file has NO class column. Each readout's class is
derived from how long before the failure event it occurred, using train_tte.csv:

  delta = failure_time - time_step     (time-before-failure, for FAILED vehicles)

  delta  > 48        -> class 0   (far from failure)
  24 < delta <= 48   -> class 1
  12 < delta <= 24   -> class 2
   6 < delta <= 12   -> class 3
   0 <= delta <= 6   -> class 4   (imminent)

Censored vehicles (in_study_repair == 0) never failed -> every readout is class 0.

Boundary convention (STATED ASSUMPTION): upper-inclusive, as above. The window
edges (e.g. whether delta==48 is class 0 or 1) are not fully pinned down by the
challenge text; this convention is explicit here and easy to change in one place.

LEAKAGE GUARDS (why this file is separate and careful):
  * `length_of_study_time_step` and `in_study_repair` are LABEL INPUTS. They must
    NEVER become model features -- they encode the answer. FEATURE_EXCLUDE below
    names every column that must be kept out of the feature matrix.
  * `delta_to_failure` is kept for inspection only and is also excluded.
  * Feature construction (Phase 2) must use only readouts up to a given time_step;
    that guard lives there. Here we only assign each existing readout its class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ingest import VEHICLE_COL, TIME_COL

REPAIR_FLAG = "in_study_repair"
REPAIR_TIME = "length_of_study_time_step"
CLASS_COL = "class"
DELTA_COL = "delta_to_failure"

# Columns that must never enter the feature matrix (they leak the label).
FEATURE_EXCLUDE = frozenset({
    VEHICLE_COL, TIME_COL, CLASS_COL, DELTA_COL, REPAIR_FLAG, REPAIR_TIME,
})

# Window upper bounds -> class. Ordered from nearest failure to farthest.
_WINDOWS = [(6, 4), (12, 3), (24, 2), (48, 1)]   # delta <= bound -> class


def delta_to_class(delta) -> np.ndarray:
    """Vectorized time-before-failure -> class {0..4}. Assumes delta >= 0."""
    d = np.asarray(delta, dtype=float)
    cls = np.zeros(d.shape, dtype=np.int8)      # default 0 (delta > 48)
    for bound, klass in reversed(_WINDOWS):     # assign widest first, then narrow
        cls = np.where(d <= bound, klass, cls)
    return cls


def build_training_labels(op_df: pd.DataFrame, tte_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach a per-readout `class` column to training operational data.

    Returns (labelled_df, report). Post-failure readouts (delta < 0) for failed
    vehicles are dropped as a data-quality step and counted in the report.
    """
    need = [VEHICLE_COL, REPAIR_FLAG, REPAIR_TIME]
    missing = [c for c in need if c not in tte_df.columns]
    if missing:
        raise ValueError(f"tte file is missing required columns: {missing}")

    m = op_df.merge(tte_df[need], on=VEHICLE_COL, how="left")

    unmatched = int(m[REPAIR_FLAG].isna().sum())
    # Vehicles with no tte row: treat as censored (class 0) but report the count.
    m[REPAIR_FLAG] = m[REPAIR_FLAG].fillna(0)

    failed = (m[REPAIR_FLAG] == 1).to_numpy()
    delta = (m[REPAIR_TIME] - m[TIME_COL]).to_numpy()

    cls = np.zeros(len(m), dtype=np.int8)               # censored -> 0
    cls[failed] = delta_to_class(delta[failed])
    # Add both derived columns in one shot -> no frame fragmentation on the big file.
    derived = pd.DataFrame(
        {DELTA_COL: np.where(failed, delta, np.nan), CLASS_COL: cls},
        index=m.index,
    )
    m = pd.concat([m, derived], axis=1)

    # Drop readouts recorded AFTER the failure time (shouldn't exist; be safe).
    drop_mask = failed & (delta < 0)
    n_dropped = int(drop_mask.sum())
    m = m.loc[~drop_mask].reset_index(drop=True)

    dist = m[CLASS_COL].value_counts().sort_index()
    report = {
        "rows_in": int(len(op_df)),
        "rows_out": int(len(m)),
        "post_failure_dropped": n_dropped,
        "vehicles_unmatched_in_tte": unmatched,
        "class_counts": {int(k): int(v) for k, v in dist.items()},
        "class_pct": {int(k): round(100 * v / len(m), 2) for k, v in dist.items()},
    }
    return m, report


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns = everything except identifiers and label-linked columns."""
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def sanity_check(labelled: pd.DataFrame, report: dict) -> list[str]:
    """Cheap invariants that should hold if labelling is correct."""
    issues = []
    if labelled[CLASS_COL].min() < 0 or labelled[CLASS_COL].max() > 4:
        issues.append("class values outside {0..4}")
    # A failed vehicle's minimum delta should map to a high class near failure.
    failed = labelled[DELTA_COL].notna()
    if failed.any():
        neg = (labelled.loc[failed, DELTA_COL] < 0).sum()
        if neg:
            issues.append(f"{neg} negative deltas survived (should be dropped)")
    # No label-linked column should be considered a feature.
    leaked = FEATURE_EXCLUDE.intersection(feature_columns(labelled))
    if leaked:
        issues.append(f"label-linked columns leaked into features: {leaked}")
    return issues
