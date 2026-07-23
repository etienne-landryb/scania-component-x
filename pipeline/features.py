"""
features.py -- Phase 2 feature engineering for SCANIA Component X.

WHY THIS FILE EXISTS
--------------------
Training data gives every readout of every vehicle. Validation/test give ONE
row per vehicle, at a randomly chosen "last readout", with the future removed.
Training on raw readouts would therefore train a different task than the one
being evaluated.

So we build training examples that mirror test conditions:

    pick a cut-point readout k for a vehicle
    -> summarize ONLY readouts 0..k into one feature vector
    -> label it with the class of readout k

LEAKAGE GUARD: every summary statistic is computed from the slice [0..k]. No
statistic may touch readout k+1 or later. `summarize_history` receives only
that slice, so leakage is prevented structurally rather than by discipline.

For validation/test the operational files are ALREADY truncated at the chosen
last readout, so the cut point is simply the final row present.

FEATURES BUILT (per example)
----------------------------
  last_<col>     current value of every operational column at the cut
  rate_<col>     (last - first) / elapsed time        -> long-run accumulation
  recent_<col>   (last - previous) / gap              -> recent rate, catches
                                                         acceleration near failure
  share_<col>    histogram bin / sum of its group     -> distribution SHAPE,
                                                         invariant to total usage
  meta           age at cut, number of readouts, history span
  spec_*         static vehicle specifications (categorical)

The `share_` block matters: counters are cumulative, so raw bin heights mostly
encode how much the truck has been used. Normalising each histogram to
proportions separates "how it was used" from "how much".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ingest import VEHICLE_COL, TIME_COL
from labels import CLASS_COL

# Histogram variable id -> number of bins (from the dataset paper).
HISTOGRAM_GROUPS: dict[str, int] = {
    "167": 10, "272": 10, "291": 11, "158": 10, "459": 20, "397": 36,
}
COUNTER_COLS = ["171_0", "666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]


def histogram_column_groups(columns) -> dict[str, list[str]]:
    """Map each histogram variable id -> its bin columns present in `columns`."""
    cols = set(map(str, columns))
    groups: dict[str, list[str]] = {}
    for vid, n_bins in HISTOGRAM_GROUPS.items():
        present = [f"{vid}_{b}" for b in range(n_bins) if f"{vid}_{b}" in cols]
        if present:
            groups[vid] = present
    return groups


def _safe_div(num, den):
    """Elementwise divide, returning 0 where the denominator is ~0."""
    den = np.where(np.abs(den) < 1e-9, np.nan, den)
    out = num / den
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_history(values: np.ndarray, times: np.ndarray,
                      group_idx: dict[str, np.ndarray]) -> np.ndarray:
    """Summarize ONE vehicle's history slice into a single feature vector.

    `values` is (n_readouts, n_cols) covering readouts 0..k inclusive, already
    sorted by time. `times` is the matching time_step array. Nothing outside
    this slice is visible here -- that is the leakage guard.
    """
    last = values[-1]
    first = values[0]
    span = float(times[-1] - times[0])

    rate = _safe_div(last - first, span if span > 0 else np.nan)

    if len(values) >= 2:
        gap = float(times[-1] - times[-2])
        recent = _safe_div(last - values[-2], gap if gap > 0 else np.nan)
    else:
        recent = np.zeros_like(last)

    shares = []
    for _vid, idx in group_idx.items():
        block = last[idx]
        total = np.nansum(block)
        shares.append(block / total if total > 0 else np.zeros_like(block))
    shares = np.concatenate(shares) if shares else np.array([], dtype=float)

    meta = np.array([times[-1], len(values), span], dtype=float)
    return np.concatenate([last, rate, recent, shares, meta])


def feature_names(op_cols: list[str], group_idx: dict[str, list[str]]) -> list[str]:
    names = [f"last_{c}" for c in op_cols]
    names += [f"rate_{c}" for c in op_cols]
    names += [f"recent_{c}" for c in op_cols]
    for _vid, cols in group_idx.items():
        names += [f"share_{c}" for c in cols]
    names += ["meta_age_at_cut", "meta_n_readouts", "meta_history_span"]
    return names


def _vehicle_slices(vehicle_ids: np.ndarray):
    """Yield (vehicle_id, start, stop) for a vehicle-sorted id array."""
    change = np.flatnonzero(np.diff(vehicle_ids)) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(vehicle_ids)]])
    for s, e in zip(starts, stops):
        yield vehicle_ids[s], int(s), int(e)


def build_examples(df: pd.DataFrame, op_cols: list[str],
                   n_cuts: int | None = 8, seed: int = 0,
                   has_labels: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build one feature row per (vehicle, cut point).

    n_cuts=None  -> use the final readout only (validation/test: already truncated)
    n_cuts=k     -> sample up to k cut points uniformly at random per vehicle,
                    which mirrors the test-time distribution (a random readout)

    Returns (features_df, labels, vehicle_ids). `labels` is all -1 when
    has_labels is False.
    """
    df = df.sort_values([VEHICLE_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    groups = histogram_column_groups(op_cols)
    col_pos = {c: i for i, c in enumerate(op_cols)}
    group_idx = {vid: np.array([col_pos[c] for c in cols]) for vid, cols in groups.items()}

    values_all = df[op_cols].to_numpy(dtype=np.float64, copy=False)
    times_all = df[TIME_COL].to_numpy(dtype=np.float64, copy=False)
    vids_all = df[VEHICLE_COL].to_numpy()
    classes_all = (df[CLASS_COL].to_numpy() if has_labels and CLASS_COL in df.columns
                   else np.full(len(df), -1))

    rng = np.random.default_rng(seed)
    rows, labels, out_vids = [], [], []

    for vid, s, e in _vehicle_slices(vids_all):
        n = e - s
        if n == 0:
            continue
        if n_cuts is None:
            cuts = [n - 1]
        else:
            k = min(n_cuts, n)
            cuts = sorted(rng.choice(n, size=k, replace=False).tolist())

        v_vals = values_all[s:e]
        v_times = times_all[s:e]
        for c in cuts:
            rows.append(summarize_history(v_vals[:c + 1], v_times[:c + 1], group_idx))
            labels.append(classes_all[s + c])
            out_vids.append(vid)

    X = pd.DataFrame(np.vstack(rows).astype(np.float32),
                     columns=feature_names(op_cols, groups))
    return X, np.asarray(labels), np.asarray(out_vids)


def attach_specifications(X: pd.DataFrame, vehicle_ids: np.ndarray,
                          spec_df: pd.DataFrame) -> pd.DataFrame:
    """Join static vehicle specifications (categorical) onto the feature frame."""
    spec_cols = [c for c in spec_df.columns if c != VEHICLE_COL]
    idx = pd.DataFrame({VEHICLE_COL: vehicle_ids})
    merged = idx.merge(spec_df, on=VEHICLE_COL, how="left")
    for c in spec_cols:
        X[f"spec_{c}"] = merged[c].astype("category").values
    return X
