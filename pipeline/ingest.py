"""
ingest.py -- Phase 1 ingestion for the SCANIA Component X pipeline.

The training operational file is ~1.19 GB. Two rules follow from that:

  1. Read it memory-smart. Every feature column is numeric; read them as
     float32 instead of the pandas default float64, which roughly halves RAM.
     A one-time Parquet cache makes every later read fast and correctly typed.

  2. This file is for OFFLINE work only. It must never be loaded by the
     deployed app (Streamlit's free tier has ~1 GB RAM). Train locally; deploy
     only the small trained model plus a few sample vehicles.

Nothing here constructs labels or features -- it only loads and joins raw
tables. Labels live in labels.py so the leakage-sensitive logic is isolated.
"""

from __future__ import annotations

import os
import pandas as pd

VEHICLE_COL = "vehicle_id"
TIME_COL = "time_step"

# Columns that identify a row but are NOT features.
ID_COLS = (VEHICLE_COL, TIME_COL)


def build_dtype_map(columns, id_int_cols=(VEHICLE_COL,)) -> dict:
    """float32 for every column except integer identifiers.

    Feature columns may contain the <1% missing values noted in the paper, so
    they must stay float (NaN-capable); only pure IDs are ints.
    """
    dtypes = {}
    for c in columns:
        dtypes[c] = "int32" if c in id_int_cols else "float32"
    return dtypes


def read_operational(path, parquet_cache: str | None = None,
                     use_cache_if_exists: bool = True) -> pd.DataFrame:
    """Load an operational_readouts CSV memory-smart, optionally caching Parquet.

    If `parquet_cache` is given and already exists, it's read directly (fast,
    correctly typed). Otherwise the CSV is read as float32 and the cache written.
    """
    if parquet_cache and use_cache_if_exists and os.path.exists(parquet_cache):
        return pd.read_parquet(parquet_cache)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = build_dtype_map(header)
    df = pd.read_csv(path, dtype=dtypes)

    if parquet_cache:
        df.to_parquet(parquet_cache, index=False)
    return df


def read_tte(path) -> pd.DataFrame:
    """Load the time-to-event (repair) table.

    Expected columns: vehicle_id, length_of_study_time_step, in_study_repair.
    (The paper describes the two content columns; the file also carries
    vehicle_id to join on.) We validate that vehicle_id is present, because
    without it the labels cannot be aligned safely.
    """
    df = pd.read_csv(path)
    if VEHICLE_COL not in df.columns:
        raise ValueError(
            f"'{VEHICLE_COL}' not found in tte file columns {list(df.columns)}. "
            "Labels can't be aligned without it -- check the file."
        )
    return df


def read_specifications(path) -> pd.DataFrame:
    """Load a specifications table. All non-id columns are categorical."""
    df = pd.read_csv(path, dtype={VEHICLE_COL: "int32"})
    return df


def read_provided_labels(path, label_col: str = "class_label") -> pd.DataFrame:
    """Load validation/test *_labels.csv (already contains the 0-4 class)."""
    df = pd.read_csv(path)
    if VEHICLE_COL not in df.columns:
        raise ValueError(f"'{VEHICLE_COL}' not found in labels file {list(df.columns)}.")
    return df


def memory_report(df: pd.DataFrame) -> str:
    mb = df.memory_usage(deep=True).sum() / 1e6
    return f"{df.shape[0]:,} rows x {df.shape[1]} cols, {mb:,.1f} MB in memory"
