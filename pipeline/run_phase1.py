"""
run_phase1.py -- run the ingestion + labelling pipeline on the real files.

Run this LOCALLY (not on Streamlit): it reads the ~1.2 GB training file.
Set DATA_DIR to the folder holding the SCANIA CSVs, then:

    python run_phase1.py

It builds per-readout training labels, prints a report and sanity checks, and
writes Parquet caches that Phase 2 (feature engineering) will read quickly.
"""

from __future__ import annotations

import os

from ingest import (read_operational, read_tte, read_provided_labels,
                    memory_report)
from labels import build_training_labels, sanity_check, feature_columns

# --- set this to your data folder -------------------------------------------
DATA_DIR = r"C:\Users\dell\Documents\00 TECH JOBS - GERMANY - 2026\02 VEHICLE FAILURE DETECTION\DATA\0 RAW DATA\2024-34-2\data"
CACHE_DIR = "cache"      # Parquet caches are written here
# ---------------------------------------------------------------------------

os.makedirs(CACHE_DIR, exist_ok=True)


def p(*parts):
    return os.path.join(*parts)


def main():
    print("Loading training operational readouts (~1.2 GB, memory-smart)...")
    train_op = read_operational(
        p(DATA_DIR, "train_operational_readouts.csv"),
        parquet_cache=p(CACHE_DIR, "train_operational.parquet"),
    )
    print("  ", memory_report(train_op))

    print("Loading train_tte.csv...")
    tte = read_tte(p(DATA_DIR, "train_tte.csv"))
    print("  ", memory_report(tte))

    print("Building per-readout training labels...")
    labelled, report = build_training_labels(train_op, tte)

    print("\n=== LABEL REPORT ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    issues = sanity_check(labelled, report)
    print("\nsanity issues:", issues or "none")
    print("feature columns:", len(feature_columns(labelled)))

    out = p(CACHE_DIR, "train_labelled.parquet")
    labelled.to_parquet(out, index=False)
    print(f"\nwrote {out}")

    # Validation/test labels are provided already -- just cache them typed.
    for split in ("validation", "test"):
        lbl_path = p(DATA_DIR, f"{split}_labels.csv")
        if os.path.exists(lbl_path):
            lbl = read_provided_labels(lbl_path)
            lbl.to_parquet(p(CACHE_DIR, f"{split}_labels.parquet"), index=False)
            counts = lbl.iloc[:, -1].value_counts().sort_index().to_dict()
            print(f"{split} labels: {len(lbl):,} rows, class counts {counts}")


if __name__ == "__main__":
    main()
    
    


