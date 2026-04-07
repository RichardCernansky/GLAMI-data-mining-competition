"""
Prepare data for the ADM 2026 competition.

Run from the project root:
    python scripts/prepare_data.py

What it does:
    1. Loads raw CSVs
    2. Parses departmentIds and colorTagIdsString into sets
    3. Normalizes prices to EUR
    4. Drops useless columns
    5. Builds validation groups from training data
    6. Saves everything to data/processed/
"""

import sys
import os
import time

# Add project root to path so we can import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import load_data, prepare_items, build_item_df
from src.evaluation.validation import build_validation_groups_fast


def main():
    DATA_DIR = "data/raw"
    OUT_DIR = "data/processed"
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. Load raw data ---
    print("Loading raw data...")
    t0 = time.time()
    train, phase1, task1 = load_data(DATA_DIR)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Train: {train.shape}, Phase1: {phase1.shape}, Task1: {task1.shape}")

    # --- 2. Prepare items (parse + normalize) ---
    print("\nPreparing train items...")
    t0 = time.time()
    train_prep = prepare_items(train)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Preparing phase1 items...")
    t0 = time.time()
    phase1_prep = prepare_items(phase1)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- 3. Verify currency normalization ---
    print("\nPrice EUR stats by geo (should all be roughly similar now):")
    print(train_prep.groupby("geo")["price_eur"].mean().sort_values(ascending=False).to_string())

    # --- 4. Save prepared items ---
    # We save as pickle to preserve the parsed sets (frozensets)
    print("\nSaving prepared data...")
    train_prep.to_pickle(os.path.join(OUT_DIR, "train_prepared.pkl"))
    phase1_prep.to_pickle(os.path.join(OUT_DIR, "phase1_prepared.pkl"))
    task1.to_csv(os.path.join(OUT_DIR, "task1.csv"), index=False)
    print(f"  Saved to {OUT_DIR}/")

    # --- 5. Build validation groups ---
    print("\nBuilding validation groups (3000 positive, 12000 negative)...")
    t0 = time.time()
    val_groups, val_labels = build_validation_groups_fast(
        train, n_positive=3000, n_negative=12000, seed=42
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Positive groups: {sum(val_labels)}, Negative groups: {len(val_labels) - sum(val_labels)}")

    val_groups["label"] = val_labels
    val_groups.to_csv(os.path.join(OUT_DIR, "validation_groups.csv"), index=False)
    print(f"  Saved validation_groups.csv")

    # --- 6. Summary ---
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles in {OUT_DIR}/:")
    for f in sorted(os.listdir(OUT_DIR)):
        size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024 / 1024
        print(f"  {f:40s} {size_mb:8.1f} MB")

    print("\nSample prepared train item:")
    row = train_prep.iloc[0]
    print(f"  itemId:         {row['itemId']}")
    print(f"  title:          {row['title']}")
    print(f"  geo:            {row['geo']}")
    print(f"  price (local):  {row['price']}")
    print(f"  price (EUR):    {row['price_eur']:.2f}")
    print(f"  department_set: {row['department_set']}")
    print(f"  color_set:      {row['color_set']}")


if __name__ == "__main__":
    main()