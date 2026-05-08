"""
Build positive training pairs for contrastive fine-tuning.

Runs locally (CPU). Reads train_prepared.pkl, samples positive pairs
(two items with the same product label), builds their text, and saves
to a parquet file you upload to Google Colab.

Usage:
    python phase2/scripts/build_pairs.py

Output:
    {processed_data}/contrastive_pairs.parquet
    Columns: text_a, text_b  (one positive pair per row)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.utils import load_config

SEED        = 42
MAX_PAIRS   = 500_000   # cap to keep file size manageable
MAX_DESC    = 500       # same as embedding config


def build_text(row, max_desc=MAX_DESC):
    parts = [str(row["title"])]
    if pd.notna(row.get("description")):
        parts.append(str(row["description"])[:max_desc])
    return ". ".join(parts)


def main():
    config   = load_config()
    proc_dir = config["paths"]["processed_data"]

    print("Loading train_prepared.pkl...")
    df = pd.read_pickle(os.path.join(proc_dir, "train_prepared.pkl"))
    df = df[df["label"].notna() & (df["label"] >= 0)].copy()
    print(f"  {len(df)} items, {df['label'].nunique()} unique products")

    print("Building texts...")
    df["text"] = df.apply(build_text, axis=1)

    print("Sampling positive pairs...")
    rng    = np.random.RandomState(SEED)
    groups = df.groupby("label")["text"].apply(list)
    # only groups with at least 2 items can form a positive pair
    groups = groups[groups.apply(len) >= 2]

    pairs = []
    for texts in groups:
        texts = list(texts)
        # sample up to 3 pairs per group to avoid large groups dominating
        n_pairs = min(3, len(texts) // 2)
        idxs = rng.permutation(len(texts))
        for i in range(n_pairs):
            a, b = idxs[2 * i], idxs[2 * i + 1]
            pairs.append((texts[a], texts[b]))

    rng.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    print(f"  {len(pairs):,} pairs from {len(groups):,} product groups")

    out = pd.DataFrame(pairs, columns=["text_a", "text_b"])
    out_path = os.path.join(proc_dir, "contrastive_pairs.parquet")
    out.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print("Upload this file to Google Colab, then run finetune_embeddings.py")


if __name__ == "__main__":
    main()
