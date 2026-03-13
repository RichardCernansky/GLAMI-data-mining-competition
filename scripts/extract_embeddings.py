"""
Extract multilingual text embeddings for all items.

Run from project root:
    python scripts/extract_embeddings.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from src.utils import load_config


def build_item_text(row, max_desc_len):
    """Combine title + description into one string for embedding."""
    parts = [str(row["title"])]
    if pd.notna(row["description"]):
        parts.append(str(row["description"])[:max_desc_len])
    return ". ".join(parts)


def main():
    config = load_config()
    out_dir = config["paths"]["embeddings"]
    proc_dir = config["paths"]["processed_data"]
    emb_config = config["embedding"]
    os.makedirs(out_dir, exist_ok=True)

    # --- Load ---
    print("Loading prepared data...")
    train = pd.read_pickle(os.path.join(proc_dir, "train_prepared.pkl"))
    phase1 = pd.read_pickle(os.path.join(proc_dir, "phase1_prepared.pkl"))
    print(f"  Train: {len(train)}, Phase1: {len(phase1)}")

    # --- Build texts ---
    max_desc = emb_config["max_description_length"]
    print(f"\nBuilding texts (title + description, max_desc={max_desc})...")
    train_texts = train.apply(lambda r: build_item_text(r, max_desc), axis=1).tolist()
    phase1_texts = phase1.apply(lambda r: build_item_text(r, max_desc), axis=1).tolist()

    print(f"  Examples:")
    for i in range(3):
        print(f"    [{i}] {train_texts[i][:120]}...")

    train_ids = train["itemId"].values
    phase1_ids = phase1["itemId"].values

    # --- Load model ---
    model_name = emb_config["model_name"]
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    # --- Encode ---
    batch_size = emb_config["batch_size"]
    normalize = emb_config["normalize"]

    # run on GPU
    import torch
    # if torch.backends.mps.is_available():
    #     model = model.to(torch.device("mps"))
    #     print("Using M1 GPU (MPS)")

    print(f"\nEncoding {len(train_texts)} train texts...")
    t0 = time.time()
    train_emb = model.encode(
        train_texts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=normalize,
    )
    print(f"  Done in {time.time() - t0:.1f}s, shape: {train_emb.shape}")

    print(f"\nEncoding {len(phase1_texts)} phase1 texts...")
    t0 = time.time()
    phase1_emb = model.encode(
        phase1_texts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=normalize,
    )
    print(f"  Done in {time.time() - t0:.1f}s, shape: {phase1_emb.shape}")

    # --- Save ---
    np.save(os.path.join(out_dir, "train_embeddings.npy"), train_emb)
    np.save(os.path.join(out_dir, "train_ids.npy"), train_ids)
    np.save(os.path.join(out_dir, "phase1_embeddings.npy"), phase1_emb)
    np.save(os.path.join(out_dir, "phase1_ids.npy"), phase1_ids)

    print(f"\nSaved to {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        size_mb = os.path.getsize(os.path.join(out_dir, f)) / 1024 / 1024
        print(f"  {f:40s} {size_mb:8.1f} MB")

    # --- Sanity checks ---
    print("\n--- Sanity checks ---")

    # Same product
    label_counts = train["label"].value_counts()
    test_label = label_counts.index[0]
    idxs = np.where((train["label"] == test_label).values)[0]
    if len(idxs) >= 2:
        sim = np.dot(train_emb[idxs[0]], train_emb[idxs[1]])
        print(f"Same product (label {test_label}):")
        print(f"  1: {train_texts[idxs[0]][:100]}")
        print(f"  2: {train_texts[idxs[1]][:100]}")
        print(f"  Cosine sim: {sim:.4f}")

    # Random pair
    rng = np.random.RandomState(42)
    ri, rj = rng.randint(0, len(train_emb), size=2)
    print(f"\nRandom pair:")
    print(f"  1: {train_texts[ri][:100]}")
    print(f"  2: {train_texts[rj][:100]}")
    print(f"  Cosine sim: {np.dot(train_emb[ri], train_emb[rj]):.4f}")

    # Cross-lingual
    multi = train.groupby("label")["geo"].nunique()
    cross_lbl = multi[multi >= 3].index[0]
    cross = train[train["label"] == cross_lbl]
    geos = cross["geo"].unique()
    if len(geos) >= 2:
        a = np.where(train.index == cross[cross["geo"] == geos[0]].index[0])[0][0]
        b = np.where(train.index == cross[cross["geo"] == geos[1]].index[0])[0][0]
        print(f"\nCross-lingual (label {cross_lbl}):")
        print(f"  1 ({geos[0]}): {train_texts[a][:100]}")
        print(f"  2 ({geos[1]}): {train_texts[b][:100]}")
        print(f"  Cosine sim: {np.dot(train_emb[a], train_emb[b]):.4f}")


if __name__ == "__main__":
    main()