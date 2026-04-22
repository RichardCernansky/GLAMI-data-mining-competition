"""
Train a pair-level LightGBM edge scorer from labeled training data.

Uses FAISS to generate candidate pairs from the training set, labels them
using ground-truth product labels, computes 6 pairwise features, and trains
a binary LightGBM classifier. The trained model replaces raw cosine similarity
as the edge weight in the Phase 2 pipeline.

Usage:
    python phase2/scripts/train_edge_scorer.py

Output:
    {processed_data}/edge_scorer.joblib

Then set in configs/default.yaml:
    scorer:
      model_path: "../phase1/data/processed/edge_scorer.joblib"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.utils import load_config
from src.graph import build_faiss_index, query_neighbors
from src.edge_scorer import compute_pair_features_batch, FEATURE_NAMES


SEED = 42
SAMPLE_SIZE = 150_000  # items to sample — increase if you have time/RAM


def main():
    config    = load_config()
    emb_dir   = config["paths"]["embeddings"]
    proc_dir  = config["paths"]["processed_data"]
    faiss_cfg = config["faiss"]

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading training data...")
    train_df  = pd.read_pickle(os.path.join(proc_dir, "train_prepared.pkl"))
    train_ids = np.load(os.path.join(emb_dir, "train_ids.npy"))
    text_emb  = np.load(os.path.join(emb_dir, "train_embeddings.npy")).astype(np.float32)

    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(len(train_ids), size=min(SAMPLE_SIZE, len(train_ids)), replace=False))
    train_ids = train_ids[idx]
    text_emb  = text_emb[idx]

    # Align DataFrame rows to match embedding order (row i = item at node i)
    aligned_df = train_df.set_index("itemId").reindex(train_ids).reset_index()
    labels_arr = aligned_df["label"].values
    print(f"  {len(train_ids)} items, {len(set(labels_arr))} unique labels")

    img_emb = None
    if faiss_cfg.get("use_image", False):
        try:
            img_emb = np.load(os.path.join(emb_dir, "train_img_embeddings.npy"))[idx]
            print(f"  Image embeddings loaded: {img_emb.shape}")
        except FileNotFoundError:
            print("  No image embeddings found, img_sim will be 0")

    # ── Generate candidate pairs via FAISS ─────────────────────────────────────
    print("\nBuilding FAISS index and querying neighbors...")
    k     = faiss_cfg["k_neighbors"]
    index = build_faiss_index(text_emb, n_cells=faiss_cfg["n_cells"], n_probe=faiss_cfg["n_probe"])
    sims, indices = query_neighbors(index, text_emb, k)

    print("Assembling pairs...")
    src_list, dst_list, sim_list, pair_labels = [], [], [], []

    for i in range(len(train_ids)):
        for j_pos in range(1, k + 1):
            j   = int(indices[i, j_pos])
            sim = float(sims[i, j_pos])
            if j == -1 or i >= j or sim < 0.5:
                continue
            li, lj = labels_arr[i], labels_arr[j]
            is_match = int(li == lj and li is not None and not pd.isna(li) and li >= 0)
            src_list.append(i)
            dst_list.append(j)
            sim_list.append(sim)
            pair_labels.append(is_match)

    src      = np.array(src_list,   dtype=np.int32)
    dst      = np.array(dst_list,   dtype=np.int32)
    sims_arr = np.array(sim_list,   dtype=np.float32)
    y        = np.array(pair_labels, dtype=np.int32)
    print(f"  Pairs: {len(y):,}   positives: {y.sum():,} ({100*y.mean():.2f}%)")

    # ── Compute features ───────────────────────────────────────────────────────
    print("\nComputing pair features...")
    X = compute_pair_features_batch(src, dst, sims_arr, aligned_df, img_emb)
    print(f"  Feature matrix: {X.shape}  features: {FEATURE_NAMES}")

    # ── Train / validate ───────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw   = n_neg / n_pos
    print(f"\nTraining LightGBM  (pos={n_pos:,}  neg={n_neg:,}  scale_pos_weight={spw:.1f})...")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=spw,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        feature_name=FEATURE_NAMES,
    )

    proba = model.predict_proba(X_val)[:, 1]
    auc   = roc_auc_score(y_val, proba)
    f1    = f1_score(y_val, (proba >= 0.5).astype(int))
    print(f"  Val AUC={auc:.4f}  sanity F1={f1:.4f}")

    print("\nThreshold sweep on validation set:")
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.9, 0.99, 0.01):
        preds = (proba >= thr).astype(int)
        f1_t  = f1_score(y_val, preds)
        prec  = precision_score(y_val, preds, zero_division=0)
        rec   = recall_score(y_val, preds, zero_division=0)
        marker = " ←" if f1_t > best_f1 else ""
        print(f"  thr={thr:.2f}  F1={f1_t:.4f}  P={prec:.4f}  R={rec:.4f}{marker}")
        if f1_t > best_f1:
            best_f1, best_thr = f1_t, float(thr)
    print(f"\nBest threshold: {best_thr:.2f}  F1={best_f1:.4f}")
    print(f"Set in configs/default.yaml:  scorer.lgbm_threshold: {best_thr:.2f}")

    print("\nFeature importance:")
    max_imp = max(model.feature_importances_)
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 30 / max_imp)
        print(f"  {name:<20} {imp:5d}  {bar}")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = os.path.join(proc_dir, "edge_scorer.joblib")
    joblib.dump(model, out_path)
    print(f"\nSaved to {out_path}")
    print(f'\nNext: set in configs/default.yaml:')
    print(f'  scorer:')
    print(f'    model_path: "{out_path}"')


if __name__ == "__main__":
    main()
