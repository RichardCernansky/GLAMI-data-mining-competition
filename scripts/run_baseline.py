"""
Run baseline models for Phase 1: duplicate detection.

Run from project root:
    python scripts/run_baseline.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src.features.pairwise import compute_group_features_batch
from src.models.classifier import (
    train_lightgbm,
    find_best_threshold,
    cross_validate,
)
from src.evaluation.metrics import print_evaluation


def build_lookups(train, phase1, train_emb, train_ids, phase1_emb, phase1_ids):
    """Build item lookup and embedding lookup dicts."""
    print("Building lookups...")
    item_lookup = {}
    for _, row in train.iterrows():
        item_lookup[row["itemId"]] = row
    for _, row in phase1.iterrows():
        item_lookup[row["itemId"]] = row

    embedding_lookup = {}
    for i, item_id in enumerate(train_ids):
        embedding_lookup[item_id] = train_emb[i]
    for i, item_id in enumerate(phase1_ids):
        embedding_lookup[item_id] = phase1_emb[i]

    print(f"  Items: {len(item_lookup)}, Embeddings: {len(embedding_lookup)}")
    return item_lookup, embedding_lookup


def main():
    config = load_config()
    proc_dir = config["paths"]["processed_data"]
    emb_dir = config["paths"]["embeddings"]
    sub_dir = config["paths"]["submissions"]
    os.makedirs(sub_dir, exist_ok=True)

    # =============================================
    # 1. LOAD
    # =============================================
    print("Loading data...")
    train = pd.read_pickle(os.path.join(proc_dir, "train_prepared.pkl"))
    phase1 = pd.read_pickle(os.path.join(proc_dir, "phase1_prepared.pkl"))
    task1 = pd.read_csv(os.path.join(proc_dir, "task1.csv"))
    val_groups = pd.read_csv(os.path.join(proc_dir, "validation_groups.csv"))

    val_labels = val_groups["label"].values
    val_groups = val_groups.drop(columns=["label"])

    print("Loading embeddings...")
    train_emb = np.load(os.path.join(emb_dir, "train_embeddings.npy"))
    train_ids = np.load(os.path.join(emb_dir, "train_ids.npy"))
    phase1_emb = np.load(os.path.join(emb_dir, "phase1_embeddings.npy"))
    phase1_ids = np.load(os.path.join(emb_dir, "phase1_ids.npy"))

    # =============================================
    # 2. LOOKUPS
    # =============================================
    item_lookup, embedding_lookup = build_lookups(
        train, phase1, train_emb, train_ids, phase1_emb, phase1_ids
    )

    # =============================================
    # 3. VALIDATION FEATURES
    # =============================================
    print(f"\nComputing features for {len(val_groups)} validation groups...")
    t0 = time.time()
    val_features = compute_group_features_batch(val_groups, item_lookup, embedding_lookup)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\nFeatures: {val_features.columns.tolist()}")
    print(val_features.describe().round(4).to_string())

    # =============================================
    # 4. BASELINE 1: Threshold on emb_max
    # =============================================
    print("\n" + "=" * 60)
    print("BASELINE 1: Threshold on emb_max")
    print("=" * 60)

    from sklearn.metrics import f1_score, precision_score, recall_score

    print(f"\n{'Threshold':>10}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}")
    for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        preds = (val_features["emb_max"] >= t).astype(int).values
        f1 = f1_score(val_labels, preds)
        prec = precision_score(val_labels, preds, zero_division=0)
        rec = recall_score(val_labels, preds)
        print(f"  {t:>10.2f}  {f1:>8.4f}  {prec:>10.4f}  {rec:>8.4f}")

    # =============================================
    # 5. BASELINE 2: LightGBM
    # =============================================
    print("\n" + "=" * 60)
    print(f"BASELINE 2: {config['classifier']['model_type'].upper()}")
    print("=" * 60)

    print(f"\n{config['classifier']['cv_folds']}-fold cross-validation:")
    fold_metrics, best_threshold, overall_metrics = cross_validate(
        val_features, val_labels, config
    )

    print("\nTraining final model on all validation data...")
    final_model = train_lightgbm(val_features, val_labels, config=config)

    print("\nFeature importance:")
    importances = final_model.feature_importances_
    for name, imp in sorted(zip(val_features.columns, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp / max(importances) * 30)
        print(f"  {name:20s} {imp:5d}  {bar}")

    # =============================================
    # 6. TASK1 FEATURES
    # =============================================
    print(f"\nComputing features for {len(task1)} task1 groups...")
    t0 = time.time()
    task1_features = compute_group_features_batch(task1, item_lookup, embedding_lookup)
    print(f"  Done in {time.time() - t0:.1f}s")

    # =============================================
    # 7. SUBMISSIONS
    # =============================================
    # A: threshold
    thresh_val, thresh_metrics = find_best_threshold(
        val_labels, val_features["emb_max"].values
    )
    task1_preds_thresh = (task1_features["emb_max"] >= thresh_val).astype(int).values
    sub_thresh = pd.DataFrame({"prediction": task1_preds_thresh})
    sub_thresh_path = os.path.join(sub_dir, "submission_threshold.csv")
    sub_thresh.to_csv(sub_thresh_path, index=False)

    # B: LightGBM
    task1_proba = final_model.predict_proba(task1_features)[:, 1]
    task1_preds_lgb = (task1_proba >= best_threshold).astype(int)
    sub_lgb = pd.DataFrame({"prediction": task1_preds_lgb})
    sub_lgb_path = os.path.join(sub_dir, "submission_lightgbm.csv")
    sub_lgb.to_csv(sub_lgb_path, index=False)

    print(f"\n{'='*60}")
    print("SUBMISSIONS")
    print(f"{'='*60}")
    print(f"\nThreshold (emb_max >= {thresh_val:.2f}):")
    print(f"  Positive: {task1_preds_thresh.sum()} ({task1_preds_thresh.mean()*100:.1f}%)")
    print(f"  Saved: {sub_thresh_path}")

    print(f"\nLightGBM (thresh={best_threshold:.2f}):")
    print(f"  Positive: {task1_preds_lgb.sum()} ({task1_preds_lgb.mean()*100:.1f}%)")
    print(f"  Saved: {sub_lgb_path}")

    # Save features
    val_features["label"] = val_labels
    val_features.to_csv(os.path.join(proc_dir, "val_features.csv"), index=False)
    task1_features.to_csv(os.path.join(proc_dir, "task1_features.csv"), index=False)
    print(f"\nFeatures saved to {proc_dir}/")


if __name__ == "__main__":
    main()