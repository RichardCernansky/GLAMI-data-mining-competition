"""
Main pipeline script for Phase 2 product grouping.

Usage:
    python scripts/run.py                        # run on phase2 data -> submission
    python scripts/run.py --validate             # run on train sample + score
    python scripts/run.py --validate --grid      # grid search over hyperparameters
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict

from src.utils import load_config
from src.graph import combine_embeddings, build_faiss_index, query_neighbors, build_edges, build_nx_graph
from src.clustering import run_louvain, compute_confidence, build_submission, print_cluster_stats


SAMPLE_SIZE = 200_000
SEED = 42


# ── Scoring ────────────────────────────────────────────────────────────────────

def bcubed_f1(true_labels, pred_labels):
    n = len(true_labels)
    precision_sum = recall_sum = 0.0
    for i in range(n):
        same_pred = (pred_labels == pred_labels[i])
        same_true = (true_labels == true_labels[i])
        correct = (same_pred & same_true).sum()
        precision_sum += correct / same_pred.sum()
        recall_sum    += correct / same_true.sum()
    p = precision_sum / n
    r = recall_sum / n
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"bcubed_f1": round(f1, 4), "bcubed_p": round(p, 4), "bcubed_r": round(r, 4)}


def confidence_calibration(clusters, G, true_labels):
    """Check purity of clusters per confidence bin."""
    rows = []
    for nodes in clusters.values():
        if len(nodes) < 2:
            continue
        conf = compute_confidence(nodes, G)
        is_pure = len(set(true_labels[nodes])) == 1
        rows.append({"confidence": conf, "is_pure": is_pure})
    df = pd.DataFrame(rows)
    df["conf_bin"] = pd.cut(df["confidence"], bins=[0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01])
    print("\n  Confidence calibration (purity per bin):")
    print(df.groupby("conf_bin", observed=True)["is_pure"].agg(["mean", "count"]).to_string())


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(embeddings, k, threshold, resolution, n_cells=400, n_probe=20):
    index = build_faiss_index(embeddings, n_cells=n_cells, n_probe=n_probe)
    sims, indices = query_neighbors(index, embeddings, k)
    src, dst, weights = build_edges(sims, indices, k, threshold)
    G = build_nx_graph(len(embeddings), src, dst, weights)
    _, clusters = run_louvain(G, resolution=resolution)
    pred_labels = np.zeros(len(embeddings), dtype=np.int32)
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            pred_labels[node] = cluster_id
    return pred_labels, clusters, G


def load_embeddings(emb_dir, faiss_cfg, prefix, idx=None):
    """Load text (+image) embeddings for a given prefix (train/phase1/phase2)."""
    text_emb = np.load(os.path.join(emb_dir, f"{prefix}_embeddings.npy"))
    ids      = np.load(os.path.join(emb_dir, f"{prefix}_ids.npy"))
    if idx is not None:
        text_emb = text_emb[idx]
        ids      = ids[idx]

    if faiss_cfg["use_image"]:
        img_emb = np.load(os.path.join(emb_dir, f"{prefix}_img_embeddings.npy"))
        has_img = np.load(os.path.join(emb_dir, f"{prefix}_img_has_image.npy"))
        if idx is not None:
            img_emb = img_emb[idx]
            has_img = has_img[idx]
        embeddings = combine_embeddings(
            text_emb, img_emb, has_img,
            faiss_cfg["text_weight"], faiss_cfg["image_weight"]
        )
    else:
        embeddings = text_emb.astype(np.float32)

    return embeddings, ids


# ── Modes ──────────────────────────────────────────────────────────────────────

def mode_submit(config):
    emb_dir   = config["paths"]["embeddings"]
    proc_dir  = config["paths"]["processed_data"]
    sub_dir   = config["paths"]["submissions"]
    faiss_cfg = config["faiss"]
    k          = faiss_cfg["k_neighbors"]
    threshold  = faiss_cfg["edge_threshold"]
    resolution = config["clustering"]["resolution"]
    n_cells    = faiss_cfg["n_cells"]
    n_probe    = faiss_cfg["n_probe"]
    os.makedirs(sub_dir, exist_ok=True)

    print("Loading phase2 embeddings...")
    embeddings, ids = load_embeddings(emb_dir, faiss_cfg, "phase1")  # swap to phase2 after April 10
    print(f"  {len(ids)} items, dim={embeddings.shape[1]}")

    print(f"\nRunning pipeline (k={k}, threshold={threshold}, resolution={resolution}, n_cells={n_cells}, n_probe={n_probe})...")
    t0 = time.time()
    _, clusters, G = run_pipeline(embeddings, k, threshold, resolution, n_cells, n_probe)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_cluster_stats(clusters)

    submission = build_submission(clusters, G, ids)
    out_path = os.path.join(sub_dir, "submission_phase2.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(submission)} rows)")


def mode_validate(config, grid=False):
    emb_dir   = config["paths"]["embeddings"]
    proc_dir  = config["paths"]["processed_data"]
    faiss_cfg = config["faiss"]

    print("Loading train embeddings + labels...")
    train_ids = np.load(os.path.join(emb_dir, "train_ids.npy"))
    train_df  = pd.read_pickle(os.path.join(proc_dir, "train_prepared.pkl"))
    id_to_label = dict(zip(train_df["itemId"].values, train_df["label"].values))
    true_labels_full = np.array([id_to_label.get(i, -1) for i in train_ids])

    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(len(train_ids), size=min(SAMPLE_SIZE, len(train_ids)), replace=False))
    true_labels = true_labels_full[idx]

    embeddings, _ = load_embeddings(emb_dir, faiss_cfg, "train", idx)
    print(f"  Sample: {len(idx)} items, {len(set(true_labels))} unique products, dim={embeddings.shape[1]}")

    n_cells = faiss_cfg["n_cells"]
    n_probe = faiss_cfg["n_probe"]

    if not grid:
        k          = faiss_cfg["k_neighbors"]
        threshold  = faiss_cfg["edge_threshold"]
        resolution = config["clustering"]["resolution"]
        configs = [{"k": k, "threshold": threshold, "resolution": resolution}]
    else:
        configs = [
            {"k": k, "threshold": thr, "resolution": res}
            for k   in [10, 20, 30]
            for thr in [0.80, 0.85, 0.90]
            for res in [0.8, 1.0, 1.2]
        ]

    results = []
    for cfg in configs:
        k, thr, res = cfg["k"], cfg["threshold"], cfg["resolution"]
        print(f"\nk={k} threshold={thr} resolution={res}")
        t0 = time.time()
        pred_labels, clusters, G = run_pipeline(embeddings, k, thr, res, n_cells, n_probe)
        print(f"  Done in {time.time()-t0:.1f}s")
        print_cluster_stats(clusters)

        scores = bcubed_f1(true_labels, pred_labels)
        print(f"  BCubed F1={scores['bcubed_f1']}  P={scores['bcubed_p']}  R={scores['bcubed_r']}")
        confidence_calibration(clusters, G, true_labels)
        results.append({**cfg, **scores})

    results_df = pd.DataFrame(results).sort_values("bcubed_f1", ascending=False)
    if grid:
        print("\nTop 5 configs:")
        print(results_df.head(5).to_string(index=False))
        out_path = os.path.join(proc_dir, "validation_results.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Validate on labeled train sample")
    parser.add_argument("--grid",     action="store_true", help="Grid search over hyperparameters (requires --validate)")
    args = parser.parse_args()

    config = load_config()

    if args.validate:
        mode_validate(config, grid=args.grid)
    else:
        mode_submit(config)
