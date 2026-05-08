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
from src.clustering import run_clustering, compute_confidence, build_submission, print_cluster_stats


SAMPLE_SIZE = 200_000
SEED = 42


# ── Scoring ────────────────────────────────────────────────────────────────────

def pairwise_f1(true_labels, pred_labels):
    """Pairwise F1 — matches the competition's evaluation metric exactly."""
    true_clusters = defaultdict(list)
    pred_clusters = defaultdict(list)
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        true_clusters[t].append(i)
        pred_clusters[p].append(i)

    tp = 0
    total_pred = 0
    total_true = 0

    for nodes in pred_clusters.values():
        n = len(nodes)
        total_pred += n * (n - 1) // 2
        sub = defaultdict(int)
        for node in nodes:
            sub[true_labels[node]] += 1
        for count in sub.values():
            tp += count * (count - 1) // 2

    for nodes in true_clusters.values():
        n = len(nodes)
        total_true += n * (n - 1) // 2

    fp = total_pred - tp
    fn = total_true - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"pairwise_f1": round(f1, 4), "pairwise_p": round(p, 4), "pairwise_r": round(r, 4)}


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


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_embeddings(emb_dir, faiss_cfg, prefix, idx=None, embed_tag=None):
    """Load combined (text+image) embeddings and raw image embeddings separately."""
    emb_name = f"{prefix}_{embed_tag}_embeddings.npy" if embed_tag else f"{prefix}_embeddings.npy"
    text_emb = np.load(os.path.join(emb_dir, emb_name))
    ids      = np.load(os.path.join(emb_dir, f"{prefix}_ids.npy"))
    if idx is not None:
        text_emb = text_emb[idx]
        ids      = ids[idx]

    img_emb_raw = None
    if faiss_cfg["use_image"]:
        img_emb = np.load(os.path.join(emb_dir, f"{prefix}_img_embeddings.npy"))
        has_img = np.load(os.path.join(emb_dir, f"{prefix}_img_has_image.npy"))
        if idx is not None:
            img_emb = img_emb[idx]
            has_img = has_img[idx]
        img_emb_raw = img_emb
        embeddings = combine_embeddings(
            text_emb, img_emb, has_img,
            faiss_cfg["text_weight"], faiss_cfg["image_weight"]
        )
    else:
        embeddings = text_emb.astype(np.float32)

    return embeddings, ids, img_emb_raw


def load_item_metadata(proc_dir, ids, prefix):
    """Load prepared item DataFrame aligned to the ids array (row i = node i)."""
    df = pd.read_pickle(os.path.join(proc_dir, f"{prefix}_prepared.pkl"))
    return df.set_index("itemId").reindex(ids).reset_index()


def load_scorer(config):
    """Load EdgeScorer if model_path is configured and the file exists."""
    scorer_cfg = config.get("scorer", {})
    model_path = scorer_cfg.get("model_path")
    if not model_path:
        return None
    from src.edge_scorer import EdgeScorer
    if not EdgeScorer.exists(model_path):
        print(f"  [INFO] No scorer found at {model_path}, running without scorer")
        return None
    return EdgeScorer(model_path)


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(embeddings, k, threshold, resolution, n_cells=400, n_probe=20,
                 algorithm="leiden", scorer=None, item_df=None,
                 img_emb_raw=None, scorer_cfg=None):

    index = build_faiss_index(embeddings, n_cells=n_cells, n_probe=n_probe)
    sims, indices = query_neighbors(index, embeddings, k)

    if scorer is not None and item_df is not None:
        from src.edge_scorer import compute_pair_features_batch
        initial_thr  = (scorer_cfg or {}).get("initial_threshold", 0.6)
        lgbm_thr     = (scorer_cfg or {}).get("lgbm_threshold", 0.5)

        src, dst, raw_sims = build_edges(sims, indices, k, initial_thr)
        print(f"  Scorer: {len(src):,} candidates (cosine ≥ {initial_thr}) → LightGBM scoring...")
        features   = compute_pair_features_batch(src, dst, raw_sims, item_df, img_emb_raw)
        new_weights = scorer.score(features)
        mask        = new_weights >= lgbm_thr
        src, dst, weights = src[mask], dst[mask], new_weights[mask]
        print(f"  Scorer: {mask.sum():,} edges kept (prob ≥ {lgbm_thr})")
    else:
        src, dst, weights = build_edges(sims, indices, k, threshold)

    G = build_nx_graph(len(embeddings), src, dst, weights)
    _, clusters = run_clustering(G, resolution=resolution, algorithm=algorithm)

    pred_labels = np.zeros(len(embeddings), dtype=np.int32)
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            pred_labels[node] = cluster_id
    return pred_labels, clusters, G


# ── Modes ──────────────────────────────────────────────────────────────────────

def mode_submit(config):
    emb_dir    = config["paths"]["embeddings"]
    proc_dir   = config["paths"]["processed_data"]
    sub_dir    = config["paths"]["submissions"]
    faiss_cfg  = config["faiss"]
    scorer_cfg = config.get("scorer", {})
    k          = faiss_cfg["k_neighbors"]
    threshold  = faiss_cfg["edge_threshold"]
    resolution = config["clustering"]["resolution"]
    algorithm  = config["clustering"].get("algorithm", "leiden")
    n_cells    = faiss_cfg["n_cells"]
    n_probe    = faiss_cfg["n_probe"]
    os.makedirs(sub_dir, exist_ok=True)

    embed_tag = config.get("embedding", {}).get("embed_tag")
    print(f"Loading phase1 embeddings (tag={embed_tag})...")
    embeddings, ids, img_emb_raw = load_embeddings(emb_dir, faiss_cfg, "phase1", embed_tag=embed_tag)
    print(f"  {len(ids)} items, dim={embeddings.shape[1]}")

    scorer  = load_scorer(config)
    item_df = load_item_metadata(proc_dir, ids, "phase1") if scorer else None

    print(f"\nRunning pipeline (k={k}, threshold={threshold}, resolution={resolution}, algorithm={algorithm})...")
    t0 = time.time()
    _, clusters, G = run_pipeline(
        embeddings, k, threshold, resolution, n_cells, n_probe, algorithm,
        scorer=scorer, item_df=item_df, img_emb_raw=img_emb_raw, scorer_cfg=scorer_cfg,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print_cluster_stats(clusters)

    lines = build_submission(clusters, G, ids, algorithm=algorithm)
    out_path = os.path.join(sub_dir, "submission_phase2.csv")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {out_path} ({len(lines)} groups)")


def mode_validate(config, grid=False):
    emb_dir    = config["paths"]["embeddings"]
    proc_dir   = config["paths"]["processed_data"]
    faiss_cfg  = config["faiss"]
    scorer_cfg = config.get("scorer", {})

    print("Loading train embeddings + labels...")
    train_ids = np.load(os.path.join(emb_dir, "train_ids.npy"))
    train_df  = pd.read_pickle(os.path.join(proc_dir, "train_prepared.pkl"))
    id_to_label      = dict(zip(train_df["itemId"].values, train_df["label"].values))
    true_labels_full = np.array([id_to_label.get(i, -1) for i in train_ids])

    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(len(train_ids), size=min(SAMPLE_SIZE, len(train_ids)), replace=False))
    true_labels = true_labels_full[idx]

    embed_tag = config.get("embedding", {}).get("embed_tag")
    embeddings, sampled_ids, img_emb_raw = load_embeddings(emb_dir, faiss_cfg, "train", idx, embed_tag=embed_tag)
    print(f"  Sample: {len(idx)} items, {len(set(true_labels))} unique products, dim={embeddings.shape[1]}")

    scorer  = load_scorer(config)
    item_df = load_item_metadata(proc_dir, sampled_ids, "train") if scorer else None

    n_cells   = faiss_cfg["n_cells"]
    n_probe   = faiss_cfg["n_probe"]
    algorithm = config["clustering"].get("algorithm", "leiden")

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
        print(f"\nk={k} threshold={thr} resolution={res} algorithm={algorithm}")
        t0 = time.time()
        pred_labels, clusters, G = run_pipeline(
            embeddings, k, thr, res, n_cells, n_probe, algorithm,
            scorer=scorer, item_df=item_df, img_emb_raw=img_emb_raw, scorer_cfg=scorer_cfg,
        )
        print(f"  Done in {time.time()-t0:.1f}s")
        print_cluster_stats(clusters)

        scores = pairwise_f1(true_labels, pred_labels)
        print(f"  Pairwise F1={scores['pairwise_f1']}  P={scores['pairwise_p']}  R={scores['pairwise_r']}")
        confidence_calibration(clusters, G, true_labels)
        results.append({**cfg, **scores})

    results_df = pd.DataFrame(results).sort_values("pairwise_f1", ascending=False)
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
