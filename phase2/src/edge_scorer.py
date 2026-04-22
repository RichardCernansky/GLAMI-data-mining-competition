"""
Pair-level LightGBM edge scorer.
Replaces raw cosine similarity as edge weight with a match probability
computed from text sim, image sim, price, department, color, and geo.
"""

import numpy as np
import joblib
from pathlib import Path


FEATURE_NAMES = ["text_sim", "img_sim", "price_sim", "dept_jaccard", "color_jaccard", "same_geo"]


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def compute_pair_features_batch(src, dst, text_sims, item_df, img_emb=None):
    """
    Compute 6 features for each (src, dst) pair.

    item_df must be row-aligned to embedding indices:
      item_df.iloc[i] == item at FAISS/graph node i

    Returns float32 array of shape (n_pairs, 6).
    """
    prices = item_df["price_eur"].fillna(0).values.astype(np.float64)
    geos   = item_df["geo"].values
    depts  = item_df["department_set"].values
    colors = item_df["color_set"].values

    p_s, p_d = prices[src], prices[dst]
    price_sim = np.where(
        (p_s > 0) & (p_d > 0),
        np.minimum(p_s, p_d) / np.maximum(p_s, p_d),
        0.0,
    ).astype(np.float32)

    same_geo   = (geos[src] == geos[dst]).astype(np.float32)
    dept_jacc  = np.array([_jaccard(depts[s],  depts[d])  for s, d in zip(src, dst)], dtype=np.float32)
    color_jacc = np.array([_jaccard(colors[s], colors[d]) for s, d in zip(src, dst)], dtype=np.float32)

    if img_emb is not None:
        img_sim = np.sum(img_emb[src] * img_emb[dst], axis=1).astype(np.float32)
    else:
        img_sim = np.zeros(len(src), dtype=np.float32)

    return np.column_stack([
        text_sims.astype(np.float32),
        img_sim,
        price_sim,
        dept_jacc,
        color_jacc,
        same_geo,
    ])


class EdgeScorer:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        print(f"  Edge scorer loaded from {model_path}")

    def score(self, features):
        """Returns match probability for each pair. Shape: (n_pairs,)"""
        return self.model.predict_proba(features)[:, 1].astype(np.float32)

    @staticmethod
    def exists(model_path):
        return bool(model_path) and Path(model_path).exists()
