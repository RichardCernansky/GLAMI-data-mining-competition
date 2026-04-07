"""
Compute pairwise features for items within groups.
"""

import numpy as np
import pandas as pd
from itertools import combinations


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets. Returns 0 if both empty."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def price_similarity(price_a, price_b):
    """Price similarity as min/max ratio. Identical prices = 1."""
    if price_a <= 0 or price_b <= 0:
        return 0.0
    return min(price_a, price_b) / max(price_a, price_b)


def compute_pairwise_features(group_item_ids, item_lookup, embedding_lookup,
                               img_embedding_lookup=None):
    """
    Compute pairwise features for all 10 pairs in a group of 5 items.

    Args:
        group_item_ids: list of 5 itemIds
        item_lookup: dict itemId -> row data
        embedding_lookup: dict itemId -> text embedding vector
        img_embedding_lookup: dict itemId -> image embedding vector (optional)

    Returns dict of aggregated group-level features.
    """
    items = []
    embeddings = []
    img_embeddings = []
    for item_id in group_item_ids:
        items.append(item_lookup.get(item_id))
        embeddings.append(embedding_lookup.get(item_id))
        if img_embedding_lookup is not None:
            img_embeddings.append(img_embedding_lookup.get(item_id))
        else:
            img_embeddings.append(None)

    emb_sims = []
    img_sims = []
    dept_sims = []
    color_sims = []
    price_sims = []
    same_geo = []

    for i, j in combinations(range(5), 2):
        if items[i] is None or items[j] is None:
            continue

        # Text cosine similarity (L2-normalized)
        if embeddings[i] is not None and embeddings[j] is not None:
            emb_sims.append(float(np.dot(embeddings[i], embeddings[j])))

        # Image cosine similarity (L2-normalized)
        if img_embeddings[i] is not None and img_embeddings[j] is not None:
            sim = float(np.dot(img_embeddings[i], img_embeddings[j]))
            img_sims.append(sim)

        # Department Jaccard
        dept_sims.append(jaccard_similarity(
            items[i]["department_set"], items[j]["department_set"]
        ))

        # Color Jaccard
        color_sims.append(jaccard_similarity(
            items[i]["color_set"], items[j]["color_set"]
        ))

        # Price similarity (EUR-normalized)
        price_sims.append(price_similarity(
            items[i]["price_eur"], items[j]["price_eur"]
        ))

        # Same geo flag
        same_geo.append(1.0 if items[i]["geo"] == items[j]["geo"] else 0.0)

    features = {}

    # Text embedding features
    if emb_sims:
        features["emb_max"] = max(emb_sims)
        features["emb_mean"] = np.mean(emb_sims)
        features["emb_min"] = min(emb_sims)
        features["emb_2nd"] = sorted(emb_sims, reverse=True)[1] if len(emb_sims) > 1 else emb_sims[0]
    else:
        features["emb_max"] = 0.0
        features["emb_mean"] = 0.0
        features["emb_min"] = 0.0
        features["emb_2nd"] = 0.0

    # Image embedding features
    if img_sims:
        features["img_max"] = max(img_sims)
        features["img_mean"] = np.mean(img_sims)
        features["img_2nd"] = sorted(img_sims, reverse=True)[1] if len(img_sims) > 1 else img_sims[0]
    else:
        features["img_max"] = 0.0
        features["img_mean"] = 0.0
        features["img_2nd"] = 0.0

    # Department features
    if dept_sims:
        features["dept_max"] = max(dept_sims)
        features["dept_mean"] = np.mean(dept_sims)
    else:
        features["dept_max"] = 0.0
        features["dept_mean"] = 0.0

    # Color features
    if color_sims:
        features["color_max"] = max(color_sims)
        features["color_mean"] = np.mean(color_sims)
    else:
        features["color_max"] = 0.0
        features["color_mean"] = 0.0

    # Price features
    if price_sims:
        features["price_max"] = max(price_sims)
        features["price_mean"] = np.mean(price_sims)
    else:
        features["price_max"] = 0.0
        features["price_mean"] = 0.0

    # Geo features
    if same_geo:
        features["same_geo_max"] = max(same_geo)
        features["same_geo_mean"] = np.mean(same_geo)
    else:
        features["same_geo_max"] = 0.0
        features["same_geo_mean"] = 0.0

    # Combined signals
    features["emb_x_dept_max"] = features["emb_max"] * features["dept_max"]
    features["img_x_dept_max"] = features["img_max"] * features["dept_max"]
    features["emb_x_img_max"] = features["emb_max"] * features["img_max"]

    return features


def compute_group_features_batch(groups_df, item_lookup, embedding_lookup,
                                  img_embedding_lookup=None):
    """
    Compute features for all groups in a DataFrame.

    Returns DataFrame with one row per group.
    """
    item_cols = [c for c in groups_df.columns if c.startswith("item")]
    all_features = []

    for idx in range(len(groups_df)):
        row = groups_df.iloc[idx]
        group_ids = [row[c] for c in item_cols]
        features = compute_pairwise_features(
            group_ids, item_lookup, embedding_lookup, img_embedding_lookup
        )
        all_features.append(features)

        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(groups_df)} groups")

    return pd.DataFrame(all_features)