"""
Louvain clustering and confidence scoring.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import community as community_louvain


def run_louvain(G, resolution=1.0):
    """
    Run Louvain community detection.
    Returns dict: node -> cluster_id, and dict: cluster_id -> [nodes]
    """
    partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)

    clusters = defaultdict(list)
    for node, cluster_id in partition.items():
        clusters[cluster_id].append(node)

    return partition, dict(clusters)


def compute_confidence(nodes, G):
    """Mean weight of intra-cluster edges. Singleton = 0.0"""
    if len(nodes) < 2:
        return 0.0
    node_set = set(nodes)
    weights = [
        G[u][v]["weight"]
        for u, v in G.edges(nodes)
        if u in node_set and v in node_set
    ]
    return float(np.mean(weights)) if weights else 0.0


def build_submission(clusters, G, ids):
    """
    Build a DataFrame with group_id, item_id, confidence.
    ids: array mapping node index -> real item_id
    """
    rows = []
    for cluster_id, nodes in clusters.items():
        conf = compute_confidence(nodes, G)
        for node in nodes:
            rows.append({
                "group_id": cluster_id,
                "item_id": int(ids[node]),
                "confidence": round(conf, 4),
            })
    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)


def print_cluster_stats(clusters):
    sizes = [len(v) for v in clusters.values()]
    print(f"  Clusters:   {len(clusters)}")
    print(f"  Singletons: {sum(1 for s in sizes if s == 1)}")
    print(f"  Size min/max/mean: {min(sizes)} / {max(sizes)} / {np.mean(sizes):.2f}")
