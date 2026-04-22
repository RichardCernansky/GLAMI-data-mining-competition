"""
Leiden/Louvain clustering, cluster-size enforcement, and confidence scoring.
Switch algorithm via configs/default.yaml: clustering.algorithm: leiden | louvain
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def run_louvain(G, resolution=1.0):
    """Louvain community detection."""
    import community as community_louvain
    partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)
    clusters = defaultdict(list)
    for node, cluster_id in partition.items():
        clusters[cluster_id].append(node)
    return partition, dict(clusters)


def run_leiden(G, resolution=1.0, seed=42):
    """
    Leiden community detection.
    Stricter than Louvain: guarantees internally connected communities.
    Requires: pip install leidenalg igraph
    """
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        print("  [WARNING] leidenalg not installed, falling back to Louvain.")
        print("            pip install leidenalg igraph")
        return run_louvain(G, resolution)

    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}, {}

    node_to_idx = {nd: i for i, nd in enumerate(nodes)}
    edges   = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    weights = [float(G[u][v]["weight"]) for u, v in G.edges()]

    g = ig.Graph(n=n, edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )

    clusters = defaultdict(list)
    for idx, cid in enumerate(partition.membership):
        clusters[cid].append(nodes[idx])
    partition_dict = {nodes[idx]: cid for idx, cid in enumerate(partition.membership)}
    return partition_dict, dict(clusters)


def run_clustering(G, resolution=1.0, algorithm="leiden"):
    """Dispatcher — select algorithm via config: clustering.algorithm: leiden | louvain"""
    if algorithm == "leiden":
        return run_leiden(G, resolution)
    return run_louvain(G, resolution)


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


def split_large_clusters(clusters, G, max_size=100, resolution=2.0, algorithm="leiden", depth=0):
    """
    Recursively re-cluster oversized communities until all fit within max_size.
    Competition hard-rejects any group with > 100 items.
    Uses the same algorithm as the main clustering pass.
    """
    oversized = [(cid, nodes) for cid, nodes in clusters.items() if len(nodes) > max_size]
    if not oversized:
        return clusters

    result = {cid: nodes for cid, nodes in clusters.items() if len(nodes) <= max_size}
    new_id = max(clusters.keys(), default=0) + 1

    for _, nodes in oversized:
        if depth >= 5:
            for i in range(0, len(nodes), max_size):
                result[new_id] = nodes[i:i + max_size]
                new_id += 1
            continue

        sub_G = G.subgraph(nodes)
        _, sub_clusters = run_clustering(sub_G, resolution=resolution, algorithm=algorithm)

        remapped = {}
        for sub_nodes in sub_clusters.values():
            remapped[new_id] = sub_nodes
            new_id += 1

        remapped = split_large_clusters(remapped, G, max_size, resolution * 1.5, algorithm, depth + 1)
        result.update(remapped)
        new_id = max(result.keys(), default=0) + 1

    return result


def build_submission(clusters, G, ids, algorithm="leiden"):
    """
    Build a DataFrame with group_id, item_id, confidence.
    ids: array mapping node index -> real item_id
    Enforces 100-item cluster cap — competition rejects larger groups.
    """
    clusters = split_large_clusters(clusters, G, algorithm=algorithm)
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
    print(f"  Clusters:         {len(clusters)}")
    print(f"  Singletons:       {sum(1 for s in sizes if s == 1)}")
    print(f"  Size min/max/mean: {min(sizes)} / {max(sizes)} / {np.mean(sizes):.2f}")
    print(f"  Oversized (>100): {sum(1 for s in sizes if s > 100)}")
