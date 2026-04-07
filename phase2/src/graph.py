"""
Graph construction from embeddings via FAISS nearest neighbor search.
"""

import numpy as np
import faiss
import networkx as nx


def combine_embeddings(text_emb, img_emb, has_img, text_w, img_w):
    """
    Concatenate weighted text + image embeddings, L2-normalized.
    Items without an image get zeros in the image part.
    """
    n = len(text_emb)
    text_part = text_emb * text_w
    img_part  = np.zeros((n, img_emb.shape[1]), dtype=np.float32)
    img_part[has_img.astype(bool)] = img_emb[has_img.astype(bool)] * img_w
    combined = np.concatenate([text_part, img_part], axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (combined / norms).astype(np.float32)


def build_faiss_index(embeddings, n_cells=400, n_probe=20, min_points_per_cell=39):
    """
    Build an approximate IndexIVFFlat index.
    n_cells: number of Voronoi cells (k-means partitions)
    n_probe: how many cells to search per query (higher = more accurate, slower)
    """
    dim = embeddings.shape[1]
    n_cells = min(n_cells, len(embeddings) // min_points_per_cell)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_cells, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings.astype(np.float32))
    index.add(embeddings.astype(np.float32))
    index.nprobe = min(n_probe, n_cells)
    return index


def query_neighbors(index, embeddings, k):
    """
    Query top-k neighbors for every item.
    Returns sims and indices arrays of shape (n, k+1) — first column is self-match.
    """
    sims, indices = index.search(embeddings.astype(np.float32), k + 1)
    return sims, indices


def build_edges(sims, indices, k, threshold):
    """
    Build edge list from kNN results.
    Skips self-matches, only keeps pairs with sim >= threshold, stores each edge once.
    Returns src, dst, weights as numpy arrays.
    """
    n = len(indices)
    src, dst, weights = [], [], []

    for i in range(n):
        for j_pos in range(1, k + 1):
            j = indices[i, j_pos]
            sim = sims[i, j_pos]
            if j == -1:
                continue
            if sim < threshold:
                break  # sorted descending
            if i < j:
                src.append(i)
                dst.append(j)
                weights.append(sim)

    return (
        np.array(src, dtype=np.int32),
        np.array(dst, dtype=np.int32),
        np.array(weights, dtype=np.float32),
    )


def build_nx_graph(n, src, dst, weights):
    """Build a networkx graph from edge arrays."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for s, d, w in zip(src, dst, weights):
        G.add_edge(int(s), int(d), weight=float(w))
    return G
