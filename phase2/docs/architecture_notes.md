# Phase 2 Architecture Notes

## The Core Problem

Everything in the pipeline (graph, clustering, F1) is only as good as the edge weights.
An edge weight currently = raw cosine similarity of general-purpose sentence embeddings.

The fundamental issue: a general sentence model was trained on semantic similarity, not product identity.
Two different "black floral dresses" in CZ and HU can score cosine=0.97 even though they are different products.
This causes Louvain/Leiden to merge them transitively into giant impure clusters.

Current results confirm this: even edges with sim > 0.95 have only ~67% cluster purity.

---

## Pipeline Overview

```
items CSV
  → prepare_data.py         parse, normalize prices to EUR, parse dept/color sets
  → extract_embeddings.py   SentenceTransformer (text) + CLIP (image) → .npy arrays
  → run.py
      load embeddings
      FAISS ANN             find top-k nearest neighbors per item (approximate)
      build_edges           filter by cosine threshold → edge list
      build_nx_graph        weighted graph: nodes=items, edges=similar pairs
      run_clustering        Louvain or Leiden → communities = product groups
      build_submission      enforce 100-item cap, write CSV
```

---

## What the Threshold Is On

At every stage, the threshold is on **cosine similarity** of the item embeddings.
With general embeddings this is noisy. With contrastive-trained embeddings the same threshold becomes clean.

---

## Similarity / Scoring Options

### 1. Raw Cosine Similarity (current)
Single number from FAISS. No training needed.
Problem: semantic similarity ≠ product identity. Threshold is ambiguous.

### 2. Structural Hard Filters on top of Cosine
Discard edges where price ratio < 0.7, no department overlap, or no color overlap.
No training needed. From Phase 1 validation, price_max was the #1 feature (importance 1478).
Works as a pre-filter to reduce false positives cheaply.
Subsumed by LightGBM if you use that — not needed for correctness, only for speed.

### 3. Pair-Level LightGBM Scorer (recommended first step)
Train a binary classifier on individual pairs using labeled training data.

Features per pair:
- text_sim       — cosine of text embeddings
- img_sim        — cosine of image embeddings
- price_sim      — min/max EUR price ratio
- dept_jaccard   — overlap of department IDs
- color_jaccard  — overlap of color tag IDs
- same_geo       — 1 if same country

Training: FAISS gives candidate pairs from train data, label = 1 if same label column.
Output: predict_proba() replaces cosine as edge weight.

Effect: a pair with cosine=0.97 but very different price gets downweighted.
        a pair with cosine=0.91 but same price/dept/color gets upweighted.

Works on CPU. Trains in minutes. No embedding retraining needed.

### 4. Better Pretrained Embedding Model
Swap paraphrase-multilingual-mpnet-base-v2 for a retrieval-optimized model:
- intfloat/multilingual-e5-large
- BAAI/bge-m3

These were trained for matching/retrieval, not general semantic similarity.
No GPU needed for inference — just re-run extract_embeddings.py (slow, one-time).
~5-10 hours on CPU for 200k items.

### 5. Contrastive Fine-Tuning (Zalando approach, best quality)

Each item is encoded independently through a shared encoder:
```
item_A → encoder → emb_A
item_B → encoder → emb_B
distance(emb_A, emb_B) → training signal
```

The encoder is trained with contrastive/triplet loss:
- Positive pair (same label): pull embeddings together
- Negative pair (different label): push embeddings apart

Triplet loss:
```
loss = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)
```

After training, same-product pairs → cosine ~0.98, different products → cosine ~0.60.
The threshold becomes clean. FAISS becomes more accurate. Purity goes up.

Requires GPU. Not feasible on CPU for 928k items.
Google Colab or Kaggle free GPU would work.

The Zalando paper extends this to multimodal: image encoder + text encoder + numeric
features all projected into the same embedding space via a small linear layer.
Contrastive loss trains the whole thing end-to-end.

At inference: same pipeline as now — embed all items, FAISS, threshold, graph, cluster.
The threshold is still cosine similarity, but the embedding space is far more discriminative.

### 6. Concatenated Pair Classifier (MLP)
```
[emb_A | emb_B | price_diff | dept_jacc | ...] → MLP → P(match)
```
Sees both items together. Can learn cross-item patterns.
Better than siamese for pair scoring but worse for retrieval:
you are stuck with candidates FAISS already found.
Siamese/contrastive improves the embeddings themselves so FAISS retrieval also improves.

---

## Proposed Order of Implementation

| Step | Method | GPU needed | Effort | Expected gain |
|------|--------|-----------|--------|--------------|
| 1 | LightGBM pair scorer | No | Low | High — adds price/dept/color signal |
| 2 | intfloat/multilingual-e5-large | No (slow) | Low | Medium — better base embeddings |
| 3 | Contrastive fine-tuning | Yes | High | Highest — directly optimizes for product matching |

Current F1 = 0.185 at threshold=0.97. Precision=0.24, Recall=0.15.
Precision is acceptable. Recall is the bottleneck — too many singletons.
LightGBM scorer would allow lowering the FAISS threshold (more candidates)
while keeping precision high through multimodal scoring.

---

## Key Insight from Phase 1 Validation

Feature importance from LightGBM trained on Phase 1 groups:
```
price_max      1478  ← most important by far
emb_max        1130
emb_2nd         799
emb_min         697
price_mean      679
emb_mean        627
emb_x_dept_max  311
color_mean      275
color_max       229
dept_mean       227
same_geo_max    154
same_geo_max     73
dept_max          0  ← near-zero variance in Phase 1, may differ in Phase 2
```

Price is the strongest signal after text similarity.
This directly motivates including price_sim as a feature in the pair-level scorer.
