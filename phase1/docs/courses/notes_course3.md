# Pairwise Learning-to-Rank (RankNet)

- **Pairwise approach**: for each query q, form pairs (dᵢ, dⱼ) where dᵢ is more relevant than dⱼ
- **Scoring function**: f(q, d) — a neural net that takes query-document features and outputs a scalar score
- **Input features**: not raw document features, but query-document relationship features (BM25, term overlap, embedding similarity, CTR, etc.)
- **Predicted preference probability**: P(dᵢ ≻ dⱼ) = σ(sᵢ − sⱼ), where sᵢ = f(q, dᵢ), sⱼ = f(q, dⱼ)
- **Loss**: binary cross-entropy, L = −log σ(sᵢ − sⱼ), pushing P toward 1 for correctly ordered pairs
- **Siamese architecture**: same network (shared weights), two forward passes (one per document in the pair), one combined loss
- **Gradient flows through both branches**:
  - ∂L/∂sᵢ is negative → pushes sᵢ up
  - ∂L/∂sⱼ is positive → pushes sⱼ down
  - Both update the same weights in one backward pass
- **Inference is cheap**: just compute f(q, d) independently for each candidate document, then sort by score — no pairwise comparisons needed, O(n log n)
- **Key limitation**: all pairs weighted equally — swapping rank 1↔2 penalized the same as swapping 99↔100

## ΔNDCG and LambdaRank

- **ΔNDCG**: the change in NDCG if you swap positions of two documents dᵢ and dⱼ in the current ranking
- **Computed per pair**: for each pair, compute NDCG with current positions, compute NDCG with swapped positions, take the absolute difference
- **NDCG recap**: DCG = Σ (2^relᵢ − 1) / log₂(rank + 1), normalized by ideal DCG; top positions contribute much more due to log discount
- **Why ΔNDCG varies by position**:
  - Swap at positions 1↔2: large discount difference (1/log₂2 vs 1/log₂3) → large ΔNDCG
  - Swap at positions 99↔100: tiny discount difference (1/log₂100 vs 1/log₂101) → tiny ΔNDCG
- **Efficient computation**: no need to recompute the full DCG sum twice — only two terms change, so it simplifies to |ΔNDCG| ∝ |(2^relᵢ − 2^relⱼ) × (1/log₂(p+1) − 1/log₂(k+1))|
- **LambdaRank**: same pairwise framework as RankNet, but multiplies each pair's gradient by |ΔNDCG|
- **Effect**: misordered pairs near the top get much stronger gradients, pairs deep in the ranking get weak gradients
- **Result**: the model focuses learning capacity on getting the top of the ranking right, which is what matters for real search/recommendation


