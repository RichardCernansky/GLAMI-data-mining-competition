# EDA Rationale — What We Observe and Why

## Label Distribution

You need to know how many items share the same product label. If most labels are singletons (1 item), that means most products in the dataset have no duplicate at all, and the duplicates that exist are rare. This tells you the base rate of the problem and affects how you construct training examples for your classifier. It also matters for Phase 2 where you're building clusters — if the typical cluster size is 2-3, your approach should be very different than if it's 10+.

## Geo Distribution

You need to know whether duplicates mostly happen within the same market or across markets. If the same product appears in CZ, SK, and PL with different titles in different languages, that tells you a multilingual embedding model is essential. If most duplicates are within the same geo, you might get away with a monolingual approach. This is probably the single most important design decision for your text pipeline.

## Text Inspection (Same Product Across Geos)

By looking at actual title/description pairs for the same product, you'll see how different they are. Maybe they're just translated versions of the same text, or maybe they're completely different descriptions written by different shops. This tells you whether text similarity alone can solve the problem or whether you need image and structured features too.

## Price Analysis

Prices might be in local currencies (CZK, PLN, EUR, HUF...) which would make raw price comparison useless, or they might be normalized. You need to know this before using price as a feature. Also, if the same product has wildly different prices across listings, price ratio becomes a weak signal. If prices are tight within a label, it becomes a strong cheap filter.

## Structured Features (departmentIds, colorTagIds)

If all items with the same label always share the same departmentIds, then department mismatch is a strong negative signal — you can instantly rule out pairs from different departments. That's a very cheap filter that saves you from expensive embedding comparisons. Same logic for color tags.

## Image Column

You need to know the format (URLs vs local filenames) to plan your image embedding pipeline. Also checking if images are unique per item or shared — if duplicates literally have the same image URL, that's a trivial feature.

## Task 1 Structure

You need to know which dataset the group items come from. If they're from phase1 (no labels), you can't validate locally. If some are from train (with labels), you can build a local validation set by checking whether any pair in a group shares a label. This determines your entire validation strategy.

## Title Similarity Heuristic

This is your "how hard is this problem" check. If simple character-level title matching already separates positive from negative groups cleanly, you might get a strong F1 just from text features. If the distributions overlap heavily, you know you need multimodal features. It also gives you a baseline number to beat.

---

**In short** — every section answers the question "what approach should I build?" rather than just being exploration for its own sake.

# Desription
1. Data sizes — 928k training items with labels, 200k phase1 items without labels. Task1 has 15,000 groups of 5. All task items come from phase1 (zero overlap with train), which means you cannot validate by looking up labels for task1 items. You'll need to build your own validation groups from the training set.
2. Labels — Almost no singletons (only 8). 152k labels have 2+ items, 53k have 5+, and the biggest label has 80 items. This is great — it means the training set is rich with duplicate examples to learn from.
3. Geos — 13 markets. Slovakia dominates (231k items), followed by CZ, BG, HR, RO. Poland and Estonia are tiny (92 and 2,135 items). The critical finding: 59.6% of labels appear in only 1 geo, while 40.4% span 2+ geos. So about 40% of the time, duplicates are cross-lingual. Many labels span 9-11 geos, meaning the same product appears across almost all markets. This confirms multilingual embeddings are essential.
4. Text — Looking at the same product across geos, titles are in completely different languages and scripts. Label 1 shows "Балеринки Clarks" (Bulgarian), "Baleríny Clarks" (Czech), "Baleríny Clarks" (Slovak). Label 3 shows "Апрески EMU Australia" (Bulgarian), "Sněhule EMU Australia" (Czech), "Μπότες Χιονιού EMU Australia" (Greek). The brand name survives across languages but the product category word is totally different. No missing titles, but 35k missing descriptions (~4%). Descriptions are useful but not always available.
5. Prices are in local currencies — This is a key finding. Mean price by geo: Hungary 38,216 (Hungarian Forint), CZ 2,511 (Czech Koruna), SK 51.79 (Euro), IT 96 (Euro). The same product (label 1, Clarks ballerinas) shows as 120 in BG, 1550 in CZ, 62.95 in SK. The coefficient of variation within labels is high (mean 0.73). This means raw price comparison is useless — you need to either normalize to a common currency, or use price as a within-geo signal only.
6. Structured features — departmentIds are very consistent within labels: 91.9% of labels have the same departmentIds across all their items. This is a powerful cheap filter — if two items have different departments, they're probably not the same product. colorTagIdsString has 27k missing values but is formatted as comma-separated IDs like "230,232".
7. brandEditionTagId is mostly useless — 925k out of 928k values are missing. Only 168 unique values exist. Ignore this feature.
8. Task1 groups — Looking at the sample groups, the negative groups (no duplicates) are obvious: random items from different categories across different geos (ballerinas, caps, boots, shorts all in one group). Group 3 is interesting — it has items 2, 3, 4 all being JACQUEMUS bags in different languages at similar prices (~740-759). That's likely a positive group, and it shows the pattern: same brand, similar price, different language for the category word.
9. Title similarity heuristic — Only 10.4% of groups have max title similarity > 0.8, and 17.6% > 0.5. Since 20% of groups are positive, this means character-level title matching catches some but not all duplicates. The cross-lingual cases (where titles look nothing alike) will be missed entirely. This confirms that simple string matching alone won't be enough — you need semantic/multilingual embeddings.

# Processed data 
**train_prepared.pkl** — The 928k training items with labels, but cleaned up: departmentIds parsed into sets, colorTags parsed into sets, prices converted to EUR, brandEditionTagId dropped. You use this to build validation groups and learn what duplicates look like.
**phase1_prepared.pkl** — The 200k unlabeled items, same cleanup applied. When your model processes the real task1 groups, it looks up each item's features from here.
**task1.csv** — Just a copy of the original 15,000 groups. The groups you submit predictions for.
**validation_groups.csv** — 15,000 fake groups you created from training data (3,000 positive + 12,000 negative), with a label column that tells you the true answer (0 or 1). You test your model on this before submitting on the real task1.