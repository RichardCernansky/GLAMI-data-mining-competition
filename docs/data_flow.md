# Data Flow: run_baseline.py

## Overview

The baseline script takes prepared data + precomputed embeddings, computes group-level features, trains a classifier, and generates submissions.

## Step-by-step Flow

### 1. LOAD

```
data/processed/train_prepared.pkl     → train        (928k items with labels, parsed fields, EUR prices)
data/processed/phase1_prepared.pkl    → phase1       (200k items without labels, same cleanup)
data/processed/task1.csv              → task1         (15,000 groups of 5 itemIds — what we submit predictions for)
data/processed/validation_groups.csv  → val_groups    (15,000 fake groups built from train)
                                      → val_labels    (0/1 ground truth, extracted from the "label" column)

data/processed/embeddings/train_embeddings.npy   → train_emb   (928k × 384 float vectors)
data/processed/embeddings/train_ids.npy          → train_ids   (928k itemIds, same row order as train_emb)
data/processed/embeddings/phase1_embeddings.npy  → phase1_emb  (200k × 384 float vectors)
data/processed/embeddings/phase1_ids.npy         → phase1_ids  (200k itemIds, same row order as phase1_emb)
```

### 2. BUILD LOOKUPS

Combines train + phase1 into two dictionaries for instant access by itemId:

```
item_lookup:       itemId → row data (title, geo, price_eur, department_set, color_set, ...)
embedding_lookup:  itemId → 384-dim vector
```

These are used by the feature computation to grab any item's data without scanning the full DataFrame.

### 3. COMPUTE VALIDATION FEATURES

```
val_groups                          val_features                     val_labels
┌─────────────────────────────┐     ┌──────────────────────────┐     ┌───┐
│ item1  item2  item3  ...    │     │ emb_max  dept_max  ...   │     │   │
├─────────────────────────────┤     ├──────────────────────────┤     ├───┤
│ row 0: [id, id, id, id, id] │ ──► │ row 0: 0.82, 1.0, ...   │ ◄── │ 1 │
│ row 1: [id, id, id, id, id] │ ──► │ row 1: 0.31, 0.5, ...   │ ◄── │ 0 │
│ row 2: [id, id, id, id, id] │ ──► │ row 2: 0.75, 1.0, ...   │ ◄── │ 1 │
│ ...                         │     │ ...                      │     │...│
│ row 14999                   │ ──► │ row 14999                │ ◄── │ 0 │
└─────────────────────────────┘     └──────────────────────────┘     └───┘
         ↓                                                            
For each row (group of 5 items):                                     
  1. Look up all 5 items in item_lookup + embedding_lookup           
  2. Compute 10 pairwise comparisons (all pairs of 5)                
  3. For each pair: text cosine sim, dept Jaccard, color Jaccard,    
     price ratio, same_geo flag                                      
  4. Aggregate: max, mean, min across all 10 pairs                   
  5. Output: one row of ~13 features                                 
```

Rows are aligned by position: val_groups row 0 → val_features row 0 → val_labels[0].

### 4. BASELINE 1: Threshold

```
val_features["emb_max"]  ──►  threshold (e.g. >= 0.7)  ──►  prediction (0 or 1)
```

Sweeps thresholds from 0.4 to 0.9, prints F1/precision/recall for each. No training, just comparing one number to a cutoff.

### 5. BASELINE 2: LightGBM

#### Cross-validation (evaluation only)

```
val_features (15,000 rows × 13 features) + val_labels (15,000 × 0/1)
                            ↓
              5-fold stratified split
                            ↓
┌──────────────────────────────────────────────────────┐
│ Fold 1: train on folds 2-5, predict fold 1           │
│ Fold 2: train on folds 1,3-5, predict fold 2         │
│ Fold 3: train on folds 1-2,4-5, predict fold 3       │
│ Fold 4: train on folds 1-3,5, predict fold 4         │
│ Fold 5: train on folds 1-4, predict fold 5            │
└──────────────────────────────────────────────────────┘
                            ↓
        Each fold reports F1, precision, recall
        Overall: find best threshold across all out-of-fold predictions
```

These 5 models are discarded. They exist only to measure how good the approach is.

#### Final model (for submission)

```
ALL val_features + ALL val_labels  ──►  train_lightgbm()  ──►  final_model
```

One model trained on everything. This is what predicts on task1.

### 6. COMPUTE TASK1 FEATURES

Same process as step 3, but on the real groups:

```
task1 (15,000 groups)  ──►  compute_group_features_batch()  ──►  task1_features (15,000 × 13)
```

Items are looked up from phase1_prepared (all task1 items come from phase1).

### 7. GENERATE SUBMISSIONS

Two submission files:

```
Submission A (threshold):
  task1_features["emb_max"] >= best_threshold  ──►  0 or 1  ──►  submission_threshold.csv

Submission B (LightGBM):
  final_model.predict_proba(task1_features)  ──►  probability  ──►  >= best_threshold  ──►  0 or 1  ──►  submission_lightgbm.csv
```

### Output Files

```
data/submissions/submission_threshold.csv   ← simple baseline submission
data/submissions/submission_lightgbm.csv    ← classifier submission
data/processed/val_features.csv             ← saved for future experiments
data/processed/task1_features.csv           ← saved for future experiments
```

## Key Insight

The training data for the classifier comes from the training items, but reformatted as groups:

```
train items (928k with labels)
        ↓ prepare_data.py
validation_groups.csv (15k groups with 0/1 ground truth)
        ↓ run_baseline.py
val_features (15k × 13 numbers)
        ↓
LightGBM learns: which feature patterns mean "has duplicate"
        ↓
Applies learned patterns to task1_features → submission
```