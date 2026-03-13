"""
Build validation groups from the labeled training set.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from src.utils import load_config


def build_validation_groups(train, config=None):
    """
    Create validation groups mimicking task1 format (20% positive, 80% negative).
    Reads n_positive, n_negative, seed from config.
    """
    if config is None:
        config = load_config()

    n_positive = config["validation"]["n_positive"]
    n_negative = config["validation"]["n_negative"]
    seed = config["validation"]["seed"]
    rng = np.random.RandomState(seed)

    # Build label lookup
    item_to_label = dict(zip(train["itemId"], train["label"]))
    label_to_items = defaultdict(list)
    for item_id, label in item_to_label.items():
        label_to_items[label].append(item_id)

    multi_labels = [lbl for lbl, items in label_to_items.items() if len(items) >= 2]
    all_item_ids = list(item_to_label.keys())

    groups = []
    ground_truth = []

    # --- Positive groups ---
    for _ in range(n_positive):
        lbl = multi_labels[rng.randint(len(multi_labels))]
        items = label_to_items[lbl]

        n_same = rng.choice([2, 3], p=[0.7, 0.3])
        n_same = min(n_same, len(items))
        same_items = list(rng.choice(items, size=n_same, replace=False))

        n_fill = 5 - n_same
        fill_items = []
        attempts = 0
        while len(fill_items) < n_fill and attempts < 500:
            candidate = all_item_ids[rng.randint(len(all_item_ids))]
            if item_to_label[candidate] != lbl and candidate not in same_items and candidate not in fill_items:
                fill_items.append(candidate)
            attempts += 1

        group = same_items + fill_items
        rng.shuffle(group)
        groups.append(group)
        ground_truth.append(1)

    # --- Negative groups ---
    for _ in range(n_negative):
        used_labels = set()
        neg_items = []
        attempts = 0
        while len(neg_items) < 5 and attempts < 500:
            candidate = all_item_ids[rng.randint(len(all_item_ids))]
            if item_to_label[candidate] not in used_labels:
                neg_items.append(candidate)
                used_labels.add(item_to_label[candidate])
            attempts += 1

        groups.append(neg_items)
        ground_truth.append(0)

    groups_df = pd.DataFrame(groups, columns=["item1", "item2", "item3", "item4", "item5"])
    return groups_df, ground_truth