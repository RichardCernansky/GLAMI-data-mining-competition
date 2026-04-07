"""
Data loading and common utilities for ADM 2026 competition.
"""

import pandas as pd
import numpy as np
import ast
import os


def load_data(config=None):
    """Load train, phase1, and task1 CSVs."""
    if config is None:
        config = load_config()
    data_dir = config["paths"]["raw_data"]

    train = pd.read_csv(os.path.join(data_dir, "items_train.csv"))
    phase1 = pd.read_csv(os.path.join(data_dir, "items_phase_1.csv"))
    task1 = pd.read_csv(os.path.join(data_dir, "task_1.csv"))
    return train, phase1, task1


def parse_department_ids(dept_str):
    """Parse departmentIds from string like "['1', '11']" to a frozenset of ints."""
    try:
        parsed = ast.literal_eval(dept_str)
        return frozenset(int(x) for x in parsed)
    except (ValueError, SyntaxError):
        return frozenset()


def parse_color_tags(color_str):
    """Parse colorTagIdsString from string like "230,232" to a frozenset of ints."""
    if pd.isna(color_str):
        return frozenset()
    try:
        return frozenset(int(x.strip()) for x in str(color_str).split(",") if x.strip())
    except ValueError:
        return frozenset()


def normalize_price(price, geo, currency_rates):
    """Convert a local currency price to EUR using rates from config."""
    rate = currency_rates.get(geo, 1.0)
    return price * rate


def prepare_items(df, config=None):
    """
    Parse and normalize all items in a DataFrame.
    Adds: department_set, color_set, price_eur
    Drops: brandEditionTagId (if present)
    """
    if config is None:
        config = load_config()
    currency_rates = config["currency"]

    df = df.copy()

    if "brandEditionTagId" in df.columns:
        df = df.drop(columns=["brandEditionTagId"])

    df["department_set"] = df["departmentIds"].apply(parse_department_ids)
    df["color_set"] = df["colorTagIdsString"].apply(parse_color_tags)
    df["price_eur"] = df.apply(
        lambda row: normalize_price(row["price"], row["geo"], currency_rates), axis=1
    )

    return df


def build_item_lookup(train, phase1):
    """Combine train + phase1 into a dict keyed by itemId for fast lookups."""
    item_lookup = {}
    for _, row in train.iterrows():
        item_lookup[row["itemId"]] = row
    for _, row in phase1.iterrows():
        item_lookup[row["itemId"]] = row
    return item_lookup


def build_item_df(train, phase1):
    """Combine train + phase1 into a single DataFrame indexed by itemId."""
    all_items = pd.concat([train, phase1], ignore_index=True)
    all_items = all_items.set_index("itemId")
    return all_items

"""
Load project config from YAML.
"""

import yaml
import os


def load_config(path=None):
    """
    Load config from YAML file.
    Default: configs/default.yaml from project root.
    """
    if path is None:
        # Try relative to project root
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.yaml"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config