import pandas as pd
import numpy as np
import ast
import os
import yaml


def load_config(path=None):
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.yaml"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_department_ids(dept_str):
    try:
        return frozenset(int(x) for x in ast.literal_eval(dept_str))
    except (ValueError, SyntaxError):
        return frozenset()


def parse_color_tags(color_str):
    if pd.isna(color_str):
        return frozenset()
    try:
        return frozenset(int(x.strip()) for x in str(color_str).split(",") if x.strip())
    except ValueError:
        return frozenset()


def prepare_items(df, config=None):
    if config is None:
        config = load_config()
    rates = config["currency"]
    df = df.copy()
    if "brandEditionTagId" in df.columns:
        df = df.drop(columns=["brandEditionTagId"])
    df["department_set"] = df["departmentIds"].apply(parse_department_ids)
    df["color_set"] = df["colorTagIdsString"].apply(parse_color_tags)
    df["price_eur"] = df.apply(
        lambda r: r["price"] * rates.get(r["geo"], 1.0), axis=1
    )
    return df
