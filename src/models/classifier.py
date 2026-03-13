"""
Group classifier for Phase 1 duplicate detection.
"""

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils import load_config

def train_lightgbm(X_train, y_train, X_val=None, y_val=None, config=None):
    """Train a LightGBM classifier using params from config."""
    if config is None:
        config = load_config()
    params = config["classifier"]["lightgbm"]

    model = lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        scale_pos_weight=params["scale_pos_weight"],
        random_state=42,
        verbose=-1,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
        )
    else:
        model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train, config=None):
    """Train a Random Forest classifier using params from config."""
    if config is None:
        config = load_config()
    params = config["classifier"]["random_forest"]

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        class_weight=params["class_weight"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(y_true, y_proba):
    """Find classification threshold that maximizes F1."""
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "f1": f1,
                "precision": precision_score(y_true, preds),
                "recall": recall_score(y_true, preds),
                "threshold": threshold,
            }

    return best_threshold, best_metrics


def cross_validate(X, y, config=None):
    """Run stratified k-fold cross-validation."""
    if config is None:
        config = load_config()

    n_splits = config["classifier"]["cv_folds"]
    model_type = config["classifier"]["model_type"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_proba = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_type == "lightgbm":
            model = train_lightgbm(X_train, y_train, X_val, y_val, config)
        else:
            model = train_random_forest(X_train, y_train, config)

        proba = model.predict_proba(X_val)[:, 1]
        all_proba[val_idx] = proba

        threshold, metrics = find_best_threshold(y_val, proba)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)

        print(f"  Fold {fold + 1}: F1={metrics['f1']:.4f}  "
              f"Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}  "
              f"Thresh={metrics['threshold']:.2f}")

    overall_threshold, overall_metrics = find_best_threshold(y, all_proba)

    print(f"\n  Overall: F1={overall_metrics['f1']:.4f}  "
          f"Prec={overall_metrics['precision']:.4f}  "
          f"Rec={overall_metrics['recall']:.4f}  "
          f"Thresh={overall_metrics['threshold']:.2f}")

    return fold_metrics, overall_threshold, overall_metrics