"""
Evaluation metrics for Phase 1 (duplicate detection).
"""

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


def evaluate_phase1(y_true, y_pred):
    """
    Evaluate phase 1 predictions.
    Returns dict with F1, precision, recall.
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def print_evaluation(y_true, y_pred):
    """Print a readable evaluation report."""
    metrics = evaluate_phase1(y_true, y_pred)
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"\nDetailed report:")
    print(classification_report(y_true, y_pred, target_names=["no_dup", "has_dup"]))