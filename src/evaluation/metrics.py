"""
Evaluation metrics for binary classification.
"""

from typing import Any

import numpy as np
import torch


def compute_binary_classification_counts(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    """
    Compute TP, TN, FP, FN counts.
    """

    predictions = predictions.cpu()
    labels = labels.cpu()

    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    true_negative = ((predictions == 0) & (labels == 0)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def compute_precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    """
    Compute precision, recall, and F1-score.
    """

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_accuracy(counts: dict[str, int]) -> float:
    """
    Compute binary classification accuracy from confusion counts.
    """

    correct = counts["true_positive"] + counts["true_negative"]
    total = correct + counts["false_positive"] + counts["false_negative"]

    return correct / total if total > 0 else 0.0


def compute_binary_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int | float]:
    """
    Compute confusion counts and common binary classification metrics.
    """

    counts = compute_binary_classification_counts(
        predictions=predictions,
        labels=labels,
    )
    precision_recall_f1 = compute_precision_recall_f1(counts)

    return {
        **counts,
        "accuracy": compute_accuracy(counts),
        **precision_recall_f1,
    }


def compute_roc_curve(
    labels: np.ndarray,
    positive_scores: np.ndarray,
) -> dict[str, Any]:
    """
    Compute ROC curve points for the positive class.

    The positive score should be the predicted probability for the cancer class.
    ROC-AUC is undefined when only one class is present in labels.
    """

    labels = np.asarray(labels).astype(int)
    positive_scores = np.asarray(positive_scores).astype(float)

    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))

    if positive_count == 0 or negative_count == 0:
        return {
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "thresholds": np.array([np.inf, -np.inf]),
            "roc_auc": None,
        }

    order = np.argsort(positive_scores)[::-1]
    sorted_labels = labels[order]
    sorted_scores = positive_scores[order]

    distinct_score_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_score_indices, sorted_labels.size - 1]

    true_positives = np.cumsum(sorted_labels == 1)[threshold_indices]
    false_positives = np.cumsum(sorted_labels == 0)[threshold_indices]

    tpr = true_positives / positive_count
    fpr = false_positives / negative_count

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    roc_auc = float(np.trapezoid(tpr, fpr))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
    }


def build_classification_report(
    counts: dict[str, int | float],
    class_names: list[str],
) -> str:
    """
    Build a small text classification report for binary predictions.
    """

    tn = counts["true_negative"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    tp = counts["true_positive"]
    total = tn + fp + fn + tp

    benign_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if (benign_precision + benign_recall) > 0
        else 0.0
    )

    cancer_precision = counts["precision"]
    cancer_recall = counts["recall"]
    cancer_f1 = counts["f1_score"]

    benign_support = tn + fp
    cancer_support = tp + fn
    accuracy = counts["accuracy"]

    lines = [
        "Classification report",
        "",
        f"{'class':<12}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}",
        (
            f"{class_names[0]:<12}"
            f"{benign_precision:>12.4f}"
            f"{benign_recall:>12.4f}"
            f"{benign_f1:>12.4f}"
            f"{benign_support:>12}"
        ),
        (
            f"{class_names[1]:<12}"
            f"{cancer_precision:>12.4f}"
            f"{cancer_recall:>12.4f}"
            f"{cancer_f1:>12.4f}"
            f"{cancer_support:>12}"
        ),
        "",
        f"{'accuracy':<36}{accuracy:>12.4f}{total:>12}",
    ]

    return "\n".join(lines) + "\n"
