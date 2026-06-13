"""
Metrics for image classification.
"""

import torch


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate classification accuracy.

    Args:
        outputs: Model logits with shape [batch_size, num_classes].
        labels: Ground-truth class labels with shape [batch_size].

    Returns:
        Accuracy value between 0.0 and 1.0.
    """

    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    return correct / total

