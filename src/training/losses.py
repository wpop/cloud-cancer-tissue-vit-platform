"""
Loss functions for image classification.
"""

from torch import nn


def create_loss_function() -> nn.Module:
    """
    Create the loss function for multi-class image classification.

    Returns:
        Cross-entropy loss.
    """

    return nn.CrossEntropyLoss()

