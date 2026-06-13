"""
Vision Transformer classifier for histopathology image classification.
"""

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def create_vit_classifier(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a ViT-B/16 classifier for transfer learning.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        Vision Transformer model with only the final head trainable.
    """

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None

    model = vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(
        in_features=in_features,
        out_features=num_classes,
    )

    freeze_vit_backbone(model)
    unfreeze_classification_head(model)
    print_model_summary(model)

    return model


def freeze_vit_backbone(model: nn.Module) -> None:
    """
    Freeze all Vision Transformer parameters.

    This is the first transfer learning stage. Later, this function can be
    extended or paired with another helper to unfreeze selected encoder blocks.
    """

    for param in model.parameters():
        param.requires_grad = False


def unfreeze_classification_head(model: nn.Module) -> None:
    """
    Leave only the final classification head trainable.
    """

    for param in model.heads.head.parameters():
        param.requires_grad = True


def print_model_summary(model: nn.Module) -> None:
    """
    Print a short parameter summary for the model.
    """

    print(f"Total parameters: {count_total_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")


def count_total_parameters(model: nn.Module) -> int:
    """
    Count all model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Total number of parameters.
    """

    return sum(param.numel() for param in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    model = create_vit_classifier(num_classes=2, pretrained=False)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(model)
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
