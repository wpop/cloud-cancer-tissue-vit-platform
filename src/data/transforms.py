"""
Image transformations for training and inference.
"""

from torchvision import transforms


def get_train_transforms(image_size: int) -> transforms.Compose:
    """
    Create image transformations for training.

    Args:
        image_size: Target image size.

    Returns:
        TorchVision transformation pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_validation_transforms(image_size: int) -> transforms.Compose:
    """
    Create image transformations for validation.

    Args:
        image_size: Target image size.

    Returns:
        TorchVision transformation pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_inference_transforms(image_size: int) -> transforms.Compose:
    """
    Create image transformations for inference.

    Args:
        image_size: Target image size.

    Returns:
        TorchVision transformation pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

