"""
DataLoader utilities for PCam histopathology classification.
"""

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.data.pcam_dataset import PCamDataset
from src.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
)


def create_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from PCam HDF5 files.
    """

    dataset_config = config["dataset"]
    dataloader_config = config["dataloader"]

    root_dir = Path(dataset_config["root_dir"])

    train_dataset = PCamDataset(
        images_path=root_dir / dataset_config["train_images"],
        labels_path=root_dir / dataset_config["train_labels"],
        transform=get_train_transforms(dataset_config["image_size"]),
    )

    val_dataset = PCamDataset(
        images_path=root_dir / dataset_config["val_images"],
        labels_path=root_dir / dataset_config["val_labels"],
        transform=get_validation_transforms(dataset_config["image_size"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle"],
        num_workers=dataloader_config["num_workers"],
        pin_memory=dataloader_config["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
        pin_memory=dataloader_config["pin_memory"],
    )

    return train_loader, val_loader
