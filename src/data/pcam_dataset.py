"""
PCam HDF5 dataset for histopathology image classification.
"""

from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PCamDataset(Dataset):
    """
    PyTorch Dataset for PatchCamelyon HDF5 files.
    """

    def __init__(
        self,
        images_path: str | Path,
        labels_path: str | Path,
        transform=None,
    ) -> None:
        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        self.transform = transform

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images file not found: {self.images_path}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        self.images_file = h5py.File(self.images_path, "r")
        self.labels_file = h5py.File(self.labels_path, "r")

        self.images = self.images_file["x"]
        self.labels = self.labels_file["y"]

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        image_array = np.asarray(self.images[index])
        label = int(self.labels[index][0][0][0])

        image = Image.fromarray(image_array).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

