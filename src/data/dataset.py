"""
Dataset utilities for histopathology image classification.
"""

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class TissueImageDataset(Dataset):
    """
    PyTorch Dataset for tissue image classification.

    Expected directory structure:

        root_dir/
        ├── benign/
        │   ├── image_001.png
        │   └── image_002.png
        └── cancer/
            ├── image_003.png
            └── image_004.png
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root_dir: Path to the dataset split directory.
            transform: Optional image transformations.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.class_names = sorted(
            [path.name for path in self.root_dir.iterdir() if path.is_dir()]
        )

        if not self.class_names:
            raise ValueError(f"No class folders found in: {self.root_dir}")

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }

        self.samples: list[tuple[Path, int]] = []
        self._collect_samples()

        if not self.samples:
            raise ValueError(f"No image files found in: {self.root_dir}")

    def _collect_samples(self) -> None:
        """
        Collect image paths and labels from class folders.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            label = self.class_to_idx[class_name]

            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() in valid_extensions:
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Load one image and its label.

        Args:
            index: Sample index.

        Returns:
            Tuple of transformed image and integer label.
        """
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

