"""
Prepare a binary histopathology image dataset for PyTorch training.

The script reads image IDs and binary labels from a CSV file, randomly splits
the images into training and validation sets, and copies each image into the
class folder expected by the training pipeline.

Example:
    python scripts/prepare_dataset.py \
        --images-dir data/raw/original_images \
        --labels-csv data/raw/train_labels.csv \
        --output-dir data/raw \
        --val-split 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LABEL_TO_CLASS_NAME = {
    0: "benign",
    1: "cancer",
}
IMAGE_ID_COLUMNS = ("image_id", "image", "filename", "file", "id")
LABEL_COLUMNS = ("label", "labels", "target", "class")


@dataclass(frozen=True)
class ImageRecord:
    """
    One image and its binary label.
    """

    image_path: Path
    label: int


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare train/validation folders for binary tissue classification."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing the original image files.",
    )
    parser.add_argument(
        "--labels-csv",
        required=True,
        type=Path,
        help="CSV file containing image IDs and binary labels.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory where train/ and validation/ folders will be created.",
    )
    parser.add_argument(
        "--val-split",
        default=0.2,
        type=float,
        help="Fraction of images to copy into the validation split.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed used for reproducible splitting.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Check that input paths and split settings are usable.
    """
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    if not args.labels_csv.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels_csv}")

    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val-split must be greater than 0 and less than 1.")


def find_column(fieldnames: list[str], choices: tuple[str, ...]) -> str:
    """
    Find a CSV column by trying common column names.
    """
    normalized_names = {name.strip().lower(): name for name in fieldnames}

    for choice in choices:
        if choice in normalized_names:
            return normalized_names[choice]

    expected = ", ".join(choices)
    found = ", ".join(fieldnames)
    raise ValueError(f"Could not find one of [{expected}] in CSV columns: {found}")


def index_image_files(images_dir: Path) -> dict[str, Path]:
    """
    Build a lookup table for images by filename and by filename without extension.

    This lets the CSV contain either "sample_001.png" or just "sample_001".
    """
    image_lookup: dict[str, Path] = {}

    for image_path in images_dir.rglob("*"):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS
        ):
            continue

        # Exact filename lookup, for IDs like "case_001.png".
        image_lookup[image_path.name.lower()] = image_path

        # Stem lookup, for IDs like "case_001". If duplicate stems exist, the
        # exact filename form in the CSV is safer and should be used instead.
        stem_key = image_path.stem.lower()
        if stem_key not in image_lookup:
            image_lookup[stem_key] = image_path

    if not image_lookup:
        extensions = ", ".join(sorted(VALID_IMAGE_EXTENSIONS))
        raise ValueError(
            f"No image files found in {images_dir} with extensions: {extensions}"
        )

    return image_lookup


def load_records(labels_csv: Path, image_lookup: dict[str, Path]) -> list[ImageRecord]:
    """
    Read image IDs and labels from the CSV file.
    """
    records: list[ImageRecord] = []
    missing_images: list[str] = []

    with labels_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Labels CSV is empty or missing a header row.")

        image_id_column = find_column(reader.fieldnames, IMAGE_ID_COLUMNS)
        label_column = find_column(reader.fieldnames, LABEL_COLUMNS)

        for row_number, row in enumerate(reader, start=2):
            image_id = row[image_id_column].strip()
            label_text = row[label_column].strip()

            if not image_id:
                raise ValueError(f"Missing image ID on CSV row {row_number}.")

            try:
                label = int(label_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid label on CSV row {row_number}: {label_text!r}. "
                    "Expected 0 or 1."
                ) from error

            if label not in LABEL_TO_CLASS_NAME:
                raise ValueError(
                    f"Invalid label on CSV row {row_number}: {label}. Expected 0 or 1."
                )

            image_path = image_lookup.get(image_id.lower())
            if image_path is None:
                missing_images.append(image_id)
                continue

            records.append(ImageRecord(image_path=image_path, label=label))

    if missing_images:
        preview = ", ".join(missing_images[:10])
        extra = (
            ""
            if len(missing_images) <= 10
            else f", and {len(missing_images) - 10} more"
        )
        raise FileNotFoundError(f"Could not find images listed in CSV: {preview}{extra}")

    if not records:
        raise ValueError("No labeled image records were loaded from the CSV.")

    return records


def create_output_folders(output_dir: Path) -> None:
    """
    Create the train/validation class folders if they do not exist.
    """
    for split_name in ("train", "validation"):
        for class_name in LABEL_TO_CLASS_NAME.values():
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)


def split_records(
    records: list[ImageRecord],
    val_split: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """
    Randomly split records into train and validation lists.
    """
    shuffled_records = records.copy()
    random.Random(seed).shuffle(shuffled_records)

    validation_count = int(len(shuffled_records) * val_split)
    validation_records = shuffled_records[:validation_count]
    train_records = shuffled_records[validation_count:]

    return train_records, validation_records


def copy_records(records: list[ImageRecord], output_dir: Path, split_name: str) -> None:
    """
    Copy records into output_dir/split_name/class_name/.
    """
    for record in records:
        class_name = LABEL_TO_CLASS_NAME[record.label]
        destination = output_dir / split_name / class_name / record.image_path.name
        shutil.copy2(record.image_path, destination)


def print_summary(
    train_records: list[ImageRecord],
    validation_records: list[ImageRecord],
) -> None:
    """
    Print a short summary of the prepared dataset.
    """
    all_records = train_records + validation_records
    benign_count = sum(record.label == 0 for record in all_records)
    cancer_count = sum(record.label == 1 for record in all_records)

    print("Dataset preparation complete.")
    print(f"Training images: {len(train_records)}")
    print(f"Validation images: {len(validation_records)}")
    print(f"Benign images: {benign_count}")
    print(f"Cancer images: {cancer_count}")


def main() -> None:
    """
    Prepare the dataset folders.
    """
    args = parse_args()
    validate_args(args)

    image_lookup = index_image_files(args.images_dir)
    records = load_records(args.labels_csv, image_lookup)
    train_records, validation_records = split_records(records, args.val_split, args.seed)

    create_output_folders(args.output_dir)
    copy_records(train_records, args.output_dir, "train")
    copy_records(validation_records, args.output_dir, "validation")

    print_summary(train_records, validation_records)


if __name__ == "__main__":
    main()
