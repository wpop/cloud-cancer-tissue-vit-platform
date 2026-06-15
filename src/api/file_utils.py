"""
Small file helpers for uploaded API images.
"""

import os
from pathlib import Path

from fastapi import UploadFile

from src.api.model_loader import get_inference_config


DEFAULT_UPLOAD_DIR = Path("outputs") / "api_uploads"


def get_upload_dir() -> Path:
    """
    Read the upload directory from config, with the original path as a fallback.
    """

    config = get_inference_config()
    upload_dir = config.get("output", {}).get("api_upload_dir", DEFAULT_UPLOAD_DIR)
    return Path(upload_dir)


def get_supported_extensions() -> set[str]:
    """
    Read the image extensions supported by the inference API.
    """

    config = get_inference_config()
    formats = config.get("input", {}).get("supported_formats", [])
    return {str(item).lower().lstrip(".") for item in formats}


def is_supported_image(filename: str) -> bool:
    """
    Check whether a filename has an allowed image extension.
    """

    extension = Path(filename).suffix.lower().lstrip(".")
    return extension in get_supported_extensions()


async def save_uploaded_image(upload_file: UploadFile) -> Path:
    """
    Save an uploaded image under the configured API upload directory.
    """

    filename = os.path.basename(upload_file.filename or "")
    if not filename:
        raise ValueError("Uploaded file must have a filename.")

    if not is_supported_image(filename):
        allowed = ", ".join(sorted(get_supported_extensions()))
        raise ValueError(f"Unsupported image type. Allowed extensions: {allowed}.")

    upload_dir = get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / filename

    with target_path.open("wb") as output_file:
        while chunk := await upload_file.read(1024 * 1024):
            output_file.write(chunk)

    return target_path
