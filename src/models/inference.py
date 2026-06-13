"""
Inference utilities for tissue image classification.
"""

from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.data.transforms import get_inference_transforms


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image and convert it to RGB.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    class_names: list[str],
    image_size: int,
    device: torch.device,
) -> dict:
    """
    Predict class probabilities for one image.
    """

    model.eval()

    image = load_image(image_path)
    transform = get_inference_transforms(image_size)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    probability_dict = {
        class_names[index]: float(probabilities[index].item())
        for index in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_dict,
    }

