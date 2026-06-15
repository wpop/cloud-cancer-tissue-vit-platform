"""
Load and cache the ViT model for API inference.
"""

from pathlib import Path

import torch

from src.config import load_config
from src.models.vit_classifier import create_vit_classifier


CONFIG_PATH = Path("configs") / "inference.yaml"
DEFAULT_CHECKPOINT_PATH = Path("models") / "checkpoints" / "best_model.pt"

CLASS_NAMES = ["benign", "cancer"]

_model = None
_device = None
_config = None


def load_model():
    """
    Load the trained ViT model once and reuse it for later requests.
    """

    global _model, _device, _config

    if _model is not None:
        return _model

    config = load_config(CONFIG_PATH)
    checkpoint_path = Path(
        config.get("model", {}).get("checkpoint", DEFAULT_CHECKPOINT_PATH)
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    use_cuda = config.get("device", {}).get("use_cuda", True)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    num_classes = config.get("model", {}).get("num_classes", len(CLASS_NAMES))
    model = create_vit_classifier(num_classes=num_classes, pretrained=False)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model = model
    _device = device
    _config = config

    return model


def get_inference_config() -> dict:
    """
    Return the loaded inference config, loading it from disk if needed.
    """

    global _config

    if _config is None:
        _config = load_config(CONFIG_PATH)

    return _config


def get_model_status() -> dict:
    """
    Return model load status without forcing a model load.
    """

    config = get_inference_config()
    checkpoint_path = Path(
        config.get("model", {}).get("checkpoint", DEFAULT_CHECKPOINT_PATH)
    )

    return {
        "model_loaded": _model is not None,
        "device": _device.type if _device is not None else None,
        "checkpoint_path": str(checkpoint_path),
    }
