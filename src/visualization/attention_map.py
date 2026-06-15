"""
Attention-map visualization utilities for torchvision ViT models.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps

from src.data.transforms import get_inference_transforms
from src.models.inference import load_image


def save_attention_overlay(
    model: torch.nn.Module,
    image_path: str | Path,
    save_path: str | Path,
    image_size: int,
    device: torch.device,
) -> None:
    """
    Save a final-layer class-token attention overlay for a ViT prediction.

    Torchvision ViT does not expose attention weights from its public forward
    output. For this first explainability feature, we temporarily wrap the final
    encoder block's MultiheadAttention call and force it to return attention
    weights. This is simple and reliable for torchvision ViT-B/16, but it should
    be treated as a visual aid rather than clinical evidence.
    """

    image_path = Path(image_path)
    save_path = Path(save_path)

    attention = _extract_final_class_attention(
        model=model,
        image_path=image_path,
        image_size=image_size,
        device=device,
    )
    overlay = _create_overlay_image(
        image_path=image_path,
        patch_attention=attention,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(save_path)


def _extract_final_class_attention(
    model: torch.nn.Module,
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Extract class-token attention from the final transformer encoder block.
    """

    model.eval()
    image = load_image(image_path)
    transform = get_inference_transforms(image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    final_block = model.encoder.layers[-1]
    attention_module = final_block.self_attention
    original_forward = attention_module.forward
    captured_attention = {}

    def forward_with_weights(query, key, value, **kwargs):
        """
        Force the attention module to return per-head attention weights.
        """

        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        output, weights = original_forward(query, key, value, **kwargs)
        captured_attention["weights"] = weights.detach().cpu()
        return output, weights

    attention_module.forward = forward_with_weights
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        attention_module.forward = original_forward

    if "weights" not in captured_attention:
        raise RuntimeError("Could not capture ViT attention weights.")

    weights = captured_attention["weights"]
    if weights.dim() != 4:
        raise RuntimeError(f"Unexpected attention shape: {tuple(weights.shape)}")

    # Shape is [batch, heads, target_tokens, source_tokens].
    class_attention = weights[0, :, 0, 1:].mean(dim=0).numpy()
    grid_size = int(np.sqrt(class_attention.shape[0]))

    if grid_size * grid_size != class_attention.shape[0]:
        raise RuntimeError("Attention tokens do not form a square patch grid.")

    attention = class_attention.reshape(grid_size, grid_size)
    attention = attention - attention.min()
    max_value = attention.max()
    if max_value > 0:
        attention = attention / max_value

    return attention.astype(np.float32)


def _create_overlay_image(image_path: Path, patch_attention: np.ndarray) -> Image.Image:
    """
    Blend a resized attention heatmap over the original RGB image.
    """

    image = load_image(image_path)
    heatmap = Image.fromarray((patch_attention * 255).astype(np.uint8))
    heatmap = heatmap.resize(image.size, resample=Image.Resampling.BILINEAR)

    heatmap_array = np.asarray(heatmap).astype(np.float32) / 255.0
    heatmap_rgb = colormaps["jet"](heatmap_array)[..., :3]

    image_array = np.asarray(image).astype(np.float32) / 255.0
    overlay = (0.6 * image_array) + (0.4 * heatmap_rgb)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
