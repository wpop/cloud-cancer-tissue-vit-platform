"""
Test inference on one PCam validation image.
"""

import torch
import h5py
from PIL import Image
import numpy as np

from src.config import load_config
from src.models.vit_classifier import create_vit_classifier
from src.models.inference import predict_image
from src.visualization.probability_plot import save_probability_plot

import json
from pathlib import Path


def main() -> None:
    config = load_config("configs/train.yaml")
    pcam_dir = config["dataset"]["root_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_vit_classifier(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt", map_location=device))
    model.to(device)

    image_path = "outputs/predictions/sample_pcam_image.png"

    with h5py.File(f"{pcam_dir}/{config['dataset']['val_images']}", "r") as f:
        image_array = np.asarray(f["x"][0])

    Image.fromarray(image_array).convert("RGB").save(image_path)

    result = predict_image(
        model=model,
        image_path=image_path,
        class_names=["benign", "cancer"],
        image_size=config["dataset"]["image_size"],
        device=device,
    )

    print(result)

    output_json_path = Path("outputs/predictions/sample_prediction.json")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Saved prediction JSON: {output_json_path}")

    save_probability_plot(
        result["probabilities"],
        "outputs/figures/sample_prediction_probabilities.png",
    )


if __name__ == "__main__":
    main()

