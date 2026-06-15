"""
Evaluate trained ViT model on the PCam validation set.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cloud_cancer_eval_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.data.datamodule import create_dataloaders
from src.evaluation.metrics import (
    build_classification_report,
    compute_binary_metrics,
    compute_roc_curve,
)
from src.models.vit_classifier import create_vit_classifier


CLASS_NAMES = ["benign", "cancer"]
CONFIG_PATH = Path("configs/train.yaml")
CHECKPOINT_PATH = Path("models/checkpoints/best_model.pt")
OUTPUT_DIR = Path("outputs/evaluation")


def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    """
    Create the ViT model and load the trained checkpoint.
    """

    model = create_vit_classifier(
        num_classes=config["dataset"]["num_classes"],
        pretrained=False,
    )

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def collect_validation_predictions(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_val_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run validation inference and collect labels, predictions, and cancer scores.
    """

    all_labels = []
    all_predictions = []
    all_cancer_scores = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_cancer_scores.append(probabilities[:, 1].cpu())

    if not all_labels:
        raise ValueError("No validation batches were evaluated.")

    labels_tensor = torch.cat(all_labels)
    predictions_tensor = torch.cat(all_predictions)
    cancer_scores_tensor = torch.cat(all_cancer_scores)

    return labels_tensor, predictions_tensor, cancer_scores_tensor


def save_json_metrics(metrics: dict, output_path: Path) -> None:
    """
    Save evaluation metrics as JSON.
    """

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_confusion_matrix_plot(
    counts: dict[str, int],
    output_path: Path,
) -> None:
    """
    Save a confusion matrix image.
    """

    matrix = np.array(
        [
            [counts["true_negative"], counts["false_positive"]],
            [counts["false_negative"], counts["true_positive"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    max_value = matrix.max() if matrix.size else 0
    text_color_threshold = max_value / 2 if max_value > 0 else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text_color = "white" if matrix[row, col] > text_color_threshold else "black"
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float | None,
    output_path: Path,
) -> None:
    """
    Save a ROC curve image.
    """

    fig, ax = plt.subplots(figsize=(5, 4))
    label = "ROC curve"
    if roc_auc is not None:
        label = f"ROC curve (AUC = {roc_auc:.4f})"

    ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], color="navy", linewidth=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if roc_auc is None:
        ax.text(
            0.5,
            0.2,
            "ROC-AUC is undefined with one class present.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = load_config(CONFIG_PATH)

    use_cuda = config.get("device", {}).get("use_cuda", True)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader = create_dataloaders(config)

    model = load_model(config=config, device=device)

    max_val_batches = config["training"].get("max_val_batches")
    labels_tensor, predictions_tensor, cancer_scores_tensor = (
        collect_validation_predictions(
            model=model,
            val_loader=val_loader,
            device=device,
            max_val_batches=max_val_batches,
        )
    )

    metrics = compute_binary_metrics(
        predictions=predictions_tensor,
        labels=labels_tensor,
    )

    labels_array = labels_tensor.numpy()
    cancer_scores_array = cancer_scores_tensor.numpy()
    roc_data = compute_roc_curve(
        labels=labels_array,
        positive_scores=cancer_scores_array,
    )

    results = {
        **metrics,
        "roc_auc": roc_data["roc_auc"],
        "num_samples": int(labels_tensor.numel()),
        "class_names": CLASS_NAMES,
        "positive_class": "cancer",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "config_path": str(CONFIG_PATH),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    report_path = OUTPUT_DIR / "classification_report.txt"
    confusion_matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    roc_curve_path = OUTPUT_DIR / "roc_curve.png"

    save_json_metrics(results, metrics_path)

    report = build_classification_report(
        counts=metrics,
        class_names=CLASS_NAMES,
    )
    with report_path.open("w", encoding="utf-8") as file:
        file.write(report)

    save_confusion_matrix_plot(metrics, confusion_matrix_path)
    save_roc_curve_plot(
        fpr=roc_data["fpr"],
        tpr=roc_data["tpr"],
        roc_auc=roc_data["roc_auc"],
        output_path=roc_curve_path,
    )

    print("\nEvaluation complete")
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
    if results["roc_auc"] is None:
        print("ROC-AUC: undefined")
    else:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Saved evaluation outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
