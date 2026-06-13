"""
Probability visualization utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def save_probability_plot(
    probabilities: dict[str, float],
    save_path: str | Path,
) -> None:
    """
    Save a bar plot with class probabilities.

    Args:
        probabilities: Dictionary with class names and probabilities.
        save_path: Output path for the plot image.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = list(probabilities.keys())
    values = [prob * 100.0 for prob in probabilities.values()]

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, values)
    plt.ylabel("Probability (%)")
    plt.xlabel("Class")
    plt.title("Classification Probabilities")
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

