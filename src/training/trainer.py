"""
Training and validation loops.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import calculate_accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    log_every_n_batches: int | None = None,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        loss_fn: Loss function.
        optimizer: Optimizer used to update the model.
        device: Device used for training.
        max_batches: Optional number of batches to run for debugging.
        log_every_n_batches: Optional interval for printing batch progress.
    """

    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(outputs, labels)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        if log_every_n_batches and batch_idx % log_every_n_batches == 0:
            print(
                f"Batch {batch_idx}/{total_batches} | "
                f"Loss: {loss.item():.4f} | Acc: {accuracy:.4f}"
            )

        if max_batches is not None and batch_idx >= max_batches:
            break

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """
    Validate the model for one epoch.

    Args:
        model: Model to validate.
        dataloader: Validation dataloader.
        loss_fn: Loss function.
        device: Device used for validation.
        max_batches: Optional number of batches to run for debugging.
    """

    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        accuracy = calculate_accuracy(outputs, labels)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        if max_batches is not None and batch_idx >= max_batches:
            break

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def save_checkpoint(
    model: nn.Module,
    save_path: str | Path,
) -> None:
    """
    Save model weights.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path)
