"""
Main training entry point.
"""

import torch

from src.config import load_config
from src.data.datamodule import create_dataloaders
from src.models.vit_classifier import create_vit_classifier
from src.training.losses import create_loss_function
from src.training.trainer import (
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
)


def main() -> None:
    config = load_config("configs/train.yaml")

    device = torch.device(
        "cuda" if config["device"]["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(config)

    model = create_vit_classifier(
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    loss_fn = create_loss_function()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_accuracy = 0.0
    training_config = config["training"]
    epochs = training_config["epochs"]
    max_train_batches = training_config.get("max_train_batches")
    max_val_batches = training_config.get("max_val_batches")
    log_every_n_batches = training_config.get("log_every_n_batches")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
            log_every_n_batches=log_every_n_batches,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            max_batches=max_val_batches,
        )

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_checkpoint(
                model=model,
                save_path="models/checkpoints/best_model.pt",
            )
            print("Saved new best model.")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
