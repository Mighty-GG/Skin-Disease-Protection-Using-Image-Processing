# evaluate.py

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from config import config
from augmentations import get_eval_transforms, get_train_transforms
from data_loader import build_dataloaders
from imbalance_utils import compute_class_counts_from_subset, build_loss_function
from metrics import summarize_classification_metrics, format_metrics_report
from model_custom_cnn import build_custom_cnn
from model_transfer import build_transfer_model


def get_device() -> torch.device:
    if config.device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    #rebuild the correct model architecture from checkpoint metadata
    model_name = checkpoint.get("model_name", config.model_name).lower()
    num_classes = checkpoint.get("num_classes", config.num_classes)

    if model_name == "custom_cnn":
        model = build_custom_cnn(
            num_classes=num_classes,
            input_channels=config.num_channels,
            dropout_rate=config.dropout_rate,
        )
    else:
        model = build_transfer_model(
            model_name=model_name,
            num_classes=num_classes,
            use_pretrained=False,
            freeze_feature_extractor=False,
        )

    return model


def evaluate_model(
    checkpoint_path: Path,
    split: str = "test",
) -> Tuple[float, dict]:
    #evaluate a saved model on validation or test split
    
    if split not in {"val", "test"}:
        raise ValueError("split must be either 'val' or 'test'")

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("Building model...")
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Building dataloaders...")
    train_transform = get_train_transforms(config.image_size)
    eval_transform = get_eval_transforms(config.image_size)

    dataloaders, datasets = build_dataloaders(
        dataset_root=config.dataset_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        val_ratio=config.val_split,
        random_seed=config.random_seed,
        use_weighted_sampler=False,
        return_paths=False,
    )

    dataloader = dataloaders[split]

    class_counts = compute_class_counts_from_subset(
        subset=datasets["train_subset"],
        num_classes=len(datasets["class_names"]),
    )

    criterion = build_loss_function(
        use_class_weights=config.use_class_weights,
        use_focal_loss=config.use_focal_loss,
        class_counts=class_counts,
        device=device.type,
        focal_loss_gamma=config.focal_loss_gamma,
    )

    print(f"Evaluating on {split} set...")

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_labels.append(labels.detach().cpu())
            all_predictions.append(preds.detach().cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    class_names = checkpoint.get("class_names", config.class_names)

    metrics_summary = summarize_classification_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
        num_classes=len(class_names),
        class_names=class_names,
    )

    return epoch_loss, metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pth",
        help="Checkpoint filename inside ...",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on.",
    )

    args = parser.parse_args()

    checkpoint_path = config.saved_models_dir / args.checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loss, metrics_summary = evaluate_model(
        checkpoint_path=checkpoint_path,
        split=args.split,
    )

    print("\nEvaluation complete.")
    print(f"Loss: {loss:.4f}")
    print("\nMetrics Report:")
    print(format_metrics_report(metrics_summary))
    print("\nConfusion Matrix:")
    print(metrics_summary["confusion_matrix"])