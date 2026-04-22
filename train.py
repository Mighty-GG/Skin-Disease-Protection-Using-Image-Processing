# train.py

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from config import config
from augmentations import get_train_transforms, get_eval_transforms
from data_loader import build_dataloaders
from imbalance_utils import compute_class_counts_from_subset, build_loss_function
from metrics import summarize_classification_metrics, format_metrics_report
from model_custom_cnn import build_custom_cnn
from model_transfer import build_transfer_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> nn.Module:
    #build the model selected in config
    model_name = config.model_name.lower()

    if model_name == "custom_cnn":
        model = build_custom_cnn(
            num_classes=config.num_classes,
            input_channels=config.num_channels,
            dropout_rate=config.dropout_rate,
        )
    else:
        model = build_transfer_model(
            model_name=model_name,
            num_classes=config.num_classes,
            use_pretrained=config.use_pretrained,
            freeze_feature_extractor=config.freeze_feature_extractor,
        )

    return model


def build_optimizer(model: nn.Module):
    #Build optimizer based on config
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer_name = config.optimizer_name.lower()

    if optimizer_name == "adam":
        return Adam(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "sgd":
        return SGD(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )

    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def build_scheduler(optimizer):
    #build learning rate scheduler based on config
    if config.scheduler_name is None:
        return None

    scheduler_name = str(config.scheduler_name).lower()

    if scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    elif scheduler_name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.gamma,
            patience=2,
        )

    raise ValueError(f"Unsupported scheduler: {config.scheduler_name}")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    
    #train the model for one epoch. return: epoch loss, all_labels, all_predictions
    model.train()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_labels.append(labels.detach().cpu())
        all_predictions.append(preds.detach().cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    return epoch_loss, all_labels, all_predictions


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    #evaluate the model for one epoch
    model.eval()

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

    return epoch_loss, all_labels, all_predictions


def build_checkpoint_name() -> str:
    #build a descriptive checkpoint filename automatically
    parts = [config.model_name]

    if config.use_focal_loss:
        parts.append("focal")
    elif config.use_class_weights:
        parts.append("class_weights")
    elif config.use_weighted_sampler:
        parts.append("weighted_sampler")
    else:
        parts.append("baseline")

    return "best_" + "_".join(parts) + ".pth"


def save_checkpoint(model: nn.Module, epoch: int, val_loss: float) -> None:
    config.saved_models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = build_checkpoint_name()
    save_path = config.saved_models_dir / checkpoint_filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "model_name": config.model_name,
        "num_classes": config.num_classes,
        "class_names": config.class_names,
        "use_class_weights": config.use_class_weights,
        "use_weighted_sampler": config.use_weighted_sampler,
        "use_focal_loss": config.use_focal_loss,
    }

    torch.save(checkpoint, save_path)
    print(f"Saved best model to: {save_path}")


def train_model():
    #main training entry point
    device = get_device()
    print(f"Using device: {device}")

    print(f"Experiment name: {config.experiment_name}")
    print(f"Model name     : {config.model_name}")
    print(f"Class weights  : {config.use_class_weights}")
    print(f"Weighted sampler: {config.use_weighted_sampler}")
    print(f"Focal loss     : {config.use_focal_loss}")

    print("Building transforms...")
    train_transform = get_train_transforms(config.image_size)
    eval_transform = get_eval_transforms(config.image_size)

    print("Building dataloaders...")
    dataloaders, datasets = build_dataloaders(
        dataset_root=config.dataset_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        val_ratio=config.val_split,
        random_seed=config.random_seed,
        use_weighted_sampler=config.use_weighted_sampler,
        return_paths=False,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print("Computing class counts...")
    class_counts = compute_class_counts_from_subset(
        subset=datasets["train_subset"],
        num_classes=len(datasets["class_names"]),
    )

    print("Building model...")
    model = build_model().to(device)
    #print(model)

    print("Building loss function...")
    criterion = build_loss_function(
        use_class_weights=config.use_class_weights,
        use_focal_loss=config.use_focal_loss,
        class_counts=class_counts,
        device=device.type,
        focal_loss_gamma=config.focal_loss_gamma,
    )

    print("Building optimizer and scheduler...")
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "train_weighted_f1": [],
        "val_weighted_f1": [],
    }

    print("\nStarting training...\n")

    for epoch in range(config.epochs):
        print(f"Epoch [{epoch + 1}/{config.epochs}]")

        train_loss, train_labels, train_preds = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_labels, val_preds = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        train_metrics = summarize_classification_metrics(
            y_true=train_labels,
            y_pred=train_preds,
            num_classes=config.num_classes,
            class_names=config.class_names,
        )

        val_metrics = summarize_classification_metrics(
            y_true=val_labels,
            y_pred=val_preds,
            num_classes=config.num_classes,
            class_names=config.class_names,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["train_weighted_f1"].append(train_metrics["weighted_f1"])
        history["val_weighted_f1"].append(val_metrics["weighted_f1"])

        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Train Accuracy : {train_metrics['accuracy']:.4f}")
        print(f"Val Accuracy   : {val_metrics['accuracy']:.4f}")
        print(f"Train Macro F1 : {train_metrics['macro_f1']:.4f}")
        print(f"Val Macro F1   : {val_metrics['macro_f1']:.4f}")
        print(f"Train Weighted F1: {train_metrics['weighted_f1']:.4f}")
        print(f"Val Weighted F1  : {val_metrics['weighted_f1']:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            if config.save_best_model:
                save_checkpoint(model, epoch + 1, val_loss)
        else:
            epochs_without_improvement += 1

        print("-" * 60)

        if config.use_early_stopping and epochs_without_improvement >= config.patience:
            print("Early stopping triggered.")
            break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print("\nFinal Validation Metrics Report:")
    print(format_metrics_report(val_metrics))

    print(f"Checkpoint filename: {build_checkpoint_name()}")

    return model, history


if __name__ == "__main__":
    config.create_directories()
    config.validate()
    train_model()