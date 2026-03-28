# data_loader.py

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from dataset import build_skin_disease_datasets
from imbalance_utils import create_weighted_sampler_from_subset

def split_train_validation(
    train_dataset,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[Subset, Subset]:
    #split the training dataset into train and validation subsets.
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    dataset_size = len(train_dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(random_seed)

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_subset, val_subset


def get_subset_class_distribution(subset, class_names) -> Dict[str, int]:
    #Compute class distribution for a torch.utils.data.Subset.
    distribution = {class_name: 0 for class_name in class_names}

    for idx in subset.indices:
        _, label = subset.dataset.samples[idx]
        class_name = class_names[label]
        distribution[class_name] += 1

    return distribution


###def create_weighted_sampler(subset, num_classes: int):
    class_counts = [0] * num_classes

    for idx in subset.indices:
        _, label = subset.dataset.samples[idx]
        class_counts[label] += 1

    class_weights = []
    for count in class_counts:
        if count == 0:
            class_weights.append(0.0)
        else:
            class_weights.append(1.0 / count)

    sample_weights = []
    for idx in subset.indices:
        _, label = subset.dataset.samples[idx]
        sample_weights.append(class_weights[label])

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def build_dataloaders(
    dataset_root: Path,
    train_transform,
    eval_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    use_weighted_sampler: bool = False,
    return_paths: bool = False,
):
    #build train, validation, and test dataloaders
    dataset_root = Path(dataset_root)

    full_train_dataset, test_dataset = build_skin_disease_datasets(
        dataset_root=dataset_root,
        train_transform=train_transform,
        test_transform=eval_transform,
        return_paths=return_paths,
    )

    train_subset, val_subset = split_train_validation(
        train_dataset=full_train_dataset,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    #validation subset should use eval transforms, not train transforms
    #we rebuild a second version of the train dataset using eval transforms and use the same validation indices
    eval_train_dataset, _ = build_skin_disease_datasets(
        dataset_root=dataset_root,
        train_transform=eval_transform,
        test_transform=eval_transform,
        return_paths=return_paths,
    )

    val_subset = Subset(eval_train_dataset, val_subset.indices)

    class_names = full_train_dataset.get_class_names()
    num_classes = full_train_dataset.get_num_classes()

    if use_weighted_sampler:
        train_sampler = create_weighted_sampler_from_subset(train_subset, num_classes)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    datasets = {
        "full_train": full_train_dataset,
        "train_subset": train_subset,
        "val_subset": val_subset,
        "test": test_dataset,
        "class_names": class_names,
    }

    return dataloaders, datasets


if __name__ == "__main__":
    from config import config
    from augmentations import get_train_transforms, get_eval_transforms

    train_tf = get_train_transforms(config.image_size)
    eval_tf = get_eval_transforms(config.image_size)

    dataloaders, datasets = build_dataloaders(
        dataset_root=config.dataset_root,
        train_transform=train_tf,
        eval_transform=eval_tf,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        val_ratio=config.val_split,
        random_seed=config.random_seed,
        use_weighted_sampler=config.use_weighted_sampler,
        return_paths=False,
    )

    print("Class names:")
    print(datasets["class_names"])

    print("\nDataset sizes:")
    print("Full train:", len(datasets["full_train"]))
    print("Train subset:", len(datasets["train_subset"]))
    print("Val subset:", len(datasets["val_subset"]))
    print("Test:", len(datasets["test"]))

    print("\nNumber of batches:")
    print("Train:", len(dataloaders["train"]))
    print("Val:", len(dataloaders["val"]))
    print("Test:", len(dataloaders["test"]))