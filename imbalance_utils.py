# imbalance_utils.py

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset, WeightedRandomSampler


def compute_class_counts_from_dataset(dataset, num_classes: int) -> List[int]:
    """
    Compute class counts from a full dataset that stores samples as (path, label).
    """
    class_counts = [0] * num_classes

    for _, label in dataset.samples:
        class_counts[label] += 1

    return class_counts


def compute_class_counts_from_subset(subset: Subset, num_classes: int) -> List[int]:
    """
    Compute class counts from a torch.utils.data.Subset.
    """
    class_counts = [0] * num_classes

    for idx in subset.indices:
        _, label = subset.dataset.samples[idx]
        class_counts[label] += 1

    return class_counts


def class_counts_to_distribution(
    class_counts: List[int],
    class_names: List[str],
) -> Dict[str, int]:
    """
    Convert class counts into a readable dictionary.
    """
    if len(class_counts) != len(class_names):
        raise ValueError("class_counts and class_names must have the same length.")

    return {class_name: count for class_name, count in zip(class_names, class_counts)}


def compute_class_weights(
    class_counts: List[int],
    method: str = "inverse",
    normalize: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute class weights for weighted loss.

    Args:
        class_counts: Number of samples in each class.
        method:
            - "inverse": weight = 1 / count
            - "balanced": weight = total_samples / (num_classes * count)
        normalize: Whether to normalize weights so they average to 1.
        device: Device to place the tensor on.

    Returns:
        Tensor of shape [num_classes]
    """
    if any(count < 0 for count in class_counts):
        raise ValueError("class_counts cannot contain negative values.")

    if all(count == 0 for count in class_counts):
        raise ValueError("All class counts are zero.")

    num_classes = len(class_counts)
    total_samples = sum(class_counts)

    weights = []

    for count in class_counts:
        if count == 0:
            weight = 0.0
        else:
            if method == "inverse":
                weight = 1.0 / count
            elif method == "balanced":
                weight = total_samples / (num_classes * count)
            else:
                raise ValueError(
                    f"Unsupported method '{method}'. Use 'inverse' or 'balanced'."
                )

        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    if normalize and weights.sum() > 0:
        weights = weights / weights.mean()

    return weights


def create_weighted_sampler_from_subset(
    subset: Subset,
    num_classes: int,
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler from a training subset.
    Each sample gets a weight inverse to its class frequency.
    """
    class_counts = compute_class_counts_from_subset(subset, num_classes=num_classes)

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

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for imbalanced classification.

    Formula:
        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    Notes:
    - Works for multi-class classification with logits input.
    - alpha can be:
        None -> no class weighting
        Tensor[num_classes] -> per-class weights
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model outputs of shape [batch_size, num_classes]
            targets: Ground-truth class indices of shape [batch_size]

        Returns:
            Scalar loss if reduction is mean/sum, otherwise per-sample loss
        """
        ce_loss = self.cross_entropy(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def build_loss_function(
    use_class_weights: bool,
    use_focal_loss: bool,
    class_counts: List[int],
    device: str = "cpu",
    focal_loss_gamma: float = 2.0,
    class_weight_method: str = "balanced",
):
    """
    Build the appropriate loss function for training.

    Options:
    - Standard CrossEntropyLoss
    - Weighted CrossEntropyLoss
    - FocalLoss with optional class weights
    """
    num_classes = len(class_counts)
    if num_classes == 0:
        raise ValueError("class_counts must not be empty.")

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(
            class_counts=class_counts,
            method=class_weight_method,
            normalize=True,
            device=device,
        )

    if use_focal_loss:
        return FocalLoss(
            alpha=class_weights,
            gamma=focal_loss_gamma,
            reduction="mean",
        )

    return nn.CrossEntropyLoss(weight=class_weights)


def summarize_imbalance(
    class_counts: List[int],
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Return a readable summary of imbalance statistics.
    """
    if len(class_counts) != len(class_names):
        raise ValueError("class_counts and class_names must have the same length.")

    total_samples = sum(class_counts)
    max_count = max(class_counts) if class_counts else 0
    min_count = min(class_counts) if class_counts else 0

    summary = {}
    for class_name, count in zip(class_names, class_counts):
        percentage = (count / total_samples * 100.0) if total_samples > 0 else 0.0
        imbalance_ratio = (max_count / count) if count > 0 else float("inf")

        summary[class_name] = {
            "count": count,
            "percentage": round(percentage, 4),
            "imbalance_ratio_vs_max": round(imbalance_ratio, 4) if count > 0 else float("inf"),
        }

    summary["_overall"] = {
        "total_samples": total_samples,
        "num_classes": len(class_names),
        "max_class_count": max_count,
        "min_class_count": min_count,
        "max_to_min_ratio": round(max_count / min_count, 4) if min_count > 0 else float("inf"),
    }

    return summary


if __name__ == "__main__":
    # Simple standalone test
    class_counts = [500, 250, 100, 50, 10]
    class_names = ["A", "B", "C", "D", "E"]

    print("Class counts:")
    print(class_counts)

    print("\nDistribution:")
    print(class_counts_to_distribution(class_counts, class_names))

    print("\nBalanced class weights:")
    weights = compute_class_weights(
        class_counts=class_counts,
        method="balanced",
        normalize=True,
        device="cpu",
    )
    print(weights)

    print("\nImbalance summary:")
    summary = summarize_imbalance(class_counts, class_names)
    for key, value in summary.items():
        print(key, value)