from typing import Dict, List, Optional

import torch


def compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.numel() == 0:
        return 0.0

    correct = (y_true == y_pred).sum().item()
    total = y_true.numel()
    return correct / total


def compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    #compute confusion matrix of shape [num_classes, num_classes].
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    return cm


def compute_per_class_metrics(
    confusion_matrix: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    #compute per-class precision, recall, and F1 from confusion matrix
    num_classes = confusion_matrix.shape[0]

    if class_names is not None and len(class_names) != num_classes:
        raise ValueError("Length of class_names must match confusion matrix size.")

    results = {}

    for i in range(num_classes):
        tp = confusion_matrix[i, i].item()
        fp = confusion_matrix[:, i].sum().item() - tp
        fn = confusion_matrix[i, :].sum().item() - tp
        support = confusion_matrix[i, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        class_key = class_names[i] if class_names is not None else str(i)

        results[class_key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return results


def compute_macro_f1(per_class_metrics: Dict[str, Dict[str, float]]) -> float:
    #compute macro-averaged F1
    if not per_class_metrics:
        return 0.0

    f1_scores = [metrics["f1"] for metrics in per_class_metrics.values()]
    return sum(f1_scores) / len(f1_scores)


def compute_weighted_f1(per_class_metrics: Dict[str, Dict[str, float]]) -> float:
    #compute weighted F1 using class support
    if not per_class_metrics:
        return 0.0

    total_support = sum(metrics["support"] for metrics in per_class_metrics.values())
    if total_support == 0:
        return 0.0

    weighted_sum = sum(
        metrics["f1"] * metrics["support"] for metrics in per_class_metrics.values()
    )
    return weighted_sum / total_support


def compute_macro_precision(per_class_metrics: Dict[str, Dict[str, float]]) -> float:
    #compute macro-averaged precision
    if not per_class_metrics:
        return 0.0

    values = [metrics["precision"] for metrics in per_class_metrics.values()]
    return sum(values) / len(values)


def compute_macro_recall(per_class_metrics: Dict[str, Dict[str, float]]) -> float:
    #Compute macro-averaged recall
    if not per_class_metrics:
        return 0.0

    values = [metrics["recall"] for metrics in per_class_metrics.values()]
    return sum(values) / len(values)


def summarize_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    #compute a full set of classification metrics. Returns: accuracy, macro_precision, macro_recall, macro_f1, weighted_f1, confusion_matrix, per_class
    confusion_matrix = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    per_class = compute_per_class_metrics(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )

    summary = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_precision": compute_macro_precision(per_class),
        "macro_recall": compute_macro_recall(per_class),
        "macro_f1": compute_macro_f1(per_class),
        "weighted_f1": compute_weighted_f1(per_class),
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
    }

    return summary


def format_metrics_report(metrics_summary: Dict[str, object]) -> str:
    #build a readable text report from summarized metrics
    lines = []
    lines.append("Overall Metrics")
    lines.append(f"Accuracy       : {metrics_summary['accuracy']:.4f}")
    lines.append(f"Macro Precision: {metrics_summary['macro_precision']:.4f}")
    lines.append(f"Macro Recall   : {metrics_summary['macro_recall']:.4f}")
    lines.append(f"Macro F1       : {metrics_summary['macro_f1']:.4f}")
    lines.append(f"Weighted F1    : {metrics_summary['weighted_f1']:.4f}")
    lines.append("")
    lines.append("Per-Class Metrics")

    for class_name, values in metrics_summary["per_class"].items():
        lines.append(
            f"{class_name}: "
            f"P={values['precision']:.4f}, "
            f"R={values['recall']:.4f}, "
            f"F1={values['f1']:.4f}, "
            f"Support={values['support']}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Small standalone test
    y_true = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 1, 0, 2, 0, 1, 2, 1])
    class_names = ["ClassA", "ClassB", "ClassC"]

    summary = summarize_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=3,
        class_names=class_names,
    )

    print(format_metrics_report(summary))
    print("\nConfusion Matrix:")
    print(summary["confusion_matrix"])