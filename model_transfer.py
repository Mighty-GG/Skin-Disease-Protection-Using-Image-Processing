import torch.nn as nn
from torchvision import models


def build_resnet50_transfer(
    num_classes: int = 22,
    use_pretrained: bool = True,
    freeze_feature_extractor: bool = False,
):
    if use_pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    if freeze_feature_extractor:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_transfer_model(
    model_name: str = "resnet50",
    num_classes: int = 22,
    use_pretrained: bool = True,
    freeze_feature_extractor: bool = False,
):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return build_resnet50_transfer(
            num_classes=num_classes,
            use_pretrained=use_pretrained,
            freeze_feature_extractor=freeze_feature_extractor,
        )

    raise ValueError(f"Unsupported transfer model: {model_name}")


if __name__ == "__main__":
    model = build_transfer_model(
        model_name="resnet50",
        num_classes=22,
        use_pretrained=True,
        freeze_feature_extractor=False,
    )

    print(model)

    # Quick output-layer check
    print("\nFinal classifier:")
    print(model.fc)