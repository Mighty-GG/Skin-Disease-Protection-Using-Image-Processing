# model_custom_cnn.py

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A reusable convolution block:
    Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomSkinDiseaseCNN(nn.Module):
    """
    Custom CNN for multi-class skin disease classification.

    Input:
        [batch_size, 3, 224, 224]

    Output:
        [batch_size, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 22,
        input_channels: int = 3,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(input_channels, 32),   # 224 -> 112
            ConvBlock(32, 64),               # 112 -> 56
            ConvBlock(64, 128),              # 56 -> 28
            ConvBlock(128, 256),             # 28 -> 14
            ConvBlock(256, 512),             # 14 -> 7
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_custom_cnn(
    num_classes: int = 22,
    input_channels: int = 3,
    dropout_rate: float = 0.5,
) -> CustomSkinDiseaseCNN:
    """
    Helper function to build the custom CNN model.
    """
    return CustomSkinDiseaseCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        dropout_rate=dropout_rate,
    )


if __name__ == "__main__":
    # Quick shape test
    model = build_custom_cnn(num_classes=22, input_channels=3, dropout_rate=0.5)
    dummy_input = torch.randn(8, 3, 224, 224)
    output = model(dummy_input)

    print(model)
    print("\nInput shape :", dummy_input.shape)
    print("Output shape:", output.shape)