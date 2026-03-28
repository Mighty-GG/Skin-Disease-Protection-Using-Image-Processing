from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    #project paths
    project_root: Path = Path(__file__).resolve().parent

    dataset_root: Path = project_root / "archive" / "SkinDisease" / "SkinDisease"

    outputs_dir: Path = project_root / "outputs"
    plots_dir: Path = outputs_dir / "plots"
    metrics_dir: Path = outputs_dir / "metrics"
    reports_dir: Path = outputs_dir / "reports"
    confusion_matrix_dir: Path = outputs_dir / "confusion_matrices"

    models_dir: Path = project_root / "models"
    checkpoints_dir: Path = models_dir / "checkpoints"
    saved_models_dir: Path = models_dir / "saved_models"

    #dataset settings
    dataset_name: str = "SkinDisease"
    num_classes: int = 22
    num_channels: int = 3
    image_size: int = 224

    class_names: List[str] = field(default_factory=lambda: [
        "Acne",
        "Actinic_Keratosis",
        "Atopic_Dermatitis",
        "Basal_Cell_Carcinoma",
        "Benign_Keratosis",
        "Bullous_Disease",
        "Cellulitis",
        "Contact_Dermatitis",
        "Drug_Eruption",
        "Eczema",
        "Fungal_Infection",
        "Herpes",
        "Lupus",
        "Melanoma",
        "Nevus",
        "Psoriasis",
        "Rosacea",
        "Scabies",
        "Seborrheic_Keratosis",
        "Tinea",
        "Urticaria",
        "Vitiligo",
    ])

    
    #dataloader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    
    #split settings
    #only validation split is needed because train/test already exist
    val_split: float = 0.15
    random_seed: int = 42

    #training settings
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    optimizer_name: str = "adam"
    scheduler_name: str = "step"
    step_size: int = 7
    gamma: float = 0.1

    use_early_stopping: bool = True
    patience: int = 5

    #model settings
    model_name: str = "custom_cnn"
    dropout_rate: float = 0.5

    use_pretrained: bool = True
    freeze_feature_extractor: bool = False

    #imbalance settings
    use_class_weights: bool = True
    use_weighted_sampler: bool = False
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0

    #saving settings
    save_best_model: bool = True
    best_model_name: str = "best_model.pth"
    save_metrics: bool = True
    save_plots: bool = True
    save_confusion_matrix: bool = True

    #device settings
    device: str = "cuda"

    def create_directories(self) -> None:
        directories = [
            self.outputs_dir,
            self.plots_dir,
            self.metrics_dir,
            self.reports_dir,
            self.confusion_matrix_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.saved_models_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be between 0 and 1, got {self.val_split}")

        if self.num_classes != len(self.class_names):
            raise ValueError(
                f"num_classes ({self.num_classes}) does not match "
                f"number of class_names ({len(self.class_names)})."
            )


config = Config()


if __name__ == "__main__":
    config.create_directories()
    config.validate()

    print("Configuration is valid.")
    print(f"Project root: {config.project_root}")
    print(f"Dataset root: {config.dataset_root}")
    print(f"Model selected: {config.model_name}")
    print(f"Number of classes: {config.num_classes}")