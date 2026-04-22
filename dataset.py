from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(file_path: Path) -> bool:
    #return true if the file has a valid image extension
    return file_path.suffix.lower() in VALID_EXTENSIONS


def find_split_dir(root_dir: Path, split_name: str) -> Path:
    #find a split directory in a case-insensitive way
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dataset directory not found: {root_dir}")

    #exact match first
    exact = root_dir / split_name
    if exact.exists() and exact.is_dir():
        return exact

    #case-insensitive fallback
    split_name_lower = split_name.lower()
    for item in root_dir.iterdir():
        if item.is_dir() and item.name.lower() == split_name_lower:
            return item

    raise FileNotFoundError(
        f"Could not find split directory '{split_name}' under: {root_dir}"
    )


def get_class_names_from_directory(split_dir: Path) -> List[str]:
    #Read class names from subfolder names and return them sorted.
    class_names = sorted([item.name for item in split_dir.iterdir() if item.is_dir()])

    if not class_names:
        raise ValueError(f"No class folders found in split directory: {split_dir}")

    return class_names


class SkinDiseaseDataset(Dataset):
    #PyTorch Dataset for folder-based skin disease classification.
    def __init__(
        self,
        split_dir: Path,
        class_names: Optional[List[str]] = None,
        transform=None,
        return_paths: bool = False,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.return_paths = return_paths

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        if class_names is None:
            self.class_names = get_class_names_from_directory(self.split_dir)
        else:
            self.class_names = list(class_names)

        self.class_to_idx: Dict[str, int] = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: class_name for class_name, idx in self.class_to_idx.items()
        }

        self.samples: List[Tuple[Path, int]] = self._build_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found in: {self.split_dir}")

    def _build_samples(self) -> List[Tuple[Path, int]]:
        #build a list of (image_path, label_idx) samples from class folders
        
        samples: List[Tuple[Path, int]] = []

        for class_name in self.class_names:
            class_dir = self.split_dir / class_name

            if not class_dir.exists() or not class_dir.is_dir():
                #skip missing class folders silently so long as mapping stays consistent
                continue

            for file_path in sorted(class_dir.rglob("*")):
                if file_path.is_file() and is_image_file(file_path):
                    samples.append((file_path, self.class_to_idx[class_name]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        if self.return_paths:
            return image, label, str(image_path)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        #Return sample count for each class in this split.
        
        distribution = {class_name: 0 for class_name in self.class_names}

        for _, label_idx in self.samples:
            class_name = self.idx_to_class[label_idx]
            distribution[class_name] += 1

        return distribution

    def get_num_classes(self) -> int:
        #Return number of classes.
        return len(self.class_names)

    def get_class_names(self) -> List[str]:
        #Return class names in label-index order.
        return self.class_names

    def get_samples(self) -> List[Tuple[Path, int]]:
        #Return raw sample list.
        return self.samples
    

def build_skin_disease_datasets(
    dataset_root: Path,
    train_transform=None,
    test_transform=None,
    return_paths: bool = False,
):
    #Build train and test datasets from the dataset root directory.
    dataset_root = Path(dataset_root)

    train_dir = find_split_dir(dataset_root, "train")
    test_dir = find_split_dir(dataset_root, "test")

    class_names = get_class_names_from_directory(train_dir)

    train_dataset = SkinDiseaseDataset(
        split_dir=train_dir,
        class_names=class_names,
        transform=train_transform,
        return_paths=return_paths,
    )

    test_dataset = SkinDiseaseDataset(
        split_dir=test_dir,
        class_names=class_names,
        transform=test_transform,
        return_paths=return_paths,
    )

    return train_dataset, test_dataset

if __name__ == "__main__":
    from config import config

    train_dataset, test_dataset = build_skin_disease_datasets(
        dataset_root=config.dataset_root,
        train_transform=None,
        test_transform=None,
        return_paths=True,
    )

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Classes:", train_dataset.get_class_names())
    print("Train distribution:", train_dataset.get_class_distribution())
    print("Test distribution:", test_dataset.get_class_distribution())

    sample = train_dataset[0]
    print("First sample path:", sample[2])
    print("First sample label:", sample[1])