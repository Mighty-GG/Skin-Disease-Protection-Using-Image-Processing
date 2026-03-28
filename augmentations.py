# augmentations.py

from torchvision import transforms


def get_train_transforms(image_size: int = 224):
    #Return training transforms with data augmentation.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transforms(image_size: int = 224):
    #return validation/test transforms without augmentation.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_light_train_transforms(image_size: int = 224):
    #a lighter augmentation pipeline. Useful if stronger augmentation hurts performance.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


if __name__ == "__main__":
    train_tf = get_train_transforms(224)
    eval_tf = get_eval_transforms(224)

    print("Train transforms:")
    print(train_tf)

    print("\nEvaluation transforms:")
    print(eval_tf)