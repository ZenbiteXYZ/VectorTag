from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from typing import Tuple, List, Optional

from src.data.tagged_dataset import TaggedImagesDataset

def get_dataloaders(
    data_dir: str,
    csv_file: str,
    batch_size: int = 32,
    top_k: int = 30,
    filter_to_top: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates training and validation DataLoaders with appropriate transforms.

    Args:
        data_dir: Path to image directory
        csv_file: Path to metadata CSV
        batch_size: Batch size for loaders
        top_k: Number of top classes to use
        filter_to_top: Whether to filter dataset to only top classes
        max_samples: Maximum number of samples to use (for debugging)
        num_workers: Number of worker threads for loading

    Returns:
        Tuple of (train_loader, val_loader, classes_list)
    """

    # 1. Define Transforms
    # Strong augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Standard preprocessing for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Create Datasets
    print("Initializing Training Dataset...")
    # Create train dataset first to determine classes
    full_train_dataset = TaggedImagesDataset(
        root_dir=data_dir,
        csv_file_path=csv_file,
        transform=train_transform,
        top_k=top_k,
        filter_to_top=filter_to_top,
        max_samples=max_samples,
    )

    classes = full_train_dataset.classes
    print(f"Selected classes ({len(classes)}): {classes}")

    print("Initializing Validation Dataset...")
    # Create validation dataset using the SAME classes
    full_val_dataset = TaggedImagesDataset(
        root_dir=data_dir,
        csv_file_path=csv_file,
        transform=val_transform,
        classes=classes,  # Force same classes
        filter_to_top=filter_to_top,
        max_samples=max_samples,
    )

    # 3. Split Indices
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Generate random split indices
    dummy_train, dummy_val = random_split(range(dataset_size), [train_size, val_size])

    # Create Subsets pointing to appropriate datasets (augmented vs clean)
    train_dataset = Subset(full_train_dataset, dummy_train.indices)
    val_dataset = Subset(full_val_dataset, dummy_val.indices)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, classes
