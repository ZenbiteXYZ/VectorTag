import numpy as np
from torch.utils.data import DataLoader, Subset
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
    num_workers: int = 4,
    stratify: bool = True
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
        stratify: Whether to use stratified split (ensures balanced distribution)

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

    # 2. Create Full Dataset (for determining classes)
    print("Initializing Dataset...")
    full_dataset = TaggedImagesDataset(
        root_dir=data_dir,
        csv_file_path=csv_file,
        transform=train_transform,
        top_k=top_k,
        filter_to_top=filter_to_top,
        max_samples=max_samples,
    )

    classes = full_dataset.classes
    print(f"Selected classes ({len(classes)}): {classes}")

    # 3. Stratified Split
    if stratify:
        print("Applying stratified split...")
        train_indices, val_indices = _stratified_split(full_dataset, test_size=0.2, classes=classes)
        print(f"Stratified split: {len(train_indices)} train, {len(val_indices)} val")
    else:
        print("Applying random split...")
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)

        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]


    # 4. Create Datasets with appropriate transforms
    # Training subset with augmentation
    train_dataset_augmented = TaggedImagesDataset(
        root_dir=data_dir,
        csv_file_path=csv_file,
        transform=train_transform,
        classes=classes,
        filter_to_top=filter_to_top,
        max_samples=max_samples,
    )
    train_dataset = Subset(train_dataset_augmented, train_indices)

    # Validation subset without augmentation
    val_dataset_clean = TaggedImagesDataset(
        root_dir=data_dir,
        csv_file_path=csv_file,
        transform=val_transform,
        classes=classes,
        filter_to_top=filter_to_top,
        max_samples=max_samples,
    )
    val_dataset = Subset(val_dataset_clean, val_indices)

    # 5. Create DataLoaders
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


def _stratified_split(
    dataset: TaggedImagesDataset,
    test_size: float = 0.2,
    classes: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split for Multi-Label dataset.

    Strategy: For each image, find the rarest tag and use it for stratification.
    This ensures that rare classes are equally distributed between train and val.
    """
    from sklearn.model_selection import train_test_split
    from collections import Counter

    # Count class frequencies
    all_tags = []
    for tags in dataset.df['parsed_tags']:
        all_tags.extend(tags)

    counts = Counter(all_tags)

    # For each sample, find the rarest tag
    stratify_labels = []
    for tags in dataset.df['parsed_tags']:
        relevant_tags = [t for t in tags if t in classes]

        if not relevant_tags:
            # Fallback: if no relevant tags, use 'unknown'
            stratify_labels.append('unknown')
        else:
            # Use the rarest tag in this image as stratification key
            rarest_tag = min(relevant_tags, key=lambda t: counts[t])
            stratify_labels.append(rarest_tag)

    stratify_labels = np.array(stratify_labels)

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_labels,
        random_state=42
    )
    return train_idx, val_idx
