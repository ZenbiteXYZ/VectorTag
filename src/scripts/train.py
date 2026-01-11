import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from safetensors.torch import save_file

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.core.config import settings
from src.models.baseline import BaselineModel
from src.data.loaders import get_dataloaders
from src.utils.plotting import plot_training_history


def main():
    print(f"Using device: {settings.DEVICE}")

    # Check paths
    if not os.path.exists(settings.DATA_DIR) or not os.path.exists(settings.CSV_FILE):
        print(f"Error: Dataset not found.")
        print(f"Looking for images in: {os.path.abspath(settings.DATA_DIR)}")
        print(f"Looking for csv in: {os.path.abspath(settings.CSV_FILE)}")
        return

    # 1. Prepare DataLoaders
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=settings.DATA_DIR,
        csv_file=settings.CSV_FILE,
        top_k=settings.TOP_K,
        filter_to_top=settings.FILTER_TO_TOP,
        max_samples=settings.MAX_SAMPLES,
        num_workers=settings.NUM_WORKERS,
        stratify=True
    )

    # 2. Initialize Model
    model = BaselineModel(num_classes=len(classes)).to(settings.DEVICE)

    # 3. Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.1, patience=2
    )

    best_val_loss = float('inf')
    weights_save_path = os.path.join(settings.STANDARD_WEIGHTS_DIR, f"{settings.DEFAULT_STANDARD_MODEL}.safetensors")
    classes_save_path = os.path.join(settings.STANDARD_CLASSES_DIR, f"{settings.DEFAULT_STANDARD_MODEL}.json")

    # Initialize history
    history = {'epochs': [], 'train_loss': [], 'val_loss': []}

    print("Starting training...")
    for epoch in range(settings.EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{settings.EPOCHS}")

        for images, labels in progress_bar:
            images = images.to(settings.DEVICE)
            labels = labels.to(settings.DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)

        # Validation loop (simple)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(settings.DEVICE)
                labels = labels.to(settings.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{settings.EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # Update history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Move state_dict to CPU for safetensors saving
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, weights_save_path)
            with open(classes_save_path, "w") as f:
                json.dump(classes, f, indent=2)
            print(f"  >>> New best model saved! (Loss: {val_loss:.4f})")

    # Save classes
    with open(classes_save_path, "w") as f:
        json.dump(classes, f)

    # Plot results
    assets_dir = os.path.join(project_root, "assets")
    plot_training_history(history, settings.DEFAULT_STANDARD_MODEL, save_dir=assets_dir)

if __name__ == "__main__":
    main()
