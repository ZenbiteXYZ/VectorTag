import os
import sys
import json
import torch
import argparse
from typing import Tuple, List
from torch import nn
from PIL import Image
from torchvision import transforms

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.core.config import settings
from src.models.baseline import BaselineModel


def load_model_and_classes():
    # 1. Load Classes
    if not os.path.exists(settings.CLASSES_PATH):
        raise FileNotFoundError(f"Classes file not found: {settings.CLASSES_PATH}")

    with open(settings.CLASSES_PATH, "r") as f:
        classes = json.load(f)

    print(f"Loaded {len(classes)} classes.")

    # 2. Initialize Model
    model = BaselineModel(num_classes=len(classes))

    # 3. Load Weights
    if not os.path.exists(settings.WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {settings.WEIGHTS_PATH}")

    # map_location ensures it loads on CPU if CUDA is not available
    state_dict = torch.load(settings.WEIGHTS_PATH, map_location=settings.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(settings.DEVICE)
    model.eval()

    print("Model loaded successfully.")
    return model, classes


def predict_image(model: nn.Module, classes: List[str], image_path: str, threshold=0.5):
    """
    Predicts tags for a single image.

    Args:
        model (nn.Module): Model to use.
        classes (List[str]): Classes which will be predicted.
        image_path (str): Path to image to get tags from.
        threshold (float): Minimal possibility to show tag.
    """
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Prepare input
    input_tensor = transform(image).unsqueeze(0).to(settings.DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0]

    # Format results
    results: List[Tuple[str]] = []
    for i, prob in enumerate(probabilities):
        if prob >= threshold:
            results.append((classes[i], prob.item()))

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Image: {os.path.basename(image_path)}")
    print("-" * 30)
    if not results:
        print("No tags found.")
    else:
        for tag, prob in results:
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"{tag:15} {bar} {prob:.1%}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict tags for an image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold (0.0-1.0)")

    args = parser.parse_args()

    try:
        model, classes = load_model_and_classes()
        predict_image(model, classes, args.image_path, args.threshold)
    except Exception as e:
        print(f"Error: {e}")
