import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.core.config import settings
from src.models.baseline import BaselineModel
from src.utils.gradcam import GradCAM, overlay_heatmap
from torchvision import transforms

# --- CONFIGURATION ---
TEST_IMAGE_PATH = "/home/zenbitex/Pictures/raspberry.jpg"
TARGET_TAG = "food"

MODELS_TO_COMPARE = {
    "Exp 002 (BCE)": "baseline_v2.pth",
    "Exp 004 (Focal)": "baseline_v4.pth",
    "Exp 005: (BCE + pos_weight)": "baseline_v5.pth"
    }
# ---------------------

def load_model_and_classes(weights_filename):
    """Loads model and its specific classes.json"""
    weights_path = settings.STANDARD_WEIGHTS_DIR / weights_filename
    classes_filename = weights_filename.replace(".pth", ".json")
    classes_path = settings.STANDARD_CLASSES_DIR / classes_filename

    if not weights_path.exists() or not classes_path.exists():
        print(f"Skipping {weights_filename}: File not found.")
        return None, None

    with open(classes_path, "r") as f:
        classes = json.load(f)

    model = BaselineModel(num_classes=len(classes))
    state_dict = torch.load(weights_path, map_location=settings.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(settings.DEVICE)
    model.eval()

    return model, classes

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Image not found at {TEST_IMAGE_PATH}")
        return

    # Prepare Image
    original_image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(original_image).unsqueeze(0).to(settings.DEVICE)

    # Setup Plot
    num_models = len(MODELS_TO_COMPARE)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))

    fig.suptitle(f"GradCAM Comparison for Tag: '{TARGET_TAG}'", fontsize=16)

    # Plot Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Process Models
    for i, (model_name, weights_file) in enumerate(MODELS_TO_COMPARE.items()):
        print(f"Processing {model_name}...")
        model, classes = load_model_and_classes(weights_file)

        ax = axes[i + 1]

        if model is None:
            ax.text(0.5, 0.5, "Model Not Found", ha='center')
            ax.axis("off")
            continue

        if TARGET_TAG not in classes:
            ax.text(0.5, 0.5, f"Tag '{TARGET_TAG}'\nnot in classes", ha='center')
            ax.axis("off")
            continue

        # GradCAM
        target_layer = model.backbone.layer4[-1]
        gradcam = GradCAM(model, target_layer)

        class_idx = classes.index(TARGET_TAG)

        # Get prediction score for this tag
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits)[0][class_idx].item()

        # Generate Heatmap
        with torch.enable_grad():
            heatmap = gradcam(input_tensor, class_idx)

        overlay = overlay_heatmap(original_image, heatmap)

        ax.imshow(overlay)
        ax.set_title(f"{model_name}\nConf: {prob:.1%}")
        ax.axis("off")

    plt.tight_layout()
    output_file = f"comparison_{TARGET_TAG}.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved comparison to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
