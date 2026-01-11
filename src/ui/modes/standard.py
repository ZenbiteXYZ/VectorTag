import torch
import json
from PIL import Image
from pathlib import Path
from torch import nn
from torchvision import transforms
from safetensors.torch import load_file
from typing import List, Tuple, Dict, Optional

from src.core.config import settings
from src.ui.modes.base import InferenceMode
from src.models.baseline import BaselineModel
from src.utils.gradcam import GradCAM, overlay_heatmap


class StandardMode(InferenceMode):
    name = "Standard Multi-Label"
    description = "Uses a fixed ResNet-18 model trained on top-K tags."

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.current_model_key: Optional[str] = None
        self.classes: List[str] = []
        self.gradcam: Optional[GradCAM] = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_available_models(self) -> Dict[str, str]:
        weights_dir = settings.STANDARD_WEIGHTS_DIR
        classes_dir = settings.STANDARD_CLASSES_DIR
        available: Dict[str, str] = {}

        if not weights_dir.exists():
            return {}

        for weights_file in sorted(weights_dir.glob("baseline_v*.safetensors")):
            model_name = weights_file.stem
            classes_file = classes_dir / f"{model_name}.json"

            if classes_file.exists():
                version = model_name.replace("baseline_v", "")
                display_name = f"Version {version}"
                available[display_name] = model_name

        return available

    def set_model(self, model_key: str):
        """Sets the active model by its key."""
        weights_path = settings.STANDARD_WEIGHTS_DIR / f"{model_key}.safetensors"
        classes_path = settings.STANDARD_CLASSES_DIR / f"{model_key}.json"

        if not weights_path.exists():
            raise FileNotFoundError(f"Model file not found: {weights_path}")
        if not classes_path.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_path}")

        # Reset current model
        self.model = None
        self.classes = []
        self.current_model_key = model_key

        # Load model and classes
        self._load_model(weights_path, classes_path)

    def _load_model(self, weights_path: Path, classes_path: Path):
        """Internal method to load the model."""
        with open(classes_path, "r") as f:
            self.classes = json.load(f)

        self.model = BaselineModel(num_classes=len(self.classes))
        state_dict = load_file(weights_path, device=str(settings.DEVICE))
        self.model.load_state_dict(state_dict)
        self.model.to(settings.DEVICE)
        self.model.eval()

        # Init GradCAM on the last ResNet block
        target_layer = self.model.backbone.layer4[-1]
        self.gradcam = GradCAM(self.model, target_layer)

    def predict(self, image: Image.Image, threshold: float) -> List[Tuple[str, float]]:
        if self.model is None:
            self.load_resources()

        # Prepare input
        input_tensor = self.transform(image).unsqueeze(0).to(settings.DEVICE)

        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.sigmoid(logits)[0]

        # Filter results
        results = []
        for i, prob in enumerate(probabilities):
            if prob >= threshold:
                results.append((self.classes[i], prob.item()))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_gradcam_image(self, image: Image.Image, tag_name: str) -> Image.Image:
        """Generates an image with Grad-CAM overlay for a specific tag."""
        if self.model is None or self.gradcam is None:
            self.load_resources()

        if tag_name not in self.classes:
            return image

        class_idx = self.classes.index(tag_name)
        input_tensor = self.transform(image).unsqueeze(0).to(settings.DEVICE)

        # Generate heatmap
        # Note: We need gradients, so we enable grad temporarily even in inference mode
        with torch.enable_grad():
            heatmap = self.gradcam(input_tensor, class_idx)

        return overlay_heatmap(image, heatmap)
