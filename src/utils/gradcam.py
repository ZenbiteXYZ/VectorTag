import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor):
        self.activations = output

    def save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, class_idx: int):
        # 1. Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 2. Backward pass for specific class
        target = output[0][class_idx]
        target.backward()

        # 3. Generate CAM
        gradients = self.gradients[0] # [C, H, W]
        activations = self.activations[0] # [C, H, W]

        # Global Average Pooling of gradients (weights)
        weights = torch.mean(gradients, dim=(1, 2)) # [C]

        # Weighted sum of activations
        # (weights * activations)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = F.relu(cam)

        # Normalize 0..1
        max_val = cam.max()
        if max_val > 1e-7:
            cam = cam / max_val

        return cam.detach().cpu().numpy()

def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """Overlays a heatmap onto the original image."""
    # Resize heatmap to image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(original_image.size, resample=Image.BICUBIC)

    # Apply colormap (Jet is standard for heatmaps)
    try:
        colormap = cm.get_cmap("jet")
    except ValueError:
        # Fallback for newer matplotlib versions if 'jet' is deprecated/moved
        colormap = cm.jet

    # Get RGB values from colormap
    heatmap_data = np.array(heatmap_img) / 255.0
    heatmap_colored = colormap(heatmap_data) # Returns RGBA (H, W, 4)

    # Create RGB image for the heatmap
    heatmap_rgb = Image.fromarray(np.uint8(255 * heatmap_colored[:, :, :3])).convert("RGB")

    # Create Alpha Mask
    # We want low heatmap values to be transparent, high values to be opaque
    # Using gamma correction (power < 1) to make mid-tones more visible
    alpha_data = (heatmap_data ** 0.6) * 255
    alpha_data = np.clip(alpha_data * 0.7, 0, 255).astype(np.uint8) # Cap opacity at 70%
    mask = Image.fromarray(alpha_data, mode='L')

    # Composite: Paste heatmap onto original using the mask
    result = original_image.copy().convert("RGB")
    result.paste(heatmap_rgb, (0, 0), mask)

    return result
