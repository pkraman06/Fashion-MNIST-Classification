"""
explainability.py
Grad-CAM and vanilla-gradient saliency maps for the trained model.
Used both by evaluate-style scripts and by app.py for the live Gradio demo.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

import config


class GradCAM:
    """
    Grad-CAM using the final residual stage (stage4) feature map.
    Usage:
        cam = GradCAM(model)
        heatmap, pred_class, probs = cam.generate(input_tensor)
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: (1, 3, H, W) already normalized, on config.DEVICE.
        Returns: heatmap (H, W) in [0,1], predicted class idx, softmax probs.
        """
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor, register_hooks_for_cam=True)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        feature_map = self.model.last_conv_features  # (1, C, h, w)
        grads = feature_map.grad  # (1, C, h, w)

        # global-average-pool the gradients -> per-channel importance weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * feature_map).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        return cam, target_class, probs.detach().cpu().numpy()[0]


def vanilla_saliency(model, input_tensor, target_class=None):
    """
    Simple gradient-based saliency map: |d(logit)/d(input)|, max over channels.
    Returns saliency map (H, W) normalized to [0,1] and predicted class idx.
    """
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)

    logits = model(input_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    grad = input_tensor.grad.abs().squeeze(0)  # (3, H, W)
    saliency, _ = torch.max(grad, dim=0)  # (H, W)
    saliency = saliency.cpu().numpy()

    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency, target_class


def overlay_heatmap(rgb_image_uint8, heatmap, alpha=0.4):
    """
    rgb_image_uint8: (H, W, 3) uint8 array in [0,255]
    heatmap: (H, W) float array in [0,1]
    Returns: (H, W, 3) uint8 overlay image.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    overlay = (alpha * color_map + (1 - alpha) * rgb_image_uint8).astype(np.uint8)
    return overlay
