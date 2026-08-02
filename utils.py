"""
utils.py
Small shared helpers used across training, probing, evaluation, and the app.
"""

import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Tracks running average of a metric (loss, accuracy, etc.)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def plot_training_curves(history, save_path):
    """history: dict with keys train_loss, val_loss, train_acc, val_acc (lists)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_probe_curve(x_values, accuracies, xlabel, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x_values, accuracies, marker="o", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def denormalize_image(tensor_img, mean, std):
    """tensor_img: (C,H,W) normalized tensor -> numpy HWC in [0,1]."""
    img = tensor_img.clone().detach().cpu().numpy()
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))
