"""
evaluate.py
Loads the best checkpoint, computes test accuracy, plots a confusion matrix,
and reports the single most-confused class pair (excluding the diagonal).
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import config
from model import build_model
from dataset import get_dataloaders


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(config.DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def top_misclassified_pair(cm, class_names):
    """Finds the off-diagonal cell with the highest count."""
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    idx = np.unravel_index(np.argmax(cm_off_diag), cm_off_diag.shape)
    true_cls, pred_cls = class_names[idx[0]], class_names[idx[1]]
    count = int(cm_off_diag[idx])
    return true_cls, pred_cls, count


def run_evaluation():
    _, _, test_loader, classes = get_dataloaders()
    model = build_model()
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred = get_predictions(model, test_loader)
    test_acc = (y_true == y_pred).mean()
    print(f"Test accuracy: {test_acc*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, os.path.join(config.RESULTS_DIR, "confusion_matrix.png"))

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    with open(os.path.join(config.RESULTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    true_cls, pred_cls, count = top_misclassified_pair(cm, classes)
    print(f"Top misclassification: true='{true_cls}' predicted as '{pred_cls}' ({count} cases)")

    with open(os.path.join(config.RESULTS_DIR, "top_confusion_pair.json"), "w") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "true_class": true_cls,
            "predicted_class": pred_cls,
            "count": count,
        }, f, indent=2)

    return test_acc, cm, (true_cls, pred_cls, count)


if __name__ == "__main__":
    run_evaluation()
