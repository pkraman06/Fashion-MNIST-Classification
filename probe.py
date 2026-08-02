"""
probe.py
Two linear-probing analyses:

1. Probing across LAYERS (depth): freeze the best trained model, extract
   global-average-pooled features at each of the 4 residual stages, and fit
   a logistic regression probe per stage. Accuracy should rise with depth
   as representations become more separable.

2. Probing across TRAINING CHECKPOINTS: for each saved epoch checkpoint,
   extract the penultimate embedding and fit a logistic regression probe.
   Accuracy over epochs quantifies the shift from memorization to
   increasingly abstract, linearly-separable representations.

Both analyses use scikit-learn's LogisticRegression as the linear probe,
trained on the (fixed, frozen) train-split embeddings and evaluated on the
test-split embeddings.
"""

import os
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config
from model import build_model
from dataset import get_dataloaders
from utils import plot_probe_curve


@torch.no_grad()
def extract_stage_embeddings(model, loader):
    """Returns dict stage_name -> (N, D) numpy array, plus labels (N,)."""
    model.eval()
    stage_feats = {"stage1": [], "stage2": [], "stage3": [], "stage4": []}
    labels_all = []

    for images, labels in loader:
        images = images.to(config.DEVICE)
        feats = model.forward_features(images)
        for stage_name, fmap in feats.items():
            pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1).cpu().numpy()
            stage_feats[stage_name].append(pooled)
        labels_all.append(labels.numpy())

    for k in stage_feats:
        stage_feats[k] = np.concatenate(stage_feats[k], axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return stage_feats, labels_all


@torch.no_grad()
def extract_penultimate_embeddings(model, loader):
    model.eval()
    embs, labels_all = [], []
    for images, labels in loader:
        images = images.to(config.DEVICE)
        pooled, _ = model.get_embedding(images)
        embs.append(pooled.cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels_all, axis=0)


def fit_linear_probe(train_X, train_y, test_X, test_y):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    clf.fit(train_X, train_y)
    return clf.score(test_X, test_y)


def probe_across_layers():
    """Analysis 1: accuracy vs. network depth on the best trained model."""
    train_loader, _, test_loader, classes = get_dataloaders()

    model = build_model()
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    train_feats, train_y = extract_stage_embeddings(model, train_loader)
    test_feats, test_y = extract_stage_embeddings(model, test_loader)

    stage_names = ["stage1", "stage2", "stage3", "stage4"]
    accuracies = []
    for s in stage_names:
        acc = fit_linear_probe(train_feats[s], train_y, test_feats[s], test_y)
        accuracies.append(acc)
        print(f"[Layer probe] {s}: {acc*100:.2f}%")

    plot_probe_curve(
        stage_names, accuracies,
        xlabel="Network depth (stage)",
        title="Linear Probe Accuracy vs. Depth",
        save_path=os.path.join(config.RESULTS_DIR, "probe_by_layer.png"),
    )
    with open(os.path.join(config.RESULTS_DIR, "probe_by_layer.json"), "w") as f:
        json.dump({s: a for s, a in zip(stage_names, accuracies)}, f, indent=2)

    return stage_names, accuracies


def probe_across_checkpoints():
    """Analysis 2: accuracy vs. training epoch (memorization -> abstraction)."""
    train_loader, _, test_loader, classes = get_dataloaders()

    ckpt_paths = sorted(glob.glob(os.path.join(config.CHECKPOINT_DIR, "epoch_*.pt")))
    if not ckpt_paths:
        raise RuntimeError(
            "No periodic checkpoints found in checkpoints/. "
            "Run train.py first (it saves epoch_XXX.pt checkpoints)."
        )

    epochs, accuracies = [], []
    model = build_model()

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        train_X, train_y = extract_penultimate_embeddings(model, train_loader)
        test_X, test_y = extract_penultimate_embeddings(model, test_loader)
        acc = fit_linear_probe(train_X, train_y, test_X, test_y)

        epoch_num = ckpt["epoch"]
        epochs.append(epoch_num)
        accuracies.append(acc)
        print(f"[Checkpoint probe] epoch {epoch_num}: {acc*100:.2f}%")

    plot_probe_curve(
        epochs, accuracies,
        xlabel="Training epoch",
        title="Linear Probe Accuracy vs. Training Time",
        save_path=os.path.join(config.RESULTS_DIR, "probe_by_checkpoint.png"),
    )
    with open(os.path.join(config.RESULTS_DIR, "probe_by_checkpoint.json"), "w") as f:
        json.dump({"epochs": epochs, "accuracies": accuracies}, f, indent=2)

    return epochs, accuracies


if __name__ == "__main__":
    print("Running layer-depth probing...")
    probe_across_layers()
    print("\nRunning training-checkpoint probing...")
    probe_across_checkpoints()
