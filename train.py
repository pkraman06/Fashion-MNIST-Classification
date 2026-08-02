"""
train.py
Trains the ResidualSE-CNN on the Brain Tumor MRI dataset with:
  - mixed-precision training (torch.cuda.amp)
  - cosine-annealing-with-warm-restarts LR scheduling
  - best-checkpoint saving (by validation accuracy)
  - periodic checkpoint saving (epoch_N.pt) used later by probe.py to
    study representation quality over training time.
"""

import os
import json
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import config
from model import build_model
from dataset import get_dataloaders
from utils import set_seed, AverageMeter, accuracy_from_logits, plot_training_curves


def evaluate(model, loader, criterion):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            with autocast(enabled=config.USE_AMP):
                logits = model(images)
                loss = criterion(logits, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy_from_logits(logits, labels), images.size(0))
    return loss_meter.avg, acc_meter.avg


def train():
    set_seed(config.SEED)
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)

    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.SCHEDULER_T0, T_mult=config.SCHEDULER_TMULT
    )
    scaler = GradScaler(enabled=config.USE_AMP)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config.USE_AMP):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy_from_logits(logits, labels), images.size(0))

        scheduler.step(epoch)

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history["train_loss"].append(loss_meter.avg)
        history["train_acc"].append(acc_meter.avg)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{config.EPOCHS} | "
              f"train_loss {loss_meter.avg:.4f} train_acc {acc_meter.avg:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
              f"lr {lr_now:.2e} | {elapsed:.1f}s")

        # Best-checkpoint saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
            }, config.BEST_MODEL_PATH)
            print(f"  -> new best model saved (val_acc={val_acc:.4f})")

        # Periodic checkpoint saving (for training-dynamics probing)
        if epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0 or epoch == 1 or epoch == config.EPOCHS:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
            }, ckpt_path)

    # Final test evaluation using the best checkpoint
    best_ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinal test accuracy (best checkpoint, epoch {best_ckpt['epoch']}): {test_acc*100:.2f}%")

    plot_training_curves(history, os.path.join(config.RESULTS_DIR, "training_curves.png"))
    with open(os.path.join(config.RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
            "epochs_trained": config.EPOCHS,
            "classes": classes,
        }, f, indent=2)

    print(f"Training curves + history saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    train()
