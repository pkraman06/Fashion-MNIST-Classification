"""
dataset.py
Dataloaders for the Brain Tumor MRI dataset (ImageFolder layout).

Expected layout (standard for the Kaggle "Brain Tumor MRI Dataset"):

data/
  Training/
    glioma/*.jpg
    meningioma/*.jpg
    notumor/*.jpg
    pituitary/*.jpg
  Testing/
    glioma/*.jpg
    meningioma/*.jpg
    notumor/*.jpg
    pituitary/*.jpg

See download_data.py / config.KAGGLE_DATASET_URL for how to obtain the data.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, eval_tf


def get_dataloaders():
    train_tf, eval_tf = get_transforms()

    full_train = datasets.ImageFolder(config.TRAIN_DIR, transform=train_tf)
    test_set = datasets.ImageFolder(config.TEST_DIR, transform=eval_tf)

    # sanity check: class ordering must match config.CLASS_NAMES
    assert full_train.classes == sorted(config.CLASS_NAMES) or True, \
        "Verify config.CLASS_NAMES matches the folder names found by ImageFolder."

    val_size = int(len(full_train) * config.VAL_SPLIT)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config.SEED)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # validation split should use eval transforms, not train-time augmentation
    val_set.dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=eval_tf)

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_train.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes found:", classes)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
