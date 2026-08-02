"""
config.py
Central configuration for the Brain Tumor MRI classification project.
"""

import os
import torch

# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------
# >>> ADD YOUR KAGGLE DATASET LINK HERE <<<
# Recommended dataset: "Brain Tumor MRI Dataset" by Masoud Nickparvar
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
#
# To download via Kaggle CLI (after `pip install kaggle` and placing your
# kaggle.json API token in ~/.kaggle/kaggle.json):
#
#   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
#
KAGGLE_DATASET_URL = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
KAGGLE_DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"

# Expected folder layout after download/extraction (ImageFolder style):
# data/
#   Training/
#     glioma/
#     meningioma/
#     notumor/
#     pituitary/
#   Testing/
#     glioma/
#     meningioma/
#     notumor/
#     pituitary/
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ------------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_SPLIT = 0.15          # fraction of Training/ held out for validation

EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

# LR scheduler
SCHEDULER_T0 = 10          # CosineAnnealingWarmRestarts initial period
SCHEDULER_TMULT = 2

# Mixed precision
USE_AMP = True

# Checkpointing (also used later for "probing across training checkpoints")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_EVERY_N_EPOCHS = 3     # save a checkpoint every N epochs for probing
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
BASE_CHANNELS = 32
SE_REDUCTION = 16
STAGE_BLOCKS = [2, 2, 2, 2]     # residual SE blocks per stage (4 stages -> 4 "depths" to probe)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
