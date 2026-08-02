# Brain Tumor MRI Classification with Explainability & Training Dynamics

A residual CNN with Squeeze-and-Excitation (SE) attention for multi-class
brain tumor classification from MRI scans (glioma, meningioma, pituitary,
no tumor), with representation-learning analysis across network depth and
training time, plus an interactive Grad-CAM / saliency-map explainability
demo.

## Dataset

> **Add your Kaggle dataset link here:**
> `config.py -> KAGGLE_DATASET_URL` and `KAGGLE_DATASET_SLUG`

Recommended dataset: *Brain Tumor MRI Dataset* (Masoud Nickparvar)
`https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`

Download it either manually (unzip into `data/`) or via:

```bash
pip install kaggle
# place kaggle.json in ~/.kaggle/kaggle.json
python download_data.py
```

Expected folder layout:

```
data/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

## Project structure

```
config.py            Central configuration (paths, hyperparameters, dataset link)
utils.py              Seeding, metric meters, plotting helpers
model.py              Residual CNN with SE-attention blocks
dataset.py            Dataloaders / transforms
download_data.py       Kaggle dataset download helper
train.py               Training loop (AMP, LR scheduling, checkpointing)
evaluate.py            Test accuracy, confusion matrix, top misclassified pair
probe.py                Linear probing across depth and training checkpoints
explainability.py       Grad-CAM and saliency map implementations
app.py                  Gradio demo (prediction + explainability + dashboards)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Get the data (see Dataset section above)
python download_data.py

# 2. Train the model
python train.py

# 3. Evaluate on the test set (confusion matrix, misclassification analysis)
python evaluate.py

# 4. Run linear probing (across depth and across training checkpoints)
python probe.py

# 5. Launch the interactive Gradio demo
python app.py
```

`train.py` saves:
- `checkpoints/best_model.pt` — best model by validation accuracy
- `checkpoints/epoch_XXX.pt` — periodic checkpoints used by `probe.py`
- `results/training_curves.png`, `results/history.json`, `results/summary.json`

`evaluate.py` saves:
- `results/confusion_matrix.png`
- `results/classification_report.json`
- `results/top_confusion_pair.json`

`probe.py` saves:
- `results/probe_by_layer.png` / `.json` — accuracy vs. network depth
- `results/probe_by_checkpoint.png` / `.json` — accuracy vs. training epoch

`app.py` reads all of the above and displays them in a "Training Dynamics"
and "Confusion Matrix" tab, alongside a live "Predict & Explain" tab.

## Method summary

- **Architecture:** stem conv + 4 stages of residual blocks, each with a
  Squeeze-and-Excitation channel-attention module, global average pooling,
  and a linear classification head.
- **Training:** AdamW, cosine-annealing-with-warm-restarts LR schedule,
  mixed-precision (`torch.cuda.amp`), label smoothing, best-checkpoint
  saving by validation accuracy.
- **Layer-depth probing:** freeze the trained model, pool features from
  each of the 4 stages, fit an independent scikit-learn logistic-regression
  probe per stage on train embeddings, evaluate on test embeddings.
- **Training-time probing:** repeat the same linear-probe procedure using
  the penultimate embedding from each periodically saved checkpoint, to
  trace how separability evolves over training.
- **Explainability:** Grad-CAM on the final residual stage's feature map,
  and vanilla-gradient saliency maps, both exposed live in the Gradio app.

## Tech

PyTorch, Scikit-learn, Matplotlib, OpenCV, Gradio
