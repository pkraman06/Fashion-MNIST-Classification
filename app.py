"""
app.py
Interactive Gradio demo for the Brain Tumor MRI classifier.

Tabs:
  1. Predict & Explain  - upload an MRI slice, get class probabilities,
                          Grad-CAM overlay, and a saliency map.
  2. Training Dynamics  - shows saved training curves and linear-probe
                          results (accuracy vs. depth, accuracy vs. epoch).
  3. Confusion Matrix    - shows the test-set confusion matrix and the
                          top misclassified class pair.

Run with:  python app.py
"""

import os
import json

import numpy as np
import torch
import gradio as gr
from PIL import Image

import config
from model import build_model
from dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD
from explainability import GradCAM, vanilla_saliency, overlay_heatmap

# ------------------------------------------------------------------
# Load model once at startup
# ------------------------------------------------------------------
_, EVAL_TRANSFORM = get_transforms()

model = build_model()
_model_loaded = False
if os.path.exists(config.BEST_MODEL_PATH):
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    CLASS_NAMES = ckpt.get("classes", config.CLASS_NAMES)
    _model_loaded = True
else:
    CLASS_NAMES = config.CLASS_NAMES
    print(f"WARNING: no checkpoint found at {config.BEST_MODEL_PATH}. "
          f"Run train.py first. The app will still launch but predictions "
          f"will use an untrained model.")

model.eval()
grad_cam = GradCAM(model)


def predict_and_explain(pil_image):
    if pil_image is None:
        return None, None, None, "Please upload an MRI image."

    pil_image = pil_image.convert("RGB")
    input_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(config.DEVICE)

    # Grad-CAM (also gives us predicted class + probabilities)
    cam_heatmap, pred_idx, probs = grad_cam.generate(input_tensor.clone())

    # Vanilla saliency map
    saliency_map, _ = vanilla_saliency(model, input_tensor.clone(), target_class=pred_idx)

    # Build a display-ready RGB base image (resized, de-normalized not needed
    # since we resize the *original* PIL image directly for visualization)
    display_img = pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    display_np = np.array(display_img).astype(np.uint8)

    gradcam_overlay = overlay_heatmap(display_np, cam_heatmap, alpha=0.45)
    saliency_overlay = overlay_heatmap(display_np, saliency_map, alpha=0.45)

    pred_class = CLASS_NAMES[pred_idx]
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    summary = f"### Prediction: **{pred_class}**\n"
    if not _model_loaded:
        summary += "\n*(Warning: model checkpoint not found - predictions are untrained.)*"

    return gradcam_overlay, saliency_overlay, prob_dict, summary


def load_image_if_exists(path):
    return path if os.path.exists(path) else None


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_training_dynamics_markdown():
    md = "## Training & Probing Results\n\n"

    summary = load_json_if_exists(os.path.join(config.RESULTS_DIR, "summary.json"))
    if summary:
        md += (f"- **Test accuracy:** {summary['test_accuracy']*100:.2f}%\n"
               f"- **Best validation accuracy:** {summary['best_val_accuracy']*100:.2f}%\n"
               f"- **Epochs trained:** {summary['epochs_trained']}\n\n")

    layer_probe = load_json_if_exists(os.path.join(config.RESULTS_DIR, "probe_by_layer.json"))
    if layer_probe:
        md += "**Linear probe accuracy by depth:**\n"
        for stage, acc in layer_probe.items():
            md += f"- {stage}: {acc*100:.2f}%\n"
        md += "\n"

    ckpt_probe = load_json_if_exists(os.path.join(config.RESULTS_DIR, "probe_by_checkpoint.json"))
    if ckpt_probe:
        first_acc = ckpt_probe["accuracies"][0] * 100
        last_acc = ckpt_probe["accuracies"][-1] * 100
        md += (f"**Linear probe accuracy over training:** "
               f"{first_acc:.2f}% (epoch {ckpt_probe['epochs'][0]}) -> "
               f"{last_acc:.2f}% (epoch {ckpt_probe['epochs'][-1]})\n\n")

    if not summary and not layer_probe and not ckpt_probe:
        md += ("*No results found yet. Run `train.py`, `probe.py`, and "
               "`evaluate.py` to populate this tab.*")

    return md


def build_confusion_markdown():
    top_pair = load_json_if_exists(os.path.join(config.RESULTS_DIR, "top_confusion_pair.json"))
    if not top_pair:
        return "*No confusion matrix results found yet. Run `evaluate.py` first.*"

    return (f"### Confusion Matrix Summary\n\n"
            f"- **Test accuracy:** {top_pair['test_accuracy']*100:.2f}%\n"
            f"- **Top misclassification:** true = `{top_pair['true_class']}` "
            f"predicted as `{top_pair['predicted_class']}` "
            f"({top_pair['count']} cases)\n")


# ------------------------------------------------------------------
# Build Gradio interface
# ------------------------------------------------------------------
with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown(
        "# 🧠 Brain Tumor MRI Classification with Explainability\n"
        "Residual CNN + SE-attention trained on brain MRI scans "
        "(glioma / meningioma / pituitary / no tumor).\n\n"
        f"**Dataset:** [Kaggle dataset]({config.KAGGLE_DATASET_URL}) "
        "— see `download_data.py` to fetch it locally.\n"
    )

    with gr.Tab("Predict & Explain"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload MRI scan")
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                pred_summary = gr.Markdown()
                prob_output = gr.Label(label="Class probabilities", num_top_classes=4)

        with gr.Row():
            gradcam_output = gr.Image(label="Grad-CAM")
            saliency_output = gr.Image(label="Saliency Map")

        predict_btn.click(
            fn=predict_and_explain,
            inputs=image_input,
            outputs=[gradcam_output, saliency_output, prob_output, pred_summary],
        )
        image_input.change(
            fn=predict_and_explain,
            inputs=image_input,
            outputs=[gradcam_output, saliency_output, prob_output, pred_summary],
        )

    with gr.Tab("Training Dynamics"):
        gr.Markdown(build_training_dynamics_markdown())
        with gr.Row():
            gr.Image(value=load_image_if_exists(os.path.join(config.RESULTS_DIR, "training_curves.png")),
                     label="Loss / Accuracy curves")
        with gr.Row():
            gr.Image(value=load_image_if_exists(os.path.join(config.RESULTS_DIR, "probe_by_layer.png")),
                     label="Probe accuracy vs. depth")
            gr.Image(value=load_image_if_exists(os.path.join(config.RESULTS_DIR, "probe_by_checkpoint.png")),
                     label="Probe accuracy vs. training epoch")

    with gr.Tab("Confusion Matrix"):
        gr.Markdown(build_confusion_markdown())
        gr.Image(value=load_image_if_exists(os.path.join(config.RESULTS_DIR, "confusion_matrix.png")),
                 label="Confusion matrix (test set)")


if __name__ == "__main__":
    demo.launch(share=True)
