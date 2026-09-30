"""
Gradio web app for the Hugging Face Space.

Upload a bird photo -> top-5 predicted species with confidence.

Put these next to app.py in the Space:
  - best.pt       (trained checkpoint from train.py)
  - classes.txt   (optional: one class name per line, in label order,
                   e.g. "001.Black_footed_Albatross". Without it, labels show as "Class N")
"""

import os

import gradio as gr
import timm
import torch
from PIL import Image

from src.data import build_transforms

CKPT_PATH = os.getenv("CKPT_PATH", "best.pt")
CLASSES_PATH = os.getenv("CLASSES_PATH", "classes.txt")

# --- Load model once at start-up (Spaces run on CPU by default) ----------------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
_, eval_tf = build_transforms(ckpt["img_size"])


def load_class_names(n):
    """Readable names like 'Black footed Albatross'; falls back to numeric labels."""
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        # Strip "001." prefixes and underscores for display
        return [n.split(".", 1)[-1].replace("_", " ") for n in names][:n]
    return [f"Class {i + ckpt['label_offset']}" for i in range(n)]


CLASS_NAMES = load_class_names(ckpt["num_classes"])


@torch.no_grad()
def predict(img: Image.Image):
    if img is None:
        return {}
    x = eval_tf(img.convert("RGB")).unsqueeze(0)
    probs = model(x).softmax(1)[0]
    top = probs.topk(5)
    # gr.Label expects {label: confidence}
    return {CLASS_NAMES[i]: float(p) for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Bird photo"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 predictions"),
    title="Bird Species Classifier (CUB-200)",
    description="Upload a photo of a bird to identify it among 200 species.",
    examples=[os.path.join("examples", f) for f in sorted(os.listdir("examples"))]
    if os.path.isdir("examples") else None,
)

if __name__ == "__main__":
    demo.launch()
