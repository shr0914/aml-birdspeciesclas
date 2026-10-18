"""
Gradio web app for the Hugging Face Space.

Upload a bird photo -> the top-5 species with confidence, a Grad-CAM heatmap of the
regions that drove the prediction, and reference photos of the top guesses so the
user can check the answer themselves.

Build a ready-to-upload Space folder with `python build_space.py` (see README). It holds:
  app.py, src/, requirements.txt
  best.pt        trained checkpoint
  classes.txt    species names in label order
  about.md       model card shown in the "About" panel (optional)
  reference/     one photo per species named <label index>.jpg (optional)
  examples/      sample photos shown under the upload box (optional)
"""

import os

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import build_transforms

CKPT_PATH = os.getenv("CKPT_PATH", "best.pt")
CLASSES_PATH = os.getenv("CLASSES_PATH", "classes.txt")
ABOUT_PATH = os.getenv("ABOUT_PATH", "about.md")
REF_DIR = os.getenv("REF_DIR", "reference")
EXAMPLES_DIR = os.getenv("EXAMPLES_DIR", "examples")
UNSURE = 0.5  # below this top-1 confidence the app says it isn't sure

# --- Load model once at start-up (Spaces run on CPU by default) ----------------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
for p in model.parameters():  # Grad-CAM only needs gradients w.r.t. the image's features
    p.requires_grad_(False)

IMG = ckpt["img_size"]
# Temperature scaling (set by build_space.py): rescales confidence to match real accuracy
TEMPERATURE = ckpt.get("temperature", 1.0)
_, eval_tf = build_transforms(IMG)
# Same resize + centre crop the model sees, at 2x resolution, so the heatmap lines up with the photo
view_tf = transforms.Compose([transforms.Resize(int(IMG * 2 * 1.14)), transforms.CenterCrop(IMG * 2)])


def load_class_names(n):
    """Readable names like 'Black footed Albatross'; falls back to numeric labels."""
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        # Strip "001." prefixes and underscores for display
        return [n.split(".", 1)[-1].replace("_", " ") for n in names][:n]
    return [f"Class {i + ckpt['label_offset']}" for i in range(n)]


CLASS_NAMES = load_class_names(ckpt["num_classes"])


def grad_cam(x, cls):
    """Grad-CAM: weight each channel of the last feature map by how strongly it pushes up
    the score for `cls`, giving a coarse map of the image regions behind the prediction."""
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        feats = model.forward_features(x)
        if feats.ndim != 4:  # transformer backbones have no spatial feature map
            return None
        feats.retain_grad()
        model.forward_head(feats)[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG * 2, IMG * 2), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy()


# Inferno-style colour stops (dark purple -> orange -> pale yellow)
_STOPS = np.array([[40, 11, 84], [101, 21, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]]) / 255


def overlay(img, cam):
    """Blend the heatmap over the photo; cold regions keep the original photo."""
    pos = cam * (len(_STOPS) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(_STOPS) - 2)
    frac = (pos - lo)[..., None]
    heat = _STOPS[lo] * (1 - frac) + _STOPS[lo + 1] * frac
    alpha = 0.65 * cam[..., None]
    out = np.asarray(img, dtype=np.float32) / 255 * (1 - alpha) + heat * alpha
    return Image.fromarray((out * 255).astype(np.uint8))


def reference_photo(idx):
    path = os.path.join(REF_DIR, f"{idx}.jpg")
    return path if os.path.exists(path) else None


def predict(img):
    if img is None:
        return "Upload a photo to get started.", {}, None, []
    img = img.convert("RGB")  # handles PNG transparency and greyscale photos
    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = (model(x) / TEMPERATURE).softmax(1)[0]
    top = probs.topk(5)
    idxs, confs = top.indices.tolist(), top.values.tolist()

    best, conf = CLASS_NAMES[idxs[0]], confs[0]
    if conf >= UNSURE:
        verdict = f"## {best}\n**{conf:.0%} confident**"
    else:
        verdict = (f"## Not sure – maybe {best}?\n**Only {conf:.0%} confident.** The bird may be small, "
                   "partly hidden or not one of the 200 species this model knows. Check the other guesses.")

    cam = grad_cam(x, idxs[0])
    cam_img = overlay(view_tf(img), cam) if cam is not None else None
    refs = [(reference_photo(i), f"{CLASS_NAMES[i]} ({c:.0%})" if c >= 0.01 else f"{CLASS_NAMES[i]} (<1%)")
            for i, c in zip(idxs[:3], confs[:3]) if reference_photo(i)]
    return verdict, {CLASS_NAMES[i]: c for i, c in zip(idxs, confs)}, cam_img, refs


def read_about():
    if os.path.exists(ABOUT_PATH):
        with open(ABOUT_PATH, encoding="utf-8") as f:
            return f.read()
    return f"Model: `{ckpt['model_name']}` fine-tuned on CUB-200 ({ckpt['num_classes']} species)."


examples = ([os.path.join(EXAMPLES_DIR, f) for f in sorted(os.listdir(EXAMPLES_DIR))]
            if os.path.isdir(EXAMPLES_DIR) else [])

with gr.Blocks(title="Bird Species Classifier") as demo:
    gr.Markdown("# Bird Species Classifier\n"
                "Upload a photo of a bird to identify it among **200 North American species** "
                "(CUB-200 dataset). See which parts of the photo the model focused on, and "
                "compare against reference photos of its top guesses.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            photo = gr.Image(type="pil", label="Bird photo", height=380)
            with gr.Row():
                clear = gr.ClearButton(value="Clear")
                identify = gr.Button("Identify", variant="primary")
            if examples:
                gr.Examples(examples, inputs=photo, label="Try an example")
        with gr.Column(scale=1):
            verdict = gr.Markdown("Upload a photo to get started.")
            top5 = gr.Label(num_top_classes=5, label="Top-5 predictions")
    with gr.Row(equal_height=True):
        heatmap = gr.Image(label="Where the model looked (Grad-CAM)", height=340, interactive=False)
        refs = gr.Gallery(label="Reference photos of the top 3 guesses", columns=3, height=340,
                          object_fit="contain", allow_preview=True)
    with gr.Accordion("About this model", open=False):
        gr.Markdown(read_about())

    # `change` fires on upload, on picking an example and on clear (which resets the outputs)
    outputs = [verdict, top5, heatmap, refs]
    photo.change(predict, inputs=photo, outputs=outputs)
    identify.click(predict, inputs=photo, outputs=outputs)
    clear.add(photo)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"))
