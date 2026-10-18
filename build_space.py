"""
Assemble a ready-to-upload Hugging Face Space folder for the Gradio app.

Example:
    python build_space.py --ckpt runs/E_strong/best.pt --out space

Creates <out>/ with app.py, src/, requirements.txt, the checkpoint (as best.pt),
classes.txt, a reference photo per species, example photos from the test set,
about.md (model card with test-set metrics) and the Space README header.
Upload the whole folder to a new Gradio Space.
"""

import argparse
import json
import os
import random
import shutil

import gradio
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.calibration import fit_temperature, validation_logits
from src.data import (BirdDataset, accuracy_metrics, build_transforms, class_names_from_split,
                      find_image, read_split_file)


def thumbnail(src, dst, size):
    img = Image.open(src).convert("RGB")
    img.thumbnail((size, size))
    img.save(dst, quality=88)


@torch.no_grad()
def test_metrics(ckpt, test_dir, test_txt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    _, eval_tf = build_transforms(ckpt["img_size"])
    ds = BirdDataset(test_dir, test_txt, transform=eval_tf, label_offset=ckpt["label_offset"])
    preds, targets = [], []
    for x, y in DataLoader(ds, batch_size=64):
        preds.append(model(x.to(device)).argmax(1).cpu())
        targets.append(y)
    top1, avg_cls, _ = accuracy_metrics(torch.cat(preds), torch.cat(targets), ckpt["num_classes"])
    return top1, avg_cls, len(ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="space")
    p.add_argument("--train-dir", default="data/Train")
    p.add_argument("--train-txt", default="data/train.txt")
    p.add_argument("--test-dir", default="data/Test")
    p.add_argument("--test-txt", default="data/test.txt")
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--title", default="Bird Species Classifier")
    args = p.parse_args()

    out = args.out
    os.makedirs(os.path.join(out, "src"), exist_ok=True)
    for f in ["app.py", "requirements.txt", "src/__init__.py", "src/data.py"]:
        shutil.copy(f, os.path.join(out, f))
    # Temperature scaling: fitted on the run's validation split so the app's confidence is honest
    ckpt = torch.load(args.ckpt, map_location="cpu")
    has_config = os.path.exists(os.path.join(os.path.dirname(args.ckpt), "config.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt["temperature"] = fit_temperature(*validation_logits(args.ckpt, device)) if has_config else 1.0
    torch.save(ckpt, os.path.join(out, "best.pt"))

    names = class_names_from_split(args.train_txt)
    with open(os.path.join(out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")

    # One reference photo per species (the first training image of each class)
    train_raw = read_split_file(args.train_txt)
    offset = min(l for _, l in train_raw)
    first = {}
    for name, label in train_raw:
        first.setdefault(label - offset, name)
    ref_dir = os.path.join(out, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    for idx, name in first.items():
        thumbnail(find_image(args.train_dir, name), os.path.join(ref_dir, f"{idx}.jpg"), 320)

    # Example photos: unseen test images from different species
    test_raw = read_split_file(args.test_txt)
    by_class = {}
    for name, label in test_raw:
        by_class.setdefault(label, []).append(name)
    rng = random.Random(0)
    ex_dir = os.path.join(out, "examples")
    shutil.rmtree(ex_dir, ignore_errors=True)
    os.makedirs(ex_dir)
    for label in rng.sample(sorted(by_class), args.n_examples):
        name = rng.choice(by_class[label])
        species = names[label - offset].replace(" ", "_")
        thumbnail(find_image(args.test_dir, name), os.path.join(ex_dir, f"{species}.jpg"), 640)

    # Model card, with the run's training settings when its config.json is available
    top1, avg_cls, n_test = test_metrics(ckpt, args.test_dir, args.test_txt)
    cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    lines = [
        f"**Backbone:** `{ckpt['model_name']}` ({'ImageNet-pretrained' if not cfg.get('no_pretrained') else 'trained from scratch'}), "
        f"fine-tuned on CUB-200 ({ckpt['num_classes']} species, {len(train_raw):,} training photos).",
        "",
        f"**Test set ({n_test:,} unseen photos):** top-1 accuracy **{top1:.1%}**, "
        f"average per-class accuracy **{avg_cls:.1%}**.",
    ]
    if cfg:
        lines += ["", f"**Training:** checkpoint from epoch {ckpt.get('epoch', '?')} of {cfg['epochs']} "
                      "(chosen by validation accuracy), AdamW, "
                      f"cosine learning rate, regularisation preset `{cfg['reg']}` (augmentation `{cfg['aug']}`, "
                      f"label smoothing {cfg['label_smoothing']}, weight decay {cfg['weight_decay']}, "
                      f"Mixup {cfg['mixup']}, CutMix {cfg['cutmix']}, random erasing {cfg['random_erasing']}, "
                      f"stochastic depth {cfg['drop_path']}, dropout {cfg['drop_rate']})."]
    if ckpt["temperature"] != 1.0:
        lines += ["", f"**Confidence:** calibrated with temperature scaling (T = {ckpt['temperature']:.2f}, "
                      "fitted on the validation split), so a shown confidence of 80% means the model is "
                      "right about 80% of the time."]
    lines += ["", "**Heatmap:** Grad-CAM highlights the regions that pushed the model towards its top answer.",
              "", "**Limits:** only knows these 200 species; photos with several birds, or tiny or hidden "
                  "birds, are harder. Some photos may overlap with the model's ImageNet pretraining data."]
    with open(os.path.join(out, "about.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Hugging Face reads the Space configuration from the README front matter
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"---\ntitle: {args.title}\nemoji: 🐦\ncolorFrom: blue\ncolorTo: green\nsdk: gradio\n"
                f"sdk_version: {gradio.__version__}\napp_file: app.py\npinned: false\n---\n\n"
                f"Fine-grained bird species classifier (CUB-200). Test top-1 accuracy {top1:.1%}.\n")

    print(f"Space folder ready: {out}/ (test top-1 {top1:.2%}, avg per class {avg_cls:.2%})")
    print(f"Run locally:  cd {out}; python app.py")


if __name__ == "__main__":
    main()
