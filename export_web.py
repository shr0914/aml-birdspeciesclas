"""
Build a static web app that runs the bird classifier entirely in the visitor's browser.

No server is needed: the model is exported to ONNX and run with onnxruntime-web, so the
site can be hosted for free on GitHub Pages (or Netlify, Vercel, Cloudflare Pages...).

Example (from the repo root, after training):
    python export_web.py --ckpt runs/E_strong/best.pt --out docs

Creates <out>/ with:
  index.html, app.js, style.css   the web app (copied from web/)
  model.onnx                      image -> (calibrated probabilities, Grad-CAM map for the top class)
  meta.json                       species names, temperature, preprocessing, test metrics
  reference/<label index>.jpg     one training photo per species (shown next to the top 3 guesses)
  examples/*.jpg                  unseen test photos to try

Grad-CAM inside the ONNX graph
------------------------------
Grad-CAM needs the gradient of the top class score with respect to the last feature map.
ONNX graphs can't run backpropagation, but for ConvNeXt's head (global average pool ->
LayerNorm -> linear layer) the gradient has a closed form, so it is computed with ordinary
forward operations and exported with the rest of the model. `check_grad_cam` confirms it
matches PyTorch autograd before anything is written.
"""

import argparse
import json
import os
import random
import shutil

import numpy as np
import onnx
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from torch.utils.data import DataLoader

from src.data import (IMAGENET_MEAN, IMAGENET_STD, BirdDataset, accuracy_metrics, build_transforms,
                      class_names_from_split, find_image, read_split_file)

WEB_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class WebModel(nn.Module):
    """Wraps the classifier so one forward pass returns calibrated probabilities and the
    Grad-CAM map (normalised to 0-1) of the most likely class."""

    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = temperature
        head = model.head
        # ConvNeXt (timm NormMlpClassifierHead): pool -> LayerNorm -> (pre_logits) -> dropout -> fc
        if not (hasattr(head, "norm") and hasattr(head, "fc") and isinstance(head.pre_logits, nn.Identity)):
            raise NotImplementedError("Grad-CAM export is written for ConvNeXt heads "
                                      "(pool -> LayerNorm -> linear).")
        self.norm, self.fc = head.norm, head.fc

    def forward(self, x):
        feats = self.model.forward_features(x)               # (B, C, H, W)
        pooled = feats.mean(dim=(2, 3))                      # global average pool
        mu = pooled.mean(1, keepdim=True)
        var = pooled.var(1, unbiased=False, keepdim=True)
        inv_std = torch.rsqrt(var + self.norm.eps)
        xhat = (pooled - mu) * inv_std
        logits = self.fc(xhat * self.norm.weight + self.norm.bias)

        # d(logit_k)/d(pooled) through the LayerNorm, for the top class k
        k = logits.argmax(1)
        g = self.norm.weight * self.fc.weight[k]             # (B, C)
        g = (g - g.mean(1, keepdim=True) - xhat * (xhat * g).mean(1, keepdim=True)) * inv_std
        # Every spatial position feeds the average pool equally, so the Grad-CAM channel
        # weights are g / (H*W); the constant scale disappears in the 0-1 normalisation.
        cam = F.relu((g[:, :, None, None] * feats).sum(1))   # (B, H, W)
        flat = cam.flatten(1)
        lo, hi = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
        cam = (cam - lo) / (hi - lo + 1e-8)

        probs = (logits / self.temperature).softmax(1)
        return probs, cam


def reference_grad_cam(model, x, cls):
    """Grad-CAM computed the usual way, with autograd: used to check the closed-form version."""
    x = x.clone().requires_grad_(True)
    feats = model.forward_features(x)
    feats.retain_grad()
    model.forward_head(feats)[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1))[0]
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach()


def check_grad_cam(model, web, img_size):
    torch.manual_seed(0)
    for _ in range(3):
        x = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            probs, cam = web(x)
            ref_logits = model(x)
        assert torch.equal(probs.argmax(1), ref_logits.argmax(1)), "prediction mismatch"
        ref_cam = reference_grad_cam(model, x, int(probs.argmax()))
        err = (cam[0] - ref_cam).abs().max().item()
        assert err < 1e-3, f"closed-form Grad-CAM differs from autograd by {err:.4f}"
    print("Grad-CAM check passed (closed form matches autograd)")


def weights_to_fp16(path):
    """Store large weights as float16 (halves the download, keeps the file under GitHub's
    100 MB limit) and cast them back to float32 when the model loads, so all maths still
    runs in float32."""
    m = onnx.load(path)
    g = m.graph
    casts, keep = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= 1024:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "_cast"))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, path)


def check_onnx(path, web, img_size):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        p_ref, c_ref = web(x)
    p, c = sess.run(None, {"image": x.numpy()})
    assert p.argmax() == p_ref.argmax().item(), "ONNX prediction differs from PyTorch"
    print(f"ONNX check passed (max prob diff {np.abs(p - p_ref.numpy()).max():.1e}, "
          f"max CAM diff {np.abs(c - c_ref.numpy()).max():.1e})")


def thumbnail(src, dst, size):
    img = Image.open(src).convert("RGB")
    img.thumbnail((size, size))
    img.save(dst, quality=85)


@torch.no_grad()
def test_metrics(model, ckpt, test_dir, test_txt):
    _, eval_tf = build_transforms(ckpt["img_size"])
    ds = BirdDataset(test_dir, test_txt, transform=eval_tf, label_offset=ckpt["label_offset"])
    preds, targets = [], []
    for x, y in DataLoader(ds, batch_size=64):
        preds.append(model(x).argmax(1))
        targets.append(y)
    top1, avg_cls, _ = accuracy_metrics(torch.cat(preds), torch.cat(targets), ckpt["num_classes"])
    return top1, avg_cls, len(ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="docs")
    p.add_argument("--train-dir", default="data/Train")
    p.add_argument("--train-txt", default="data/train.txt")
    p.add_argument("--test-dir", default="data/Test")
    p.add_argument("--test-txt", default="data/test.txt")
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--temperature", type=float,
                   help="Skip fitting and use this temperature (default: fit on the validation split)")
    p.add_argument("--skip-test", action="store_true", help="Don't compute test metrics for the model card")
    args = p.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(WEB_TEMPLATE):
        shutil.copy(os.path.join(WEB_TEMPLATE, f), os.path.join(out, f))
    open(os.path.join(out, ".nojekyll"), "w").close()  # GitHub Pages: serve files as-is

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = ckpt["img_size"]

    # Temperature scaling, fitted on the run's validation split so the test set stays unseen
    cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if args.temperature is not None:
        temperature = args.temperature
    elif cfg:
        from src.calibration import fit_temperature, validation_logits
        temperature = fit_temperature(*validation_logits(args.ckpt))
    else:
        temperature = 1.0
    print(f"Temperature: {temperature:.3f}")

    # Export model + Grad-CAM to ONNX
    web = WebModel(model, temperature).eval()
    check_grad_cam(model, web, img_size)
    onnx_path = os.path.join(out, "model.onnx")
    torch.onnx.export(web, torch.randn(1, 3, img_size, img_size), onnx_path,
                      input_names=["image"], output_names=["probs", "cam"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    weights_to_fp16(onnx_path)
    check_onnx(onnx_path, web, img_size)
    print(f"model.onnx: {os.path.getsize(onnx_path) / 1e6:.1f} MB")

    # Species names, reference photos, examples
    names = class_names_from_split(args.train_txt)  # e.g. "Black footed Albatross"
    train_raw = read_split_file(args.train_txt)
    offset = min(l for _, l in train_raw)
    first = {}
    for name, label in train_raw:
        first.setdefault(label - offset, name)
    ref_dir = os.path.join(out, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    for idx, name in first.items():
        thumbnail(find_image(args.train_dir, name), os.path.join(ref_dir, f"{idx}.jpg"), 320)

    test_raw = read_split_file(args.test_txt)
    by_class = {}
    for name, label in test_raw:
        by_class.setdefault(label, []).append(name)
    rng = random.Random(0)
    ex_dir = os.path.join(out, "examples")
    shutil.rmtree(ex_dir, ignore_errors=True)
    os.makedirs(ex_dir)
    examples = []
    for label in rng.sample(sorted(by_class), min(args.n_examples, len(by_class))):
        name = rng.choice(by_class[label])
        fname = names[label - offset].replace(" ", "_") + ".jpg"
        thumbnail(find_image(args.test_dir, name), os.path.join(ex_dir, fname), 640)
        examples.append({"file": f"examples/{fname}", "species": names[label - offset]})

    meta = {
        "model_name": ckpt["model_name"],
        "img_size": img_size,
        "resize": int(img_size * 1.14),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "temperature": temperature,
        "classes": names,
        "examples": examples,
        "n_train": len(train_raw),
        "epoch": ckpt.get("epoch"),
        "config": {k: cfg[k] for k in ["epochs", "reg", "aug", "label_smoothing", "weight_decay", "mixup",
                                       "cutmix", "random_erasing", "drop_path", "drop_rate"] if k in cfg},
    }
    if not args.skip_test:
        top1, avg_cls, n_test = test_metrics(model, ckpt, args.test_dir, args.test_txt)
        meta["test"] = {"top1": top1, "avg_class": avg_cls, "n": n_test}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"Web app ready in {out}/")
    print(f"Try it locally:  python -m http.server 8000 --directory {out}   then open http://localhost:8000")


if __name__ == "__main__":
    main()
