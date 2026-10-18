"""
Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural Networks").

Label smoothing and Mixup make the model under-confident: it might say "50% sure" while
being right 90% of the time. Dividing the logits by one learnt number T > 0 fixes this
without changing any prediction (the ranking of classes stays the same). T is fitted on
the validation split only, so the test set stays unseen.
"""

import os
import json

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data import BirdDataset, build_transforms


def fit_temperature(logits, targets):
    """Find T minimising the validation cross-entropy of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


@torch.no_grad()
def validation_logits(ckpt_path, device="cpu"):
    """Logits and labels on the run's own validation split, rebuilt from its config.json."""
    from train import stratified_split  # same split function and seed as training

    with open(os.path.join(os.path.dirname(ckpt_path), "config.json")) as f:
        cfg = json.load(f)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    _, eval_tf = build_transforms(ckpt["img_size"])
    ds = BirdDataset(cfg["train_dir"], cfg["train_txt"], transform=eval_tf, label_offset=ckpt["label_offset"])
    # Runs made before --split-seed existed used --seed for the split
    _, val_idx = stratified_split([l for _, l in ds.samples], cfg["val_split"], cfg.get("split_seed", cfg["seed"]))
    logits, targets = [], []
    for x, y in DataLoader(Subset(ds, val_idx), batch_size=64):
        logits.append(model(x.to(device)).float().cpu())
        targets.append(y)
    return torch.cat(logits), torch.cat(targets)
