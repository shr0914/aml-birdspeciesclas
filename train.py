"""
Fine-tune a pretrained CNN/ViT on CUB-200.

Example:
    python train.py --train-dir data/Train --train-txt data/train.txt --epochs 30

The best checkpoint (by validation average per-class accuracy) is saved to
checkpoints/best.pt, per-epoch metrics go to checkpoints/history.csv so you can
plot learning curves, and the full run settings go to checkpoints/config.json.

Regularisation (fighting overfitting) is controlled by a preset, and any single
setting can be overridden on top of it:
    --reg none      pretrained weights only – the overfitting baseline
    --reg default   basic augmentation + label smoothing + weight decay
    --reg strong    default + RandAugment, random erasing, Mixup/CutMix,
                    stochastic depth and dropout
    e.g. --reg none --aug basic   (augmentation only, for an ablation study)
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader, Subset

from src.data import BirdDataset, build_transforms, accuracy_metrics

# Regularisation presets. Each key is also a command-line flag that overrides the preset.
PRESETS = {
    "none": dict(aug="none", label_smoothing=0.0, weight_decay=0.0, random_erasing=0.0,
                 mixup=0.0, cutmix=0.0, mixup_prob=0.0, drop_path=0.0, drop_rate=0.0),
    "default": dict(aug="basic", label_smoothing=0.1, weight_decay=0.05, random_erasing=0.0,
                    mixup=0.0, cutmix=0.0, mixup_prob=0.0, drop_path=0.0, drop_rate=0.0),
    "strong": dict(aug="strong", label_smoothing=0.1, weight_decay=0.05, random_erasing=0.25,
                   mixup=0.8, cutmix=1.0, mixup_prob=0.5, drop_path=0.1, drop_rate=0.2),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True, help="Folder of training images")
    p.add_argument("--train-txt", required=True, help="train.txt split file")
    p.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k",
                   help="Any timm model name, e.g. resnet50, efficientnet_b0")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from random init (baseline to show the value of transfer learning)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone learning rate")
    p.add_argument("--head-lr-mult", type=float, default=10.0,
                   help="New classifier head learns this many times faster")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of train held out for model selection (keeps test unseen)")
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping: stop after this many epochs without a new best (0 = off)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42,
                   help="Training randomness: head initialisation, batch order, augmentation, Mixup")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Which images form the validation split; kept fixed so runs with different "
                        "--seed values are scored on the same validation images")

    # --- Regularisation (unset flags take their value from the --reg preset) ---
    g = p.add_argument_group("regularisation")
    g.add_argument("--reg", choices=PRESETS, default="default", help="Regularisation preset")
    g.add_argument("--aug", choices=["none", "basic", "strong"], help="Augmentation level")
    g.add_argument("--label-smoothing", type=float)
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--random-erasing", type=float, help="Probability of erasing a random patch")
    g.add_argument("--mixup", type=float, help="Mixup alpha (0 = off)")
    g.add_argument("--cutmix", type=float, help="CutMix alpha (0 = off)")
    g.add_argument("--mixup-prob", type=float, help="Chance a batch gets Mixup/CutMix")
    g.add_argument("--drop-path", type=float, help="Stochastic depth rate")
    g.add_argument("--drop-rate", type=float, help="Dropout before the classifier")

    args = p.parse_args()
    for k, v in PRESETS[args.reg].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def set_seed(seed):
    """Make runs reproducible (important when comparing models in the report)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(labels, val_frac, seed):
    """Hold out val_frac of each class so every class appears in validation."""
    rng = random.Random(seed)
    by_class = {}
    for i, l in enumerate(labels):
        by_class.setdefault(l, []).append(i)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac)) if val_frac > 0 else 0
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Top-1, average per-class accuracy and plain cross-entropy loss (no label
    smoothing, so the loss is comparable between runs with different settings)."""
    model.eval()
    preds, targets, loss_sum = [], [], 0.0
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True)).float()
        loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
        preds.append(logits.argmax(1).cpu())
        targets.append(y)
    targets = torch.cat(targets)
    top1, avg_cls, _ = accuracy_metrics(torch.cat(preds), targets, num_classes)
    return top1, avg_cls, loss_sum / len(targets)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Regularisation [{args.reg}]: aug={args.aug} label_smoothing={args.label_smoothing} "
          f"weight_decay={args.weight_decay} random_erasing={args.random_erasing} "
          f"mixup={args.mixup} cutmix={args.cutmix} mixup_prob={args.mixup_prob} "
          f"drop_path={args.drop_path} drop_rate={args.drop_rate}")

    # --- Data -------------------------------------------------------------
    train_tf, eval_tf = build_transforms(args.img_size, args.aug, args.random_erasing)
    # Two views of the same file list: augmented for training, clean for evaluation
    full_train = BirdDataset(args.train_dir, args.train_txt, transform=train_tf)
    full_eval = BirdDataset(args.train_dir, args.train_txt, transform=eval_tf,
                            label_offset=full_train.label_offset)
    labels = [l for _, l in full_train.samples]
    num_classes = max(labels) + 1

    tr_idx, va_idx = stratified_split(labels, args.val_split, args.split_seed)
    train_loader = DataLoader(Subset(full_train, tr_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(full_eval, va_idx), batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=args.workers, pin_memory=True) if va_idx else None
    # Training accuracy measured *during* training is on augmented/mixed images, so it
    # understates how well the model fits the training set. A fixed subset of training
    # images scored with the clean eval transform gives the true train-vs-val gap.
    clean_idx = random.Random(args.split_seed).sample(tr_idx, min(1000, len(tr_idx)))
    train_clean_loader = DataLoader(Subset(full_eval, clean_idx), batch_size=args.batch_size * 2,
                                    shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"Classes: {num_classes} | train: {len(tr_idx)} | val: {len(va_idx)}")

    # --- Model ------------------------------------------------------------
    # timm replaces the ImageNet head with a fresh num_classes-way classifier.
    # drop_rate = dropout before the head; drop_path = stochastic depth (randomly skips
    # whole residual blocks during training so no single block can be relied on).
    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=num_classes,
                              drop_rate=args.drop_rate, drop_path_rate=args.drop_path).to(device)

    # Pretrained layers get a small LR; the new head gets a larger one
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=args.weight_decay)

    # Cosine decay with a 1-epoch linear warm-up
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup = steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup)))
    ))

    # Mixup blends two images and their labels; CutMix pastes a patch of one image onto
    # another. Labels become soft (e.g. 70% sparrow / 30% finch), so the loss changes too.
    mixup_fn = None
    if args.mixup_prob > 0 and (args.mixup > 0 or args.cutmix > 0):
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=args.mixup_prob,
                         label_smoothing=args.label_smoothing, num_classes=num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device == "cuda"  # mixed precision: faster + less VRAM on NVIDIA GPUs
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "num_classes": num_classes}, f, indent=2)
    history_path = os.path.join(args.out_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "train_clean_acc", "train_clean_loss",
                                "val_loss", "val_top1", "val_avg_class", "gap", "time_s"])

    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        # --- Train one epoch ----------------------------------------------
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        lr_now = optimizer.param_groups[0]["lr"]
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            targets = y
            if mixup_fn is not None:
                x, targets = mixup_fn(x, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)

        train_loss, train_acc = loss_sum / seen, correct / seen

        # --- Measure overfitting: clean train subset vs validation ---------
        train_clean_acc, _, train_clean_loss = evaluate(model, train_clean_loader, device, num_classes)
        if val_loader:
            val_top1, val_avg, val_loss = evaluate(model, val_loader, device, num_classes)
        else:
            val_top1 = val_avg = val_loss = float("nan")
        gap = train_clean_acc - val_top1  # big gap = memorising rather than generalising
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | train(clean) {train_clean_acc:.4f} "
              f"| val loss {val_loss:.4f} | val top-1 {val_top1:.4f} | val avg/class {val_avg:.4f} "
              f"| gap {gap:+.4f} | {elapsed:.0f}s")
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr_now:.2e}", f"{train_loss:.4f}", f"{train_acc:.4f}",
                                    f"{train_clean_acc:.4f}", f"{train_clean_loss:.4f}", f"{val_loss:.4f}",
                                    f"{val_top1:.4f}", f"{val_avg:.4f}", f"{gap:.4f}", f"{elapsed:.1f}"])

        # --- Save best (or last, if no validation split) ------------------
        score = val_avg if val_loader else epoch
        if score > best:
            best, best_epoch = score, epoch
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "label_offset": full_train.label_offset,
                "img_size": args.img_size,
                "epoch": epoch,
                "reg": args.reg,
            }, os.path.join(args.out_dir, "best.pt"))
            print("  -> saved new best checkpoint")
        elif args.patience and epoch - best_epoch >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    print(f"Done. Best val avg per-class accuracy: {best:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
