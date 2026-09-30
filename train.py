"""
Fine-tune a pretrained CNN/ViT on CUB-200.

Example:
    python train.py --train-dir data/Train --train-txt data/train.txt --epochs 30

The best checkpoint (by validation average per-class accuracy) is saved to
checkpoints/best.pt, and per-epoch metrics go to checkpoints/history.csv
so you can plot learning curves for the report.
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import BirdDataset, build_transforms, accuracy_metrics


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
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of train held out for model selection (keeps test unseen)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(1).cpu())
        targets.append(y)
    top1, avg_cls, _ = accuracy_metrics(torch.cat(preds), torch.cat(targets), num_classes)
    return top1, avg_cls


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data -------------------------------------------------------------
    train_tf, eval_tf = build_transforms(args.img_size)
    # Two views of the same file list: augmented for training, clean for validation
    full_train = BirdDataset(args.train_dir, args.train_txt, transform=train_tf)
    full_eval = BirdDataset(args.train_dir, args.train_txt, transform=eval_tf,
                            label_offset=full_train.label_offset)
    labels = [l for _, l in full_train.samples]
    num_classes = max(labels) + 1

    tr_idx, va_idx = stratified_split(labels, args.val_split, args.seed)
    train_loader = DataLoader(Subset(full_train, tr_idx), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(full_eval, va_idx), batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=args.workers, pin_memory=True) if va_idx else None
    print(f"Classes: {num_classes} | train: {len(tr_idx)} | val: {len(va_idx)}")

    # --- Model ------------------------------------------------------------
    # timm replaces the ImageNet head with a fresh num_classes-way classifier
    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=num_classes).to(device)

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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device == "cuda"  # mixed precision: faster + less VRAM on NVIDIA GPUs
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    history_path = os.path.join(args.out_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_top1", "val_avg_class", "time_s"])

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        # --- Train one epoch ----------------------------------------------
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)

        train_loss, train_acc = loss_sum / seen, correct / seen

        # --- Validate -----------------------------------------------------
        val_top1, val_avg = evaluate(model, val_loader, device, num_classes) if val_loader else (float("nan"),) * 2
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | train acc {train_acc:.4f} "
                f"| val top-1 {val_top1:.4f} | val avg/class {val_avg:.4f} | {elapsed:.0f}s")
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                                    f"{val_top1:.4f}", f"{val_avg:.4f}", f"{elapsed:.1f}"])

        # --- Save best (or last, if no validation split) ------------------
        score = val_avg if val_loader else epoch
        if score > best:
            best = score
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "label_offset": full_train.label_offset,
                "img_size": args.img_size,
            }, os.path.join(args.out_dir, "best.pt"))
            print("  ↳ saved new best checkpoint")

    print(f"Done. Best val avg per-class accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
