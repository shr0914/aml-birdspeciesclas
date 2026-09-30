"""
Evaluate a trained checkpoint on the test set.

Reports the two metrics the assignment asks for:
  - Top-1 accuracy
  - Average accuracy per class

Example:
    python evaluate.py --ckpt checkpoints/best.pt --test-dir data/Test --test-txt data/test.txt
"""

import argparse
import csv

import timm
import torch
from torch.utils.data import DataLoader

from src.data import BirdDataset, build_transforms, accuracy_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--test-dir", required=True)
    p.add_argument("--test-txt", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--per-class-csv", default="per_class_accuracy.csv",
                   help="Where to write each class's accuracy (handy for the report)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Rebuild the same architecture and load trained weights
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # Reuse the training label offset so test labels map identically
    _, eval_tf = build_transforms(ckpt["img_size"])
    test_ds = BirdDataset(args.test_dir, args.test_txt, transform=eval_tf,
                          label_offset=ckpt["label_offset"])
    loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            targets.append(y)

    top1, avg_cls, per_class = accuracy_metrics(torch.cat(preds), torch.cat(targets), ckpt["num_classes"])
    print(f"Test images:                {len(test_ds)}")
    print(f"Top-1 accuracy:             {top1 * 100:.2f}%")
    print(f"Average accuracy per class: {avg_cls * 100:.2f}%")

    # Classes present in the test set, in label order, with their accuracy
    present = sorted(set(torch.cat(targets).tolist()))
    with open(args.per_class_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_label", "accuracy"])
        for c, acc in zip(present, per_class):
            w.writerow([c + ckpt["label_offset"], f"{acc:.4f}"])  # original label numbering
    print(f"Per-class accuracies written to {args.per_class_csv}")


if __name__ == "__main__":
    main()
