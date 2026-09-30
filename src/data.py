"""
Dataset utilities for CUB-200 bird species classification.

The assignment provides:
  - Train/ and Test/ image folders (extracted from Train.zip / Test.zip)
  - train.txt / test.txt, one line per image:  "<image_name> <class_label>"
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics – the pretrained backbones were trained with these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_split_file(txt_path):
    """Parse a split file into a list of (image_name, int_label) tuples."""
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # rsplit so image names containing spaces still work
            name, label = line.rsplit(maxsplit=1)
            samples.append((name, int(label)))
    return samples


def find_image(img_dir, name):
    """Locate an image whether the txt lists a bare filename or a relative path."""
    direct = os.path.join(img_dir, name)
    if os.path.exists(direct):
        return direct
    # Fallback: zips sometimes extract into a nested folder – search once by basename
    base = os.path.basename(name)
    for root, _, files in os.walk(img_dir):
        if base in files:
            return os.path.join(root, base)
    raise FileNotFoundError(f"Image '{name}' not found under '{img_dir}'")


class BirdDataset(Dataset):
    """Returns (image_tensor, label) with labels shifted to start at 0."""

    def __init__(self, img_dir, txt_path, transform=None, label_offset=None):
        raw = read_split_file(txt_path)
        # Labels may be 0- or 1-indexed; the offset is learnt from the training set
        # and passed to the test set so both use the same mapping.
        self.label_offset = min(l for _, l in raw) if label_offset is None else label_offset
        self.samples = [(find_image(img_dir, n), l - self.label_offset) for n, l in raw]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")  # some CUB images are greyscale/CMYK
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(img_size=224):
    """Augmentation for training; deterministic resize + centre crop for eval."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def accuracy_metrics(preds, targets, num_classes):
    """
    Top-1 accuracy: correct / total.
    Average per-class accuracy: mean over classes of (correct in class / images in class).
    The second metric isn't skewed by classes that have more test images.
    """
    preds, targets = torch.as_tensor(preds), torch.as_tensor(targets)
    correct = preds == targets
    top1 = correct.float().mean().item()

    per_class = []
    for c in range(num_classes):
        mask = targets == c
        if mask.any():  # skip classes absent from this split
            per_class.append(correct[mask].float().mean().item())
    avg_per_class = sum(per_class) / len(per_class)
    return top1, avg_per_class, per_class
