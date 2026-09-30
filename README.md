# Bird Species Classification (CUB-200)

Fine-grained classification of 200 bird species using transfer learning with a pretrained backbone (default: ConvNeXt-Tiny from `timm`).

## Setup

```bash
pip install -r requirements.txt
```

Extract the dataset into `data/`:

```
data/
├── Train/        # from Train.zip
├── Test/         # from Test.zip
├── train.txt     # "<image_name> <class_label>" per line
└── test.txt
```

## Train

```bash
python train.py --train-dir data/Train --train-txt data/train.txt --epochs 30
```

- 10% of the training set is held out (stratified) for model selection, so the test set stays unseen until final evaluation.
- The best checkpoint goes to `checkpoints/best.pt`; per-epoch metrics go to `checkpoints/history.csv` for learning-curve plots.
- Try other backbones with `--model` (any `timm` name, e.g. `resnet50`, `efficientnet_b0`, `vit_small_patch16_224`).
- `--no-pretrained` trains from scratch, a useful baseline for showing the benefit of transfer learning.

## Evaluate

```bash
python evaluate.py --ckpt checkpoints/best.pt --test-dir data/Test --test-txt data/test.txt
```

Prints **Top-1 accuracy** and **average accuracy per class**, and writes each class's accuracy to `per_class_accuracy.csv`.

## Analysis & graphs

Open `analysis.ipynb` after training and *Run All* (needs `matplotlib`, `pandas`, `scikit-learn`, `jupyter`). It plots the dataset overview, learning curves, test metrics (top-1, average per class, top-5), per-species and per-family accuracy, the confusion matrix, the most common mix-ups, and the most confident mistakes. Figures are saved to `figures/`, and it writes `classes.txt` (species names for the web app).

## Web app (Hugging Face Spaces)

Create a Gradio Space and upload `app.py`, `src/`, `requirements.txt`, `best.pt`, and optionally `classes.txt` (one species name per line in label order) and an `examples/` folder of sample images.

Run locally:

```bash
cp checkpoints/best.pt . && python app.py
```

## Project structure

| File | Purpose |
|---|---|
| `src/data.py` | Dataset loading, transforms, accuracy metrics |
| `train.py` | Fine-tuning with AdamW, cosine LR, label smoothing, mixed precision |
| `evaluate.py` | Test-set Top-1 and per-class accuracy |
| `app.py` | Gradio demo showing top-5 predictions |
