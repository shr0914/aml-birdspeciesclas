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

### Regularisation (combating overfitting)

`--reg` picks a preset; any individual flag overrides it (e.g. `--reg none --aug basic`).

| Flag | `none` | `default` | `strong` | What it does |
|---|---|---|---|---|
| `--aug` | none | basic | strong | basic = random crop, flip, colour jitter; strong = crop, flip, RandAugment |
| `--label-smoothing` | 0 | 0.1 | 0.1 | Softens targets so the model can't become 100% confident |
| `--weight-decay` | 0 | 0.05 | 0.05 | Keeps weights small (AdamW) |
| `--random-erasing` | 0 | 0 | 0.25 | Blanks a random patch so no single body part is relied on |
| `--mixup` / `--cutmix` / `--mixup-prob` | off | off | 0.8 / 1.0 / 0.5 | Blends pairs of images and their labels |
| `--drop-path` | 0 | 0 | 0.1 | Stochastic depth: randomly skips residual blocks |
| `--drop-rate` | 0 | 0 | 0.2 | Dropout before the classifier |

`--patience N` stops training after N epochs without a new best. Each epoch also scores a fixed subset of training images with the clean (eval) transform, so `history.csv` records the true train-vs-validation **gap**, plus validation loss, which are the overfitting indicators. Run settings are saved to `config.json`.

### Experiments

Two scripts reproduce every run in the report (outputs go to `runs/<name>/`):

```powershell
powershell -ExecutionPolicy Bypass -File run_ablation.ps1                  # A–E: regularisation added step by step
powershell -ExecutionPolicy Bypass -File run_comparison.ps1 -Reg strong    # F–J: architecture & pretraining
```

| Run | Model | Pretraining | Regularisation |
|---|---|---|---|
| A_none | ConvNeXt-Tiny | ImageNet-22k | none |
| B_aug | ConvNeXt-Tiny | ImageNet-22k | basic augmentation only |
| C_default | ConvNeXt-Tiny | ImageNet-22k | `default` |
| D_mixup | ConvNeXt-Tiny | ImageNet-22k | `default` + Mixup/CutMix |
| E_strong | ConvNeXt-Tiny | ImageNet-22k | `strong` |
| F_convnext_in1k | ConvNeXt-Tiny | ImageNet-1k | chosen preset |
| G_resnet50_none / H_resnet50 | ResNet-50 | ImageNet-1k | none / chosen preset |
| I_scratch_none / J_scratch_reg | ConvNeXt-Tiny | none (30 epochs, LR 1e-3) | none / chosen preset |

## Evaluate

```bash
python evaluate.py --ckpt checkpoints/best.pt --test-dir data/Test --test-txt data/test.txt
```

Prints **Top-1 accuracy** and **average accuracy per class**, and writes each class's accuracy to `per_class_accuracy.csv`.

## Analysis & graphs

Open `analysis.ipynb` after training and *Run All* (needs `matplotlib`, `pandas`, `scikit-learn`, `jupyter`). It covers the dataset, the overfitting ablation (validation accuracy, gap and validation loss per run, plus test scores), the architecture/pretraining comparison, and a detailed analysis of the final model, which is picked by **validation** score: test metrics, per-species and per-family accuracy, the confusion matrix, the most common mix-ups and the most confident mistakes. Figures and results tables are saved to `figures/`.

## Web app (Hugging Face Spaces)

The app shows the top-5 species, a **Grad-CAM heatmap** of where the model looked, reference photos of the top 3 guesses, a low-confidence warning, example photos and a model card.

```bash
python build_space.py --ckpt runs/E_strong/best.pt --out space   # use the final model's checkpoint
cd space && python app.py                                         # try it locally
```

`build_space.py` assembles everything the Space needs into `space/`: app, checkpoint, species names, one reference photo per species, test-set examples, a model card with test metrics and the Space README header. Create a new Gradio Space on Hugging Face and upload the contents of `space/`.

## Project structure

| File | Purpose |
|---|---|
| `src/data.py` | Dataset loading, augmentation levels, species names, accuracy metrics |
| `train.py` | Fine-tuning with AdamW, cosine LR, regularisation presets, overfitting tracking, early stopping, mixed precision |
| `evaluate.py` | Test-set Top-1 and per-class accuracy |
| `run_ablation.ps1`, `run_comparison.ps1` | Reproduce all experiment runs |
| `analysis.ipynb` | Graphs and results tables for the report |
| `app.py` | Gradio app: top-5 predictions, Grad-CAM, reference photos |
| `build_space.py` | Builds the Hugging Face Space folder |
