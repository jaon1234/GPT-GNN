# Medical image masked pretraining

This package adds an image pretraining path for hematology analyzer plots, such
as scattergrams and histograms. It is intentionally separate from the GPT-GNN
examples so it can be used on instrument image exports without changing the
graph code.

## What is implemented

- Unlabeled recursive image loader for PNG/JPG/BMP/TIFF plots.
- A hybrid masked autoencoder backbone:
  - ViT-style patch tokens reconstruct masked image patches.
  - ResNet18 extracts local morphology context from the masked image only, so
    hidden patches do not leak into the reconstruction target.
  - The encoder can be reused as a feature extractor for downstream classifiers.
- A pretraining CLI for unlabeled plot folders.
- A classifier fine-tuning CLI using `torchvision.datasets.ImageFolder`.

## Suggested data layout

Unlabeled pretraining can point at any directory tree:

```text
data/unlabeled_plots/
  analyzer_a/run_001/xxx.png
  analyzer_a/run_002/yyy.tif
  analyzer_b/histograms/zzz.jpg
```

Fine-tuning expects class subdirectories:

```text
data/classifier/train/
  normal/*.png
  flag_blast/*.png
  flag_clump/*.png

data/classifier/val/
  normal/*.png
  flag_blast/*.png
  flag_clump/*.png
```

## Pretrain

```bash
python -m medical_image_pretraining.pretrain \
  --data_dir data/unlabeled_plots \
  --output_dir runs/hct_mask_pretrain \
  --image_size 224 \
  --patch_size 16 \
  --mask_ratio 0.6 \
  --batch_size 32 \
  --epochs 100
```

Useful outputs:

- `checkpoint_latest.pt`: latest pretraining checkpoint.
- `checkpoint_epoch_XXXX.pt`: periodic checkpoints.
- `previews/epoch_XXXX.png`: original, masked, reconstructed image rows.
- `pretrain_args.json`: reproducibility metadata.

## Fine-tune a classifier

```bash
python -m medical_image_pretraining.finetune_classifier \
  --train_dir data/classifier/train \
  --val_dir data/classifier/val \
  --checkpoint runs/hct_mask_pretrain/checkpoint_latest.pt \
  --output_dir runs/hct_classifier \
  --batch_size 32 \
  --epochs 50
```

The fine-tuning script saves:

- `checkpoint_best.pt`
- `checkpoint_latest.pt`
- `class_to_idx.json`
- `finetune_args.json`

## Notes for hematology plot training

- Keep a separate validation split by instrument, reagent lot, or acquisition
  window when possible. Random image-level splits can overestimate performance
  if near-duplicate plots from the same run appear in both train and validation.
- Start with mask ratios between `0.5` and `0.7`. Scattergrams with sparse
  populations often benefit from slightly lower masking than natural images.
- Avoid applying heavy color jitter unless the instrument rendering pipeline is
  known to vary in color. The plotted geometry is usually more important than
  visual style augmentation.
- Exported images should be de-identified and should not include patient
  metadata in labels, legends, or filenames.
