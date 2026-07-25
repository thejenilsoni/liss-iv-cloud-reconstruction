# Data preparation

## Preferred source product

Use radiometrically corrected, coregistered LISS-IV multispectral scenes with
green, red, and near-infrared bands. The official mission material describes
LISS-IV as a three-band VNIR instrument with 5.8 m spatial resolution.

Do not assume that every downloaded file has identical:

- band ordering;
- scale and offset;
- no-data value;
- radiometric depth;
- projection;
- spatial alignment.

Read the accompanying product metadata before preparing patches.

## Training patch contract

The dataset loader reads compressed NumPy files:

```text
patch_00001.npz
  clear   float32 [C, H, W], normalized to [0, 1]
  cloudy  float32 [C, H, W], optional
  mask    float32 [H, W], optional, 1 = reconstruct
```

When only `clear` is present, the loader generates multiscale procedural
clouds and corresponding opacity masks. Paired cloudy/clear data should be
used whenever reliable coregistration is available.

## Preparing clear scenes

```bash
python scripts/prepare_dataset.py \
  --input data/raw/clear \
  --output data/processed/train \
  --patch-size 256 \
  --stride 192 \
  --bands 3
```

Recommended dataset partitions:

```text
data/processed/
  train/
  validation/
  test/
```

Create partitions by geographic scene or acquisition region before
extracting patches. Patch-level random splits cause spatial leakage.

## Training

```bash
python -m ml.train \
  --data data/processed/train \
  --output checkpoints \
  --epochs 80 \
  --batch-size 8 \
  --patch-size 256 \
  --bands 3
```

The best validation checkpoint is saved as `best.pt`. A traced deployment
artifact is saved as `generator.ts`.

## Evaluation

Evaluation files must contain `clear`, `cloudy`, and `mask`:

```bash
python -m ml.evaluate \
  --model checkpoints/generator.ts \
  --data data/processed/test
```

## Dataset hygiene

- Retain acquisition date, path/row or tile ID, and processing level in a
  separate manifest.
- Record cloud-mask provenance.
- Exclude saturated, corrupt, or badly registered samples.
- Keep coastlines, snow, salt pans, and bright roofs represented because they
  are common cloud-mask confounders.
- Measure class and geography distribution across splits.
- Never commit licensed scenes or restricted products to this repository.

