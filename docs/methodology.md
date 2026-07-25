# Reconstruction methodology

## Sensor assumptions

Resourcesat-2A LISS-IV operates in three visible and near-infrared bands at
5.8 m spatial resolution. The pipeline therefore assumes the common
green-red-NIR ordering when three bands are supplied, while keeping the model
band count configurable. The exact product metadata and ordering should
always be checked before training.

Primary references:

- [ISRO Resourcesat-2A mission page](https://www.isro.gov.in/RESOURCESAT_2A.html)
- [NRSC Resourcesat-2 Data Users' Handbook](https://bhoonidhi.nrsc.gov.in/bhoonidhi_resources/help/docs/R2_data_user_handbook.pdf)

## Problem formulation

Let \(X \in [0,1]^{C \times H \times W}\) be the cloudy observation,
\(M \in \{0,1\}^{1 \times H \times W}\) the reconstruction mask, and
\(Y\) the unavailable clear target at inference time. The generator predicts:

\[
\hat{Y} = (1-M) \odot X + M \odot G([X, M]).
\]

This residual merge is an important invariant: values outside the declared
cloud and shadow region are not altered.

## Cloud-mask baseline

LISS-IV does not provide blue or SWIR bands, so common multi-band cloud tests
cannot be copied directly. The transparent baseline combines:

- green/red visible brightness;
- total VNIR brightness;
- cross-band whiteness;
- low vegetation contrast;
- low local texture;
- scene-adaptive percentile thresholding;
- morphological opening, closing, and edge dilation.

Possible cloud shadows are dark components near or directionally displaced
from detected clouds. The mask implementation is replaceable so a
segmentation network can be introduced without changing the API.

## Generator

The generator uses gated convolutions, an encoder-decoder topology, dilated
residual bottleneck blocks, skip connections, and dropout. Gating learns how
much corrupted versus contextual information should pass through each
feature channel.

The PatchGAN-style discriminator observes:

\[
[X, \hat{Y}, M]
\]

and judges local spatial realism while remaining conditioned on the input
scene and reconstruction region.

## Training losses

The objective is:

\[
\mathcal{L} =
6.0\mathcal{L}_{masked}
+ 1.0\mathcal{L}_{context}
+ 1.4\mathcal{L}_{SAM}
+ 0.8\mathcal{L}_{structure}
+ 0.6\mathcal{L}_{edge}
+ 0.08\mathcal{L}_{adv}.
\]

| Term | Purpose |
|---|---|
| Masked L1 | Recover clear reflectance in obscured pixels |
| Context L1 | Penalize changes to observable pixels |
| Spectral angle | Preserve inter-band spectral direction |
| Multiscale structure | Match low- and mid-frequency spatial structure |
| Edge | Preserve field, road, river, and settlement boundaries |
| Adversarial | Improve plausible local texture |

## Uncertainty

For a learned checkpoint, repeated stochastic forward passes estimate
epistemic variance. For the baseline, uncertainty combines distance from
observable context and local texture complexity. These values are useful for
ranking pixels for review; they are not calibrated probabilities until
validated on held-out scenes.

## Evaluation protocol

### Paired validation

The defensible primary evaluation uses a clear target with a synthetically
or temporally derived cloud mask:

- masked-region MAE;
- masked-region MSE and PSNR;
- spectral angle mapper (SAM);
- structural similarity (SSIM);
- boundary continuity;
- downstream-task retention, such as segmentation agreement.

Split data geographically—not merely by random patches—to avoid the same
land-cover structures leaking into both training and validation.

### Real cloudy scenes

A true target does not exist underneath real clouds. Therefore:

- the UI labels its displayed PSNR as an estimate;
- spectral angle compares reconstructed and observed neighbourhood
  signatures, not a hidden clear pixel;
- uncertainty and seam indicators guide review;
- claims about absolute reconstruction accuracy must come from paired
  validation, not real-scene reference-free indicators.

## Recommended ablations

1. L1-only generator versus the full loss.
2. Standard convolutions versus gated convolutions.
3. No shadow mask versus combined cloud-shadow mask.
4. No spectral loss versus SAM-constrained training.
5. Single-date input versus auxiliary temporal context.
6. Baseline inpainting versus learned reconstruction.

