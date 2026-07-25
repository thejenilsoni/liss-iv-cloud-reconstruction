# LISS-IV Cloud Reconstruction

An end-to-end geospatial AI workspace for detecting clouds and reconstructing
cloud-obscured regions in multispectral LISS-IV satellite imagery.

The system combines a mask-guided generative model, spectral-consistency
constraints, uncertainty estimation, and an operator-focused web interface.
It is designed around ISRO Bharatiya Antariksh Hackathon 2026 Problem
Statement 2.

> Current status: application, inference baseline, training pipeline, tests,
> containers, and CI are implemented. Trained weights are intentionally not
> bundled; supply paired data and a validated checkpoint for learned output.

## What is included

- Multispectral GeoTIFF and common image ingestion
- Automated cloud and cloud-shadow mask estimation
- Mask-guided generative reconstruction model
- Spectral-aware deterministic fallback for checkpoint-free demos
- Per-pixel confidence and uncertainty output
- PSNR, SSIM, SAM, MAE, and cloud-region coverage metrics
- Next.js reconstruction workspace with upload and result inspection
- FastAPI inference service and typed API contract
- Reproducible training, evaluation, container, and CI workflows

## Repository layout

```text
apps/
  api/        FastAPI inference and raster-processing service
  web/        Next.js operator workspace
ml/           Model, dataset, losses, training, and evaluation
docs/         Architecture, methodology, API, and data guidance
```

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:3000` for the workspace and
`http://localhost:8000/docs` for the API explorer.

For local development without containers:

```bash
make install
make dev
```

## Input expectations

The preferred input is a georeferenced LISS-IV GeoTIFF with green, red, and
near-infrared bands. Four-band imagery is also supported. PNG and JPEG files
can be used for interface demonstrations but do not preserve geospatial
metadata.

The repository does not redistribute satellite scenes or trained weights.
Place local data under `data/raw/` and model checkpoints under `checkpoints/`;
both paths are ignored by Git.

## Model strategy

The learning pipeline receives the cloudy observation and a cloud mask. A
gated encoder-decoder predicts missing reflectance while retaining observed
pixels outside the mask. Training combines masked reconstruction,
spectral-angle, structural, edge, and adversarial objectives. At inference
time, Monte Carlo passes estimate epistemic uncertainty.

When no checkpoint is configured, the API uses a deterministic
spectral-spatial inpainting baseline. This makes the full product testable
without presenting baseline output as a trained result.

## Development

```bash
make test
make lint
make build
```

See [docs/architecture.md](docs/architecture.md) and
[docs/methodology.md](docs/methodology.md) for implementation details. Dataset
preparation is covered in [docs/data.md](docs/data.md), and the HTTP contract
is documented in [docs/api.md](docs/api.md).

## License

MIT
