# API reference

Interactive OpenAPI documentation is available at `/docs` while the service
is running.

## Health

```http
GET /health
```

Example:

```json
{
  "status": "ok",
  "service": "liss-iv-reconstruction",
  "model_ready": true,
  "inference_mode": "baseline"
}
```

## Reconstruct a scene

```http
POST /v1/reconstruct
Content-Type: multipart/form-data
```

| Form field | Type | Required | Notes |
|---|---|---:|---|
| `scene` | file | yes | GeoTIFF, TIFF, PNG, or JPEG |
| `cloud_sensitivity` | number | no | Range 0.2–0.9; default 0.58 |

Example:

```bash
curl -X POST http://localhost:8000/v1/reconstruct \
  -F 'scene=@scene.tif' \
  -F 'cloud_sensitivity=0.58'
```

Response:

```json
{
  "requestId": "4a57f265-8992-44d6-a6bf-d2866a4bf63f",
  "originalPreview": "data:image/png;base64,...",
  "reconstructedPreview": "data:image/png;base64,...",
  "maskPreview": "data:image/png;base64,...",
  "uncertaintyPreview": "data:image/png;base64,...",
  "metrics": {
    "cloudCoverage": 34.21,
    "confidence": 87.65,
    "processingTimeMs": 1832,
    "psnrEstimate": 30.42,
    "spectralAngle": 3.62
  },
  "model": "spectral-spatial-baseline",
  "mode": "baseline"
}
```

Preview fields are data URLs for immediate rendering. A production extension
should expose a separate export endpoint for full-resolution GeoTIFF outputs
and preserve the original CRS, affine transform, and no-data metadata.

## Error responses

| Status | Meaning |
|---:|---|
| 400 | Empty upload |
| 413 | Upload exceeds `MAX_UPLOAD_MB` |
| 422 | Unsupported format, invalid raster, invalid sensitivity, or unusable mask |

