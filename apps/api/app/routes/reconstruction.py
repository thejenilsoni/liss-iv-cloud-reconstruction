import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.schemas import ReconstructionMetrics, ReconstructionResponse
from app.services.cloud_mask import estimate_cloud_mask
from app.services.metrics import quality_summary
from app.services.raster_io import mask_data_uri, preview_data_uri, read_scene

router = APIRouter(prefix="/v1", tags=["reconstruction"])


@router.post(
    "/reconstruct",
    response_model=ReconstructionResponse,
    status_code=status.HTTP_200_OK,
)
async def reconstruct(
    request: Request,
    scene: Annotated[UploadFile, File(description="GeoTIFF, TIFF, PNG, or JPEG scene")],
    cloud_sensitivity: Annotated[float, Form(ge=0.2, le=0.9)] = 0.58,
) -> ReconstructionResponse:
    started = time.perf_counter()
    payload = await scene.read()
    max_bytes = request.app.state.settings.max_upload_mb * 1024 * 1024
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded scene is empty.")
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"The scene exceeds the {request.app.state.settings.max_upload_mb} MB limit.",
        )

    try:
        raster = read_scene(payload, scene.filename or "scene.tif")
        mask_result = estimate_cloud_mask(raster.bands, cloud_sensitivity)
        output = request.app.state.reconstructor.reconstruct(
            raster.bands,
            mask_result.mask,
        )
        summary = quality_summary(
            raster.bands,
            output.reconstructed,
            mask_result.mask,
            output.uncertainty,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return ReconstructionResponse(
        requestId=str(uuid4()),
        originalPreview=preview_data_uri(raster.bands),
        reconstructedPreview=preview_data_uri(output.reconstructed),
        maskPreview=mask_data_uri(mask_result.mask),
        uncertaintyPreview=mask_data_uri(output.uncertainty, colour=True),
        metrics=ReconstructionMetrics(
            cloudCoverage=summary.cloud_coverage,
            confidence=summary.confidence,
            processingTimeMs=elapsed_ms,
            psnrEstimate=summary.psnr_estimate,
            spectralAngle=summary.spectral_angle,
        ),
        model=output.model_name,
        mode=output.mode,
    )

