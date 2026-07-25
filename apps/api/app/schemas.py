from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_ready: bool
    inference_mode: Literal["learned", "baseline"]


class ReconstructionMetrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cloud_coverage: float = Field(alias="cloudCoverage")
    confidence: float
    processing_time_ms: int = Field(alias="processingTimeMs")
    psnr_estimate: float = Field(alias="psnrEstimate")
    spectral_angle: float = Field(alias="spectralAngle")


class ReconstructionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    original_preview: str = Field(alias="originalPreview")
    reconstructed_preview: str = Field(alias="reconstructedPreview")
    mask_preview: str = Field(alias="maskPreview")
    uncertainty_preview: str = Field(alias="uncertaintyPreview")
    metrics: ReconstructionMetrics
    model: str
    mode: Literal["learned", "baseline"]

