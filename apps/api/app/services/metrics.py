from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(slots=True)
class QualitySummary:
    cloud_coverage: float
    confidence: float
    psnr_estimate: float
    spectral_angle: float


def quality_summary(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    uncertainty: np.ndarray,
) -> QualitySummary:
    """Calculate reference-free operational quality indicators.

    PSNR is estimated at the observed/reconstructed boundary because a clear
    ground-truth pixel is not available under real clouds. Full-reference
    metrics are provided separately by the training evaluation pipeline.
    """

    mask_bool = mask >= 0.5
    coverage = float(mask_bool.mean() * 100.0)
    confidence = (
        float((1.0 - uncertainty[mask_bool].mean()) * 100.0)
        if mask_bool.any()
        else 100.0
    )

    outer = ndimage.binary_dilation(mask_bool, iterations=5) & ~mask_bool
    inner = mask_bool & ~ndimage.binary_erosion(mask_bool, iterations=5)
    if outer.any() and inner.any():
        outside_spectrum = original[:, outer].mean(axis=1)
        inside_spectrum = reconstructed[:, inner].mean(axis=1)
        boundary_error = float(np.mean(np.abs(outside_spectrum - inside_spectrum)))
    else:
        boundary_error = 0.025
    psnr = float(np.clip(20.0 * np.log10(1.0 / (boundary_error + 1e-6)), 0, 60))

    valid = ~mask_bool
    observed_signature = (
        original[:, valid].mean(axis=1) if valid.any() else original.mean(axis=(1, 2))
    )
    reconstructed_signature = (
        reconstructed[:, mask_bool].mean(axis=1)
        if mask_bool.any()
        else reconstructed.mean(axis=(1, 2))
    )
    spectral_angle = spectral_angle_degrees(observed_signature, reconstructed_signature)

    return QualitySummary(
        cloud_coverage=round(coverage, 2),
        confidence=round(float(np.clip(confidence, 0, 100)), 2),
        psnr_estimate=round(psnr, 2),
        spectral_angle=round(spectral_angle, 3),
    )


def spectral_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.dot(first, second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def full_reference_metrics(
    clear: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Metrics for validation sets where a clear target exists."""

    target = mask >= 0.5
    if not target.any():
        target = np.ones_like(mask, dtype=bool)
    difference = clear[:, target] - prediction[:, target]
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = float(20 * np.log10(1.0 / np.sqrt(mse + 1e-12)))

    angles = []
    for row in range(clear.shape[1]):
        active_columns = np.flatnonzero(target[row])
        for column in active_columns[::8]:
            angles.append(
                spectral_angle_degrees(clear[:, row, column], prediction[:, row, column])
            )
    sam = float(np.mean(angles)) if angles else 0.0
    return {"mae": mae, "mse": mse, "psnr": psnr, "sam_degrees": sam}

