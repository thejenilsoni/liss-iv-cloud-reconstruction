from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(slots=True)
class CloudMaskResult:
    mask: np.ndarray
    cloud_probability: np.ndarray
    shadow_probability: np.ndarray


def estimate_cloud_mask(
    bands: np.ndarray,
    sensitivity: float = 0.58,
) -> CloudMaskResult:
    """Estimate clouds and adjacent shadows without blue or SWIR bands.

    LISS-IV scenes generally provide green, red, and near-infrared bands. The
    heuristic therefore combines visible brightness, low vegetation contrast,
    local texture, and adjacency-aware shadow evidence. It is designed as a
    transparent preprocessing baseline; learned masks can replace it later.
    """

    if bands.ndim != 3 or bands.shape[0] < 1:
        raise ValueError("Expected a C×H×W raster with at least one band.")
    sensitivity = float(np.clip(sensitivity, 0.2, 0.9))

    green = bands[0]
    red = bands[min(1, bands.shape[0] - 1)]
    nir = bands[min(2, bands.shape[0] - 1)]

    visible = (green + red) / 2.0
    brightness = (green + red + nir) / 3.0
    ndvi = (nir - red) / (nir + red + 1e-6)
    whiteness = 1.0 - np.clip(np.std(np.stack([green, red, nir]), axis=0) * 3.2, 0, 1)

    local_mean = ndimage.uniform_filter(brightness, size=7, mode="reflect")
    local_sq_mean = ndimage.uniform_filter(brightness**2, size=7, mode="reflect")
    texture = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    smoothness = 1.0 - np.clip(texture * 5.0, 0, 1)
    low_vegetation = np.clip(1.0 - (ndvi + 0.15) / 0.65, 0, 1)

    cloud_probability = np.clip(
        0.42 * brightness
        + 0.24 * visible
        + 0.17 * whiteness
        + 0.11 * smoothness
        + 0.06 * low_vegetation,
        0,
        1,
    )

    # Higher sensitivity lowers the cutoff while remaining scene-adaptive.
    percentile = 94.0 - (sensitivity - 0.2) * 31.0
    adaptive_threshold = float(np.percentile(cloud_probability, percentile))
    threshold = np.clip(
        0.72 * adaptive_threshold + 0.28 * (0.78 - sensitivity * 0.22),
        0.48,
        0.82,
    )
    cloud = cloud_probability >= threshold

    cloud = ndimage.binary_opening(cloud, structure=np.ones((3, 3)))
    cloud = ndimage.binary_closing(cloud, structure=np.ones((5, 5)))
    cloud = ndimage.binary_fill_holes(cloud)
    cloud = ndimage.binary_dilation(cloud, iterations=2)

    # Dark pixels close to cloud components are plausible projected shadows.
    dark_score = np.clip((0.32 - brightness) / 0.32, 0, 1)
    cloud_neighbourhood = ndimage.binary_dilation(cloud, iterations=18)
    displaced_cloud = np.roll(np.roll(cloud, 9, axis=0), 6, axis=1)
    shadow_probability = dark_score * (
        0.55 * cloud_neighbourhood.astype(np.float32)
        + 0.45 * displaced_cloud.astype(np.float32)
    )
    shadow = shadow_probability > (0.48 + (1.0 - sensitivity) * 0.12)
    shadow = ndimage.binary_opening(shadow, structure=np.ones((3, 3)))

    combined = ndimage.binary_dilation(cloud | shadow, iterations=2)
    return CloudMaskResult(
        mask=combined.astype(np.float32),
        cloud_probability=cloud_probability.astype(np.float32),
        shadow_probability=shadow_probability.astype(np.float32),
    )

