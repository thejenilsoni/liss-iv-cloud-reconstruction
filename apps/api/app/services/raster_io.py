import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(slots=True)
class RasterScene:
    """Normalized multispectral raster and its optional geospatial metadata."""

    bands: np.ndarray
    profile: dict[str, Any] = field(default_factory=dict)
    source_dtype: str = "uint8"

    @property
    def height(self) -> int:
        return int(self.bands.shape[1])

    @property
    def width(self) -> int:
        return int(self.bands.shape[2])


def read_scene(payload: bytes, filename: str) -> RasterScene:
    """Read a GeoTIFF or common image into a normalized C×H×W array."""

    suffix = Path(filename).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _read_geotiff(payload)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _read_image(payload)
    raise ValueError("Unsupported scene format. Use GeoTIFF, TIFF, PNG, or JPEG.")


def _read_geotiff(payload: bytes) -> RasterScene:
    try:
        from rasterio.io import MemoryFile
    except ImportError as exc:  # pragma: no cover - dependency is present in production
        raise RuntimeError("GeoTIFF support requires rasterio.") from exc

    with MemoryFile(payload) as memory_file, memory_file.open() as dataset:
        count = min(dataset.count, 4)
        if count < 1:
            raise ValueError("The GeoTIFF does not contain any raster bands.")
        raw = dataset.read(indexes=list(range(1, count + 1)))
        profile = {
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "transform": tuple(dataset.transform),
            "bounds": tuple(dataset.bounds),
            "nodata": dataset.nodata,
            "count": dataset.count,
        }
    bands = _normalize(raw, profile["nodata"])
    return RasterScene(bands=bands, profile=profile, source_dtype=str(raw.dtype))


def _read_image(payload: bytes) -> RasterScene:
    with Image.open(io.BytesIO(payload)) as image:
        image = image.convert("RGB")
        raw = np.asarray(image)
    bands = np.moveaxis(raw, -1, 0).astype(np.float32) / 255.0
    return RasterScene(bands=bands, source_dtype=str(raw.dtype))


def _normalize(raw: np.ndarray, nodata_value: float | None = None) -> np.ndarray:
    bands = raw.astype(np.float32)
    normalized = np.zeros_like(bands, dtype=np.float32)
    nodata = ~np.isfinite(bands)
    if nodata_value is not None and np.isfinite(nodata_value):
        nodata |= bands == nodata_value

    for index, band in enumerate(bands):
        valid = band[~nodata[index]]
        if valid.size == 0:
            continue
        low, high = np.percentile(valid, [1, 99])
        if high <= low:
            high = low + 1.0
        normalized[index] = np.clip((band - low) / (high - low), 0.0, 1.0)
        normalized[index][nodata[index]] = 0.0

    return normalized


def preview_data_uri(bands: np.ndarray, *, false_colour: bool = True) -> str:
    """Create a compact PNG preview from a C×H×W normalized raster."""

    display = _display_bands(bands, false_colour=false_colour)
    display = _resize_for_preview(display)
    image = Image.fromarray((np.clip(display, 0.0, 1.0) * 255).astype(np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def mask_data_uri(mask: np.ndarray, *, colour: bool = False) -> str:
    values = np.clip(mask, 0.0, 1.0)
    if colour:
        red = np.clip(values * 1.4, 0, 1)
        green = np.clip(1.0 - np.abs(values - 0.5) * 1.5, 0, 1)
        blue = np.clip(1.1 - values * 1.5, 0, 1)
        display = np.stack([red, green, blue], axis=-1)
    else:
        display = np.repeat(values[..., None], 3, axis=-1)
    display = _resize_for_preview(display)
    image = Image.fromarray((display * 255).astype(np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _display_bands(bands: np.ndarray, *, false_colour: bool) -> np.ndarray:
    count = bands.shape[0]
    if count == 1:
        return np.repeat(bands[0][..., None], 3, axis=-1)
    if false_colour and count >= 3:
        # LISS-IV commonly arrives as green, red, NIR. NIR-R-G highlights vegetation.
        indexes = [2, 1, 0]
    else:
        indexes = [0, min(1, count - 1), min(2, count - 1)]
    return np.stack([bands[index] for index in indexes], axis=-1)


def _resize_for_preview(display: np.ndarray, max_size: int = 960) -> np.ndarray:
    height, width = display.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale == 1.0:
        return display
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray((display * 255).astype(np.uint8), "RGB")
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.float32) / 255.0
