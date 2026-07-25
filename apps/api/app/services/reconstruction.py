from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from scipy import ndimage


@dataclass(slots=True)
class ReconstructionOutput:
    reconstructed: np.ndarray
    uncertainty: np.ndarray
    model_name: str
    mode: Literal["learned", "baseline"]


class Reconstructor(Protocol):
    name: str
    mode: Literal["learned", "baseline"]

    def reconstruct(self, bands: np.ndarray, mask: np.ndarray) -> ReconstructionOutput: ...


class SpectralSpatialInpainter:
    """Checkpoint-free baseline that preserves unmasked reflectance exactly."""

    name = "spectral-spatial-baseline"
    mode: Literal["baseline"] = "baseline"

    def reconstruct(self, bands: np.ndarray, mask: np.ndarray) -> ReconstructionOutput:
        mask_bool = mask >= 0.5
        if not mask_bool.any():
            return ReconstructionOutput(
                reconstructed=bands.copy(),
                uncertainty=np.zeros_like(mask, dtype=np.float32),
                model_name=self.name,
                mode=self.mode,
            )
        if mask_bool.all():
            raise ValueError("The detected mask covers the complete scene.")

        distance, indexes = ndimage.distance_transform_edt(
            mask_bool,
            return_distances=True,
            return_indices=True,
        )
        reconstructed = np.empty_like(bands, dtype=np.float32)

        for band_index, band in enumerate(bands):
            nearest = band[indexes[0], indexes[1]]
            fine = ndimage.gaussian_filter(nearest, sigma=2.2, mode="reflect")
            medium = ndimage.gaussian_filter(nearest, sigma=6.0, mode="reflect")
            broad = ndimage.gaussian_filter(nearest, sigma=14.0, mode="reflect")

            blend = np.clip(distance / 28.0, 0.0, 1.0)
            recovered = (
                nearest * (0.38 * (1.0 - blend))
                + fine * (0.42 - 0.12 * blend)
                + medium * (0.16 + 0.22 * blend)
                + broad * (0.04 + 0.28 * blend)
            )

            # Match low-frequency tone at the feathered mask boundary.
            boundary = ndimage.binary_dilation(mask_bool, iterations=3) & ~mask_bool
            if boundary.any():
                observed_mean = float(np.mean(band[boundary]))
                recovered_mean = float(np.mean(recovered[boundary]))
                recovered = recovered + (observed_mean - recovered_mean) * 0.35

            reconstructed[band_index] = np.where(mask_bool, recovered, band)

        reconstructed = np.clip(reconstructed, 0.0, 1.0).astype(np.float32)

        local_mean = ndimage.uniform_filter(np.mean(reconstructed, axis=0), size=9)
        local_sq = ndimage.uniform_filter(np.mean(reconstructed, axis=0) ** 2, size=9)
        texture = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
        texture = texture / (float(np.percentile(texture, 95)) + 1e-6)
        normalized_distance = distance / (float(distance.max()) + 1e-6)
        uncertainty = np.where(
            mask_bool,
            np.clip(0.16 + 0.58 * normalized_distance + 0.18 * texture, 0, 1),
            0.03,
        ).astype(np.float32)

        return ReconstructionOutput(
            reconstructed=reconstructed,
            uncertainty=uncertainty,
            model_name=self.name,
            mode=self.mode,
        )


class TorchScriptReconstructor:
    """Deployment adapter for a traced mask-guided generator checkpoint."""

    name = "mask-guided-generator"
    mode: Literal["learned"] = "learned"

    def __init__(self, checkpoint_path: str, monte_carlo_passes: int = 4) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "A model checkpoint is configured, but the API was installed without the 'ml' extra."
            ) from exc

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(str(path), map_location=self._device)
        self._model.eval()
        self._passes = max(1, monte_carlo_passes)

    def reconstruct(self, bands: np.ndarray, mask: np.ndarray) -> ReconstructionOutput:
        torch = self._torch
        image = torch.from_numpy(bands).unsqueeze(0).to(self._device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self._device)
        model_input = torch.cat([image, mask_tensor], dim=1)

        predictions = []
        with torch.inference_mode():
            for _ in range(self._passes):
                prediction = self._model(model_input)
                if isinstance(prediction, (tuple, list)):
                    prediction = prediction[0]
                predictions.append(prediction.clamp(0, 1))

        stack = torch.stack(predictions)
        mean = stack.mean(dim=0)
        variance = stack.var(dim=0, unbiased=False).mean(dim=1)
        merged = image * (1.0 - mask_tensor) + mean * mask_tensor

        return ReconstructionOutput(
            reconstructed=merged.squeeze(0).cpu().numpy().astype(np.float32),
            uncertainty=variance.squeeze(0).cpu().numpy().astype(np.float32),
            model_name=self.name,
            mode=self.mode,
        )


def create_reconstructor(model_path: str | None) -> Reconstructor:
    if model_path:
        return TorchScriptReconstructor(model_path)
    return SpectralSpatialInpainter()

