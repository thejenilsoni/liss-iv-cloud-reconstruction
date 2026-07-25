from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


class LissCloudDataset(Dataset[dict[str, torch.Tensor]]):
    """Clear-scene patch dataset with procedural cloud augmentation.

    Each `.npz` file must contain `clear` as a C×H×W float array. Optional
    paired `cloudy` and `mask` arrays are used when available.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 256,
        bands: int = 3,
        augment: bool = True,
        seed: int = 2026,
    ) -> None:
        self.files = sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz scene patches found in {root}")
        if len(self.files) < 2:
            raise ValueError("At least two patches are required for training and validation.")
        self.patch_size = patch_size
        self.bands = bands
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        with np.load(self.files[index]) as sample:
            clear = sample["clear"][: self.bands].astype(np.float32)
            cloudy = sample["cloudy"][: self.bands].astype(np.float32) if "cloudy" in sample else None
            mask = sample["mask"].astype(np.float32) if "mask" in sample else None

        clear = self._crop(clear, rng)
        if cloudy is not None and mask is not None:
            cloudy = self._crop(cloudy, rng, coordinates=self._last_crop)
            mask = self._crop_mask(mask, self._last_crop)
        else:
            mask = procedural_cloud_mask(self.patch_size, self.patch_size, rng)
            cloudy = apply_cloud(clear, mask, rng)

        if self.augment:
            clear, cloudy, mask = self._augment(clear, cloudy, mask, rng)

        return {
            "clear": torch.from_numpy(np.ascontiguousarray(clear)),
            "cloudy": torch.from_numpy(np.ascontiguousarray(cloudy)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])),
        }

    def _crop(
        self,
        array: np.ndarray,
        rng: np.random.Generator,
        coordinates: tuple[int, int] | None = None,
    ) -> np.ndarray:
        _, height, width = array.shape
        if height < self.patch_size or width < self.patch_size:
            pad_height = max(0, self.patch_size - height)
            pad_width = max(0, self.patch_size - width)
            array = np.pad(
                array,
                ((0, 0), (0, pad_height), (0, pad_width)),
                mode="reflect",
            )
            _, height, width = array.shape
        if coordinates is None:
            top = int(rng.integers(0, height - self.patch_size + 1))
            left = int(rng.integers(0, width - self.patch_size + 1))
            self._last_crop = (top, left)
        else:
            top, left = coordinates
        return array[:, top : top + self.patch_size, left : left + self.patch_size]

    def _crop_mask(self, mask: np.ndarray, coordinates: tuple[int, int]) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask[0]
        top, left = coordinates
        if mask.shape[0] < self.patch_size or mask.shape[1] < self.patch_size:
            mask = np.pad(
                mask,
                (
                    (0, max(0, self.patch_size - mask.shape[0])),
                    (0, max(0, self.patch_size - mask.shape[1])),
                ),
                mode="reflect",
            )
        return mask[top : top + self.patch_size, left : left + self.patch_size]

    @staticmethod
    def _augment(
        clear: np.ndarray,
        cloudy: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            clear, cloudy, mask = clear[:, :, ::-1], cloudy[:, :, ::-1], mask[:, ::-1]
        if rng.random() < 0.5:
            clear, cloudy, mask = clear[:, ::-1, :], cloudy[:, ::-1, :], mask[::-1, :]
        rotations = int(rng.integers(0, 4))
        clear = np.rot90(clear, rotations, axes=(1, 2))
        cloudy = np.rot90(cloudy, rotations, axes=(1, 2))
        mask = np.rot90(mask, rotations)
        return clear, cloudy, mask


def procedural_cloud_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate contiguous cloud structures at multiple spatial scales."""

    fine = ndimage.gaussian_filter(rng.normal(size=(height, width)), sigma=8)
    broad = ndimage.gaussian_filter(rng.normal(size=(height, width)), sigma=24)
    field = 0.58 * _normalize(fine) + 0.42 * _normalize(broad)
    target_coverage = rng.uniform(0.12, 0.62)
    threshold = np.quantile(field, 1.0 - target_coverage)
    mask = field >= threshold
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_dilation(mask, iterations=int(rng.integers(1, 5)))
    return mask.astype(np.float32)


def apply_cloud(
    clear: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    opacity = ndimage.gaussian_filter(mask, sigma=float(rng.uniform(2.5, 6.0)))
    opacity = np.clip(opacity * rng.uniform(0.72, 1.0), 0, 1)
    cloud_colour = rng.uniform(0.78, 0.98, size=(clear.shape[0], 1, 1))
    cloud_texture = ndimage.gaussian_filter(
        rng.normal(0, 0.025, size=clear.shape),
        sigma=(0, 3, 3),
    )
    cloudy = clear * (1.0 - opacity) + (cloud_colour + cloud_texture) * opacity
    return np.clip(cloudy, 0, 1).astype(np.float32)


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [1, 99])
    return np.clip((values - low) / (high - low + 1e-8), 0, 1)
