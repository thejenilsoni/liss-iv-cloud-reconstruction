import numpy as np

from app.services.reconstruction import SpectralSpatialInpainter


def make_scene(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.stack(
        [
            0.2 + x / size * 0.5,
            0.15 + y / size * 0.55,
            0.1 + (x + y) / (size * 2) * 0.7,
        ]
    ).astype(np.float32)


def test_inpainter_preserves_observed_pixels() -> None:
    bands = make_scene()
    mask = np.zeros((96, 96), dtype=np.float32)
    mask[28:70, 34:76] = 1

    output = SpectralSpatialInpainter().reconstruct(bands, mask)

    observed = mask == 0
    np.testing.assert_allclose(output.reconstructed[:, observed], bands[:, observed])
    assert np.isfinite(output.reconstructed).all()
    assert output.reconstructed.min() >= 0
    assert output.reconstructed.max() <= 1
    assert output.uncertainty[mask == 1].mean() > output.uncertainty[mask == 0].mean()
    assert output.mode == "baseline"


def test_empty_mask_returns_identity() -> None:
    bands = make_scene(32)
    output = SpectralSpatialInpainter().reconstruct(
        bands,
        np.zeros((32, 32), dtype=np.float32),
    )

    np.testing.assert_array_equal(output.reconstructed, bands)
    assert output.uncertainty.sum() == 0

