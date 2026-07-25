import numpy as np

from app.services.cloud_mask import estimate_cloud_mask


def test_bright_smooth_region_is_detected_as_cloud() -> None:
    rng = np.random.default_rng(7)
    bands = rng.normal(0.31, 0.025, size=(3, 128, 128)).astype(np.float32)
    bands[:, 36:94, 42:101] = 0.91

    result = estimate_cloud_mask(np.clip(bands, 0, 1), sensitivity=0.62)

    assert result.mask.shape == (128, 128)
    assert result.cloud_probability.shape == result.mask.shape
    assert result.mask[48:82, 54:88].mean() > 0.85
    assert 0.05 < result.mask.mean() < 0.65


def test_mask_sensitivity_is_bounded() -> None:
    bands = np.full((3, 32, 32), 0.35, dtype=np.float32)
    low = estimate_cloud_mask(bands, sensitivity=-4)
    high = estimate_cloud_mask(bands, sensitivity=4)

    assert np.isfinite(low.cloud_probability).all()
    assert np.isfinite(high.cloud_probability).all()

