import numpy as np
import pytest

from app.services.metrics import full_reference_metrics, spectral_angle_degrees


def test_spectral_angle_for_identical_signatures_is_zero() -> None:
    signature = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    assert spectral_angle_degrees(signature, signature) == pytest.approx(0.0, abs=1e-3)


def test_full_reference_metrics_react_to_error() -> None:
    clear = np.full((3, 16, 16), 0.5, dtype=np.float32)
    prediction = clear.copy()
    prediction[:, 4:12, 4:12] += 0.1
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1

    metrics = full_reference_metrics(clear, prediction, mask)

    assert metrics["mae"] == pytest.approx(0.1, abs=1e-5)
    assert metrics["mse"] == pytest.approx(0.01, abs=1e-5)
    assert metrics["psnr"] == pytest.approx(20.0, abs=1e-3)

