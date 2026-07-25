import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def make_png() -> bytes:
    y, x = np.mgrid[0:96, 0:96]
    scene = np.stack(
        [
            45 + x * 1.2,
            55 + y * 1.1,
            60 + (x + y) * 0.7,
        ],
        axis=-1,
    )
    scene[30:68, 35:78] = 238
    buffer = io.BytesIO()
    Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_reconstruction_contract() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post(
            "/v1/reconstruct",
            files={"scene": ("scene.png", make_png(), "image/png")},
            data={"cloud_sensitivity": "0.62"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requestId"]
    assert payload["originalPreview"].startswith("data:image/png;base64,")
    assert payload["reconstructedPreview"].startswith("data:image/png;base64,")
    assert payload["mode"] in {"baseline", "learned"}
    assert 0 <= payload["metrics"]["confidence"] <= 100

