"""Live integration tests against a running ComfyUI instance on the DGX.

These are skipped unless ``COMFYUI_BASE_URL`` points at a reachable ComfyUI.
Set it to run them:

    COMFYUI_BASE_URL=http://192.168.2.123:8188 uv pytest tests/integration/
"""

from __future__ import annotations

import os
from io import BytesIO

import pytest
from PIL import Image

from store_listing.orchestration.comfyui_client import ComfyUIClient

BASE_URL = os.environ.get("COMFYUI_BASE_URL", "")

pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="requires a live ComfyUI (set COMFYUI_BASE_URL)"
)


@pytest.fixture
def client() -> ComfyUIClient:
    return ComfyUIClient(base_url=BASE_URL, timeout_seconds=300.0)


def test_comfyui_is_healthy(client: ComfyUIClient) -> None:
    assert client.health() is True


def test_flux_generation_returns_a_png(client: ComfyUIClient) -> None:
    result = client.generate_image(
        prompt="a red bicycle leaning against a white wall, product photo, no text",
        width=512,
        height=512,
    )

    assert result.error is None
    assert result.image_bytes is not None
    with Image.open(BytesIO(result.image_bytes)) as image:
        assert image.format == "PNG"
        assert image.size == (512, 512)


def test_birefnet_returns_a_transparent_png(client: ComfyUIClient) -> None:
    source = client.generate_image(
        prompt="a single orange pumpkin on a plain white background",
        width=512,
        height=512,
    )
    assert source.image_bytes is not None

    result = client.remove_background(source.image_bytes)

    assert result.error is None
    assert result.image_bytes is not None
    with Image.open(BytesIO(result.image_bytes)) as image:
        assert image.mode == "RGBA"
        alpha_bytes = image.getchannel("A").tobytes()
        assert min(alpha_bytes) < 255


def test_a_missing_model_is_reported_clearly(client: ComfyUIClient) -> None:
    from store_listing.orchestration.comfyui_client import ComfyUIError

    with pytest.raises(ComfyUIError) as excinfo:
        client.generate_image(prompt="test", unet_name="does-not-exist.safetensors")

    assert "does-not-exist.safetensors" in str(excinfo.value)
