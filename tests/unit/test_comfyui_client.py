"""Tests for the ComfyUI REST client used by the DGX image pipeline."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from PIL import Image

from store_listing.orchestration.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
    _alpha_polarity_is_inverted,
    _build_calibration_image,
    _invert_alpha,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _png_bytes(mode: str = "RGB") -> bytes:
    image = Image.new(mode, (8, 8), (255, 0, 0) if mode == "RGB" else (255, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _prompt_ok(prompt_id: str = "prompt-123") -> httpx.Response:
    return httpx.Response(200, json={"prompt_id": prompt_id, "number": 1, "node_errors": {}})


def _history_ok(filename: str = "store_listing_00001_.png", node_id: str = "9") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "prompt-123": {
                "outputs": {
                    node_id: {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}
                },
                "status": {"status_str": "success", "completed": True, "messages": []},
            }
        },
    )


def _client(handler: Handler, **kwargs: Any) -> ComfyUIClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return ComfyUIClient(base_url="http://comfy.test:8188", http_client=http_client, **kwargs)


class TestGenerateImage:
    def test_returns_image_bytes_on_the_happy_path(self) -> None:
        png = _png_bytes()
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            seen[path] = request.method
            if path == "/prompt":
                return _prompt_ok()
            if path.startswith("/history/"):
                return _history_ok()
            if path == "/view":
                return httpx.Response(200, content=png)
            return httpx.Response(404)

        client = _client(handler)
        result = client.generate_image(prompt="a red bicycle")

        assert result.error is None
        assert result.image_bytes == png
        assert result.prompt_id == "prompt-123"
        assert seen["/prompt"] == "POST"
        assert seen["/view"] == "GET"

    def test_returns_a_decodable_png(self) -> None:
        png = _png_bytes()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/prompt":
                return _prompt_ok()
            if request.url.path.startswith("/history/"):
                return _history_ok()
            if request.url.path == "/view":
                return httpx.Response(200, content=png)
            return httpx.Response(404)

        result = _client(handler).generate_image(prompt="a red bicycle")
        assert result.image_bytes is not None
        assert Image.open(io.BytesIO(result.image_bytes)).format == "PNG"

    def test_posts_the_flux_graph(self) -> None:
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/prompt":
                import json

                bodies.append(json.loads(request.content))
                return _prompt_ok()
            if request.url.path.startswith("/history/"):
                return _history_ok()
            if request.url.path == "/view":
                return httpx.Response(200, content=_png_bytes())
            return httpx.Response(404)

        _client(handler).generate_image(prompt="a red bicycle", width=512, height=512, seed=7)

        assert len(bodies) == 1
        graph = bodies[0]["prompt"]
        assert graph["7"]["inputs"]["steps"] == 4
        assert graph["7"]["inputs"]["cfg"] == 1.0
        assert graph["7"]["inputs"]["seed"] == 7
        assert graph["6"]["inputs"]["width"] == 512
        assert graph["4"]["inputs"]["text"] == "a red bicycle"

    def test_raises_with_the_node_detail_when_the_graph_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "prompt_outputs_failed_validation",
                        "message": "Prompt outputs failed validation",
                    },
                    "node_errors": {
                        "1": {
                            "class_type": "UNETLoader",
                            "errors": [
                                {
                                    "type": "value_not_in_list",
                                    "message": "Value not in list",
                                    "details": ("unet_name: 'flux1-schnell.safetensors' not in []"),
                                }
                            ],
                        }
                    },
                },
            )

        with pytest.raises(ComfyUIError) as excinfo:
            _client(handler).generate_image(prompt="a red bicycle")

        assert "flux1-schnell.safetensors" in str(excinfo.value)
        assert "UNETLoader" in str(excinfo.value)

    def test_raises_when_execution_reports_an_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/prompt":
                return _prompt_ok()
            return httpx.Response(
                200,
                json={
                    "prompt-123": {
                        "outputs": {},
                        "status": {
                            "status_str": "error",
                            "completed": True,
                            "messages": [
                                [
                                    "execution_error",
                                    {"exception_message": "CUDA out of memory"},
                                ]
                            ],
                        },
                    }
                },
            )

        with pytest.raises(ComfyUIError) as excinfo:
            _client(handler).generate_image(prompt="a red bicycle")

        assert "CUDA out of memory" in str(excinfo.value)

    def test_raises_when_the_workflow_never_completes(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/prompt":
                return _prompt_ok()
            return httpx.Response(
                200,
                json={
                    "prompt-123": {
                        "outputs": {},
                        "status": {"status_str": "running", "completed": False, "messages": []},
                    }
                },
            )

        with pytest.raises(ComfyUIError) as excinfo:
            _client(handler, timeout_seconds=0.2, poll_interval_seconds=0.0).generate_image(
                prompt="a red bicycle"
            )

        assert "timed out" in str(excinfo.value).lower()

    def test_raises_when_the_view_endpoint_returns_no_bytes(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/prompt":
                return _prompt_ok()
            if request.url.path.startswith("/history/"):
                return _history_ok()
            return httpx.Response(200, content=b"")

        with pytest.raises(ComfyUIError):
            _client(handler).generate_image(prompt="a red bicycle")


class TestRemoveBackground:
    def test_uploads_the_image_then_runs_the_graph(self) -> None:
        rgba = _png_bytes("RGBA")
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            calls.append(path)
            if path == "/upload/image":
                return httpx.Response(
                    200, json={"name": "uploaded_abc.png", "subfolder": "", "type": "input"}
                )
            if path == "/prompt":
                return _prompt_ok()
            if path.startswith("/history/"):
                return _history_ok(filename="bg_removed_00001_.png", node_id="5")
            if path == "/view":
                return httpx.Response(200, content=rgba)
            return httpx.Response(404)

        result = _client(handler, bg_model_name="birefnet.safetensors").remove_background(
            _png_bytes()
        )

        assert result.error is None
        assert result.image_bytes == rgba
        assert calls[0] == "/upload/image"
        assert "/prompt" in calls

    def test_uploads_as_a_multipart_image_field(self) -> None:
        content_types: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/upload/image":
                content_types.append(request.headers.get("content-type", ""))
                return httpx.Response(
                    200, json={"name": "uploaded_abc.png", "subfolder": "", "type": "input"}
                )
            if request.url.path == "/prompt":
                return _prompt_ok()
            if request.url.path.startswith("/history/"):
                return _history_ok(filename="bg_removed_00001_.png", node_id="5")
            return httpx.Response(200, content=_png_bytes("RGBA"))

        _client(handler, bg_model_name="birefnet.safetensors").remove_background(_png_bytes())

        assert content_types
        assert content_types[0].startswith("multipart/form-data")


class TestBackgroundModelDiscovery:
    def _object_info(self, options: list[str]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "LoadBackgroundRemovalModel": {
                    "input": {
                        "required": {
                            "bg_removal_name": [
                                "COMBO",
                                {"options": options, "multiselect": False, "tooltip": "x"},
                            ]
                        }
                    }
                }
            },
        )

    def test_discovers_the_installed_model_when_none_is_configured(self) -> None:
        posted: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/object_info/LoadBackgroundRemovalModel":
                return self._object_info(["model.safetensors"])
            if request.url.path == "/upload/image":
                return httpx.Response(
                    200, json={"name": "uploaded_abc.png", "subfolder": "", "type": "input"}
                )
            if request.url.path == "/prompt":
                import json

                posted.append(json.loads(request.content)["prompt"])
                return _prompt_ok()
            if request.url.path.startswith("/history/"):
                return _history_ok(filename="bg_removed_00001_.png", node_id="5")
            return httpx.Response(200, content=_png_bytes("RGBA"))

        _client(handler).remove_background(_png_bytes())

        assert posted[0]["1"]["inputs"]["bg_removal_name"] == "model.safetensors"

    def test_an_explicitly_configured_model_is_used_as_is(self) -> None:
        posted: list[dict] = []
        object_info_calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/object_info/LoadBackgroundRemovalModel":
                object_info_calls.append(request.url.path)
                return self._object_info(["model.safetensors"])
            if request.url.path == "/upload/image":
                return httpx.Response(
                    200, json={"name": "uploaded_abc.png", "subfolder": "", "type": "input"}
                )
            if request.url.path == "/prompt":
                import json

                posted.append(json.loads(request.content)["prompt"])
                return _prompt_ok()
            if request.url.path.startswith("/history/"):
                return _history_ok(filename="bg_removed_00001_.png", node_id="5")
            return httpx.Response(200, content=_png_bytes("RGBA"))

        _client(handler, bg_model_name="birefnet.safetensors").remove_background(_png_bytes())

        assert posted[0]["1"]["inputs"]["bg_removal_name"] == "birefnet.safetensors"
        assert object_info_calls == []

    def test_raises_when_no_model_is_installed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/object_info/LoadBackgroundRemovalModel":
                return self._object_info([])
            return httpx.Response(404)

        with pytest.raises(ComfyUIError) as excinfo:
            _client(handler).remove_background(_png_bytes())

        assert "No background removal model" in str(excinfo.value)


def _mean_luma(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    region = image.crop(box).tobytes()
    return sum(region) / max(1, len(region))


def _alpha_at(image: Image.Image, x: int, y: int) -> int:
    channel = image.getchannel("A") if image.mode == "RGBA" else image
    return channel.crop((x, y, x + 1, y + 1)).tobytes()[0]


class TestCalibrationImage:
    def test_is_a_png(self) -> None:
        image = Image.open(io.BytesIO(_build_calibration_image()))
        assert image.format == "PNG"

    def test_has_a_dark_subject_on_a_light_background(self) -> None:
        image = Image.open(io.BytesIO(_build_calibration_image())).convert("L")
        size = image.size[0]
        centre = _mean_luma(image, (size // 2 - 16, size // 2 - 16, size // 2 + 16, size // 2 + 16))
        corner = _mean_luma(image, (0, 0, 16, 16))
        assert centre < 128
        assert corner > 128

    def test_the_subject_covers_the_middle_of_the_canvas(self) -> None:
        image = Image.open(io.BytesIO(_build_calibration_image())).convert("L")
        size = image.size[0]
        assert _mean_luma(image, (0, 0, 16, 16)) > 128
        assert (
            _mean_luma(image, (size // 2 - 16, size // 2 - 16, size // 2 + 16, size // 2 + 16))
            < 128
        )


class TestAlphaPolarity:
    def _rgba(self, centre: int, corner: int) -> bytes:
        image = Image.new("RGBA", (64, 64), (255, 0, 0, corner))
        for x in range(16, 48):
            for y in range(16, 48):
                image.putpixel((x, y), (255, 0, 0, centre))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_detects_an_inverted_mask(self) -> None:
        # subject transparent, background opaque
        assert _alpha_polarity_is_inverted(self._rgba(centre=0, corner=255)) is True

    def test_detects_a_correct_mask(self) -> None:
        # subject opaque, background transparent
        assert _alpha_polarity_is_inverted(self._rgba(centre=255, corner=0)) is False


class TestInvertAlpha:
    def test_flips_the_alpha_channel(self) -> None:
        original = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
        original.putpixel((1, 1), (10, 20, 30, 255))
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")

        inverted = Image.open(io.BytesIO(_invert_alpha(buffer.getvalue()))).convert("RGBA")

        assert _alpha_at(inverted, 0, 0) == 255
        assert _alpha_at(inverted, 1, 1) == 0

    def test_leaves_the_colour_channels_alone(self) -> None:
        original = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")

        inverted = Image.open(io.BytesIO(_invert_alpha(buffer.getvalue()))).convert("RGBA")

        assert inverted.crop((0, 0, 1, 1)).tobytes()[:3] == bytes((10, 20, 30))
        assert _alpha_at(inverted, 0, 0) == 127


class TestRemoveBackgroundPolarity:
    """The client measures the server's mask polarity instead of assuming it."""

    def _handler(self, views: list[bytes], counter: list[int]) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/upload/image":
                return httpx.Response(
                    200, json={"name": "uploaded_abc.png", "subfolder": "", "type": "input"}
                )
            if path == "/prompt":
                return _prompt_ok()
            if path.startswith("/history/"):
                return _history_ok(filename="bg_removed_00001_.png", node_id="5")
            if path == "/view":
                index = min(counter[0], len(views) - 1)
                counter[0] += 1
                return httpx.Response(200, content=views[index])
            return httpx.Response(404)

        return handler

    def _rgba_with(self, centre: int, corner: int, size: int = 64) -> bytes:
        image = Image.new("RGBA", (size, size), (255, 0, 0, corner))
        half = size // 4
        for x in range(half, size - half):
            for y in range(half, size - half):
                image.putpixel((x, y), (255, 0, 0, centre))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_inverts_the_alpha_when_the_server_returns_a_background_map(self) -> None:
        probe = self._rgba_with(centre=0, corner=255)
        result_image = self._rgba_with(centre=0, corner=255)
        counter = [0]
        client = _client(
            self._handler([probe, result_image], counter), bg_model_name="birefnet.safetensors"
        )

        result = client.remove_background(_png_bytes())

        assert result.image_bytes is not None
        alpha = Image.open(io.BytesIO(result.image_bytes)).convert("RGBA").getchannel("A")
        # the subject is now opaque and the background transparent
        assert _alpha_at(alpha, 32, 32) == 255
        assert _alpha_at(alpha, 2, 2) == 0

    def test_leaves_the_alpha_alone_when_the_server_returns_a_foreground_map(self) -> None:
        probe = self._rgba_with(centre=255, corner=0)
        result_image = self._rgba_with(centre=255, corner=0)
        counter = [0]
        client = _client(
            self._handler([probe, result_image], counter), bg_model_name="birefnet.safetensors"
        )

        result = client.remove_background(_png_bytes())

        assert result.image_bytes is not None
        alpha = Image.open(io.BytesIO(result.image_bytes)).convert("RGBA").getchannel("A")
        assert _alpha_at(alpha, 32, 32) == 255
        assert _alpha_at(alpha, 2, 2) == 0

    def test_the_probe_runs_only_once(self) -> None:
        probe = self._rgba_with(centre=0, corner=255)
        counter = [0]
        client = _client(self._handler([probe], counter), bg_model_name="birefnet.safetensors")

        client.remove_background(_png_bytes())
        client.remove_background(_png_bytes())

        # one probe + two removals
        assert counter[0] == 3

    def test_a_failed_probe_falls_back_to_inverting(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/upload/image":
                return httpx.Response(
                    200, json={"name": "uploaded_abc.png", "subfolder": "", "type": "input"}
                )
            if request.url.path == "/prompt":
                return _prompt_ok()
            if request.url.path.startswith("/history/"):
                return _history_ok(filename="bg_removed_00001_.png", node_id="5")
            if request.url.path == "/view":
                return httpx.Response(200, content=self._rgba_with(centre=0, corner=255))
            return httpx.Response(404)

        client = _client(handler, bg_model_name="birefnet.safetensors")
        # The probe and the removal share the same /view response here, so the
        # probe measures an inverted mask and the removal is corrected.
        result = client.remove_background(_png_bytes())

        assert result.image_bytes is not None
        alpha = Image.open(io.BytesIO(result.image_bytes)).convert("RGBA").getchannel("A")
        assert _alpha_at(alpha, 32, 32) == 255


class TestHealth:
    def test_returns_true_on_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"system": {}})

        assert _client(handler).health() is True

    def test_returns_false_on_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        assert _client(handler).health() is False

    def test_returns_false_when_unreachable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        assert _client(handler).health() is False
