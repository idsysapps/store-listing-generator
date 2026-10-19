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
    _clean_alpha,
    _invert_alpha,
    _mask_is_inverted,
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


class TestCleanAlpha:
    def _rgba(self, pixels: dict[tuple[int, int], int], size: int = 8) -> bytes:
        image = Image.new("RGBA", (size, size), (10, 20, 30, 0))
        for (x, y), alpha in pixels.items():
            image.putpixel((x, y), (10, 20, 30, alpha))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_removes_the_faint_haze_over_the_background(self) -> None:
        cleaned = Image.open(
            io.BytesIO(_clean_alpha(self._rgba({(0, 0): 10, (7, 7): 10})))
        ).convert("RGBA")
        assert _alpha_at(cleaned, 0, 0) == 0
        assert _alpha_at(cleaned, 7, 7) == 0

    def test_keeps_the_subject_fully_opaque(self) -> None:
        pixels: dict[tuple[int, int], int] = {}
        for x in range(2, 6):
            for y in range(2, 6):
                pixels[(x, y)] = 255
        cleaned = Image.open(io.BytesIO(_clean_alpha(self._rgba(pixels)))).convert("RGBA")
        assert _alpha_at(cleaned, 3, 3) == 255

    def test_leaves_the_colour_channels_alone(self) -> None:
        cleaned = Image.open(io.BytesIO(_clean_alpha(self._rgba({(0, 0): 255})))).convert("RGBA")
        assert cleaned.crop((0, 0, 1, 1)).tobytes()[:3] == bytes((10, 20, 30))

    def test_erodes_the_soft_edge_so_the_cut_falls_inside_the_subject(self) -> None:
        # A 4x4 solid block of alpha 255 in an 8x8 canvas, with one ring of
        # half-transparent pixels around it.
        pixels: dict[tuple[int, int], int] = {}
        for x in range(2, 6):
            for y in range(2, 6):
                pixels[(x, y)] = 255
        for x in range(1, 7):
            for y in (1, 6):
                pixels[(x, y)] = 128
        for y in range(1, 7):
            for x in (1, 6):
                pixels[(x, y)] = 128

        cleaned = Image.open(io.BytesIO(_clean_alpha(self._rgba(pixels)))).convert("RGBA")

        # the half-transparent ring is gone
        assert _alpha_at(cleaned, 1, 1) == 0
        # and the solid core survives
        assert _alpha_at(cleaned, 3, 3) == 255

    def test_erosion_can_be_disabled(self) -> None:
        pixels: dict[tuple[int, int], int] = {}
        for x in range(2, 6):
            for y in range(2, 6):
                pixels[(x, y)] = 255
        for x in range(1, 7):
            for y in (1, 6):
                pixels[(x, y)] = 128

        cleaned = Image.open(io.BytesIO(_clean_alpha(self._rgba(pixels), erode=0))).convert("RGBA")

        assert _alpha_at(cleaned, 1, 1) > 0


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


class TestMaskPolarity:
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
        assert _mask_is_inverted(self._rgba(centre=0, corner=255)) is True

    def test_detects_a_correct_mask(self) -> None:
        # subject opaque, background transparent
        assert _mask_is_inverted(self._rgba(centre=255, corner=0)) is False


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

    def _source(self, size: int = 64) -> bytes:
        """A design on a flat magenta background with an orange square in the middle."""
        image = Image.new("RGB", (size, size), (247, 0, 102))
        half = size // 4
        for x in range(half, size - half):
            for y in range(half, size - half):
                image.putpixel((x, y), (250, 120, 10))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _server_output(self, source: bytes, *, inverted: bool) -> bytes:
        """What the server returns: a mask with the requested polarity."""
        with Image.open(io.BytesIO(source)) as opened:
            image = opened.convert("RGB")
        values = image.tobytes()
        width, height = image.size
        half = width // 4
        out = image.convert("RGBA")
        data: list[tuple[int, int, int, int]] = []
        for y in range(height):
            for x in range(width):
                is_artwork = half <= x < width - half and half <= y < height - half
                alpha = 255 if is_artwork else 0
                if inverted:
                    alpha = 255 - alpha
                offset = (y * width + x) * 3
                data.append((values[offset], values[offset + 1], values[offset + 2], alpha))
        out.putdata(data)
        buffer = io.BytesIO()
        out.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_a_background_map_is_corrected(self) -> None:
        source = self._source()
        counter = [0]
        client = _client(
            self._handler([self._server_output(source, inverted=True)], counter),
            bg_model_name="birefnet.safetensors",
        )

        result = client.remove_background(source)

        assert result.image_bytes is not None
        alpha = Image.open(io.BytesIO(result.image_bytes)).convert("RGBA").getchannel("A")
        assert _alpha_at(alpha, 32, 32) == 255, "the artwork should be opaque"
        assert _alpha_at(alpha, 2, 2) == 0, "the background should be transparent"

    def test_a_foreground_map_is_left_alone(self) -> None:
        source = self._source()
        counter = [0]
        client = _client(
            self._handler([self._server_output(source, inverted=False)], counter),
            bg_model_name="birefnet.safetensors",
        )

        result = client.remove_background(source)

        assert result.image_bytes is not None
        alpha = Image.open(io.BytesIO(result.image_bytes)).convert("RGBA").getchannel("A")
        assert _alpha_at(alpha, 32, 32) == 255
        assert _alpha_at(alpha, 2, 2) == 0

    def test_the_background_colour_is_punched_out_of_the_artwork(self) -> None:
        # A letter's enclosed counter keeps the background colour in the mask.
        # Those pixels are punched out.
        source = self._source()
        with Image.open(io.BytesIO(source)) as opened:
            image = opened.convert("RGB")
        image.putpixel((32, 32), (247, 0, 102))  # a counter that the model kept
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        source = buffer.getvalue()

        counter = [0]
        client = _client(
            self._handler([self._server_output(source, inverted=False)], counter),
            bg_model_name="birefnet.safetensors",
        )

        result = client.remove_background(source)

        assert result.image_bytes is not None
        alpha = Image.open(io.BytesIO(result.image_bytes)).convert("RGBA").getchannel("A")
        assert _alpha_at(alpha, 32, 32) == 0, (
            "the background-coloured counter should be punched out"
        )

    def test_only_one_workflow_is_run_per_removal(self) -> None:
        source = self._source()
        counter = [0]
        client = _client(
            self._handler([self._server_output(source, inverted=False)], counter),
            bg_model_name="birefnet.safetensors",
        )

        client.remove_background(source)
        client.remove_background(source)

        assert counter[0] == 2

    def test_a_failed_removal_reports_the_error(self) -> None:
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
                return httpx.Response(200, content=b"")
            return httpx.Response(404)

        client = _client(handler, bg_model_name="birefnet.safetensors")
        with pytest.raises(ComfyUIError):
            client.remove_background(_png_bytes())


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
