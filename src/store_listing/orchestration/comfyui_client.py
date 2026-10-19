"""ComfyUI client for DGX-hosted image generation (Flux Schnell).

Speaks the ComfyUI REST API: ``POST /prompt`` to queue a workflow, poll
``GET /history/{prompt_id}`` until it completes, then fetch the bytes from
``GET /view``.

The API shapes used here were verified against a live ComfyUI 0.37.1 instance on
the DGX Spark. The client is deliberately synchronous: the surrounding image
pipeline is synchronous and is driven from Celery tasks.
"""

from __future__ import annotations

import io
import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Self

import httpx
from PIL import Image, ImageFilter

from store_listing.orchestration.chroma_key import punch_out_key_colour
from store_listing.orchestration.comfyui_workflows import (
    build_birefnet_background_graph,
    build_flux_txt2img_graph,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8188"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0

DEFAULT_UNET_NAME = "flux1-schnell.safetensors"
DEFAULT_CLIP_NAME1 = "t5xxl_fp8_e4m3fn.safetensors"
DEFAULT_CLIP_NAME2 = "clip_l.safetensors"
DEFAULT_VAE_NAME = "ae.safetensors"

# Alpha cleanup applied to the background removal result.
ALPHA_BLACK_POINT = 24
ALPHA_WHITE_POINT = 240
ALPHA_ERODE_PIXELS = 1


def _mask_is_inverted(rgba_bytes: bytes) -> bool:
    """True when the mask treats the background as foreground.

    ComfyUI 0.37.1's ``RemoveBackground`` returns a background map (1 = background)
    for some renders and a foreground map for others, so the polarity has to be
    measured per image rather than assumed. The border of a design is background
    by construction, so an opaque border means the mask has to be flipped.
    """
    with Image.open(io.BytesIO(rgba_bytes)) as opened:
        alpha = opened.convert("RGBA").getchannel("A")
        width, height = alpha.size
        values = alpha.tobytes()
        step = max(1, min(width, height) // 64)
        opaque = 0
        total = 0
        for x in range(0, width, step):
            for y in (0, height - 1):
                total += 1
                if values[y * width + x] > 127:
                    opaque += 1
        for y in range(0, height, step):
            for x in (0, width - 1):
                total += 1
                if values[y * width + x] > 127:
                    opaque += 1
    return opaque * 2 > total


def _invert_alpha(rgba_bytes: bytes) -> bytes:
    """Flip the alpha channel of an RGBA PNG, leaving the colour channels alone."""
    with Image.open(io.BytesIO(rgba_bytes)) as opened:
        image = opened.convert("RGBA")
        image.putalpha(Image.eval(image.getchannel("A"), lambda value: 255 - value))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def _clean_alpha(
    rgba_bytes: bytes,
    *,
    black: int = ALPHA_BLACK_POINT,
    white: int = ALPHA_WHITE_POINT,
    erode: int = ALPHA_ERODE_PIXELS,
) -> bytes:
    """Strip the residual background haze and the dark halo around the subject.

    The mask the background removal model returns is soft in two ways: a faint
    haze over the whole background, and a one or two pixel band at the
    silhouette whose colour is still blended with the original background.
    Printed onto a shirt, that band shows up as a dark outline around the
    design. Stretching the alpha channel drops the haze, and eroding it by a
    pixel pulls the cut inside the subject so the edge pixels carry the
    subject's own colour rather than the background's.
    """
    with Image.open(io.BytesIO(rgba_bytes)) as opened:
        image = opened.convert("RGBA")
        alpha = image.getchannel("A")
        table = [min(255, max(0, (value - black) * 255 // (white - black))) for value in range(256)]
        alpha = alpha.point(table)
        if erode > 0:
            alpha = alpha.filter(ImageFilter.MinFilter(2 * erode + 1))
        image.putalpha(alpha)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return buffer.getvalue()


class ComfyUIError(Exception):
    """Raised when ComfyUI rejects a request or a workflow fails to execute."""


@dataclass
class ComfyUIResult:
    """Outcome of a single ComfyUI workflow execution."""

    image_bytes: bytes | None = None
    prompt_id: str | None = None
    error: str | None = None
    seed: int | None = None


class ComfyUIClient:
    """Thin, synchronous client for the ComfyUI REST API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        http_client: httpx.Client | None = None,
        unet_name: str = DEFAULT_UNET_NAME,
        clip_name1: str = DEFAULT_CLIP_NAME1,
        clip_name2: str = DEFAULT_CLIP_NAME2,
        vae_name: str = DEFAULT_VAE_NAME,
        bg_model_name: str | None = None,
        alpha_black_point: int = ALPHA_BLACK_POINT,
        alpha_white_point: int = ALPHA_WHITE_POINT,
        alpha_erode_pixels: int = ALPHA_ERODE_PIXELS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=httpx.Timeout(timeout_seconds))
        self._client_id = f"store-listing-{uuid.uuid4().hex[:12]}"
        self._unet_name = unet_name
        self._clip_name1 = clip_name1
        self._clip_name2 = clip_name2
        self._vae_name = vae_name
        self._bg_model_name = bg_model_name
        self._discovered_bg_model: str | None = None
        self._alpha_black_point = alpha_black_point
        self._alpha_white_point = alpha_white_point
        self._alpha_erode_pixels = alpha_erode_pixels

    def close(self) -> None:
        """Close the underlying HTTP client, if this instance created it."""
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def generate_image(
        self,
        *,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        steps: int = 4,
        cfg: float = 1.0,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        negative_prompt: str = "",
        unet_name: str | None = None,
        clip_name1: str | None = None,
        clip_name2: str | None = None,
        vae_name: str | None = None,
    ) -> ComfyUIResult:
        """Queue a Flux Schnell text-to-image workflow and block until it is ready."""
        actual_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        graph = build_flux_txt2img_graph(
            prompt=prompt,
            width=width,
            height=height,
            seed=actual_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            negative_prompt=negative_prompt,
            unet_name=unet_name or self._unet_name,
            clip_name1=clip_name1 or self._clip_name1,
            clip_name2=clip_name2 or self._clip_name2,
            vae_name=vae_name or self._vae_name,
        )
        return self._run_graph(graph, seed=actual_seed)

    def remove_background(
        self,
        image_bytes: bytes,
        *,
        bg_model_name: str | None = None,
    ) -> ComfyUIResult:
        """Upload the image to ComfyUI, run BiRefNet, return a transparent RGBA PNG."""
        model_name = bg_model_name or self._bg_model_name or self._discover_bg_model()

        uploaded = self._upload_image(image_bytes)
        graph = build_birefnet_background_graph(
            image_name=uploaded["name"],
            bg_model_name=model_name,
        )
        result = self._run_graph(graph)
        if result.image_bytes is None:
            return result

        if _mask_is_inverted(result.image_bytes):
            result.image_bytes = _invert_alpha(result.image_bytes)
        result.image_bytes = _clean_alpha(
            result.image_bytes,
            black=self._alpha_black_point,
            white=self._alpha_white_point,
            erode=self._alpha_erode_pixels,
        )
        # The model treats the enclosed counter of a letter as part of the
        # subject, so those pixels keep the background colour. Punch them out.
        result.image_bytes = punch_out_key_colour(result.image_bytes, image_bytes)
        return result

    def health(self) -> bool:
        """True when GET /system_stats succeeds."""
        try:
            response = self._http.get(f"{self.base_url}/system_stats")
        except httpx.HTTPError as exc:
            logger.warning("ComfyUI health check failed: %s", exc)
            return False
        return response.status_code == 200

    # -- internals ---------------------------------------------------------

    def _discover_bg_model(self) -> str:
        """Find the background removal model the server actually has installed.

        The file name on disk is not something we control (it depends on how the
        checkpoint was installed), so ask the server which models it can load
        rather than hard-coding a name.
        """
        if self._discovered_bg_model is not None:
            return self._discovered_bg_model

        try:
            response = self._http.get(f"{self.base_url}/object_info/LoadBackgroundRemovalModel")
        except httpx.HTTPError as exc:
            raise ComfyUIError(
                f"Could not reach ComfyUI to discover the background removal model: {exc}"
            ) from exc
        if response.status_code != 200:
            raise ComfyUIError(
                f"ComfyUI does not expose LoadBackgroundRemovalModel - HTTP {response.status_code}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyUIError("ComfyUI returned a non-JSON response for /object_info") from exc

        node = body.get("LoadBackgroundRemovalModel") or {}
        required = (node.get("input") or {}).get("required") or {}
        spec = required.get("bg_removal_name") or []
        options = spec[1].get("options", []) if len(spec) > 1 and isinstance(spec[1], dict) else []

        if not options:
            raise ComfyUIError(
                "No background removal model is installed on the ComfyUI server - "
                "run deploy/dgx/download-flux.sh on the DGX"
            )

        self._discovered_bg_model = str(options[0])
        logger.info("Using background removal model %s", self._discovered_bg_model)
        return self._discovered_bg_model

    def _run_graph(self, graph: dict[str, Any], *, seed: int | None = None) -> ComfyUIResult:
        save_node_id = self._find_save_node(graph)
        prompt_id = self._queue(graph)
        logger.info("Queued ComfyUI workflow %s (save node %s)", prompt_id, save_node_id)
        descriptor = self._await_image(prompt_id, save_node_id)
        image_bytes = self._fetch_image(descriptor)
        return ComfyUIResult(image_bytes=image_bytes, prompt_id=prompt_id, seed=seed)

    @staticmethod
    def _find_save_node(graph: dict[str, Any]) -> str:
        for node_id, node in graph.items():
            if node.get("class_type") == "SaveImage":
                return str(node_id)
        raise ComfyUIError("Workflow does not contain a SaveImage node")

    def _queue(self, graph: dict[str, Any]) -> str:
        payload = {"prompt": graph, "client_id": self._client_id}
        try:
            response = self._http.post(f"{self.base_url}/prompt", json=payload)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Could not reach ComfyUI at {self.base_url}: {exc}") from exc
        if response.status_code != 200:
            raise ComfyUIError(self._describe_rejection(response))
        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyUIError("ComfyUI returned a non-JSON response for /prompt") from exc
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError("ComfyUI accepted the workflow but returned no prompt_id")
        return str(prompt_id)

    def _await_image(self, prompt_id: str, save_node_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                response = self._http.get(f"{self.base_url}/history/{prompt_id}")
            except httpx.HTTPError as exc:
                raise ComfyUIError(
                    f"Could not read ComfyUI history for {prompt_id}: {exc}"
                ) from exc

            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                entry = body.get(prompt_id) or {}
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyUIError(self._describe_execution_error(status))
                images = ((entry.get("outputs") or {}).get(save_node_id) or {}).get("images") or []
                if images:
                    return images[0]
                if status.get("completed"):
                    raise ComfyUIError(
                        f"ComfyUI workflow {prompt_id} finished without producing an image"
                    )

            if time.monotonic() >= deadline:
                raise ComfyUIError(
                    f"ComfyUI workflow {prompt_id} timed out after {self.timeout_seconds}s"
                )
            time.sleep(self.poll_interval_seconds)

    def _fetch_image(self, descriptor: dict[str, Any]) -> bytes:
        filename = str(descriptor.get("filename", ""))
        params = {
            "filename": filename,
            "subfolder": str(descriptor.get("subfolder", "")),
            "type": str(descriptor.get("type", "output")),
        }
        try:
            response = self._http.get(f"{self.base_url}/view", params=params)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Could not fetch the generated image {filename}: {exc}") from exc
        if response.status_code != 200:
            raise ComfyUIError(
                f"ComfyUI returned HTTP {response.status_code} for the generated image {filename}"
            )
        if not response.content:
            raise ComfyUIError(f"ComfyUI returned an empty image for {filename}")
        return response.content

    def _upload_image(self, image_bytes: bytes) -> dict[str, str]:
        # A unique name per upload: reusing one name with overwrite=true can
        # leave ComfyUI's input directory pointing at a file it can no longer read.
        name = f"input_{uuid.uuid4().hex[:12]}.png"
        files = {"image": (name, image_bytes, "image/png")}
        data = {"overwrite": "true", "type": "input"}
        try:
            response = self._http.post(f"{self.base_url}/upload/image", files=files, data=data)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Could not upload the image to ComfyUI: {exc}") from exc
        if response.status_code != 200:
            raise ComfyUIError(
                f"ComfyUI rejected the image upload with HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyUIError("ComfyUI returned a non-JSON response for /upload/image") from exc
        name = body.get("name")
        if not name:
            raise ComfyUIError("ComfyUI uploaded the image but returned no name")
        return {
            "name": str(name),
            "subfolder": str(body.get("subfolder", "")),
            "type": str(body.get("type", "input")),
        }

    @staticmethod
    def _describe_rejection(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return (
                f"ComfyUI rejected the workflow with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        parts: list[str] = []
        for node_id, node_error in (body.get("node_errors") or {}).items():
            class_type = (node_error or {}).get("class_type")
            label = f"node {node_id}" + (f" ({class_type})" if class_type else "")
            for error in (node_error or {}).get("errors", []):
                # ComfyUI's "message" is generic ("Value not in list"); the
                # "details" field carries the offending value and the allowed list.
                message = error.get("details") or error.get("message") or str(error)
                parts.append(f"{label}: {message}")
        if not parts:
            error = body.get("error")
            if isinstance(error, dict):
                parts.append(str(error.get("message") or error))
            elif error:
                parts.append(str(error))
        if not parts:
            parts.append(f"HTTP {response.status_code}: {response.text[:500]}")
        return "ComfyUI rejected the workflow - " + "; ".join(parts)

    @staticmethod
    def _describe_execution_error(status: dict[str, Any]) -> str:
        details: list[str] = []
        for message in status.get("messages") or []:
            if isinstance(message, (list, tuple)) and len(message) >= 2:
                info = message[1]
                if isinstance(info, dict):
                    details.append(
                        str(info.get("exception_message") or info.get("exception_type") or info)
                    )
                    continue
            details.append(str(message))
        if not details:
            details.append(str(status.get("status_str", "unknown error")))
        return "ComfyUI workflow failed - " + "; ".join(details)
