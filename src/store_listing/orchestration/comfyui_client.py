"""ComfyUI client for DGX-hosted image generation (Flux Schnell).

Speaks the ComfyUI REST API: ``POST /prompt`` to queue a workflow, poll
``GET /history/{prompt_id}`` until it completes, then fetch the bytes from
``GET /view``.

The API shapes used here were verified against a live ComfyUI 0.37.1 instance on
the DGX Spark. The client is deliberately synchronous: the surrounding image
pipeline is synchronous and is driven from Celery tasks.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Self

import httpx

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
DEFAULT_BG_MODEL_NAME = "birefnet.safetensors"


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=httpx.Timeout(timeout_seconds))
        self._client_id = f"store-listing-{uuid.uuid4().hex[:12]}"

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
        unet_name: str = DEFAULT_UNET_NAME,
        clip_name1: str = DEFAULT_CLIP_NAME1,
        clip_name2: str = DEFAULT_CLIP_NAME2,
        vae_name: str = DEFAULT_VAE_NAME,
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
            unet_name=unet_name,
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            vae_name=vae_name,
        )
        return self._run_graph(graph, seed=actual_seed)

    def remove_background(
        self,
        image_bytes: bytes,
        *,
        bg_model_name: str = DEFAULT_BG_MODEL_NAME,
    ) -> ComfyUIResult:
        """Upload the image to ComfyUI, run BiRefNet, return a transparent RGBA PNG."""
        uploaded = self._upload_image(image_bytes)
        graph = build_birefnet_background_graph(
            image_name=uploaded["name"],
            bg_model_name=bg_model_name,
        )
        return self._run_graph(graph)

    def health(self) -> bool:
        """True when GET /system_stats succeeds."""
        try:
            response = self._http.get(f"{self.base_url}/system_stats")
        except httpx.HTTPError as exc:
            logger.warning("ComfyUI health check failed: %s", exc)
            return False
        return response.status_code == 200

    # -- internals ---------------------------------------------------------

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
        files = {"image": ("input.png", image_bytes, "image/png")}
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
            for error in (node_error or {}).get("errors", []):
                message = error.get("message") or error.get("details") or str(error)
                parts.append(f"node {node_id}: {message}")
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
