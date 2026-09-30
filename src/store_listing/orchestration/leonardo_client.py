"""Leonardo.ai REST client for Flux image generation.

Temporary bridge until DGX hardware is available for self-hosted
ComfyUI + Flux Schnell. See issue #111.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

LEONARDO_API_BASE = "https://cloud.leonardo.ai/api/rest"
LEONARDO_API_KEY: str = os.environ.get("LEONARDO_API_KEY", "")
LEONARDO_MODEL: str = os.environ.get("LEONARDO_MODEL", "flux-schnell")

MAX_POLL_ATTEMPTS = 30
DEFAULT_POLL_INTERVAL = 2.0


class LeonardoError(Exception):
    pass


@dataclass
class LeonardoGenerationResult:
    image_bytes: bytes | None
    prompt: str
    error: str | None = None
    generation_id: str | None = None


class LeonardoClient:
    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "",
        api_base: str = LEONARDO_API_BASE,
    ) -> None:
        self._api_key = api_key or LEONARDO_API_KEY
        self._model = model or LEONARDO_MODEL
        self._api_base = api_base

    def _build_payload(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        quantity: int = 1,
    ) -> dict:
        return {
            "model": self._model,
            "public": False,
            "parameters": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "quantity": quantity,
            },
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        http_client: httpx.AsyncClient | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_polls: int = MAX_POLL_ATTEMPTS,
    ) -> LeonardoGenerationResult:
        own_client = http_client is None
        if own_client:
            http_client = httpx.AsyncClient(timeout=60.0)

        try:
            return await self._do_generate(
                prompt=prompt,
                width=width,
                height=height,
                http_client=http_client,
                poll_interval=poll_interval,
                max_polls=max_polls,
            )
        finally:
            if own_client:
                await http_client.aclose()

    async def _do_generate(
        self,
        prompt: str,
        width: int,
        height: int,
        http_client: httpx.AsyncClient,
        poll_interval: float,
        max_polls: int,
    ) -> LeonardoGenerationResult:
        payload = self._build_payload(prompt=prompt, width=width, height=height)

        resp = await http_client.post(
            f"{self._api_base}/v2/generations",
            json=payload,
            headers=self._headers(),
        )

        if resp.status_code != 200:
            error_msg = f"Leonardo API error {resp.status_code}: {resp.text}"
            logger.warning(error_msg)
            return LeonardoGenerationResult(image_bytes=None, prompt=prompt, error=error_msg)

        data = resp.json()
        generation_id = data.get("generationId")
        if not generation_id:
            return LeonardoGenerationResult(
                image_bytes=None,
                prompt=prompt,
                error="No generationId in response",
            )

        for _ in range(max_polls):
            if poll_interval > 0:
                await asyncio.sleep(poll_interval)

            poll_resp = await http_client.get(
                f"{self._api_base}/v1/generations/{generation_id}",
                headers=self._headers(),
            )

            if poll_resp.status_code != 200:
                continue

            poll_data = poll_resp.json()
            gen = poll_data.get("generations_by_pk", {})
            status = gen.get("status", "")

            if status == "COMPLETE":
                images = gen.get("generated_images", [])
                if not images:
                    return LeonardoGenerationResult(
                        image_bytes=None,
                        prompt=prompt,
                        error="Generation complete but no images returned",
                        generation_id=generation_id,
                    )

                image_url = images[0].get("url", "")
                img_resp = await http_client.get(image_url)
                return LeonardoGenerationResult(
                    image_bytes=img_resp.content,
                    prompt=prompt,
                    generation_id=generation_id,
                )

            if status == "FAILED":
                return LeonardoGenerationResult(
                    image_bytes=None,
                    prompt=prompt,
                    error=f"Leonardo generation FAILED (id={generation_id})",
                    generation_id=generation_id,
                )

        return LeonardoGenerationResult(
            image_bytes=None,
            prompt=prompt,
            error=f"Timed out polling for generation {generation_id}",
            generation_id=generation_id,
        )
