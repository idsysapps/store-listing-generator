"""Tests for Leonardo.ai Flux image generation client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from store_listing.orchestration.leonardo_client import (
    LeonardoClient,
    LeonardoGenerationResult,
)


def _generation_response(generation_id: str = "gen-123") -> dict[str, Any]:
    return {
        "generate": {"generationId": generation_id, "cost": {"amount": "0.003", "unit": "DOLLARS"}}
    }


def _error_list_response(message: str = "Insufficient tokens") -> list[dict[str, Any]]:
    return [{"message": message, "extensions": {"statusCode": 402}, "locations": [], "path": []}]


def _poll_response_pending() -> dict[str, Any]:
    return {
        "generations_by_pk": {
            "id": "gen-123",
            "status": "PENDING",
            "generated_images": [],
        }
    }


def _poll_response_complete(image_url: str = "https://cdn.leonardo.ai/test.png") -> dict[str, Any]:
    return {
        "generations_by_pk": {
            "id": "gen-123",
            "status": "COMPLETE",
            "generated_images": [{"url": image_url, "id": "img-456"}],
        }
    }


def _poll_response_failed() -> dict[str, Any]:
    return {
        "generations_by_pk": {
            "id": "gen-123",
            "status": "FAILED",
            "generated_images": [],
        }
    }


class TestLeonardoClient:
    def test_create_requires_api_key(self) -> None:
        client = LeonardoClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_build_generation_payload_flux_schnell(self) -> None:
        client = LeonardoClient(api_key="test-key")
        payload = client._build_payload(
            prompt="A skeleton doing yoga",
            width=1024,
            height=1024,
        )
        assert payload["model"] == "flux-schnell"
        assert payload["parameters"]["prompt"] == "A skeleton doing yoga"
        assert payload["parameters"]["width"] == 1024
        assert payload["parameters"]["height"] == 1024

    def test_build_payload_custom_model(self) -> None:
        client = LeonardoClient(api_key="test-key", model="flux-dev")
        payload = client._build_payload(prompt="test", width=1024, height=1024)
        assert payload["model"] == "flux-dev"


class TestLeonardoGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_image_bytes(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = httpx.Response(
            200,
            json=_generation_response("gen-123"),
            request=httpx.Request("POST", "https://example.com"),
        )
        mock_client.get.side_effect = [
            httpx.Response(
                200,
                json=_poll_response_complete("https://cdn.leonardo.ai/test.png"),
                request=httpx.Request("GET", "https://example.com"),
            ),
            httpx.Response(
                200,
                content=fake_png,
                request=httpx.Request("GET", "https://cdn.leonardo.ai/test.png"),
            ),
        ]

        client = LeonardoClient(api_key="test-key")
        result = await client.generate(
            prompt="A skeleton doing yoga",
            http_client=mock_client,
        )

        assert isinstance(result, LeonardoGenerationResult)
        assert result.image_bytes == fake_png
        assert result.error is None

    @pytest.mark.asyncio
    async def test_generate_polls_until_complete(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = httpx.Response(
            200,
            json=_generation_response("gen-123"),
            request=httpx.Request("POST", "https://example.com"),
        )
        mock_client.get.side_effect = [
            httpx.Response(
                200,
                json=_poll_response_pending(),
                request=httpx.Request("GET", "https://example.com"),
            ),
            httpx.Response(
                200,
                json=_poll_response_complete(),
                request=httpx.Request("GET", "https://example.com"),
            ),
            httpx.Response(
                200,
                content=fake_png,
                request=httpx.Request("GET", "https://cdn.leonardo.ai/test.png"),
            ),
        ]

        client = LeonardoClient(api_key="test-key")
        result = await client.generate(
            prompt="test",
            http_client=mock_client,
            poll_interval=0,
        )

        assert result.image_bytes == fake_png
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_returns_error_on_failure(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = httpx.Response(
            200,
            json=_generation_response("gen-123"),
            request=httpx.Request("POST", "https://example.com"),
        )
        mock_client.get.return_value = httpx.Response(
            200,
            json=_poll_response_failed(),
            request=httpx.Request("GET", "https://example.com"),
        )

        client = LeonardoClient(api_key="test-key")
        result = await client.generate(
            prompt="test",
            http_client=mock_client,
        )

        assert result.image_bytes is None
        assert result.error is not None
        assert "FAILED" in result.error

    @pytest.mark.asyncio
    async def test_generate_returns_error_on_api_error(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = httpx.Response(
            429,
            json={"error": "Rate limited"},
            request=httpx.Request("POST", "https://example.com"),
        )

        client = LeonardoClient(api_key="test-key")
        result = await client.generate(
            prompt="test",
            http_client=mock_client,
        )

        assert result.image_bytes is None
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_generate_sends_auth_header(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = httpx.Response(
            200,
            json=_generation_response("gen-123"),
            request=httpx.Request("POST", "https://example.com"),
        )
        mock_client.get.side_effect = [
            httpx.Response(
                200,
                json=_poll_response_complete(),
                request=httpx.Request("GET", "https://example.com"),
            ),
            httpx.Response(
                200,
                content=b"\x89PNG",
                request=httpx.Request("GET", "https://cdn.leonardo.ai/test.png"),
            ),
        ]

        client = LeonardoClient(api_key="my-secret-key")
        await client.generate(prompt="test", http_client=mock_client)

        post_call = mock_client.post.call_args
        headers = post_call.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_generate_handles_error_list_response(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = httpx.Response(
            200,
            json=_error_list_response("Insufficient tokens"),
            request=httpx.Request("POST", "https://example.com"),
        )

        client = LeonardoClient(api_key="test-key")
        result = await client.generate(
            prompt="test",
            http_client=mock_client,
        )

        assert result.image_bytes is None
        assert result.error is not None
        assert "Insufficient tokens" in result.error
