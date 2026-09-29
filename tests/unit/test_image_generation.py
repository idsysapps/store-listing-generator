"""Tests for Bedrock-based image generation from design briefs."""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import MagicMock

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.image_generation import (
    ImageResult,
    build_prompt_from_brief,
    generate_image,
)


def _make_brief(**overrides: Any) -> DesignBrief:
    defaults = {
        "concept": "Skeleton doing yoga poses",
        "product_type": "dtf_apparel",
        "specific_products": ["t-shirt", "hoodie"],
        "audience": "Yoga enthusiasts who love dark humor",
        "visual_style": "Retro vintage, distressed texture, orange and teal on dark background",
        "confidence": 85,
        "reasoning": "Skeleton yoga trending on social media",
        "source_seed_ids": [1, 2],
    }
    defaults.update(overrides)
    return DesignBrief(**defaults)


class TestBuildPromptFromBrief:
    def test_includes_concept(self) -> None:
        brief = _make_brief(concept="Skeleton doing yoga poses")
        prompt = build_prompt_from_brief(brief)
        assert "Skeleton doing yoga poses" in prompt

    def test_includes_visual_style(self) -> None:
        brief = _make_brief(visual_style="Watercolor pastel, soft pink and blue tones")
        prompt = build_prompt_from_brief(brief)
        assert "Watercolor pastel" in prompt

    def test_includes_product_context(self) -> None:
        brief = _make_brief(product_type="dtf_apparel")
        prompt = build_prompt_from_brief(brief)
        assert "print" in prompt.lower() or "apparel" in prompt.lower()

    def test_includes_print_ready_instruction(self) -> None:
        brief = _make_brief()
        prompt = build_prompt_from_brief(brief)
        assert "print" in prompt.lower()


class TestGenerateImage:
    def test_returns_image_result(self) -> None:
        mock_client = MagicMock()
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_client.invoke_model.return_value = {
            "body": MagicMock(
                read=lambda: json.dumps({"images": [base64.b64encode(fake_png).decode()]}).encode()
            ),
        }

        brief = _make_brief()
        result = generate_image(brief, bedrock_client=mock_client)

        assert isinstance(result, ImageResult)
        assert result.image_bytes == fake_png
        assert result.prompt != ""

    def test_calls_bedrock_with_correct_model(self) -> None:
        mock_client = MagicMock()
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_client.invoke_model.return_value = {
            "body": MagicMock(
                read=lambda: json.dumps({"images": [base64.b64encode(fake_png).decode()]}).encode()
            ),
        }

        brief = _make_brief()
        generate_image(brief, bedrock_client=mock_client)

        call_kwargs = mock_client.invoke_model.call_args
        assert "stability" in call_kwargs.kwargs.get("modelId", call_kwargs[1].get("modelId", ""))

    def test_sends_prompt_in_body(self) -> None:
        mock_client = MagicMock()
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_client.invoke_model.return_value = {
            "body": MagicMock(
                read=lambda: json.dumps({"images": [base64.b64encode(fake_png).decode()]}).encode()
            ),
        }

        brief = _make_brief(concept="Galaxy cat astronaut")
        generate_image(brief, bedrock_client=mock_client)

        call_kwargs = mock_client.invoke_model.call_args
        body_str = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", ""))
        body = json.loads(body_str)
        assert "Galaxy cat astronaut" in str(body)

    def test_returns_error_on_bedrock_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("Bedrock timeout")

        brief = _make_brief()
        result = generate_image(brief, bedrock_client=mock_client)

        assert isinstance(result, ImageResult)
        assert result.image_bytes is None
        assert "Bedrock timeout" in (result.error or "")

    def test_uses_square_aspect_ratio(self) -> None:
        mock_client = MagicMock()
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_client.invoke_model.return_value = {
            "body": MagicMock(
                read=lambda: json.dumps({"images": [base64.b64encode(fake_png).decode()]}).encode()
            ),
        }

        brief = _make_brief()
        generate_image(brief, bedrock_client=mock_client)

        call_kwargs = mock_client.invoke_model.call_args
        body_str = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", ""))
        body = json.loads(body_str)
        assert body.get("aspect_ratio") == "1:1"
        assert body.get("mode") == "text-to-image"
        assert body.get("output_format") == "png"
