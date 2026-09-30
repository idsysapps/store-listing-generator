"""Tests for Bedrock-based image generation from design briefs."""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import MagicMock

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.image_generation import (
    REMOVE_BG_MODEL_ID,
    ImageResult,
    build_prompt_from_brief,
    generate_image,
    remove_background,
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

    def test_headline_text_wrapped_in_double_quotes(self) -> None:
        brief = _make_brief(headline_text="SPOOKY SEASON")
        prompt = build_prompt_from_brief(brief)
        assert '"SPOOKY SEASON"' in prompt

    def test_tagline_text_wrapped_in_double_quotes(self) -> None:
        brief = _make_brief(headline_text="Born To Be", tagline_text="Spooky")
        prompt = build_prompt_from_brief(brief)
        assert '"Born To Be"' in prompt
        assert '"Spooky"' in prompt

    def test_typography_style_derived_from_visual_style(self) -> None:
        brief = _make_brief(
            visual_style="Retro vintage, distressed texture",
            headline_text="COFFEE",
        )
        prompt = build_prompt_from_brief(brief)
        assert "retro" in prompt.lower()
        assert "lettering" in prompt.lower()

    def test_long_concept_with_separator_quotes_short_part(self) -> None:
        brief = _make_brief(
            concept="This Meeting Could Have Been An Email - Skeleton Edition",
            headline_text=None,
        )
        prompt = build_prompt_from_brief(brief)
        assert '"Skeleton Edition"' in prompt
        assert "This Meeting Could Have Been An Email" in prompt

    def test_long_concept_no_separator_suppresses_text(self) -> None:
        brief = _make_brief(
            concept="A very long concept with many words and no separator",
            headline_text=None,
        )
        prompt = build_prompt_from_brief(brief)
        assert "do not include any text" in prompt.lower()

    def test_short_concept_quotes_itself(self) -> None:
        brief = _make_brief(concept="Yoga Cat", headline_text=None)
        prompt = build_prompt_from_brief(brief)
        assert '"Yoga Cat"' in prompt

    def test_neon_style_uses_neon_typography(self) -> None:
        brief = _make_brief(visual_style="neon glow effect", headline_text="OPEN")
        prompt = build_prompt_from_brief(brief)
        assert "neon" in prompt.lower()
        assert "lettering" in prompt.lower()

    def test_regeneration_feedback_appended_to_prompt(self) -> None:
        brief = _make_brief(regeneration_feedback="make the skeleton more cartoonish")
        prompt = build_prompt_from_brief(brief)
        assert "make the skeleton more cartoonish" in prompt

    def test_no_feedback_section_when_feedback_is_none(self) -> None:
        brief = _make_brief(regeneration_feedback=None)
        prompt = build_prompt_from_brief(brief)
        assert "additional creative direction" not in prompt.lower()

    def test_dtf_apparel_prompt_mentions_isolated_subject(self) -> None:
        brief = _make_brief(product_type="dtf_apparel")
        prompt = build_prompt_from_brief(brief)
        assert "isolated" in prompt.lower() or "single subject" in prompt.lower()

    def test_dtf_apparel_prompt_mentions_no_background_scene(self) -> None:
        brief = _make_brief(product_type="dtf_apparel")
        prompt = build_prompt_from_brief(brief)
        assert "no background" in prompt.lower() or "plain white" in prompt.lower()


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

        brief = _make_brief(concept="Galaxy cat astronaut", product_type="sublimation")
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

        brief = _make_brief(product_type="sublimation")
        generate_image(brief, bedrock_client=mock_client)

        call_kwargs = mock_client.invoke_model.call_args
        body_str = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", ""))
        body = json.loads(body_str)
        assert body.get("aspect_ratio") == "1:1"
        assert body.get("mode") == "text-to-image"
        assert body.get("output_format") == "png"


class TestRemoveBackground:
    def _mock_bg_response(self, output_bytes: bytes) -> MagicMock:
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": MagicMock(
                read=lambda: json.dumps(
                    {"images": [base64.b64encode(output_bytes).decode()]}
                ).encode()
            ),
        }
        return mock_client

    def test_calls_remove_background_model(self) -> None:
        input_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        output_png = b"\x89PNG\r\n\x1a\n" + b"\x01" * 100
        mock_client = self._mock_bg_response(output_png)

        result = remove_background(input_png, bedrock_client=mock_client)

        call_kwargs = mock_client.invoke_model.call_args
        assert call_kwargs.kwargs["modelId"] == REMOVE_BG_MODEL_ID
        assert result == output_png

    def test_sends_image_as_base64(self) -> None:
        input_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        output_png = b"\x89PNG\r\n\x1a\n" + b"\x01" * 100
        mock_client = self._mock_bg_response(output_png)

        remove_background(input_png, bedrock_client=mock_client)

        call_kwargs = mock_client.invoke_model.call_args
        body = json.loads(call_kwargs.kwargs["body"])
        assert body["image"] == base64.b64encode(input_png).decode()

    def test_dtf_generate_calls_remove_background(self) -> None:
        fake_gen_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        fake_nobg_png = b"\x89PNG\r\n\x1a\n" + b"\x01" * 100
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = [
            {
                "body": MagicMock(
                    read=lambda: json.dumps(
                        {"images": [base64.b64encode(fake_gen_png).decode()]}
                    ).encode()
                ),
            },
            {
                "body": MagicMock(
                    read=lambda: json.dumps(
                        {"images": [base64.b64encode(fake_nobg_png).decode()]}
                    ).encode()
                ),
            },
        ]

        brief = _make_brief(product_type="dtf_apparel")
        result = generate_image(brief, bedrock_client=mock_client)

        assert mock_client.invoke_model.call_count == 2
        assert result.image_bytes == fake_nobg_png

    def test_sublimation_skips_remove_background(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": MagicMock(
                read=lambda: json.dumps({"images": [base64.b64encode(fake_png).decode()]}).encode()
            ),
        }

        brief = _make_brief(product_type="sublimation")
        result = generate_image(brief, bedrock_client=mock_client)

        assert mock_client.invoke_model.call_count == 1
        assert result.image_bytes == fake_png

    def test_bg_removal_failure_returns_raw_image(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = [
            {
                "body": MagicMock(
                    read=lambda: json.dumps(
                        {"images": [base64.b64encode(fake_png).decode()]}
                    ).encode()
                ),
            },
            Exception("BG removal failed"),
        ]

        brief = _make_brief(product_type="dtf_apparel")
        result = generate_image(brief, bedrock_client=mock_client)

        assert result.image_bytes == fake_png
        assert result.error is None
