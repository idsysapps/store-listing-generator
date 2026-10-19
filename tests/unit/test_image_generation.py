"""Tests for ComfyUI/Flux image generation from design briefs."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from store_listing.orchestration.comfyui_client import ComfyUIError, ComfyUIResult
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.image_generation import (
    build_flux_prompt,
    generate_image,
)


def _make_brief(**overrides: Any) -> DesignBrief:
    defaults: dict[str, Any] = {
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


def _client(image_bytes: bytes | None = b"\x89PNG\r\n\x1a\n", **kwargs: Any) -> MagicMock:
    client = MagicMock()
    client.generate_image.return_value = ComfyUIResult(image_bytes=image_bytes, prompt_id="p1")
    client.remove_background.return_value = ComfyUIResult(
        image_bytes=b"transparent", prompt_id="p2"
    )
    for key, value in kwargs.items():
        setattr(client, key, value)
    return client


class TestBuildFluxPrompt:
    def test_full_bleed_includes_the_scene_style_and_product_context(self) -> None:
        brief = _make_brief(
            layout_type="full_bleed",
            headline_text=None,
            tagline_text=None,
            scene_description="a skeleton doing yoga on a beach towel",
        )
        prompt = build_flux_prompt(brief)

        assert "a skeleton doing yoga on a beach towel" in prompt
        assert "Retro vintage" in prompt
        assert "DTF" in prompt

    def test_full_bleed_explicitly_excludes_lettering(self) -> None:
        brief = _make_brief(layout_type="full_bleed", headline_text=None, tagline_text=None)
        prompt = build_flux_prompt(brief)

        assert "do not include any text or lettering" in prompt.lower()

    def test_full_bleed_does_not_contain_the_headline(self) -> None:
        brief = _make_brief(
            layout_type="full_bleed",
            headline_text="SPOOKY SEASON",
            tagline_text="TRICK OR TREAT",
        )
        prompt = build_flux_prompt(brief)

        assert "SPOOKY SEASON" not in prompt
        assert "TRICK OR TREAT" not in prompt

    def test_text_top_includes_the_quoted_headline_and_colour(self) -> None:
        brief = _make_brief(
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            tagline_text=None,
            font_color="#FF8C42",
        )
        prompt = build_flux_prompt(brief)

        assert 'Bold text reading "SPOOKY SEASON"' in prompt
        assert "#FF8C42" in prompt
        assert "at the top" in prompt

    def test_text_top_does_not_contain_the_tagline(self) -> None:
        brief = _make_brief(
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            tagline_text="TRICK OR TREAT",
        )
        prompt = build_flux_prompt(brief)

        assert "TRICK OR TREAT" not in prompt

    def test_text_top_bottom_includes_both_strings(self) -> None:
        brief = _make_brief(
            layout_type="text_top_bottom",
            headline_text="SPOOKY SEASON",
            tagline_text="TRICK OR TREAT",
            font_color="#FFFFFF",
        )
        prompt = build_flux_prompt(brief)

        assert 'Bold text reading "SPOOKY SEASON"' in prompt
        assert 'Text reading "TRICK OR TREAT"' in prompt
        assert "at the top" in prompt
        assert "at the bottom" in prompt

    def test_font_colour_defaults_to_black(self) -> None:
        brief = _make_brief(
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color=None,
        )
        prompt = build_flux_prompt(brief)

        assert "#000000" in prompt

    def test_a_white_font_colour_is_kept_for_dtf_apparel(self) -> None:
        # The dtf background is magenta and is removed before printing, so
        # white text is visible.
        brief = _make_brief(
            product_type="dtf_apparel",
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color="#FFFFFF",
        )
        prompt = build_flux_prompt(brief)

        assert "#FFFFFF" in prompt

    def test_a_white_font_colour_falls_back_to_black_on_a_white_background(self) -> None:
        # The print-through product types keep their white background, so
        # white text on it would be invisible.
        brief = _make_brief(
            product_type="sublimation",
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color="#FFFFFF",
        )
        prompt = build_flux_prompt(brief)

        assert "#FFFFFF" not in prompt
        assert "#000000" in prompt

    def test_a_near_white_font_colour_falls_back_to_black(self) -> None:
        brief = _make_brief(
            product_type="sublimation",
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color="#F5F5F0",
        )
        prompt = build_flux_prompt(brief)

        assert "#F5F5F0" not in prompt
        assert "#000000" in prompt

    def test_regeneration_feedback_is_appended(self) -> None:
        brief = _make_brief(
            layout_type="full_bleed",
            headline_text=None,
            regeneration_feedback="more vibrant colors please",
        )
        prompt = build_flux_prompt(brief)

        assert "more vibrant colors please" in prompt

    def test_never_asks_for_extra_elements(self) -> None:
        brief = _make_brief(layout_type="full_bleed", headline_text=None)
        prompt = build_flux_prompt(brief)

        assert "Do not draw anything else." in prompt


class TestGenerateImage:
    def test_calls_the_client_with_the_folded_prompt(self) -> None:
        client = _client()
        brief = _make_brief(
            product_type="sublimation",
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color="#FF0000",
        )

        result = generate_image(brief, comfyui_client=client)

        assert result.error is None
        assert result.image_bytes == b"\x89PNG\r\n\x1a\n"
        prompt = client.generate_image.call_args.kwargs["prompt"]
        assert 'Bold text reading "SPOOKY SEASON"' in prompt

    def test_a_client_error_becomes_an_error_result_instead_of_raising(self) -> None:
        client = _client()
        client.generate_image.side_effect = ComfyUIError("ComfyUI rejected the workflow")
        brief = _make_brief(layout_type="full_bleed", headline_text=None)

        result = generate_image(brief, comfyui_client=client)

        assert result.image_bytes is None
        assert result.error is not None
        assert "ComfyUI rejected the workflow" in result.error

    def test_dtf_brief_returns_the_raw_and_transparent_bytes(self) -> None:
        client = _client(image_bytes=b"raw-bytes")
        brief = _make_brief(
            product_type="dtf_apparel", layout_type="full_bleed", headline_text=None
        )

        result = generate_image(brief, comfyui_client=client)

        assert result.raw_bytes == b"raw-bytes"
        assert result.image_bytes == b"transparent"
        client.remove_background.assert_called_once_with(b"raw-bytes")

    def test_non_dtf_brief_does_not_remove_the_background(self) -> None:
        client = _client(image_bytes=b"raw-bytes")
        brief = _make_brief(
            product_type="sublimation", layout_type="full_bleed", headline_text=None
        )

        result = generate_image(brief, comfyui_client=client)

        assert result.image_bytes == b"raw-bytes"
        client.remove_background.assert_not_called()

    def test_a_background_removal_failure_falls_back_to_the_raw_image(self) -> None:
        client = _client(image_bytes=b"raw-bytes")
        client.remove_background.side_effect = ComfyUIError("BiRefNet exploded")
        brief = _make_brief(
            product_type="dtf_apparel", layout_type="full_bleed", headline_text=None
        )

        result = generate_image(brief, comfyui_client=client)

        assert result.image_bytes == b"raw-bytes"
        assert result.raw_bytes == b"raw-bytes"

    def test_the_seed_is_passed_through(self) -> None:
        client = _client()
        brief = _make_brief(layout_type="full_bleed", headline_text=None)

        generate_image(brief, comfyui_client=client, seed=1234)

        assert client.generate_image.call_args.kwargs["seed"] == 1234

    def test_an_extra_prompt_is_appended_for_retries(self) -> None:
        client = _client()
        brief = _make_brief(layout_type="full_bleed", headline_text=None)

        generate_image(
            brief,
            comfyui_client=client,
            extra_prompt='The text must read exactly "SPOOKY SEASON".',
        )

        prompt = client.generate_image.call_args.kwargs["prompt"]
        assert 'The text must read exactly "SPOOKY SEASON".' in prompt
