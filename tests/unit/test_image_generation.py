"""Tests for ComfyUI/Flux image generation from design briefs."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.comfyui_client import ComfyUIError, ComfyUIResult
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.design_templates import (
    ART_TEMPLATE_IDS,
    FULL_BLEED_COMPOSITIONS,
    TEMPLATES,
)
from store_listing.orchestration.image_generation import (
    LIGHT_INK,
    PromptBuildError,
    build_flux_prompt,
    build_negative_prompt,
    font_colour,
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
        "scene_description": "a skeleton doing yoga poses on a beach towel",
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


class TestTheGarmentMaterial:
    def test_a_dark_font_on_a_dark_garment_falls_back_to_light(self) -> None:
        brief = _make_brief(
            font_color="#111111",
            recommended_materials=[{"external_id": "shirt-jet-black-m", "color": "Jet Black"}],
        )

        assert font_colour(brief) == LIGHT_INK

    def test_a_light_font_on_a_dark_garment_is_kept(self) -> None:
        brief = _make_brief(
            font_color="#F2C14E",
            recommended_materials=[{"external_id": "shirt-jet-black-m", "color": "Jet Black"}],
        )

        assert font_colour(brief) == "#F2C14E"

    def test_a_dark_font_on_a_light_garment_is_kept(self) -> None:
        brief = _make_brief(
            font_color="#111111",
            recommended_materials=[{"external_id": "shirt-white-m", "color": "White"}],
        )

        assert font_colour(brief) == "#111111"


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

    def test_a_dtf_prompt_keeps_the_garment_as_the_background(self) -> None:
        prompt = build_flux_prompt(_make_brief())

        assert "no printed background" in prompt
        assert "garment colour is the background" in prompt

    def test_a_dtf_prompt_does_not_ask_for_a_sticker_or_a_full_bleed_plate(self) -> None:
        prompt = build_flux_prompt(
            _make_brief(layout_type="mid_ribbon", headline_text="POWERED BY COFFEE")
        )

        assert "sticker illustration" not in prompt
        assert "the subject is isolated on the garment" in prompt
        assert "fills the whole canvas edge to edge" not in prompt

    def test_an_art_only_dtf_prompt_is_not_a_full_bleed_plate(self) -> None:
        prompt = build_flux_prompt(
            _make_brief(
                layout_type="full_bleed",
                copy_mode="no_text",
                headline_text=None,
                tagline_text=None,
            )
        )

        assert "full-bleed" not in prompt.lower()
        assert "edge to edge" not in prompt.lower()
        assert "garment colour around it" in prompt

    def test_a_dtf_negative_keeps_the_background_and_sticker_border_out(self) -> None:
        negative = build_negative_prompt(_make_brief())

        assert "no background" in negative
        assert "no sticker border" in negative

    def test_the_preset_palette_reaches_the_prompt(self) -> None:
        brief = _make_brief(
            style_preset="hand_inked_woodcut",
            visual_style="dark green and amber",
        )

        prompt = build_flux_prompt(brief)

        assert "forest green" in prompt

    def test_a_dtf_prompt_says_the_magenta_is_only_the_backdrop(self) -> None:
        prompt = build_flux_prompt(_make_brief())

        assert "magenta is only the empty key backdrop" in prompt

    def test_a_dtf_negative_forbids_magenta_when_the_palette_has_none(self) -> None:
        brief = _make_brief(style_preset="hand_inked_woodcut")

        assert "no hot pink" in build_negative_prompt(brief)

    def test_a_pink_palette_keeps_its_pink(self) -> None:
        brief = _make_brief(style_preset="kawaii_flat")

        assert "no hot pink" not in build_negative_prompt(brief)

    def test_a_dark_garment_prompt_asks_for_light_highlights_without_a_black_background(
        self,
    ) -> None:
        # The old rule said "the black areas are the shirt", and Flux painted a
        # black half-canvas next to the magenta key (briefs 205, 206). The dark
        # ink is removed after keying instead.
        brief = _make_brief(
            recommended_materials=[
                {
                    "external_id": "shirt-jet-black-m",
                    "name": "Core Adult",
                    "color": "Jet Black",
                }
            ]
        )

        prompt = build_flux_prompt(brief)

        assert "light cream/white highlights" in prompt
        assert "black areas are the shirt" not in prompt
        assert "do not paint a black" in prompt.lower()
        assert "only background is the magenta key" in prompt.lower()

    def test_a_light_garment_does_not_get_the_dark_rule(self) -> None:
        brief = _make_brief(
            recommended_materials=[
                {"external_id": "shirt-white-m", "name": "Core Adult", "color": "White"}
            ]
        )

        assert "black areas are the shirt" not in build_flux_prompt(brief)

    def test_a_dark_garment_negative_forbids_a_black_fill(self) -> None:
        brief = _make_brief(
            recommended_materials=[{"external_id": "shirt-jet-black-m", "color": "Jet Black"}]
        )

        assert "no large black fill" in build_negative_prompt(brief)

    def test_a_no_text_brief_says_the_illustration_carries_the_idea(self) -> None:
        brief = _make_brief(
            copy_mode="no_text",
            layout_type="full_bleed",
            headline_text=None,
            tagline_text=None,
        )
        prompt = build_flux_prompt(brief)

        assert "do not include any text or lettering" in prompt.lower()
        assert "illustration alone carries the idea" in prompt

    def test_full_bleed_does_not_contain_the_headline(self) -> None:
        brief = _make_brief(
            layout_type="full_bleed",
            headline_text="SPOOKY SEASON",
            tagline_text="TRICK OR TREAT",
        )
        prompt = build_flux_prompt(brief)

        assert "SPOOKY SEASON" not in prompt
        assert "TRICK OR TREAT" not in prompt

    def test_the_prompt_never_contains_the_headline_or_the_tagline(self) -> None:
        # The text is composited onto the artwork, so the model must not be
        # asked to spell it (issue #187).
        for layout in ("full_bleed", "text_top", "text_top_bottom"):
            brief = _make_brief(
                layout_type=layout,
                headline_text="SPOOKY SEASON",
                tagline_text="TRICK OR TREAT",
            )
            prompt = build_flux_prompt(brief)

            assert "SPOOKY SEASON" not in prompt, layout
            assert "TRICK OR TREAT" not in prompt, layout

    def test_the_prompt_asks_for_artwork_without_lettering(self) -> None:
        brief = _make_brief(layout_type="text_top", headline_text="SPOOKY SEASON")
        prompt = build_flux_prompt(brief)

        assert "without any lettering" in prompt.lower()

    def test_the_prompt_never_carries_the_words_of_the_scene(self) -> None:
        # The image model cannot spell: a scene that quotes a punchline makes it
        # draw the words wrong (issue #208).
        brief = _make_brief(
            scene_description=(
                "A grinning pumpkin with a speech bubble saying "
                "'To get to the other side!' with a small ghost peeking out"
            )
        )
        prompt = build_flux_prompt(brief)

        assert "other side" not in prompt
        assert "speech bubble" not in prompt
        assert "pumpkin" in prompt

    def test_the_prompt_asks_for_clear_space_where_the_text_goes(self) -> None:
        brief = _make_brief(
            layout_type="text_top_bottom",
            headline_text="SPOOKY SEASON",
            tagline_text="TRICK OR TREAT",
        )
        prompt = build_flux_prompt(brief).lower()

        assert "top" in prompt
        assert "bottom" in prompt
        assert "empty" in prompt

    def test_the_background_instruction_comes_last(self) -> None:
        # A visual style that asks for a black background must not win over the
        # white paper background of a print-through product (issue #188).
        brief = _make_brief(
            product_type="sublimation",
            layout_type="full_bleed",
            headline_text=None,
            visual_style="High-contrast black and orange palette",
        )
        prompt = build_flux_prompt(brief)

        assert prompt.index("white background") > prompt.index("print artwork only")

    def test_the_prompt_forbids_a_product_mockup(self) -> None:
        brief = _make_brief(product_type="dtf_apparel", layout_type="full_bleed")
        prompt = build_flux_prompt(brief).lower()

        assert "mockup" in prompt

    def test_the_prompt_says_the_artwork_is_not_the_object_it_is_printed_on(self) -> None:
        # A scene that names the product makes the model draw the product
        # (issue #195).
        brief = _make_brief(product_type="dtf_apparel", layout_type="full_bleed")
        prompt = build_flux_prompt(brief).lower()

        assert "print artwork" in prompt
        assert "not the object it is printed on" in prompt

    def test_the_prompt_names_the_products_that_must_not_be_drawn(self) -> None:
        brief = _make_brief(product_type="dtf_apparel", layout_type="full_bleed")
        prompt = build_flux_prompt(brief).lower()

        for product in ("bag", "tote", "mug", "shirt"):
            assert f"no {product}" in prompt

    def test_a_brief_without_a_scene_does_not_fall_back_to_the_concept(self) -> None:
        # The concept names the design idea - "Trick or Treat Bag with Candy
        # Scatter" - and sending it to the model draws the bag (issue #195).
        brief = _make_brief(
            concept="Trick or Treat Bag with Candy Scatter",
            scene_description=None,
            layout_type="text_top",
            headline_text="Trick or Treat",
        )

        with pytest.raises(PromptBuildError):
            build_flux_prompt(brief)

    def test_a_text_only_brief_is_not_sent_to_the_image_model(self) -> None:
        brief = _make_brief(layout_type="text_only", headline_text="SPOOKY SEASON")

        with pytest.raises(PromptBuildError):
            build_flux_prompt(brief)

    def test_a_brief_without_a_scene_reports_an_error_instead_of_raising(self) -> None:
        client = _client()
        brief = _make_brief(scene_description=None, layout_type="full_bleed")

        result = generate_image(brief, comfyui_client=client)

        assert result.image_bytes is None
        assert result.error is not None
        assert "scene" in result.error
        client.generate_image.assert_not_called()

    def test_font_colour_defaults_to_black(self) -> None:
        brief = _make_brief(
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color=None,
        )

        assert font_colour(brief) == "#000000"

    def test_a_white_font_colour_is_kept_for_dtf_apparel(self) -> None:
        # The dtf background is magenta and is removed before printing, so
        # white text is visible.
        brief = _make_brief(
            product_type="dtf_apparel",
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color="#FFFFFF",
        )

        assert font_colour(brief) == "#FFFFFF"

    def test_a_white_font_colour_falls_back_to_black_on_a_white_background(self) -> None:
        # The print-through product types keep their white background, so
        # white text on it would be invisible.
        brief = _make_brief(
            product_type="sublimation",
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color="#FFFFFF",
        )

        assert font_colour(brief) == "#000000"

    def test_a_near_white_font_colour_falls_back_to_black(self) -> None:
        brief = _make_brief(
            product_type="sublimation",
            layout_type="text_top",
            headline_text="SPOOKY SEASON",
            font_color="#F5F5F0",
        )

        assert font_colour(brief) == "#000000"

    def test_regeneration_feedback_is_appended(self) -> None:
        brief = _make_brief(
            layout_type="full_bleed",
            headline_text=None,
            regeneration_feedback="more vibrant colors please",
        )
        prompt = build_flux_prompt(brief)

        assert "more vibrant colors please" in prompt

    def test_the_prompt_does_not_tell_the_model_to_stop(self) -> None:
        # "Do not draw anything else" was the last thing the model read, and
        # every design came back as one small centred object (issue #238).
        brief = _make_brief(layout_type="full_bleed", headline_text=None)
        prompt = build_flux_prompt(brief)

        assert "Do not draw anything else." not in prompt


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
        assert "SPOOKY SEASON" not in prompt
        assert "without any lettering" in prompt.lower()

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


class TestTheLookOfTheDesign:
    def test_the_look_of_the_preset_is_in_the_prompt(self) -> None:
        brief = _make_brief(
            layout_type="full_bleed",
            style_preset="halftone_comic",
            visual_style="retro vintage, orange and teal",
        )

        prompt = build_flux_prompt(brief)

        assert "comic book illustration" in prompt.lower()
        assert "halftone dot shading" not in prompt.lower(), "the hard screen read as squares"
        assert "no halftone dots" in prompt.lower()

    def test_every_design_asks_for_crisp_art_without_a_halftone(self) -> None:
        brief = _make_brief(layout_type="hero_bottom", headline_text="SLOW DOWN")

        prompt = build_flux_prompt(brief)

        assert "no halftone dots" in prompt.lower()
        assert "no dot screen" in prompt.lower()

    def test_the_free_style_of_the_brief_does_not_fight_the_look(self) -> None:
        # Free text made every design come out as the same flat vector
        # clip-art (issue #227).
        brief = _make_brief(
            layout_type="full_bleed",
            style_preset="kawaii_flat",
            visual_style="photorealistic 3d render, glossy gradient",
        )

        prompt = build_flux_prompt(brief)

        assert "kawaii" in prompt.lower()
        assert "glossy gradient" not in prompt.lower()

    def test_a_brief_without_a_look_keeps_its_own_style(self) -> None:
        # The briefs that were written before the looks existed still render.
        brief = _make_brief(
            layout_type="full_bleed",
            style_preset=None,
            visual_style="retro vintage, orange and teal",
        )

        prompt = build_flux_prompt(brief)

        assert "orange and teal" in prompt.lower()

    def test_the_negative_prompt_keeps_the_paper_clean(self) -> None:
        brief = _make_brief(style_preset="riso_two_color")

        negative = build_negative_prompt(brief)

        assert "paper texture" in negative
        assert "photorealistic" in negative

    def test_the_negative_prompt_reaches_the_model(self) -> None:
        client = _client()

        generate_image(_make_brief(style_preset="vintage_screenprint"), comfyui_client=client)

        negative = client.generate_image.call_args.kwargs["negative_prompt"]
        assert "paper texture" in negative

    def test_the_background_stays_the_last_word(self) -> None:
        # The background comes last so that a look cannot win over it
        # (issue #188).
        brief = _make_brief(
            layout_type="full_bleed",
            style_preset="distressed_band_tee",
        )

        prompt = build_flux_prompt(brief)

        assert (
            prompt.lower()
            .rstrip(".")
            .endswith("plain magenta backdrop, chroma key background, uniform flat magenta color")
        )


class TestTheNewLayouts:
    def test_a_banner_reserves_the_middle_of_the_canvas(self) -> None:
        brief = _make_brief(layout_type="banner", headline_text="GOBBLE")

        prompt = build_flux_prompt(brief).lower()

        assert "across the middle" in prompt

    def test_a_stack_side_reserves_the_right_of_the_canvas(self) -> None:
        brief = _make_brief(layout_type="stack_side", headline_text="GOBBLE")

        prompt = build_flux_prompt(brief).lower()

        assert "right side" in prompt

    def test_a_badge_is_drawn_as_a_roundel(self) -> None:
        brief = _make_brief(layout_type="badge", headline_text="GOBBLE")

        prompt = build_flux_prompt(brief).lower()

        assert "bold circle" in prompt


class TestTheTemplates:
    def test_no_template_asks_for_a_centred_composition(self) -> None:
        # "compact centered composition" was in every prompt, which is why
        # every design was an object in the middle of the canvas (issue #238).
        for template_id in ART_TEMPLATE_IDS:
            brief = _make_brief(
                layout_type=template_id,
                headline_text="GOBBLE",
                tagline_text="TILL THE BELL",
            )

            prompt = build_flux_prompt(brief).lower()

            assert "centered composition" not in prompt, template_id
            assert "centred composition" not in prompt, template_id

    def test_the_composition_of_the_template_is_in_the_prompt(self) -> None:
        for template_id in ART_TEMPLATE_IDS:
            # DTF replaces the edge-to-edge composition with an isolated
            # subject, so this test uses a print-through product.
            brief = _make_brief(
                product_type="sublimation",
                layout_type=template_id,
                headline_text="GOBBLE",
                tagline_text="TILL THE BELL",
            )

            prompt = build_flux_prompt(brief)

            assert TEMPLATES[template_id].composition in prompt, template_id

    def test_an_art_only_design_gets_a_full_bleed_composition(self) -> None:
        brief = _make_brief(
            product_type="sublimation",
            layout_type="full_bleed",
            headline_text=None,
            tagline_text=None,
            concept="a raccoon king",
        )

        prompt = build_flux_prompt(brief)

        assert any(composition in prompt for composition in FULL_BLEED_COMPOSITIONS)

    def test_two_art_only_designs_can_get_different_compositions(self) -> None:
        prompts = {
            build_flux_prompt(
                _make_brief(
                    concept=f"concept {index}",
                    layout_type="full_bleed",
                    headline_text=None,
                    tagline_text=None,
                )
            )
            for index in range(20)
        }

        assert len(prompts) > 1

    def test_the_template_size_reaches_the_model(self) -> None:
        client = _client()

        generate_image(
            _make_brief(layout_type="wide_scene", headline_text="GOBBLE"),
            comfyui_client=client,
        )

        assert client.generate_image.call_args.kwargs["width"] == 1216
        assert client.generate_image.call_args.kwargs["height"] == 832

    def test_a_square_template_is_rendered_square(self) -> None:
        client = _client()

        generate_image(
            _make_brief(layout_type="roundel", headline_text="GOBBLE"),
            comfyui_client=client,
        )

        assert client.generate_image.call_args.kwargs["width"] == 1024
        assert client.generate_image.call_args.kwargs["height"] == 1024
