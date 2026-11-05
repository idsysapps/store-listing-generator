"""Tests for DTF-safe text embedding.

The font owns spelling. These tests only assert the ink rules: every inked
pixel stays at full opacity, distress is binary, and the carrier outside the
text is not repainted.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from store_listing.orchestration.text_compositor import LayoutSpec, composite_text
from store_listing.orchestration.text_embed import (
    PROFILES,
    composite_embedded,
    embed_layer,
    profile_for,
)
from store_listing.orchestration.text_integration import TextPlan, infer_surface


def _text_layer() -> Image.Image:
    layer = Image.new("RGBA", (240, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((20, 20, 220, 70), fill=(252, 244, 199, 255))
    return layer


def _background() -> Image.Image:
    image = Image.new("RGB", (240, 100), (190, 20, 45))
    draw = ImageDraw.Draw(image)
    for x in range(0, 240, 7):
        draw.line((x, 0, x, 100), fill=(150, 10, 35), width=1)
    return image


def _carrier_canvas() -> Image.Image:
    image = Image.new("RGB", (600, 400), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 145, 450, 255), fill=(246, 241, 232))
    return image


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestProfiles:
    def test_off_values_resolve_to_none(self) -> None:
        for value in (None, "", "off", "OFF", "none", "false", "0"):
            assert profile_for(value) is None

    def test_known_profiles_resolve(self) -> None:
        assert profile_for("v11") is PROFILES["v11"]
        assert profile_for("v13") is PROFILES["v13"]

    def test_an_unknown_profile_fails_safe_to_off(self) -> None:
        assert profile_for("not-a-profile") is None

    def test_a_red_carrier_is_inferred_from_its_description(self) -> None:
        assert infer_surface("banner", "a blank red ribbon across the shield") == "red"
        assert TextPlan(method="printed", carrier="banner", surface="red").surface == "red"


class TestEmbedLayer:
    def test_alpha_is_only_full_ink_or_no_ink(self) -> None:
        layer = embed_layer(_text_layer(), _background(), PROFILES["v13"])

        alpha = np.asarray(layer.getchannel("A"))
        assert set(np.unique(alpha)).issubset({0, 255})
        assert (alpha == 255).any(), "the glyph fill must still contain ink"
        assert (alpha == 0).any(), "the distress must create empty pixels"

    def test_heavy_wear_has_at_least_as_many_knockouts_as_light_wear(self) -> None:
        light = embed_layer(_text_layer(), _background(), PROFILES["v12"])
        heavy = embed_layer(_text_layer(), _background(), PROFILES["v13"])

        light_holes = int((np.asarray(light.getchannel("A")) == 0).sum())
        heavy_holes = int((np.asarray(heavy.getchannel("A")) == 0).sum())
        assert heavy_holes >= light_holes

    def test_the_ink_colour_is_shaded_without_changing_the_alpha_rule(self) -> None:
        layer = embed_layer(_text_layer(), _background(), PROFILES["v11"])

        colours = {
            tuple(int(channel) for channel in pixel[:3])
            for pixel in np.asarray(layer).reshape(-1, 4)
            if pixel[3] == 255
        }
        assert len(colours) > 1, "the carrier luminance should vary the ink"

    def test_the_same_input_is_deterministic(self) -> None:
        first = embed_layer(_text_layer(), _background(), PROFILES["v13"])
        second = embed_layer(_text_layer(), _background(), PROFILES["v13"])

        assert first.tobytes() == second.tobytes()

    def test_the_background_and_foreground_must_match(self) -> None:
        with pytest.raises(ValueError):
            embed_layer(_text_layer(), Image.new("RGB", (10, 10)), PROFILES["v11"])

    def test_compositing_keeps_the_size(self) -> None:
        result = composite_embedded(_background(), _text_layer(), PROFILES["v11"])

        assert result.size == _background().size


class TestCompositeTextEmbed:
    def _layout(self, profile: str | None) -> LayoutSpec:
        return LayoutSpec(
            layout_type="text_top",
            headline_text="FUELED BY CAFFEINE",
            tagline_text="AND DUTY",
            font_color="#FCF4C7",
            font_family="anton",
            text_plan=TextPlan(
                method="printed",
                carrier="label",
                surface="paper",
                region="center",
                ink="dark",
            ),
            text_embed_profile=profile,
        )

    def test_the_carrier_text_is_embedded(self) -> None:
        canvas = _png(_carrier_canvas())

        plain = composite_text(canvas, self._layout(None))
        embedded = composite_text(canvas, self._layout("v13"))

        assert embedded != plain

    def test_the_area_outside_the_text_is_not_repainted(self) -> None:
        canvas = _carrier_canvas()
        embedded = composite_text(_png(canvas), self._layout("v13"))

        image = Image.open(io.BytesIO(embedded)).convert("RGB")
        assert image.getpixel((10, 10)) == canvas.getpixel((10, 10))
        assert image.getpixel((590, 390)) == canvas.getpixel((590, 390))

    def test_the_lockup_is_drawn_inside_the_carrier(self) -> None:
        canvas = _carrier_canvas()
        embedded = composite_text(_png(canvas), self._layout("v13"))

        image = Image.open(io.BytesIO(embedded)).convert("RGB")
        carrier = image.crop((150, 145, 451, 256))
        colours = {
            (int(colour[0]), int(colour[1]), int(colour[2]))
            for _count, colour in (carrier.getcolors(maxcolors=1_000_000) or [])
            if isinstance(colour, tuple) and len(colour) == 3
        }
        assert any(sum(colour) > 600 for colour in colours)

    def test_a_profile_without_a_carrier_keeps_the_clean_render(self) -> None:
        canvas = _png(_carrier_canvas())
        layout = self._layout("v13")
        layout.text_plan = None
        clean = self._layout(None)
        clean.text_plan = None

        plain = composite_text(canvas, clean)
        embedded = composite_text(canvas, layout)

        assert embedded == plain

    def test_the_final_transparent_alpha_is_binary(self) -> None:
        canvas = Image.new("RGBA", (600, 400), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((150, 145, 450, 255), fill=(246, 241, 232, 255))
        result = Image.open(io.BytesIO(composite_text(_png(canvas), self._layout("v13")))).convert(
            "RGBA"
        )

        assert set(np.unique(np.asarray(result.getchannel("A")))).issubset({0, 255})
