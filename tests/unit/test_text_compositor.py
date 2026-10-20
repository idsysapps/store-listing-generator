"""Tests for the text compositor.

The diffusion model draws the artwork, the compositor draws the text on top of
it: Flux Schnell cannot spell, so the text it produced was misspelled,
duplicated or dropped (issue #187).
"""

from __future__ import annotations

import io

from PIL import Image

from store_listing.orchestration.text_compositor import (
    HEADLINE_ZONE,
    TAGLINE_ZONE,
    LayoutSpec,
    composite_text,
)


def _canvas(
    size: tuple[int, int] = (1024, 1024), color: tuple[int, int, int] = (255, 255, 255)
) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _distinct_colours(image: Image.Image) -> set[tuple[int, int, int]]:
    """The colours that occur in the image."""
    colours = image.getcolors(maxcolors=1_000_000) or []
    return {
        (int(colour[0]), int(colour[1]), int(colour[2]))
        for _count, colour in colours
        if isinstance(colour, tuple) and len(colour) == 3
    }


def _zone_colours(data: bytes, top: bool) -> set[tuple[int, int, int]]:
    """The distinct colours of the zone the text is drawn in."""
    image = _image(data)
    width, height = image.size
    if top:
        box = (0, 0, width, int(height * HEADLINE_ZONE))
    else:
        box = (0, int(height * (1 - TAGLINE_ZONE)), width, height)
    return _distinct_colours(image.crop(box))


class TestFullBleed:
    def test_a_full_bleed_image_is_returned_untouched(self) -> None:
        canvas = _canvas()

        result = composite_text(
            canvas, LayoutSpec(layout_type="full_bleed", headline_text="SPOOKY")
        )

        assert result == canvas

    def test_an_image_without_text_is_returned_untouched(self) -> None:
        canvas = _canvas()

        result = composite_text(canvas, LayoutSpec(layout_type="text_top", headline_text=None))

        assert result == canvas


class TestHeadline:
    def test_the_headline_is_drawn_in_the_top_zone(self) -> None:
        result = composite_text(
            _canvas(), LayoutSpec(layout_type="text_top", headline_text="SPOOKY SEASON")
        )

        assert any(colour != (255, 255, 255) for colour in _zone_colours(result, top=True))

    def test_the_bottom_zone_is_left_alone(self) -> None:
        result = composite_text(
            _canvas(), LayoutSpec(layout_type="text_top", headline_text="SPOOKY SEASON")
        )

        assert all(colour == (255, 255, 255) for colour in _zone_colours(result, top=False))

    def test_the_font_colour_is_used(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(layout_type="text_top", headline_text="SPOOKY SEASON", font_color="#FF0000"),
        )

        assert (255, 0, 0) in _zone_colours(result, top=True)

    def test_the_image_keeps_its_size(self) -> None:
        result = composite_text(
            _canvas((800, 600)), LayoutSpec(layout_type="text_top", headline_text="SPOOKY SEASON")
        )

        assert _image(result).size == (800, 600)

    def test_a_long_headline_stays_inside_the_zone(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="Fixer Upper Ghosts Included For Haunted Nights Everywhere",
            ),
        )

        image = _image(result)
        width, height = image.size
        drawn = [
            (x, y)
            for y in range(int(height * HEADLINE_ZONE))
            for x in range(width)
            if image.getpixel((x, y)) != (255, 255, 255)
        ]
        assert drawn, "the headline should have been drawn"
        assert max(x for x, _ in drawn) < width
        assert max(y for _, y in drawn) < int(height * HEADLINE_ZONE)


class TestTagline:
    def test_the_tagline_is_drawn_in_the_bottom_zone(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top_bottom", headline_text="SPOOKY", tagline_text="TRICK OR TREAT"
            ),
        )

        assert any(colour != (255, 255, 255) for colour in _zone_colours(result, top=False))

    def test_both_zones_are_used_for_text_top_bottom(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top_bottom", headline_text="SPOOKY", tagline_text="TRICK OR TREAT"
            ),
        )

        assert any(colour != (255, 255, 255) for colour in _zone_colours(result, top=True))
        assert any(colour != (255, 255, 255) for colour in _zone_colours(result, top=False))

    def test_a_brief_with_a_tagline_gets_one(self) -> None:
        # A brief that asks for a tagline gets one, even when the layout says
        # text_top - otherwise the tagline would be dropped on the floor.
        spec = LayoutSpec(
            layout_type="text_top", headline_text="SPOOKY", tagline_text="TRICK OR TREAT"
        )

        assert spec.layout_type == "text_top_bottom"

        result = composite_text(_canvas(), spec)

        assert any(colour != (255, 255, 255) for colour in _zone_colours(result, top=False))


class TestUnsupportedGlyphs:
    def test_a_character_the_font_cannot_render_is_dropped(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(layout_type="text_top", headline_text="🎃 SPOOKY SEASON"),
        )

        image = _image(result)
        width, height = image.size
        drawn = [
            (x, y)
            for y in range(int(height * HEADLINE_ZONE))
            for x in range(width)
            if image.getpixel((x, y)) != (255, 255, 255)
        ]
        assert drawn, "the readable part of the headline should still be drawn"
        # The emoji would be drawn as an empty box, so the drawn area is much
        # narrower than a box glyph would be.
        assert (max(x for x, _ in drawn) - min(x for x, _ in drawn)) < width * 0.9

    def test_a_headline_of_only_unsupported_characters_draws_nothing(self) -> None:
        result = composite_text(_canvas(), LayoutSpec(layout_type="text_top", headline_text="🎃"))

        assert all(colour == (255, 255, 255) for colour in _zone_colours(result, top=True))
