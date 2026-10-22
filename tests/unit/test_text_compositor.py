"""Tests for the text compositor.

The diffusion model draws the artwork, the compositor draws the text on top of
it: Flux Schnell cannot spell, so the text it produced was misspelled,
duplicated or dropped (issue #187).
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from store_listing.orchestration.design_templates import ART_TEMPLATE_IDS, TEMPLATES
from store_listing.orchestration.text_compositor import (
    HEADLINE_ZONE,
    TAGLINE_ZONE,
    TEXT_ONLY_HEADLINE_HEIGHT,
    TEXT_ONLY_HEADLINE_TOP,
    LayoutSpec,
    blank_canvas,
    composite_text,
)
from store_listing.orchestration.typography import TextTreatment


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
        # A brief that asks for a tagline gets one - it is not dropped on the
        # floor, and it no longer turns text_top into text_top_bottom either
        # (issue #238).
        spec = LayoutSpec(
            layout_type="text_top", headline_text="SPOOKY", tagline_text="TRICK OR TREAT"
        )

        assert spec.layout_type == "hero_bottom"
        assert spec.tagline_text == "TRICK OR TREAT"

        result = composite_text(_canvas(), spec)

        assert any(colour != (255, 255, 255) for colour in _zone_colours(result, top=True))

    def test_text_top_keeps_the_copy_at_the_top(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top", headline_text="SPOOKY", tagline_text="TRICK OR TREAT"
            ),
        )

        assert all(colour == (255, 255, 255) for colour in _zone_colours(result, top=False))


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


class TestTextOnly:
    def _text_only(self, **overrides: Any) -> LayoutSpec:
        defaults: dict[str, Any] = {
            "layout_type": "text_only",
            "headline_text": "SPOOKY SEASON",
            "tagline_text": None,
        }
        defaults.update(overrides)
        return LayoutSpec(**defaults)

    def test_the_layout_is_kept(self) -> None:
        assert self._text_only().layout_type == "text_only"

    def test_the_headline_is_drawn_in_the_middle_of_the_canvas(self) -> None:
        result = composite_text(_canvas(), self._text_only())

        image = _image(result)
        width, height = image.size
        drawn = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if image.getpixel((x, y)) != (255, 255, 255)
        ]
        assert drawn, "the headline should have been drawn"
        top = min(y for _, y in drawn)
        bottom = max(y for _, y in drawn)
        assert top >= int(height * TEXT_ONLY_HEADLINE_TOP)
        assert bottom <= int(height * (TEXT_ONLY_HEADLINE_TOP + TEXT_ONLY_HEADLINE_HEIGHT))

    def test_the_tagline_is_drawn_below_the_headline(self) -> None:
        result = composite_text(_canvas(), self._text_only(tagline_text="TRICK OR TREAT"))

        image = _image(result)
        width, height = image.size
        rows = [
            y
            for y in range(height)
            if any(image.getpixel((x, y)) != (255, 255, 255) for x in range(width))
        ]
        assert rows, "the text should have been drawn"
        headline_bottom = int(height * (TEXT_ONLY_HEADLINE_TOP + TEXT_ONLY_HEADLINE_HEIGHT))
        assert max(rows) > headline_bottom

    def test_the_font_colour_is_used(self) -> None:
        result = composite_text(_canvas(), self._text_only(font_color="#FF0000"))

        image = _image(result)
        width, height = image.size
        colours = {
            image.getpixel((x, y))
            for y in range(height)
            for x in range(width)
            if image.getpixel((x, y)) != (255, 255, 255)
        }
        assert (255, 0, 0) in colours

    def test_a_text_only_design_without_text_is_returned_untouched(self) -> None:
        canvas = _canvas()

        result = composite_text(canvas, self._text_only(headline_text=None, tagline_text=None))

        assert result == canvas


class TestBlankCanvas:
    def test_a_dtf_canvas_is_the_key_colour(self) -> None:
        # The magenta background is the one the DTF render is keyed out from.
        image = Image.open(io.BytesIO(blank_canvas("dtf_apparel"))).convert("RGB")

        assert image.getpixel((0, 0)) == (255, 0, 255)

    def test_a_print_through_canvas_is_white_paper(self) -> None:
        image = Image.open(io.BytesIO(blank_canvas("sublimation"))).convert("RGB")

        assert image.getpixel((0, 0)) == (255, 255, 255)

    def test_a_transparent_canvas_is_transparent(self) -> None:
        image = Image.open(io.BytesIO(blank_canvas("dtf_apparel", transparent=True))).convert(
            "RGBA"
        )

        assert image.getpixel((0, 0)) == (0, 0, 0, 0)

    def test_the_canvas_has_the_size_of_a_render(self) -> None:
        image = Image.open(io.BytesIO(blank_canvas("dtf_apparel")))

        assert image.size == (1024, 1024)


def _drawn_box(data: bytes, top: bool) -> tuple[int, int, int, int] | None:
    """The box the text was drawn in."""
    image = _image(data)
    width, height = image.size
    if top:
        box = (0, 0, width, int(height * HEADLINE_ZONE))
    else:
        box = (0, int(height * (1 - TAGLINE_ZONE)), width, height)
    drawn = [
        (x, y)
        for y in range(box[1], box[3])
        for x in range(box[0], box[2])
        if image.getpixel((x, y)) != (255, 255, 255)
    ]
    if not drawn:
        return None
    return (
        min(x for x, _ in drawn),
        min(y for _, y in drawn),
        max(x for x, _ in drawn),
        max(y for _, y in drawn),
    )


class TestTheType:
    def test_two_families_draw_different_text(self) -> None:
        # One font for every design is why the shop looks like one shop.
        one = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top", headline_text="GOBBLE TIL THE BELL", font_family="bebas"
            ),
        )
        other = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top", headline_text="GOBBLE TIL THE BELL", font_family="anton"
            ),
        )

        assert one != other

    def test_the_text_stays_inside_the_zone_with_every_family(self) -> None:
        headline = "Gobble Til The Drawbridge Drops"
        for family in ("bebas", "anton", "alfa_slab", "pacifico", "bungee", "rye", "abril"):
            result = composite_text(
                _canvas(),
                LayoutSpec(layout_type="text_top", headline_text=headline, font_family=family),
            )

            box = _drawn_box(result, top=True)
            assert box is not None, f"{family} drew nothing"
            width, height = _image(result).size
            assert box[2] < width, f"{family} ran out of the canvas"
            assert box[3] < int(height * HEADLINE_ZONE), f"{family} ran out of its zone"

    def test_a_character_the_family_cannot_render_is_dropped(self) -> None:
        # The script family has no emoji either.
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="🎃 GOBBLE",
                font_family="pacifico",
            ),
        )

        assert _drawn_box(result, top=True) is not None

    def test_the_voice_uppercases_the_headline(self) -> None:
        lower = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="gobble til the bell",
                font_family="alfa_slab",
                voice="deadpan",
            ),
        )
        upper = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="GOBBLE TIL THE BELL",
                font_family="alfa_slab",
                voice="deadpan",
            ),
        )

        assert lower == upper

    def test_the_cozy_voice_keeps_the_case(self) -> None:
        lower = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="gobble til the bell",
                font_family="alfa_slab",
                voice="cozy",
            ),
        )
        upper = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="GOBBLE TIL THE BELL",
                font_family="alfa_slab",
                voice="cozy",
            ),
        )

        assert lower != upper

    def test_the_tracking_widens_the_text(self) -> None:
        narrow = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="GOBBLE TIL THE BELL",
                treatment_override=TextTreatment(uppercase=True, tracking=0.0),
            ),
        )
        wide = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="GOBBLE TIL THE BELL",
                treatment_override=TextTreatment(uppercase=True, tracking=0.12),
            ),
        )

        narrow_box = _drawn_box(narrow, top=True)
        wide_box = _drawn_box(wide, top=True)
        assert narrow_box is not None
        assert wide_box is not None
        assert (wide_box[2] - wide_box[0]) > (narrow_box[2] - narrow_box[0])

    def test_the_accent_word_is_drawn_in_the_accent_colour(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="GOBBLE TIL THE BELL",
                font_color="#101010",
                treatment_override=TextTreatment(
                    uppercase=True,
                    tracking=0.04,
                    style="shadow",
                    accent_word=0,
                    accent_color="#F2B233",
                ),
            ),
        )

        assert (242, 178, 51) in _zone_colours(result, top=True)

    def test_the_block_style_draws_a_bar_behind_the_text(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="text_top",
                headline_text="GOBBLE TIL THE BELL",
                font_color="#101010",
                treatment_override=TextTreatment(
                    uppercase=True,
                    tracking=0.04,
                    style="block",
                    accent_color="#F2B233",
                ),
            ),
        )

        assert (242, 178, 51) in _zone_colours(result, top=True)

    def test_the_text_is_readable_over_the_artwork(self) -> None:
        # The OCR gate has to read the headline back, so the text needs to be
        # lighter than the background it is drawn on.
        result = composite_text(
            _canvas(color=(255, 255, 255)),
            LayoutSpec(
                layout_type="text_top",
                headline_text="GOBBLE TIL THE BELL",
                font_color="#101010",
            ),
        )

        colours = _zone_colours(result, top=True)
        assert (16, 16, 16) in colours


class TestEveryFamilyCanDraw:
    def test_the_alphabet_is_drawn_in_every_family(self) -> None:
        # The missing glyph of Anton has the same box as its capitals, so
        # every headline came out empty until the glyphs were compared as
        # bitmaps (issue #226).
        for family in ("bebas", "anton", "alfa_slab", "pacifico", "bungee", "rye", "abril"):
            result = composite_text(
                _canvas(),
                LayoutSpec(
                    layout_type="text_top",
                    headline_text="GOBBLE TIL THE BELL",
                    font_family=family,
                ),
            )

            box = _drawn_box(result, top=True)
            assert box is not None, f"{family} drew nothing"
            width = _image(result).size[0]
            assert (box[2] - box[0]) > width * 0.3, f"{family} dropped most of the headline"


def _drawn_box_anywhere(data: bytes) -> tuple[int, int, int, int] | None:
    """The box the text was drawn in, anywhere on the canvas."""
    image = _image(data)
    width, height = image.size
    drawn = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if image.getpixel((x, y)) != (255, 255, 255)
    ]
    if not drawn:
        return None
    return (
        min(x for x, _ in drawn),
        min(y for _, y in drawn),
        max(x for x, _ in drawn),
        max(y for _, y in drawn),
    )


class TestTheNewLayouts:
    def _drawn(self, layout: str, top: bool = True) -> tuple[int, int, int, int] | None:
        spec = LayoutSpec(
            layout_type=layout,
            headline_text="GOBBLE TIL THE BELL",
            tagline_text="FEAST AT THE OLD CASTLE",
        )
        result = composite_text(_canvas(), spec)
        return _drawn_box(result, top=top)

    def test_a_badge_draws_the_headline_at_the_top(self) -> None:
        box = self._drawn("badge", top=True)

        assert box is not None
        height = _image(_canvas()).size[1]
        assert box[3] < int(height * 0.2)

    def test_a_badge_draws_the_tagline_at_the_bottom(self) -> None:
        box = self._drawn("badge", top=False)

        assert box is not None
        height = _image(_canvas()).size[1]
        assert box[1] > int(height * 0.8)

    def test_a_badge_draws_a_band_behind_the_text(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(layout_type="banner", headline_text="GOBBLE TIL THE BELL"),
        )

        # A row just inside the top of the ribbon carries the ribbon colour and
        # nothing else: the ribbon is a solid fill.
        image = _image(result)
        width, height = image.size
        row = int(height * 0.41)
        colours = {image.getpixel((x, row)) for x in range(int(width * 0.1), int(width * 0.9))}
        assert len(colours) == 1, "the ribbon should be a solid fill"

    def test_a_banner_draws_the_headline_in_the_middle(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(layout_type="banner", headline_text="GOBBLE TIL THE BELL"),
        )

        box = _drawn_box_anywhere(result)
        assert box is not None
        height = _image(_canvas()).size[1]
        assert box[1] > int(height * 0.3)
        assert box[3] < int(height * 0.7)

    def test_an_arch_keeps_the_headline_narrow(self) -> None:
        box = self._drawn("arch", top=True)

        assert box is not None
        width = _image(_canvas()).size[0]
        assert (box[2] - box[0]) < width * 0.8

    def test_a_stack_side_draws_the_text_on_the_right(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="stack_side",
                headline_text="GOBBLE TIL THE BELL",
                tagline_text="FEAST AT THE OLD CASTLE",
            ),
        )

        box = _drawn_box_anywhere(result)
        assert box is not None
        width = _image(_canvas()).size[0]
        assert box[0] > int(width * 0.5)

    def test_a_layout_with_a_tagline_keeps_its_shape(self) -> None:
        # The shape of a layout is the layout: a badge with a tagline is still
        # a roundel and not a text_top_bottom.
        spec = LayoutSpec(
            layout_type="badge",
            headline_text="GOBBLE TIL THE BELL",
            tagline_text="FEAST AT THE OLD CASTLE",
        )

        assert spec.layout_type == "roundel"

    def test_the_text_stays_inside_its_band_in_every_layout(self) -> None:
        for layout in ("badge", "banner", "arch", "stack_side"):
            result = composite_text(
                _canvas(),
                LayoutSpec(
                    layout_type=layout,
                    headline_text="Gobble Til The Drawbridge Drops",
                    tagline_text="Feast at the old castle",
                ),
            )

            image = _image(result)
            width, height = image.size
            drawn = [
                (x, y)
                for y in range(height)
                for x in range(width)
                if image.getpixel((x, y)) != (255, 255, 255)
            ]
            assert drawn, f"{layout} drew nothing"
            assert max(x for x, _ in drawn) < width, f"{layout} ran out of the canvas"


class TestTheDesignTemplates:
    def _draw(self, template_id: str, size: tuple[int, int] = (1024, 1024)) -> bytes:
        return composite_text(
            _canvas(size),
            LayoutSpec(
                layout_type=template_id,
                headline_text="GOBBLE TIL THE BELL",
                tagline_text="FEAST AT THE OLD CASTLE",
            ),
        )

    def test_every_template_draws_something(self) -> None:
        for template_id in ART_TEMPLATE_IDS:
            width, height = TEMPLATES[template_id].size
            result = self._draw(template_id, (width, height))

            assert _drawn_box_anywhere(result) is not None, template_id

    def test_the_templates_do_not_all_draw_in_the_same_place(self) -> None:
        boxes = set()
        for template_id in ART_TEMPLATE_IDS:
            width, height = TEMPLATES[template_id].size
            box = _drawn_box_anywhere(self._draw(template_id, (width, height)))
            assert box is not None, template_id
            boxes.add((template_id, box))

        assert len({box for _template, box in boxes}) >= 8

    def test_the_text_stays_inside_the_canvas_in_every_template(self) -> None:
        for template_id in ART_TEMPLATE_IDS:
            width, height = TEMPLATES[template_id].size
            box = _drawn_box_anywhere(self._draw(template_id, (width, height)))

            assert box is not None, template_id
            assert 0 <= box[0] and box[2] < width, template_id
            assert 0 <= box[1] and box[3] < height, template_id

    def test_a_tower_draws_the_headline_down_the_left_side(self) -> None:
        result = self._draw("tower", (832, 1216))

        image = _image(result)
        width, height = image.size
        column = [
            (x, y)
            for y in range(height)
            for x in range(int(width * 0.4))
            if image.getpixel((x, y)) != (255, 255, 255)
        ]
        assert column, "the headline should be drawn down the left side"
        assert min(x for x, _ in column) < int(width * 0.1)
        # The copy runs down the canvas, not across a single band.
        assert max(y for _, y in column) - min(y for _, y in column) > height * 0.2

    def test_a_side_column_draws_the_copy_on_the_right(self) -> None:
        result = self._draw("side_column")

        box = _drawn_box_anywhere(result)
        assert box is not None
        width, _height = _image(result).size
        assert box[0] > int(width * 0.5)

    def test_a_wide_scene_draws_the_copy_in_the_bottom_left(self) -> None:
        result = self._draw("wide_scene", (1216, 832))

        box = _drawn_box_anywhere(result)
        assert box is not None
        width, height = _image(result).size
        assert box[0] < int(width * 0.2)
        assert box[1] > int(height * 0.6)

    def test_a_roundel_arcs_the_tagline_below_the_headline(self) -> None:
        result = self._draw("roundel")

        image = _image(result)
        width, height = image.size
        bottom = [
            (x, y)
            for y in range(int(height * 0.8), height)
            for x in range(width)
            if image.getpixel((x, y)) != (255, 255, 255)
        ]
        assert bottom, "the tagline should be drawn below the headline"
        # A curve does not reach the corners of its band the way a straight
        # line does.
        assert min(x for x, _ in bottom) > int(width * 0.05)

    def test_the_image_keeps_the_size_of_the_template(self) -> None:
        result = self._draw("wide_scene", (1216, 832))

        assert _image(result).size == (1216, 832)
