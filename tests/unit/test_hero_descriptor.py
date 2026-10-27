"""Tests for the descriptor compositor beneath a Qwen hero wordmark (issue #256).

Qwen draws the kicker and the hero word, but garbles a third block. The
descriptor is therefore drawn here, after the strict OCR gate, on a canvas that
has room for it.
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw

from store_listing.orchestration.hero_descriptor import (
    HERO_CANVAS_SIZE,
    compose_hero_descriptor,
    content_bbox,
    descriptor_enabled,
    plan_descriptor_layout,
)

Pixel = tuple[int, int, int, int]


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _load(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGBA")


def _opaque_artwork(
    width: int = 160,
    height: int = 160,
    *,
    background: str = "white",
    square: tuple[int, int, int, int] = (40, 40, 120, 120),
) -> bytes:
    image = Image.new("RGBA", (width, height), background)
    ImageDraw.Draw(image).rectangle(square, fill="red")
    return _png(image)


def _transparent_artwork(
    width: int = 160,
    height: int = 160,
    *,
    square: tuple[int, int, int, int] = (40, 40, 120, 120),
) -> bytes:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(square, fill=(255, 0, 0, 255))
    return _png(image)


def _is_red(pixel: Pixel) -> bool:
    red, green, blue, alpha = pixel
    return alpha > 0 and red > 150 and green < 100 and blue < 100


def _is_dark(pixel: Pixel) -> bool:
    red, green, blue, alpha = pixel
    return alpha > 0 and max(red, green, blue) < 80


def _pixel_bbox(
    image: Image.Image, predicate: Callable[[Pixel], bool]
) -> tuple[int, int, int, int] | None:
    pixels = image.load()
    assert pixels is not None
    xs: list[int] = []
    ys: list[int] = []
    for y in range(image.height):
        for x in range(image.width):
            if predicate(cast(Pixel, pixels[x, y])):
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _row_runs(
    image: Image.Image,
    box: tuple[int, int, int, int],
    predicate: Callable[[Pixel], bool],
) -> int:
    """Count separate horizontal ink bands in an (x, y, width, height) box."""
    x, y, width, height = box
    rows = []
    pixels = image.load()
    assert pixels is not None
    for row in range(y, y + height):
        if any(predicate(cast(Pixel, pixels[column, row])) for column in range(x, x + width)):
            rows.append(row)
    runs = 0
    previous: int | None = None
    for row in rows:
        if previous is None or row != previous + 1:
            runs += 1
        previous = row
    return runs


class TestContentBbox:
    def test_a_plain_white_image_has_no_content(self) -> None:
        image = Image.new("RGBA", (100, 100), "white")
        assert content_bbox(image) is None

    def test_finds_the_opaque_content(self) -> None:
        image = _load(_opaque_artwork())
        assert content_bbox(image) == (40, 40, 121, 121)

    def test_finds_the_transparent_content(self) -> None:
        image = _load(_transparent_artwork())
        assert content_bbox(image) == (40, 40, 121, 121)

    def test_uses_the_background_colour_not_only_white(self) -> None:
        image = _load(_opaque_artwork(background="#101010"))
        assert content_bbox(image) == (40, 40, 121, 121)

    def test_content_at_the_edge_is_not_dropped(self) -> None:
        image = Image.new("RGBA", (100, 100), "white")
        ImageDraw.Draw(image).rectangle((0, 0, 20, 20), fill="red")
        assert content_bbox(image) == (0, 0, 21, 21)


class TestPlanDescriptorLayout:
    def test_a_square_artwork_keeps_its_aspect(self) -> None:
        layout = plan_descriptor_layout((200, 200), canvas_size=(400, 400))
        x, y, width, height = layout.artwork_box
        assert width == height
        assert x >= 0
        assert y >= 0
        assert x + width <= 400
        assert y + height <= 400

    def test_the_descriptor_sits_below_the_artwork(self) -> None:
        layout = plan_descriptor_layout((200, 200), canvas_size=(400, 400))
        _art_x, art_y, _art_w, art_h = layout.artwork_box
        desc_x, desc_y, desc_w, desc_h = layout.descriptor_box
        assert desc_y >= art_y + art_h
        assert desc_x >= 0
        assert desc_x + desc_w <= 400
        assert desc_y + desc_h <= 400

    def test_a_wide_artwork_keeps_its_aspect(self) -> None:
        layout = plan_descriptor_layout((400, 100), canvas_size=(400, 400))
        _x, _y, width, height = layout.artwork_box
        assert width / height == pytest.approx(4.0, abs=0.02)

    def test_the_artwork_never_exceeds_the_content_area(self) -> None:
        layout = plan_descriptor_layout((2000, 1000), canvas_size=(400, 400))
        _x, _y, width, height = layout.artwork_box
        assert width <= 400
        assert height <= 400 - layout.descriptor_box[3]

    def test_the_descriptor_box_has_a_usable_height(self) -> None:
        layout = plan_descriptor_layout((200, 200), canvas_size=(400, 400))
        assert layout.descriptor_box[3] >= 40

    def test_a_canvas_without_a_dimension_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            plan_descriptor_layout((200, 200), canvas_size=(0, 400))
        with pytest.raises(ValueError):
            plan_descriptor_layout((200, 200), canvas_size=(400, 0))

    def test_the_default_canvas_matches_qwen(self) -> None:
        assert HERO_CANVAS_SIZE == (1328, 1328)


class TestComposeHeroDescriptor:
    def _compose(self, descriptor: str | None = "GROOVY GHOST VIBES", **kwargs: Any) -> Image.Image:
        output = compose_hero_descriptor(
            _opaque_artwork(),
            descriptor,
            canvas_size=(320, 320),
            **kwargs,
        )
        return _load(output)

    def test_without_a_descriptor_the_artwork_is_untouched(self) -> None:
        artwork = _opaque_artwork()
        assert compose_hero_descriptor(artwork, None) == artwork
        assert compose_hero_descriptor(artwork, "   ") == artwork
        assert compose_hero_descriptor(artwork, "") == artwork

    def test_the_output_uses_the_requested_canvas(self) -> None:
        image = self._compose()
        assert image.size == (320, 320)

    def test_the_artwork_is_above_the_descriptor_and_not_overlapped(self) -> None:
        image = self._compose()
        layout = plan_descriptor_layout((160, 160), canvas_size=(320, 320))
        artwork = _pixel_bbox(image, _is_red)
        ink = _pixel_bbox(image, _is_dark)
        assert artwork is not None
        assert ink is not None
        assert artwork[3] <= layout.descriptor_box[1]
        assert ink[1] >= layout.descriptor_box[1]
        assert ink[3] <= layout.descriptor_box[1] + layout.descriptor_box[3]

    def test_the_descriptor_is_centred(self) -> None:
        image = self._compose()
        ink = _pixel_bbox(image, _is_dark)
        assert ink is not None
        ink_centre = (ink[0] + ink[2]) / 2
        assert ink_centre == pytest.approx(image.width / 2, abs=5)

    def test_the_descriptor_sits_inside_its_box(self) -> None:
        image = self._compose("A VERY LONG DESCRIPTOR THAT NEEDS TO WRAP ACROSS TWO LINES")
        layout = plan_descriptor_layout((160, 160), canvas_size=(320, 320))
        desc_x, desc_y, desc_w, desc_h = layout.descriptor_box
        ink = _pixel_bbox(image, _is_dark)
        assert ink is not None
        assert ink[0] >= desc_x
        assert ink[2] <= desc_x + desc_w
        assert ink[1] >= desc_y
        assert ink[3] <= desc_y + desc_h

    def test_a_long_descriptor_wraps_instead_of_overflowing(self) -> None:
        image = self._compose("A VERY LONG DESCRIPTOR THAT NEEDS TO WRAP ACROSS TWO LINES")
        layout = plan_descriptor_layout((160, 160), canvas_size=(320, 320))
        assert _row_runs(image, layout.descriptor_box, _is_dark) >= 2

    def test_a_wide_artwork_keeps_the_subject_square(self) -> None:
        artwork = _opaque_artwork(width=200, height=100, square=(80, 25, 120, 65))
        image = _load(
            compose_hero_descriptor(artwork, "GROOVY GHOST VIBES", canvas_size=(320, 320))
        )
        red = _pixel_bbox(image, _is_red)
        assert red is not None
        assert (red[2] - red[0]) == pytest.approx(red[3] - red[1], abs=2)

    def test_the_raw_background_colour_is_kept(self) -> None:
        artwork = _opaque_artwork(background="#ffffcc")
        image = _load(
            compose_hero_descriptor(artwork, "GROOVY GHOST VIBES", canvas_size=(320, 320))
        )
        red, green, blue, alpha = cast(Pixel, image.getpixel((0, 0)))
        assert (red, green, blue, alpha) == (255, 255, 204, 255)

    def test_the_transparent_output_has_alpha_and_ink(self) -> None:
        image = _load(
            compose_hero_descriptor(
                _transparent_artwork(),
                "GROOVY GHOST VIBES",
                canvas_size=(320, 320),
                transparent=True,
            )
        )
        assert cast(Pixel, image.getpixel((0, 0)))[3] == 0
        assert _pixel_bbox(image, _is_red) is not None
        ink = _pixel_bbox(image, _is_dark)
        assert ink is not None
        assert all(cast(Pixel, image.getpixel((x, y)))[3] > 0 for x, y in [(ink[0], ink[1])])

    def test_the_transparent_descriptor_gets_an_outline(self) -> None:
        image = _load(
            compose_hero_descriptor(
                _transparent_artwork(),
                "GROOVY GHOST VIBES",
                canvas_size=(320, 320),
                transparent=True,
            )
        )
        # The outline is white, so there must be light opaque pixels next to the
        # dark glyphs inside the descriptor band.
        layout = plan_descriptor_layout((160, 160), canvas_size=(320, 320))
        box = layout.descriptor_box

        def is_light(pixel: Pixel) -> bool:
            red, green, blue, alpha = pixel
            return alpha > 0 and min(red, green, blue) > 200

        assert _pixel_bbox(image, is_light) is not None
        assert _row_runs(image, box, is_light) >= 1


class TestDescriptorFlag:
    def test_it_is_on_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPOGRAPHIC_HERO_DESCRIPTOR", raising=False)
        assert descriptor_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "off", "no"])
    def test_the_documented_false_values_disable_it(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_DESCRIPTOR", value)
        assert descriptor_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable_it(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_DESCRIPTOR", value)
        assert descriptor_enabled() is True
