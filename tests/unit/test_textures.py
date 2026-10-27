"""Tests for the texture of a design.

The grain, the halftone and the cracked ink are what make a flat vector read
as a print. They cannot be asked of the image model, because a texture it
paints lands on the background too, and the background of a DTF render is
keyed out (issue #227).
"""

from __future__ import annotations

import io
from typing import cast

from PIL import Image, ImageDraw

from store_listing.orchestration.textures import apply_texture

KEY = (255, 0, 255)


def _canvas(
    size: tuple[int, int] = (128, 128), colour: tuple[int, int, int] = (255, 255, 255)
) -> bytes:
    image = Image.new("RGB", size, colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _pixel(image: Image.Image, x: int, y: int) -> tuple[int, int, int, int]:
    """The colour of a pixel as four numbers."""
    return cast(tuple[int, int, int, int], image.getpixel((x, y)))


def _rgb(image: Image.Image, x: int, y: int) -> tuple[int, int, int]:
    """The colour of a pixel as a tuple of three numbers."""
    pixel = _pixel(image, x, y)
    return (pixel[0], pixel[1], pixel[2])


class TestTheTexture:
    def test_an_artwork_without_a_texture_is_untouched(self) -> None:
        canvas = _canvas()

        assert apply_texture(canvas, None) == canvas

    def test_the_grain_changes_the_artwork(self) -> None:
        canvas = _canvas()

        result = apply_texture(canvas, "grain")

        assert result != canvas

    def test_the_grain_leaves_the_background_alone(self) -> None:
        # A texture on the background of a DTF render would survive the
        # cutout as a grey rectangle.
        canvas = _canvas(colour=KEY)

        result = apply_texture(canvas, "grain", key_colour=KEY)

        image = _image(result)
        assert _rgb(image, 2, 2) == KEY

    def test_the_texture_reaches_the_artwork(self) -> None:
        canvas = _canvas(colour=(0, 0, 0))

        result = apply_texture(canvas, "grain", key_colour=KEY)

        image = _image(result)
        changed = sum(
            1
            for y in range(image.size[1])
            for x in range(image.size[0])
            if _rgb(image, x, y) != (0, 0, 0)
        )
        assert changed > 0, "the grain should reach the artwork"

    def test_the_halftone_leaves_the_background_alone(self) -> None:
        canvas = _canvas(colour=KEY)

        result = apply_texture(canvas, "halftone", key_colour=KEY)

        image = _image(result)
        assert _rgb(image, 2, 2) == KEY

    def test_an_alpha_mask_keeps_a_non_key_background_clean(self) -> None:
        # The DTF background is often pink rather than the exact key magenta;
        # a distance mask would stamp the halftone onto it. The keyed render's
        # alpha is the exact foreground.
        background = (251, 114, 183)
        raw = Image.new("RGB", (64, 64), background)
        draw = ImageDraw.Draw(raw)
        draw.rectangle([20, 20, 44, 44], fill=(255, 255, 255))
        raw_buffer = io.BytesIO()
        raw.save(raw_buffer, format="PNG")

        transparent = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        transparent_draw = ImageDraw.Draw(transparent)
        transparent_draw.rectangle([20, 20, 44, 44], fill=(255, 255, 255, 255))
        transparent_buffer = io.BytesIO()
        transparent.save(transparent_buffer, format="PNG")

        result = apply_texture(
            raw_buffer.getvalue(),
            "halftone",
            key_colour=KEY,
            alpha_mask_bytes=transparent_buffer.getvalue(),
        )

        image = _image(result)
        assert _rgb(image, 2, 2) == background
        changed = sum(
            1 for y in range(20, 45) for x in range(20, 45) if _rgb(image, x, y) != (255, 255, 255)
        )
        assert changed > 0, "the texture should still reach the artwork"

    def test_the_crackle_leaves_the_background_alone(self) -> None:
        canvas = _canvas(colour=KEY)

        result = apply_texture(canvas, "crackle", key_colour=KEY)

        image = _image(result)
        assert _rgb(image, 2, 2) == KEY

    def test_the_transparent_artwork_keeps_its_alpha(self) -> None:
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = apply_texture(buffer.getvalue(), "grain")

        assert _pixel(_image(result), 32, 32)[3] == 0

    def test_the_same_artwork_always_gets_the_same_texture(self) -> None:
        canvas = _canvas()

        assert apply_texture(canvas, "grain") == apply_texture(canvas, "grain")

    def test_an_unknown_texture_leaves_the_artwork_alone(self) -> None:
        canvas = _canvas()

        assert apply_texture(canvas, "sparkles") == canvas
