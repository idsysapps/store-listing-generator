"""Tests for the chroma-key background removal."""

from __future__ import annotations

import io

from PIL import Image

from store_listing.orchestration.chroma_key import (
    background_fraction,
    despill_key_spill,
    estimate_key_colour,
    key_out_background,
    punch_out_key_colour,
)

KEY = (247, 0, 102)


def _png(pixels: dict[tuple[int, int], tuple[int, int, int]], size: int = 8) -> bytes:
    image = Image.new("RGB", (size, size), KEY)
    for (x, y), colour in pixels.items():
        image.putpixel((x, y), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _alpha_at(image: Image.Image, x: int, y: int) -> int:
    channel = image.getchannel("A") if image.mode == "RGBA" else image
    return channel.crop((x, y, x + 1, y + 1)).tobytes()[0]


def _rgb_at(image: Image.Image, x: int, y: int) -> bytes:
    return bytes(image.convert("RGB").crop((x, y, x + 1, y + 1)).tobytes()[:3])


class TestEstimateKeyColour:
    def test_takes_the_median_of_the_border(self) -> None:
        image = Image.new("RGB", (10, 10), KEY)
        assert estimate_key_colour(image) == KEY

    def test_ignores_a_piece_of_artwork_on_the_border(self) -> None:
        image = Image.new("RGB", (10, 10), KEY)
        for x in range(4):
            image.putpixel((x, 0), (0, 0, 0))
        assert estimate_key_colour(image) == KEY


class TestKeyOutBackground:
    def test_the_background_becomes_transparent(self) -> None:
        result = key_out_background(_png({}))
        with Image.open(io.BytesIO(result)) as image:
            assert _alpha_at(image, 4, 4) == 0

    def test_the_artwork_stays_opaque(self) -> None:
        result = key_out_background(_png({(3, 3): (250, 120, 10)}))
        with Image.open(io.BytesIO(result)) as image:
            assert _alpha_at(image, 3, 3) == 255

    def test_white_artwork_is_preserved(self) -> None:
        result = key_out_background(_png({(3, 3): (255, 255, 255)}))
        with Image.open(io.BytesIO(result)) as image:
            assert _alpha_at(image, 3, 3) == 255

    def test_a_darker_shade_of_the_key_is_kept_as_artwork(self) -> None:
        # Flux tints artwork towards the background colour; a dark shade of the
        # key is artwork, not background, and must survive.
        result = key_out_background(_png({(3, 3): (120, 0, 60)}))
        with Image.open(io.BytesIO(result)) as image:
            assert _alpha_at(image, 3, 3) == 255

    def test_the_result_is_rgba(self) -> None:
        result = key_out_background(_png({}))
        with Image.open(io.BytesIO(result)) as image:
            assert image.mode == "RGBA"

    def test_an_explicit_key_colour_is_used(self) -> None:
        result = key_out_background(_png({}), key_colour=KEY)
        with Image.open(io.BytesIO(result)) as image:
            assert _alpha_at(image, 4, 4) == 0

    def test_despill_removes_the_key_colour_from_the_edges(self) -> None:
        # A half-transparent edge pixel is a blend of the artwork and the key.
        blend = (
            (250 + KEY[0]) // 2,
            (120 + KEY[1]) // 2,
            (10 + KEY[2]) // 2,
        )
        result = key_out_background(_png({(3, 3): blend}), despill=True)
        with Image.open(io.BytesIO(result)) as image:
            alpha = _alpha_at(image, 3, 3)
            assert 0 < alpha < 255
            red, green, blue = _rgb_at(image, 3, 3)
            # the key's blue channel must not survive in the edge pixel
            assert red < 250
            assert green < 250
            assert blue < 40

    def test_without_despill_the_edge_keeps_the_key_colour(self) -> None:
        blend = (
            (250 + KEY[0]) // 2,
            (120 + KEY[1]) // 2,
            (10 + KEY[2]) // 2,
        )
        result = key_out_background(_png({(3, 3): blend}), despill=False)
        with Image.open(io.BytesIO(result)) as image:
            _red, _green, blue = _rgb_at(image, 3, 3)
            assert blue > 40


class TestPunchOutHoles:
    """A letter's counter is a hole in the artwork, and the punch-out finds it."""

    def _letter(self) -> tuple[bytes, bytes]:
        """A white ring on a magenta background, and a mask that fills the hole."""
        image = Image.new("RGB", (32, 32), KEY)
        for y in range(4, 28):
            for x in range(4, 28):
                image.putpixel((x, y), (255, 255, 255))
        for y in range(10, 22):
            for x in range(10, 22):
                image.putpixel((x, y), KEY)  # the counter
        source = io.BytesIO()
        image.save(source, format="PNG")

        # the model's mask keeps the counter as part of the subject
        mask = image.convert("RGBA")
        pixels = mask.load()
        assert pixels is not None
        for y in range(4, 28):
            for x in range(4, 28):
                pixels[x, y] = (255, 255, 255, 255)
        masked = io.BytesIO()
        mask.save(masked, format="PNG")
        return source.getvalue(), masked.getvalue()

    def test_clears_a_counter_the_model_filled_in(self) -> None:
        source, masked = self._letter()

        result = punch_out_key_colour(masked, source)

        with Image.open(io.BytesIO(result)) as image:
            alpha = image.getchannel("A")
            assert _alpha_at(alpha, 16, 16) == 0, "the counter should be transparent"
            assert _alpha_at(alpha, 5, 16) == 255, "the letter should stay opaque"

    def test_keeps_the_outer_background_transparent(self) -> None:
        source, masked = self._letter()

        result = punch_out_key_colour(masked, source)

        with Image.open(io.BytesIO(result)) as image:
            alpha = image.getchannel("A")
            assert _alpha_at(alpha, 1, 1) == 0

    def test_does_not_eat_a_darker_shade_of_the_key_in_the_artwork(self) -> None:
        image = Image.new("RGB", (32, 32), KEY)
        for y in range(4, 28):
            for x in range(4, 28):
                image.putpixel((x, y), (120, 0, 60))  # a dark shade of the key
        source = io.BytesIO()
        image.save(source, format="PNG")
        masked = io.BytesIO()
        image.convert("RGBA").save(masked, format="PNG")

        result = punch_out_key_colour(masked.getvalue(), source.getvalue())

        with Image.open(io.BytesIO(result)) as opened:
            alpha = opened.getchannel("A")
            assert _alpha_at(alpha, 16, 16) == 255


class TestDespillKeySpill:
    def _source(self) -> bytes:
        image = Image.new("RGB", (8, 8), KEY)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _rgba(self, alpha: int, colour: tuple[int, int, int]) -> bytes:
        image = Image.new("RGBA", (8, 8), (*colour, alpha))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_removes_the_key_share_from_a_partial_pixel(self) -> None:
        source = self._source()
        rgba = self._rgba(128, (200, 120, 120))

        result = despill_key_spill(rgba, source)

        with Image.open(io.BytesIO(result)) as image:
            red, green, blue = _rgb_at(image, 2, 2)
        # Half of the key's contribution is removed from each channel it has:
        # the key is (247, 0, 102), so red and blue drop and green does not.
        assert abs(red - (200 - 247 // 2)) <= 2
        assert abs(green - 120) <= 1
        assert abs(blue - (120 - 102 // 2)) <= 2

    def test_leaves_the_opaque_pixels_alone(self) -> None:
        source = self._source()
        rgba = self._rgba(255, (200, 120, 120))

        result = despill_key_spill(rgba, source)

        with Image.open(io.BytesIO(result)) as image:
            assert _rgb_at(image, 2, 2) == bytes((200, 120, 120))

    def test_leaves_the_alpha_channel_alone(self) -> None:
        source = self._source()
        rgba = self._rgba(128, (200, 120, 120))

        result = despill_key_spill(rgba, source)

        with Image.open(io.BytesIO(result)) as image:
            assert _alpha_at(image, 2, 2) == 128

    def test_never_produces_a_negative_colour(self) -> None:
        source = self._source()
        rgba = self._rgba(16, (10, 10, 10))

        result = despill_key_spill(rgba, source)

        with Image.open(io.BytesIO(result)) as image:
            red, green, blue = _rgb_at(image, 2, 2)
        assert red >= 0 and green >= 0 and blue >= 0


class TestBackgroundFraction:
    def test_reports_the_keyed_area(self) -> None:
        assert background_fraction(_png({})) == 1.0

    def test_reports_less_when_the_artwork_covers_part_of_the_frame(self) -> None:
        pixels = {(x, y): (250, 120, 10) for x in range(6, 10) for y in range(6, 10)}
        assert background_fraction(_png(pixels, size=16)) == 1 - 16 / 256
