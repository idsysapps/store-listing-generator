"""Tests for the chroma-key background removal."""

from __future__ import annotations

import io

from PIL import Image

from store_listing.orchestration.chroma_key import (
    background_fraction,
    estimate_key_colour,
    key_out_background,
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
    return image.getchannel("A").crop((x, y, x + 1, y + 1)).tobytes()[0]


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


class TestBackgroundFraction:
    def test_reports_the_keyed_area(self) -> None:
        assert background_fraction(_png({})) == 1.0

    def test_reports_less_when_the_artwork_covers_part_of_the_frame(self) -> None:
        pixels = {(x, y): (250, 120, 10) for x in range(6, 10) for y in range(6, 10)}
        assert background_fraction(_png(pixels, size=16)) == 1 - 16 / 256
