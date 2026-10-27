"""Draw the descriptor beneath a Qwen hero wordmark.

The reference design (`32_transparent.png`) is three blocks: a kicker, a giant
hero word with the subject woven through it, and a descriptor underneath. Qwen
draws the first two reliably, but a third block in the same prompt garbled 0/3
times in the Phase 0 tests. The descriptor is therefore drawn deterministically
here, after the strict OCR gate has accepted the artwork.

Qwen fills its frame, so there is no ready-made band for the descriptor. The
compositor finds the artwork's content bounding box, reflows it into the upper
part of a square canvas, and reserves the bottom band. The artwork is never
scaled non-uniformly and the descriptor never overlaps it.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Final, cast

from PIL import Image, ImageChops, ImageDraw, ImageFont

from store_listing.orchestration.typography import font_path

__all__ = [
    "HERO_CANVAS_SIZE",
    "DescriptorLayout",
    "compose_hero_descriptor",
    "content_bbox",
    "descriptor_enabled",
    "plan_descriptor_layout",
]

HERO_CANVAS_SIZE: Final[tuple[int, int]] = (1328, 1328)
HERO_DESCRIPTOR_FAMILY: Final[str] = "anton"
HERO_DESCRIPTOR_INK: Final[str] = "#000000"
ENABLED_ENV: Final[str] = "TYPOGRAPHIC_HERO_DESCRIPTOR"
_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "off", "no"})
_MIN_FONT_SIZE = 18
_MAX_FONT_FRACTION = 0.65
_TRACKING = 0.03
_LINE_SPACING = 1.12
_TOP_MARGIN = 0.035
_SIDE_MARGIN = 0.06
_GAP = 0.035
_DESCRIPTOR_BAND = 0.16
_BOTTOM_MARGIN = 0.045
_CONTENT_THRESHOLD = 18


@dataclass(frozen=True)
class DescriptorLayout:
    """Where the artwork and the descriptor go on the final canvas."""

    canvas_size: tuple[int, int]
    # (x, y, width, height) of the scaled artwork.
    artwork_box: tuple[int, int, int, int]
    # (x, y, width, height) of the band the descriptor is drawn in.
    descriptor_box: tuple[int, int, int, int]


def descriptor_enabled() -> bool:
    """Whether the hero path should composite the descriptor.

    It is on by default because the descriptor is the third block of the
    reference design. It can be switched off to A/B the Phase 1 image.
    """
    raw = os.environ.get(ENABLED_ENV, "1").strip().lower()
    return raw not in _FALSE_VALUES


def content_bbox(
    image: Image.Image, *, threshold: int = _CONTENT_THRESHOLD
) -> tuple[int, int, int, int] | None:
    """The box of the pixels that are not the background.

    A transparent image is measured by its alpha channel. An opaque one is
    measured against the most common colour on its edge, which is the plain
    background Qwen draws on. The edge survives a subject in one corner, where
    the top-left pixel alone would mistake the subject for the background. The
    threshold keeps a soft drop shadow in the box while ignoring JPEG-ish noise
    in the white.
    """
    if image.mode not in ("RGBA", "LA"):
        image = image.convert("RGBA")

    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        minimum, _maximum = cast(tuple[int, int], alpha.getextrema())
        if minimum < 255:
            return alpha.getbbox()

    rgb = image.convert("RGB")
    width, height = rgb.size
    background = _border_colour(rgb)
    difference = ImageChops.difference(rgb, Image.new("RGB", (width, height), background)).convert(
        "L"
    )
    lut = [255 if value > threshold else 0 for value in range(256)]
    mask = difference.point(lut)
    return mask.getbbox()


def _border_colour(image: Image.Image) -> tuple[int, int, int]:
    """The most common colour on the edge of the image.

    The top-left pixel is enough when the background is clear, but a subject
    that touches a corner would become the "background" and the real
    background would be measured as content. The edge as a whole survives a
    subject in one corner.
    """
    width, height = image.size
    pixels = image.load()
    assert pixels is not None
    counts: dict[tuple[int, int, int], int] = {}
    for x in range(width):
        for colour in (pixels[x, 0], pixels[x, height - 1]):
            key = cast(tuple[int, int, int], colour)
            counts[key] = counts.get(key, 0) + 1
    for y in range(height):
        for colour in (pixels[0, y], pixels[width - 1, y]):
            key = cast(tuple[int, int, int], colour)
            counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.__getitem__)


def plan_descriptor_layout(
    artwork_size: tuple[int, int],
    *,
    canvas_size: tuple[int, int] = HERO_CANVAS_SIZE,
) -> DescriptorLayout:
    """Fit the artwork above a reserved descriptor band, without distortion."""
    canvas_width, canvas_height = canvas_size
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("The descriptor canvas needs a positive width and height")

    artwork_width, artwork_height = artwork_size
    if artwork_width <= 0 or artwork_height <= 0:
        raise ValueError("The artwork needs a positive width and height")

    top = round(canvas_height * _TOP_MARGIN)
    side = round(canvas_width * _SIDE_MARGIN)
    gap = round(canvas_height * _GAP)
    descriptor_height = round(canvas_height * _DESCRIPTOR_BAND)
    bottom = round(canvas_height * _BOTTOM_MARGIN)

    max_width = canvas_width - 2 * side
    max_height = canvas_height - top - gap - descriptor_height - bottom
    if max_width <= 0 or max_height <= 0:
        raise ValueError("The canvas is too small for a descriptor band")

    scale = min(max_width / artwork_width, max_height / artwork_height)
    draw_width = max(1, round(artwork_width * scale))
    draw_height = max(1, round(artwork_height * scale))
    x = (canvas_width - draw_width) // 2
    y = top + (max_height - draw_height) // 2

    descriptor_box = (
        side,
        y + draw_height + gap,
        max_width,
        descriptor_height,
    )
    return DescriptorLayout(
        canvas_size=canvas_size,
        artwork_box=(x, y, draw_width, draw_height),
        descriptor_box=descriptor_box,
    )


def compose_hero_descriptor(
    image_bytes: bytes,
    descriptor: str | None,
    *,
    canvas_size: tuple[int, int] = HERO_CANVAS_SIZE,
    family: str = HERO_DESCRIPTOR_FAMILY,
    ink: str = HERO_DESCRIPTOR_INK,
    transparent: bool = False,
) -> bytes:
    """Return the hero artwork with the descriptor drawn beneath it.

    Without a descriptor the bytes are returned untouched, so the Phase 1
    behaviour is preserved. ``transparent`` draws the descriptor on the keyed
    DTF artwork and gives it a light outline so it survives on any garment.
    """
    text = " ".join((descriptor or "").split())
    if not text:
        return image_bytes

    try:
        artwork = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except OSError:
        return image_bytes

    box = content_bbox(artwork)
    if box is None:
        box = (0, 0, artwork.width, artwork.height)

    padding = max(2, round(min(artwork.width, artwork.height) * 0.01))
    left = max(0, box[0] - padding)
    top = max(0, box[1] - padding)
    right = min(artwork.width, box[2] + padding)
    bottom = min(artwork.height, box[3] + padding)
    cropped = artwork.crop((left, top, right, bottom))

    layout = plan_descriptor_layout(cropped.size, canvas_size=canvas_size)

    if transparent:
        background: tuple[int, int, int, int] = (0, 0, 0, 0)
    else:
        corner = cast(tuple[int, int, int, int], artwork.getpixel((0, 0)))
        background = (
            (corner[0], corner[1], corner[2], 255) if corner[3] > 0 else (255, 255, 255, 255)
        )

    canvas = Image.new("RGBA", canvas_size, background)
    artwork_x, artwork_y, artwork_width, artwork_height = layout.artwork_box
    resized = cropped.resize((artwork_width, artwork_height), Image.Resampling.LANCZOS)
    canvas.alpha_composite(resized, (artwork_x, artwork_y))

    _draw_descriptor(
        canvas,
        text,
        layout.descriptor_box,
        family=family,
        ink=ink,
        transparent=transparent,
    )

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_descriptor(
    canvas: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    *,
    family: str,
    ink: str,
    transparent: bool,
) -> None:
    box_x, box_y, box_width, box_height = box
    upper = text.upper()
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    font, tracking, lines = _fit_descriptor(upper, family, box_width, box_height, measure)
    line_height = _line_height(font)
    total_height = len(lines) * line_height
    y = box_y + max(0, (box_height - total_height) // 2)

    stroke = max(2, int(font.size) // 28) if transparent else 0
    draw = ImageDraw.Draw(canvas)
    for line in lines:
        line_width = _line_width(line, font, tracking, measure)
        x = box_x + max(0, (box_width - line_width) // 2)
        _draw_tracked(draw, line, x, y, font, tracking, ink, stroke)
        y += line_height


def _fit_descriptor(
    text: str,
    family: str,
    max_width: int,
    max_height: int,
    measure: ImageDraw.ImageDraw,
) -> tuple[ImageFont.FreeTypeFont, int, list[str]]:
    """Largest size that fits, preferring one line over two."""
    one_line_start = max(_MIN_FONT_SIZE, round(max_height * _MAX_FONT_FRACTION))
    for size in range(one_line_start, _MIN_FONT_SIZE - 1, -1):
        font = _font(family, size)
        tracking = round(size * _TRACKING)
        if (
            _line_width(text, font, tracking, measure) <= max_width
            and _line_height(font) <= max_height
        ):
            return font, tracking, [text]

    for size in range(max(max_height // 2, _MIN_FONT_SIZE), _MIN_FONT_SIZE - 1, -1):
        font = _font(family, size)
        tracking = round(size * _TRACKING)
        lines = _wrap(text, font, max_width, tracking, measure, max_lines=2)
        if lines is None:
            continue
        line_height = _line_height(font)
        if len(lines) * line_height <= max_height:
            return font, tracking, lines

    # A small canvas can have a band too short for two lines at the preferred
    # floor. Shrink further rather than dropping words: a small descriptor is
    # better than a descriptor that says something else.
    for size in range(_MIN_FONT_SIZE - 1, 7, -1):
        font = _font(family, size)
        tracking = round(size * _TRACKING)
        lines = _wrap(text, font, max_width, tracking, measure, max_lines=3)
        if lines is None:
            continue
        line_height = _line_height(font)
        if len(lines) * line_height <= max_height:
            return font, tracking, lines

    font = _font(family, 8)
    tracking = round(8 * _TRACKING)
    lines = _wrap(text, font, max_width, tracking, measure)
    return font, tracking, lines if lines is not None else text.split()


def _font(family: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(family)), size)


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return max(1, round((ascent + descent) * _LINE_SPACING))


def _line_width(
    line: str,
    font: ImageFont.FreeTypeFont,
    tracking: int,
    measure: ImageDraw.ImageDraw,
) -> int:
    if not line:
        return 0
    glyphs = sum(measure.textlength(character, font=font) for character in line)
    return round(glyphs + tracking * max(0, len(line) - 1))


def _wrap(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    tracking: int,
    measure: ImageDraw.ImageDraw,
    *,
    max_lines: int | None = None,
) -> list[str] | None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or _line_width(candidate, font, tracking, measure) <= max_width:
            current = candidate
            continue
        if max_lines is not None and len(lines) + 1 >= max_lines:
            return None
        lines.append(current)
        current = word
    if current:
        if max_lines is not None and len(lines) + 1 > max_lines:
            return None
        lines.append(current)
    return lines


def _draw_tracked(
    draw: ImageDraw.ImageDraw,
    line: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    tracking: int,
    ink: str,
    stroke: int,
) -> None:
    cursor = x
    for character in line:
        if stroke:
            draw.text(
                (cursor, y),
                character,
                font=font,
                fill=ink,
                stroke_width=stroke,
                stroke_fill="white",
                anchor="lt",
            )
        else:
            draw.text((cursor, y), character, font=font, fill=ink, anchor="lt")
        cursor += draw.textlength(character, font=font) + tracking
