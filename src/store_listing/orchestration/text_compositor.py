"""Composite the brief's headline and tagline onto the generated artwork.

The diffusion model draws the artwork and nothing else. It cannot spell, so the
text it produced was misspelled, duplicated or dropped, and the briefs that
lost the race were thrown away by the OCR gate (issue #187). The headline and
the tagline are drawn here instead: the artwork comes from the model, the text
comes from a font.

The text is drawn with a halo so it stays readable over busy artwork - and so
the OCR gate can read it back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "DejaVuSans-Bold.ttf"

# Share of the canvas height the headline and the tagline are drawn in.
HEADLINE_ZONE = 0.15
TAGLINE_ZONE = 0.12
# Horizontal margin, as a share of the canvas width.
MARGIN_X = 0.05
# Smallest font size that is still worth drawing.
MIN_FONT_SIZE = 10
# Distance between two lines of a wrapped headline, as a multiple of the size.
LINE_SPACING = 1.15
# Width of the halo around the text, as a multiple of the font size.
HALO_RATIO = 0.06
# Share of the zone that stays empty around the text.
ZONE_PADDING = 0.2


def resolve_layout(layout_type: str, headline_text: str | None, tagline_text: str | None) -> str:
    """The layout that is actually drawn.

    A brief that asks for a tagline gets one, even when the layout says
    ``text_top`` - otherwise the tagline would be dropped on the floor. The
    artwork prompt has to follow the same rule, so the space for the text is
    reserved where the text ends up.
    """
    if layout_type == "full_bleed":
        return "full_bleed"
    if tagline_text and tagline_text.strip():
        return "text_top_bottom"
    return "text_top"


@dataclass
class LayoutSpec:
    """What has to be drawn, and where.

    The layout decides what is drawn: a ``text_top`` brief has a headline at
    the top and nothing else, a ``text_top_bottom`` brief has both. The
    specification is normalised so that it is the single source of truth for
    the pipeline - the text that is drawn and the text that is validated have
    to be the same text.
    """

    layout_type: str  # "full_bleed", "text_top", "text_top_bottom"
    headline_text: str | None = None
    tagline_text: str | None = None
    font_color: str = "#000000"

    def __post_init__(self) -> None:
        self.layout_type = resolve_layout(self.layout_type, self.headline_text, self.tagline_text)
        if self.layout_type == "full_bleed":
            self.headline_text = None
            self.tagline_text = None
        elif self.layout_type == "text_top":
            self.tagline_text = None

    @property
    def needs_text(self) -> bool:
        return bool(self.headline_text or self.tagline_text)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def _missing_glyph(font: ImageFont.FreeTypeFont) -> tuple[float, float, float, float]:
    """The bounding box the font draws for a character it has no glyph for.

    The private use area is a safe probe: no font has a glyph there.
    """
    return font.getbbox("\ue000")


def _drop_unsupported(text: str) -> str:
    """Remove the characters the font cannot draw.

    A missing glyph is drawn as an empty box, which is worse than leaving the
    character out. The vendored font has no emoji and no CJK.
    """
    font = _font(32)
    missing = _missing_glyph(font)
    kept: list[str] = []
    for character in text:
        if character.isspace():
            kept.append(character)
        elif font.getbbox(character) == missing:
            logger.warning("Dropping %r from the text: the font has no glyph for it", character)
        else:
            kept.append(character)
    return "".join(kept)


def _break_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> list[str]:
    """Split a word that is wider than the line into pieces that fit."""
    pieces: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _wrap(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> list[str]:
    """Break the text into lines that fit the width."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)

    wrapped: list[str] = []
    for line in lines:
        if draw.textlength(line, font=font) <= max_width:
            wrapped.append(line)
        else:
            wrapped.extend(_break_to_width(draw, line, font, max_width))
    return wrapped


def _fit(
    draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Find the biggest font size at which the text fits the zone."""
    size = max(MIN_FONT_SIZE, max_height)
    while size >= MIN_FONT_SIZE:
        font = _font(size)
        lines = _wrap(draw, text, font, max_width)
        if len(lines) * size * LINE_SPACING <= max_height:
            return font, lines
        size -= 2

    font = _font(MIN_FONT_SIZE)
    return font, _wrap(draw, text, font, max_width)


def _halo_colour(color: str) -> str:
    """The halo colour that keeps the text readable on any artwork.

    Light text gets a dark halo and dark text a light one, so the text stays
    legible over busy artwork and the OCR gate can read it back.
    """
    hex_value = color.lstrip("#")
    if len(hex_value) < 6:
        return "#000000"
    try:
        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)
    except ValueError:
        return "#000000"

    brightness = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    return "#000000" if brightness > 0.5 else "#FFFFFF"


def _draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    color: str,
    box: tuple[int, int, int, int],
) -> None:
    """Draw the text centred in the given box."""
    text = _drop_unsupported(text.strip())
    if not text:
        return

    x, y, width, height = box
    font, lines = _fit(draw, text, width, int(height * (1 - ZONE_PADDING)))

    line_height = int(font.size * LINE_SPACING)
    total_height = line_height * len(lines)
    top = y + max(0, (height - total_height) // 2)

    halo = max(1, int(font.size * HALO_RATIO))
    halo_color = _halo_colour(color)
    for index, line in enumerate(lines):
        line_width = draw.textlength(line, font=font)
        line_x = int(x + (width - line_width) / 2)
        line_y = top + index * line_height
        draw.text(
            (line_x, line_y),
            line,
            font=font,
            fill=color,
            stroke_width=halo,
            stroke_fill=halo_color,
        )


def composite_text(image_bytes: bytes, layout: LayoutSpec) -> bytes:
    """Draw the brief's headline and tagline onto the artwork."""
    if layout.layout_type == "full_bleed" or not layout.needs_text:
        return image_bytes

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except OSError as error:
        logger.warning("Cannot composite the text onto the artwork: %s", error)
        return image_bytes

    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    margin_x = int(width * MARGIN_X)
    text_width = width - 2 * margin_x

    if layout.headline_text:
        _draw_text(
            draw,
            layout.headline_text,
            color=layout.font_color,
            box=(margin_x, 0, text_width, int(height * HEADLINE_ZONE)),
        )
    if layout.tagline_text:
        _draw_text(
            draw,
            layout.tagline_text,
            color=layout.font_color,
            box=(
                margin_x,
                int(height * (1 - TAGLINE_ZONE)),
                text_width,
                int(height * TAGLINE_ZONE),
            ),
        )

    result = Image.alpha_composite(image, overlay)
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
