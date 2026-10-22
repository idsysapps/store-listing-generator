"""Composite the brief's headline and tagline onto the generated artwork.

The diffusion model draws the artwork and nothing else. It cannot spell, so the
text it produced was misspelled, duplicated or dropped, and the briefs that
lost the race were thrown away by the OCR gate (issue #187). The headline and
the tagline are drawn here instead: the artwork comes from the model, the text
comes from a font.

The text is drawn with a halo so it stays readable over busy artwork - and so
the OCR gate can read it back.

A typography-only design has no artwork at all: the canvas is blank and only
the text is drawn on it (issue #196).

The type itself is decided in ``typography``: the family comes from the art
direction and the treatment from the voice of the brief. The text used to be
drawn in one font with an outline around it, which is the "meme font" look
(issue #226).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont

from store_listing.orchestration.typography import (
    DEFAULT_FONT,
    TextTreatment,
    accent_for,
    family_for_voice,
    font_path,
    tagline_family_for_voice,
    treatment_for_voice,
)

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
# How far the shadow of the text sits from the text, as a share of the size.
SHADOW_RATIO = 0.045
# The empty space around a word that is drawn in the accent colour.
BLOCK_PADDING = 0.18
# The margin the text keeps inside a band, as a share of the band.
BAND_INSET = 0.08

# A typography-only design has no artwork to make room for, so the text sits
# in the middle of the canvas instead of at its edges.
TEXT_ONLY_HEADLINE_TOP = 0.25
TEXT_ONLY_HEADLINE_HEIGHT = 0.30
TEXT_ONLY_TAGLINE_TOP = 0.55
TEXT_ONLY_TAGLINE_HEIGHT = 0.20

# The size of a render, and the colours a blank canvas comes in.
CANVAS_SIZE = (1024, 1024)
# The DTF renders are keyed out of this colour, so the blank DTF canvas uses
# it too.
DTF_KEY_COLOUR = (255, 0, 255)
# The print-through products print the background as-is, so their canvas is
# the white of the paper.
PAPER_COLOUR = (255, 255, 255)


@dataclass(frozen=True)
class TextPlacement:
    """Where the text of a layout goes, as a share of the canvas.

    The box is (x, y, width, height) in shares of the canvas, so the same
    placement works on a render of any size.
    """

    headline: tuple[float, float, float, float] | None = None
    tagline: tuple[float, float, float, float] | None = None
    # Whether a band is drawn behind the text.
    bar: bool = False


# The layouts that put the text in a shape of its own. Every other layout
# puts the text above and below the art, which is why every design looked the
# same (issue #227).
SHAPED_LAYOUTS: tuple[str, ...] = ("badge", "banner", "arch", "stack_side")

# The placements of the layouts that draw text.
PLACEMENTS: dict[str, TextPlacement] = {
    "text_top": TextPlacement(headline=(0.05, 0.00, 0.90, 0.15)),
    "text_top_bottom": TextPlacement(
        headline=(0.05, 0.00, 0.90, 0.15), tagline=(0.05, 0.88, 0.90, 0.12)
    ),
    # A label: a band at the top, a band at the bottom, an emblem between.
    "badge": TextPlacement(
        headline=(0.08, 0.01, 0.84, 0.14), tagline=(0.08, 0.85, 0.84, 0.13), bar=True
    ),
    # A ribbon across the middle of the art.
    "banner": TextPlacement(
        headline=(0.06, 0.36, 0.88, 0.22), tagline=(0.05, 0.88, 0.90, 0.12), bar=True
    ),
    # A tall arch with the headline in the head of the arch.
    "arch": TextPlacement(
        headline=(0.20, 0.00, 0.60, 0.14), tagline=(0.05, 0.88, 0.90, 0.12), bar=True
    ),
    # The art on the left, the text stacked on the right.
    "stack_side": TextPlacement(
        headline=(0.56, 0.28, 0.40, 0.32), tagline=(0.56, 0.62, 0.40, 0.16)
    ),
}


def resolve_layout(layout_type: str, headline_text: str | None, tagline_text: str | None) -> str:
    """The layout that is actually drawn.

    A brief that asks for a tagline gets one, even when the layout says
    ``text_top`` - otherwise the tagline would be dropped on the floor. The
    artwork prompt has to follow the same rule, so the space for the text is
    reserved where the text ends up.
    """
    if layout_type == "full_bleed":
        return "full_bleed"
    if layout_type == "text_only":
        return "text_only"
    if layout_type in SHAPED_LAYOUTS:
        return layout_type
    if tagline_text and tagline_text.strip():
        return "text_top_bottom"
    return "text_top"


@dataclass
class LayoutSpec:
    """What has to be drawn, and where.

    The layout decides what is drawn: a ``text_top`` brief has a headline at
    the top and nothing else, a ``text_top_bottom`` brief has both. A
    ``text_only`` brief is typography and nothing else, so it is drawn in the
    middle of the canvas. The specification is normalised so that it is the
    single source of truth for the pipeline - the text that is drawn and the
    text that is validated have to be the same text.
    """

    layout_type: str  # "full_bleed", "text_top", "text_top_bottom", "text_only"
    headline_text: str | None = None
    tagline_text: str | None = None
    font_color: str = "#000000"
    # The family the headline is drawn in, and the one the tagline is drawn in.
    # Without one the voice of the brief decides.
    font_family: str | None = None
    tagline_font_family: str | None = None
    # The voice of the brief, which decides the treatment of the text.
    voice: str = "deadpan"
    # A treatment that replaces the one of the voice.
    treatment_override: TextTreatment | None = None

    def __post_init__(self) -> None:
        self.layout_type = resolve_layout(self.layout_type, self.headline_text, self.tagline_text)
        if self.layout_type == "full_bleed":
            self.headline_text = None
            self.tagline_text = None
        elif self.layout_type == "text_top":
            self.tagline_text = None
        if not self.font_family:
            self.font_family = family_for_voice(self.voice)
        if not self.tagline_font_family:
            self.tagline_font_family = tagline_family_for_voice(self.voice)

    @property
    def needs_text(self) -> bool:
        return bool(self.headline_text or self.tagline_text)

    @property
    def family(self) -> str:
        """The family the headline is drawn in."""
        return self.font_family or family_for_voice(self.voice)

    @property
    def tagline_family(self) -> str:
        """The family the tagline is drawn in."""
        return self.tagline_font_family or tagline_family_for_voice(self.voice)

    @property
    def treatment(self) -> TextTreatment:
        """How the text is drawn: the case, the tracking, the colours."""
        if self.treatment_override is not None:
            return self.treatment_override
        return treatment_for_voice(self.voice)


def blank_canvas(
    product_type: str,
    *,
    transparent: bool = False,
    size: tuple[int, int] = CANVAS_SIZE,
) -> bytes:
    """A blank canvas to draw a typography-only design on.

    The DTF design is keyed out of a magenta background, so its canvas is
    magenta; the print-through products print the background as the paper, so
    theirs is white. The transparent canvas is what the DTF print file is
    made of.
    """
    if transparent:
        colour = (0, 0, 0, 0)
    elif product_type == "dtf_apparel":
        colour = (*DTF_KEY_COLOUR, 255)
    else:
        colour = (*PAPER_COLOUR, 255)

    image = Image.new("RGBA", size, colour)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _font(size: int, family: str = DEFAULT_FONT) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(family)), size)


def _glyph_pixels(mask: Any) -> bytes:
    """The pixels of a glyph mask.

    The mask is an image core, which carries its pixels as a buffer. The
    stubs of Pillow do not say so, so the buffer is read through ``Any``.
    """
    return bytes(cast(Any, mask))


def _missing_glyph(font: ImageFont.FreeTypeFont) -> tuple[tuple[int, int], bytes]:
    """The bitmap the font draws for a character it has no glyph for.

    The private use area is a safe probe: no font has a glyph there. The
    bitmap is compared and not the bounding box, because the box of the
    missing glyph is the box of a real letter in some families - Anton drew
    every capital letter as the missing glyph that way, and every headline
    came out empty.
    """
    mask = font.getmask("\ue000", mode="L")
    return (mask.size, _glyph_pixels(mask))


def _drop_unsupported(text: str, family: str = DEFAULT_FONT) -> str:
    """Remove the characters the font cannot draw.

    A missing glyph is drawn as an empty box, which is worse than leaving the
    character out. The vendored fonts have no emoji and no CJK.
    """
    font = _font(32, family)
    missing = _missing_glyph(font)
    kept: list[str] = []
    for character in text:
        if character.isspace():
            kept.append(character)
            continue
        mask = font.getmask(character, mode="L")
        if (mask.size, _glyph_pixels(mask)) == missing:
            logger.warning("Dropping %r from the text: the font has no glyph for it", character)
        else:
            kept.append(character)
    return "".join(kept)


def _measure_run(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: float
) -> float:
    """The width of a word, with the tracking between the letters."""
    if not text:
        return 0.0
    if tracking <= 0:
        return draw.textlength(text, font=font)
    width = sum(draw.textlength(character, font=font) for character in text)
    return width + tracking * (len(text) - 1)


def _measure_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.FreeTypeFont,
    tracking: float,
) -> float:
    """The width of a line, as it is drawn."""
    words = line.split(" ")
    if not words:
        return 0.0
    space = draw.textlength(" ", font=font) + tracking
    width = _measure_run(draw, words[0], font, tracking)
    for word in words[1:]:
        width += space + _measure_run(draw, word, font, tracking)
    return width


def _break_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    tracking: float,
) -> list[str]:
    """Split a word that is wider than the line into pieces that fit."""
    pieces: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and _measure_run(draw, candidate, font, tracking) > max_width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    tracking: float,
) -> list[str]:
    """Break the text into lines that fit the width."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _measure_line(draw, candidate, font, tracking) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)

    wrapped: list[str] = []
    for line in lines:
        if _measure_line(draw, line, font, tracking) <= max_width:
            wrapped.append(line)
        else:
            wrapped.extend(_break_to_width(draw, line, font, max_width, tracking))
    return wrapped


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    """The height of a line of text, from the metrics of the family.

    A script family has long ascenders and descenders, so the height of a
    line is taken from the font instead of from its size.
    """
    ascent, descent = font.getmetrics()
    return max(1, int((ascent + descent) * LINE_SPACING))


def _fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    family: str,
    treatment: TextTreatment,
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Find the biggest font size at which the text fits the zone."""
    size = max(MIN_FONT_SIZE, max_height)
    while size >= MIN_FONT_SIZE:
        font = _font(size, family)
        lines = _wrap(draw, text, font, max_width, treatment.tracking * size)
        if len(lines) * _line_height(font) <= max_height:
            return font, lines
        size -= 2

    font = _font(MIN_FONT_SIZE, family)
    return font, _wrap(draw, text, font, max_width, treatment.tracking * MIN_FONT_SIZE)


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


def _contrast_colour(color: str) -> str:
    """The colour that stands out against the given colour."""
    return _halo_colour(color)


def _segments(line: str, start_word: int, accent_word: int | None) -> list[tuple[str, bool]]:
    """The words of a line, and which of them is the accent word."""
    segments: list[tuple[str, bool]] = []
    index = start_word
    for word in line.split(" "):
        segments.append((word, accent_word is not None and index == accent_word))
        index += 1
    return segments


def _draw_run(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    tracking: float,
    halo: int = 0,
    halo_fill: str = "#000000",
) -> float:
    """Draw a word, letter by letter so the tracking can be applied."""
    for index, character in enumerate(text):
        draw.text(
            (x, y),
            character,
            font=font,
            fill=fill,
            stroke_width=halo,
            stroke_fill=halo_fill,
        )
        x += draw.textlength(character, font=font)
        if index < len(text) - 1:
            x += tracking
    return x


def _draw_line(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: int,
    line: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill_for,
    tracking: float,
    start_word: int,
    accent_word: int | None,
    halo: int = 0,
    halo_fill: str = "#000000",
) -> None:
    """Draw one line of text, word by word."""
    space = draw.textlength(" ", font=font) + tracking
    for index, (word, is_accent) in enumerate(_segments(line, start_word, accent_word)):
        if index:
            x += space
        x = _draw_run(
            draw,
            x,
            y,
            word,
            font,
            fill_for(is_accent),
            tracking,
            halo,
            halo_fill,
        )


def _draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    color: str,
    box: tuple[int, int, int, int],
    family: str = DEFAULT_FONT,
    treatment: TextTreatment | None = None,
    bar: str | None = None,
) -> None:
    """Draw the text centred in the given box.

    ``bar`` is the colour of a band that is drawn behind the text. The text on
    a band is drawn in the colour that stands out against it, so it stays
    readable whatever the band is.
    """
    treatment = treatment or TextTreatment()
    text = text.strip()
    if treatment.uppercase:
        text = text.upper()
    text = _drop_unsupported(text, family)
    if not text:
        return

    x, y, width, height = box
    # The text on a band keeps a margin inside it, so it does not touch the
    # edge of the band.
    fit_width = int(width * (1 - BAND_INSET)) if bar is not None else width
    font, lines = _fit(draw, text, fit_width, int(height * (1 - ZONE_PADDING)), family, treatment)

    if bar is not None:
        color = _contrast_colour(bar)
        treatment = TextTreatment(
            uppercase=treatment.uppercase,
            tracking=treatment.tracking,
            style=treatment.style,
        )

    line_height = _line_height(font)
    total_height = line_height * len(lines)
    top = y + max(0, (height - total_height) // 2)

    tracking = treatment.tracking * font.size
    accent = treatment.accent_color or accent_for(color)
    shadow_color = _halo_colour(color)
    offset = max(1, int(font.size * SHADOW_RATIO))
    halo = max(1, int(font.size * HALO_RATIO)) if treatment.style == "halo" else 0

    if bar is not None:
        draw.rectangle([x, y, x + width, y + height], fill=bar)

    def fill_for(is_accent: bool) -> str:
        return accent if is_accent else color

    word_index = 0
    for index, line in enumerate(lines):
        line_width = _measure_line(draw, line, font, tracking)
        line_x = int(x + (width - line_width) / 2)
        line_y = top + index * line_height

        if treatment.style == "block":
            padding = int(font.size * BLOCK_PADDING)
            draw.rectangle(
                [
                    line_x - padding,
                    line_y - padding // 2,
                    line_x + line_width + padding,
                    line_y + line_height - padding // 2,
                ],
                fill=accent,
            )
        elif treatment.style == "shadow":
            _draw_line(
                draw,
                line_x + offset,
                line_y + offset,
                line,
                font,
                fill_for=lambda _is_accent: shadow_color,
                tracking=tracking,
                start_word=word_index,
                accent_word=treatment.accent_word,
            )

        _draw_line(
            draw,
            line_x,
            line_y,
            line,
            font,
            fill_for=fill_for,
            tracking=tracking,
            start_word=word_index,
            accent_word=treatment.accent_word,
            halo=halo,
            halo_fill=shadow_color,
        )
        word_index += len(line.split())


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
    treatment = layout.treatment

    if layout.layout_type == "text_only":
        _draw_text_only(draw, layout, margin_x=margin_x, text_width=text_width, height=height)
        return _composite(image, overlay)

    placement = PLACEMENTS.get(layout.layout_type, PLACEMENTS["text_top"])
    bar = accent_for(layout.font_color) if placement.bar else None

    if layout.headline_text and placement.headline:
        box = _box(placement.headline, width, height)
        _draw_text(
            draw,
            layout.headline_text,
            color=layout.font_color,
            box=box,
            family=layout.family,
            treatment=treatment,
            bar=bar,
        )
    if layout.tagline_text and placement.tagline:
        box = _box(placement.tagline, width, height)
        _draw_text(
            draw,
            layout.tagline_text,
            color=layout.font_color,
            box=box,
            family=layout.tagline_family,
            treatment=treatment,
            bar=bar,
        )

    return _composite(image, overlay)


def _box(
    placement: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """A placement in shares of the canvas as a box in pixels."""
    x, y, box_width, box_height = placement
    return (
        int(x * width),
        int(y * height),
        int(box_width * width),
        int(box_height * height),
    )


def _composite(image: Image.Image, overlay: Image.Image) -> bytes:
    result = Image.alpha_composite(image, overlay)
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_text_only(
    draw: ImageDraw.ImageDraw,
    layout: LayoutSpec,
    *,
    margin_x: int,
    text_width: int,
    height: int,
) -> None:
    """Draw a typography-only design in the middle of the canvas.

    There is no artwork to leave room for, so the headline sits above the
    middle of the canvas and the tagline below it.
    """
    treatment = layout.treatment
    if layout.headline_text:
        _draw_text(
            draw,
            layout.headline_text,
            color=layout.font_color,
            box=(
                margin_x,
                int(height * TEXT_ONLY_HEADLINE_TOP),
                text_width,
                int(height * TEXT_ONLY_HEADLINE_HEIGHT),
            ),
            family=layout.family,
            treatment=treatment,
        )
    if layout.tagline_text:
        _draw_text(
            draw,
            layout.tagline_text,
            color=layout.font_color,
            box=(
                margin_x,
                int(height * TEXT_ONLY_TAGLINE_TOP),
                text_width,
                int(height * TEXT_ONLY_TAGLINE_HEIGHT),
            ),
            family=layout.tagline_family,
            treatment=treatment,
        )
