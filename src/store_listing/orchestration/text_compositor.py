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
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from store_listing.orchestration.design_templates import resolve_template, template_for
from store_listing.orchestration.text_integration import TextPlan
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


@dataclass
class LayoutSpec:
    """What has to be drawn, and where.

    The template decides what is drawn: ``hero_top`` has the art above the copy,
    ``hero_bottom`` has the copy above the art, ``wide_scene`` is a landscape
    canvas, and so on. A ``text_only`` brief is typography and nothing else, so
    it is drawn in the middle of the canvas. The specification is normalised so
    that it is the single source of truth for the pipeline - the text that is
    drawn and the text that is validated have to be the same text.
    """

    layout_type: str  # a template id, or a legacy layout id that maps to one
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
    # How the words live in the design. Without a plan the design keeps the
    # flat compositor treatment it had before (issue #246).
    text_plan: TextPlan | None = None

    def __post_init__(self) -> None:
        self.layout_type = resolve_template(self.layout_type, self.headline_text, self.tagline_text)
        if self.layout_type == "full_bleed":
            self.headline_text = None
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
    """Find the biggest font size at which the text fits the zone.

    A word is never split down the middle: a narrow column shrinks the type
    instead of breaking ``GOBBLE`` into ``GOBBL`` and ``E``, which reads as a
    typo to the buyer and to the text gate.
    """
    words = text.split()
    size = max(MIN_FONT_SIZE, max_height)
    while size >= MIN_FONT_SIZE:
        font = _font(size, family)
        tracking = treatment.tracking * size
        longest_word = max(
            (_measure_run(draw, word, font, tracking) for word in words),
            default=0.0,
        )
        lines = _wrap(draw, text, font, max_width, tracking)
        if longest_word <= max_width and len(lines) * _line_height(font) <= max_height:
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
    align: str = "center",
    force_outline: bool = False,
) -> None:
    """Draw the text centred (or left-aligned) in the given box.

    ``bar`` is the colour of a band that is drawn behind the text. The text on
    a band is drawn in the colour that stands out against it, so it stays
    readable whatever the band is.

    ``force_outline`` adds a contrasting outline even when the treatment does
    not ask for one. The transparent DTF render needs it: its text is shown on
    white, and a light brief colour would otherwise disappear. The band
    templates are left alone because the band already guarantees contrast.
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
    if force_outline and bar is None:
        halo = max(halo, max(2, int(font.size * HALO_RATIO)))

    if bar is not None:
        draw.rectangle([x, y, x + width, y + height], fill=bar)

    def fill_for(is_accent: bool) -> str:
        return accent if is_accent else color

    word_index = 0
    for index, line in enumerate(lines):
        line_width = _measure_line(draw, line, font, tracking)
        line_x = int(x) if align == "left" else int(x + (width - line_width) / 2)
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


def _draw_arc_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    color: str,
    box: tuple[int, int, int, int],
    family: str = DEFAULT_FONT,
    treatment: TextTreatment | None = None,
    force_outline: bool = False,
) -> None:
    """Draw the text along a downward curve.

    A ``roundel`` uses it for the tagline, which follows the bottom of the
    circle. The characters stay upright so the arc is still readable by the
    text gate; the tagline is secondary, so a curve the gate cannot read does
    not fail the design.
    """
    treatment = treatment or TextTreatment()
    text = text.strip()
    if treatment.uppercase:
        text = text.upper()
    text = _drop_unsupported(text, family)
    if not text:
        return

    x, y, width, height = box
    if width < 1 or height < 1:
        return

    size = max(MIN_FONT_SIZE, int(height * 0.55))
    tracking = treatment.tracking
    while size > MIN_FONT_SIZE:
        font = _font(size, family)
        if _measure_run(draw, text, font, tracking * size) <= width:
            break
        size -= 2
    font = _font(size, family)
    tracking = treatment.tracking * size

    widths = [draw.textlength(character, font=font) for character in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    if total <= 0:
        return

    # A shallow arc: a radius of 2.5x the width keeps the curve gentle, so the
    # text stays inside its band and the vision model can still read it.
    radius = width * 2.5
    half = math.asin(min(1.0, width / (2 * radius)))
    arc = 2 * half
    centre_x = x + width / 2
    centre_y = y + height - radius
    halo = max(1, int(size * HALO_RATIO)) if treatment.style == "halo" else 0
    if force_outline:
        halo = max(halo, max(2, int(size * HALO_RATIO)))

    position = math.pi / 2 + half
    for character, character_width in zip(text, widths):
        angle = position - arc * ((character_width / 2) / total)
        draw.text(
            (
                centre_x + radius * math.cos(angle),
                centre_y + radius * math.sin(angle),
            ),
            character,
            font=font,
            fill=color,
            anchor="mm",
            stroke_width=halo,
            stroke_fill=_halo_colour(color),
        )
        position -= arc * ((character_width + tracking) / total)


def composite_text(image_bytes: bytes, layout: LayoutSpec, *, transparent: bool = False) -> bytes:
    """Draw the brief's headline and tagline onto the artwork.

    ``transparent`` marks the keyed DTF render, which is shown on whatever the
    garment or the catalog page is. The text gets a contrasting outline there
    so a light brief colour does not disappear into the background.
    """
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

    if layout.text_plan is not None and (
        layout.text_plan.method != "typographic" or layout.text_plan.needs_carrier
    ):
        planned = _composite_planned(image, layout, transparent=transparent)
        if planned is not None:
            return planned

    if layout.layout_type == "text_only":
        _draw_text_only(
            draw,
            layout,
            margin_x=margin_x,
            text_width=text_width,
            height=height,
            force_outline=transparent,
        )
        return _composite(image, overlay)

    template = template_for(layout.layout_type)
    bar = accent_for(layout.font_color) if template.bar else None

    if layout.headline_text and template.headline:
        box = _box(template.headline, width, height)
        _draw_text(
            draw,
            layout.headline_text,
            color=layout.font_color,
            box=box,
            family=layout.family,
            treatment=treatment,
            bar=bar,
            align=template.align,
            force_outline=transparent,
        )
    if layout.tagline_text and template.tagline:
        box = _box(template.tagline, width, height)
        if template.arc_tagline:
            _draw_arc_text(
                draw,
                layout.tagline_text,
                color=layout.font_color,
                box=box,
                family=layout.tagline_family,
                treatment=treatment,
                force_outline=transparent,
            )
        else:
            _draw_text(
                draw,
                layout.tagline_text,
                color=layout.font_color,
                box=box,
                family=layout.tagline_family,
                treatment=treatment,
                bar=bar,
                align=template.align,
                force_outline=transparent,
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


# The colours the compositor looks for when it has to find a carrier. The
# names are the surface vocabulary of the text plan. A surface has several
# shades because the image model draws the same wood light or dark, and a
# single swatch would miss half of them.
SURFACE_COLOURS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "paper": ((246, 241, 232), (255, 255, 255), (228, 218, 200)),
    "kraft": ((193, 154, 107), (160, 120, 80), (210, 175, 130)),
    "wood": ((139, 94, 60), (101, 67, 33), (180, 130, 90), (80, 55, 35)),
    "chalkboard": ((35, 40, 38), (50, 55, 52), (20, 25, 24)),
    "metal": ((168, 173, 178), (120, 125, 130), (200, 205, 210)),
    "stone": ((140, 140, 135), (110, 110, 105), (170, 170, 165), (90, 90, 85)),
    "fabric": ((201, 181, 155), (150, 130, 110), (220, 205, 185), (245, 240, 230)),
    "glass": ((190, 214, 220), (150, 180, 190), (220, 235, 240)),
    "none": ((246, 241, 232),),
}

# The nine zones of the canvas, as (x, y, width, height) shares. They overlap
# so a carrier near a boundary is still found.
_REGION_RECTS: dict[str, tuple[float, float, float, float]] = {
    "top_left": (0.0, 0.0, 0.5, 0.5),
    "top_center": (0.25, 0.0, 0.5, 0.5),
    "top_right": (0.5, 0.0, 0.5, 0.5),
    "middle_left": (0.0, 0.25, 0.5, 0.5),
    "center": (0.25, 0.25, 0.5, 0.5),
    "middle_right": (0.5, 0.25, 0.5, 0.5),
    "bottom_left": (0.0, 0.5, 0.5, 0.5),
    "bottom_center": (0.25, 0.5, 0.5, 0.5),
    "bottom_right": (0.5, 0.5, 0.5, 0.5),
    "any": (0.0, 0.0, 1.0, 1.0),
}


def _composite_planned(
    image: Image.Image, layout: LayoutSpec, *, transparent: bool
) -> bytes | None:
    """Draw the text the way the brief planned it.

    Without a carrier the template's text zone is the surface. With one, the
    compositor looks for the carrier in the planned region and fits the text
    to it; a carrier that is not there falls back to the zone, so the design
    never loses its words.
    """
    plan = layout.text_plan
    if plan is None:
        return None

    width, height = image.size
    template = template_for(layout.layout_type)
    margin_x = int(width * MARGIN_X)
    text_width = width - 2 * margin_x

    headline_box: tuple[int, int, int, int] | None = None
    tagline_box: tuple[int, int, int, int] | None = None
    if layout.layout_type == "text_only":
        headline_box = (
            margin_x,
            int(height * TEXT_ONLY_HEADLINE_TOP),
            text_width,
            int(height * TEXT_ONLY_HEADLINE_HEIGHT),
        )
        tagline_box = (
            margin_x,
            int(height * TEXT_ONLY_TAGLINE_TOP),
            text_width,
            int(height * TEXT_ONLY_TAGLINE_HEIGHT),
        )
    else:
        if template.headline is not None:
            headline_box = _box(template.headline, width, height)
        if template.tagline is not None:
            tagline_box = _box(template.tagline, width, height)

    surface_colour: tuple[int, int, int] | None = None
    method = plan.method
    if plan.needs_carrier:
        # The planner named the region; that is where the carrier is. The
        # template zone is only where the fallback text goes if the carrier
        # is not there. A line-art carrier has no surface colour to match, so
        # the enclosed white shape is the second chance.
        detected = _detect_carrier(image, plan) or _detect_line_art_carrier(image, plan)
        carrier = detected or _fallback_band(plan, headline_box, tagline_box, width, height)
        if detected is None:
            # The carrier was not drawn, or not where the plan said. The
            # model's own non-carrier alternative is the fallback; the words
            # still go where the carrier was meant to be.
            method = plan.non_carrier_method or plan.method
        surface_colour = _mean_colour(image, carrier) or _surface_colour(plan.surface)
        if layout.headline_text and layout.tagline_text:
            headline_box = (
                carrier[0],
                carrier[1] + int(carrier[3] * 0.06),
                carrier[2],
                int(carrier[3] * 0.58),
            )
            tagline_box = (
                carrier[0],
                carrier[1] + int(carrier[3] * 0.68),
                carrier[2],
                int(carrier[3] * 0.24),
            )
        elif layout.headline_text:
            headline_box = carrier
        elif layout.tagline_text:
            tagline_box = carrier
    else:
        sample = headline_box or tagline_box
        if sample is not None:
            surface_colour = _mean_colour(image, sample)

    ink = _resolve_ink(plan, layout.font_color, surface_colour)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    align = "center" if plan.needs_carrier else template.align
    if layout.headline_text and headline_box is not None:
        _draw_text(
            draw,
            layout.headline_text,
            color=ink,
            box=headline_box,
            family=layout.family,
            treatment=layout.treatment,
            align=align,
            force_outline=transparent,
        )
    if layout.tagline_text and tagline_box is not None:
        if template.arc_tagline and not plan.needs_carrier:
            _draw_arc_text(
                draw,
                layout.tagline_text,
                color=ink,
                box=tagline_box,
                family=layout.tagline_family,
                treatment=layout.treatment,
                force_outline=transparent,
            )
        else:
            _draw_text(
                draw,
                layout.tagline_text,
                color=ink,
                box=tagline_box,
                family=layout.tagline_family,
                treatment=layout.treatment,
                align=align,
                force_outline=transparent,
            )

    overlay = apply_material(overlay, method, ink, surface_colour)
    return _composite(image, overlay)


def _crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """A crop of a (x, y, width, height) box.

    Pillow's crop takes the right and bottom edges, not a size, and mixing the
    two conventions silently crops nothing.
    """
    x, y, width, height = box
    return image.crop((x, y, x + width, y + height))


def _region_box(region: str, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = _REGION_RECTS.get(region, _REGION_RECTS["any"])
    return (
        int(x * width),
        int(y * height),
        int(box_width * width),
        int(box_height * height),
    )


def _expand(
    box: tuple[int, int, int, int], share: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """A box with a margin, clamped to the canvas."""
    dx = int(box[2] * share)
    dy = int(box[3] * share)
    x = max(0, box[0] - dx)
    y = max(0, box[1] - dy)
    return (x, y, min(width, box[0] + box[2] + dx) - x, min(height, box[1] + box[3] + dy) - y)


def _intersection(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x = max(first[0], second[0])
    y = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    if right <= x or bottom <= y:
        return None
    return (x, y, right - x, bottom - y)


def _fallback_band(
    plan: TextPlan,
    headline_box: tuple[int, int, int, int] | None,
    tagline_box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Where the words go when the carrier is not there.

    A template zone that overlaps the planned region wins; otherwise a band
    in the middle of the region, which is where the brief put the carrier.
    """
    region = _region_box(plan.region, width, height)
    for zone in (headline_box, tagline_box):
        if zone is None:
            continue
        overlap = _intersection(region, zone)
        if overlap is not None:
            return overlap
    return (
        region[0] + int(region[2] * 0.08),
        region[1] + int(region[3] * 0.35),
        int(region[2] * 0.84),
        int(region[3] * 0.30),
    )


def _largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """The largest all-true rectangle in a boolean mask, as x, y, width, height.

    The bounding box of every matching pixel is not the carrier: a stone
    headstone standing in front of a stone wall shares a colour with the
    wall, and the union spans the whole scene. The largest solid block is the
    object itself (issue #246).
    """
    if mask.ndim != 2 or mask.size == 0:
        return None
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for y in range(mask.shape[0]):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[int] = []
        for x in range(mask.shape[1] + 1):
            height = int(heights[x]) if x < mask.shape[1] else 0
            while stack and height < int(heights[stack[-1]]):
                top = stack.pop()
                bar_height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                width = x - left
                area = bar_height * width
                if area > best_area:
                    best_area = area
                    best = (left, y - bar_height + 1, width, bar_height)
            stack.append(x)
    return best


def _trim_rectangle(
    mask: np.ndarray,
    rectangle: tuple[int, int, int, int],
    max_share: float = 0.5,
) -> tuple[int, int, int, int]:
    """Strip rows and columns that belong to a wider object.

    A wall running behind a headstone shares its colour, so the largest
    rectangle can include the strip of wall directly above the stone. That
    row continues far outside the box; the headstone's rows do not. Trimming
    those rows leaves the object the words belong on (issue #246).
    """
    x0, y0, width, height = rectangle
    while height > 0:
        row = mask[y0, x0 : x0 + width]
        inside = int(row.sum())
        outside = int(mask[y0, :].sum()) - inside
        if outside <= width * max_share:
            break
        y0 += 1
        height -= 1
    while height > 0:
        row = mask[y0 + height - 1, x0 : x0 + width]
        inside = int(row.sum())
        outside = int(mask[y0 + height - 1, :].sum()) - inside
        if outside <= width * max_share:
            break
        height -= 1
    while width > 0:
        column = mask[y0 : y0 + height, x0]
        inside = int(column.sum())
        outside = int(mask[:, x0].sum()) - inside
        if outside <= height * max_share:
            break
        x0 += 1
        width -= 1
    while width > 0:
        column = mask[y0 : y0 + height, x0 + width - 1]
        inside = int(column.sum())
        outside = int(mask[:, x0 + width - 1].sum()) - inside
        if outside <= height * max_share:
            break
        width -= 1
    return (x0, y0, width, height)


def _grow_rectangle(
    mask: np.ndarray,
    rectangle: tuple[int, int, int, int],
    min_share: float = 0.5,
) -> tuple[int, int, int, int]:
    """A rectangle grown to the edges of the surface around it.

    The largest solid block sits inside the carrier; a crack or a grain line
    can stop it short. A row or column that is still mostly surface belongs
    to the carrier, so the box grows one step at a time until the surface
    ends. A row that continues far outside the box is a different object (the
    stone wall behind a headstone), so it does not count.
    """
    x0, y0, width, height = rectangle
    rows, columns = mask.shape
    changed = True
    while changed:
        changed = False
        if x0 > 0:
            column = mask[y0 : y0 + height, x0 - 1]
            inside = int(column.sum())
            outside = int(mask[:, x0 - 1].sum()) - inside
            if inside >= height * min_share and outside <= height * 0.5:
                x0 -= 1
                width += 1
                changed = True
        if x0 + width < columns:
            column = mask[y0 : y0 + height, x0 + width]
            inside = int(column.sum())
            outside = int(mask[:, x0 + width].sum()) - inside
            if inside >= height * min_share and outside <= height * 0.5:
                width += 1
                changed = True
        if y0 > 0:
            row = mask[y0 - 1, x0 : x0 + width]
            inside = int(row.sum())
            outside = int(mask[y0 - 1, :].sum()) - inside
            if inside >= width * min_share and outside <= width * 0.5:
                y0 -= 1
                height += 1
                changed = True
        if y0 + height < rows:
            row = mask[y0 + height, x0 : x0 + width]
            inside = int(row.sum())
            outside = int(mask[y0 + height, :].sum()) - inside
            if inside >= width * min_share and outside <= width * 0.5:
                height += 1
                changed = True
    return (x0, y0, width, height)


def _block_max(mask: np.ndarray, factor: int) -> np.ndarray:
    """A mask reduced by taking the maximum of each factor x factor block."""
    height, width = mask.shape
    if height < factor or width < factor:
        return mask.copy()
    rows = height // factor
    columns = width // factor
    trimmed = mask[: rows * factor, : columns * factor]
    return trimmed.reshape(rows, factor, columns, factor).max(axis=(1, 3))


def _detect_line_art_carrier(
    image: Image.Image,
    plan: TextPlan,
) -> tuple[int, int, int, int] | None:
    """Find a flat carrier drawn inside a dark outline.

    The linocut and ink presets draw a headstone or a sign as a flat area
    inside black linework, and the riso preset paints it a flat colour the
    surface palette did not expect. The carrier is still the largest flat
    area that the linework encloses: the sky touches the edge of the search,
    and the stones of a wall are small. That enclosed area is the carrier
    (issue #246).
    """
    width, height = image.size
    search = _expand(_region_box(plan.region, width, height), 0.15, width, height)
    if search[2] < 1 or search[3] < 1:
        return None
    gray = np.asarray(_crop(image, search).convert("L"), dtype=np.int16)
    dark = gray < 100
    factor = 4
    open_cells = ~_block_max(dark, factor)
    rows, columns = open_cells.shape
    if rows < 2 or columns < 2:
        return None

    visited = np.zeros_like(open_cells, dtype=bool)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for start_y in range(rows):
        for start_x in range(columns):
            if not open_cells[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            area = 0
            min_x = max_x = start_x
            min_y = max_y = start_y
            touches_border = False
            while stack:
                y, x = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                if x == 0 or y == 0 or x == columns - 1 or y == rows - 1:
                    touches_border = True
                if y > 0 and open_cells[y - 1, x] and not visited[y - 1, x]:
                    visited[y - 1, x] = True
                    stack.append((y - 1, x))
                if y + 1 < rows and open_cells[y + 1, x] and not visited[y + 1, x]:
                    visited[y + 1, x] = True
                    stack.append((y + 1, x))
                if x > 0 and open_cells[y, x - 1] and not visited[y, x - 1]:
                    visited[y, x - 1] = True
                    stack.append((y, x - 1))
                if x + 1 < columns and open_cells[y, x + 1] and not visited[y, x + 1]:
                    visited[y, x + 1] = True
                    stack.append((y, x + 1))
            if touches_border or area < 200:
                continue
            if area > best_area:
                best_area = area
                best = (
                    search[0] + min_x * factor,
                    search[1] + min_y * factor,
                    (max_x - min_x + 1) * factor,
                    (max_y - min_y + 1) * factor,
                )
    if best is None:
        return None
    x0, y0, box_width, box_height = best
    if box_width < 40 or box_height < 20:
        return None
    inset_x = max(2, int(box_width * 0.04))
    inset_y = max(2, int(box_height * 0.04))
    return (
        x0 + inset_x,
        y0 + inset_y,
        max(1, box_width - 2 * inset_x),
        max(1, box_height - 2 * inset_y),
    )


def _detect_carrier(
    image: Image.Image,
    plan: TextPlan,
) -> tuple[int, int, int, int] | None:
    """The box of the carrier the scene was asked to draw, or None.

    The planner named the region, so that is the search area. The surface
    colour is the hint; the largest solid block of that colour is the carrier.
    A flat canvas that happens to match the hint is still bounded by the
    region, so it cannot swallow the whole design.
    """
    width, height = image.size
    search = _expand(_region_box(plan.region, width, height), 0.15, width, height)
    if search[2] < 1 or search[3] < 1:
        return None

    # int32: a squared channel difference is up to 65025, which overflows
    # int16 and turns the distance into garbage.
    targets = np.array(_surface_colours(plan.surface), dtype=np.int32)
    rgba = np.asarray(_crop(image, search).convert("RGBA"), dtype=np.int32)
    rgb = rgba[:, :, :3]
    differences = rgb[:, :, None, :] - targets[None, None, :, :]
    distance = np.sqrt((differences**2).sum(axis=3)).min(axis=2)
    mask = (distance <= 60) & (rgba[:, :, 3] > 20)
    if int(mask.sum()) < 200 or int(mask.sum()) < int(mask.size * 0.005):
        return None

    rectangle = _largest_rectangle(mask)
    if rectangle is None:
        return None
    x0, y0, box_width, box_height = _trim_rectangle(mask, rectangle)
    if box_width < 40 or box_height < 20:
        return None
    x0, y0, box_width, box_height = _grow_rectangle(mask, (x0, y0, box_width, box_height))
    if box_width < 40 or box_height < 20:
        return None

    # Stay inside the surface: the text should not spill over its edge.
    inset_x = max(2, int(box_width * 0.04))
    inset_y = max(2, int(box_height * 0.04))
    return (
        search[0] + x0 + inset_x,
        search[1] + y0 + inset_y,
        max(1, box_width - 2 * inset_x),
        max(1, box_height - 2 * inset_y),
    )


def _surface_colours(surface: str) -> tuple[tuple[int, int, int], ...]:
    return SURFACE_COLOURS.get(surface, SURFACE_COLOURS["paper"])


def _surface_colour(surface: str) -> tuple[int, int, int]:
    return _surface_colours(surface)[0]


def _mean_colour(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int] | None:
    """The average visible colour of a box, ignoring transparent pixels."""
    if box[2] < 1 or box[3] < 1:
        return None
    pixels = np.asarray(_crop(image, box).convert("RGBA"), dtype=np.float32)
    visible = pixels[pixels[:, :, 3] > 20]
    if visible.size == 0:
        return None
    mean = visible[:, :3].mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return (0, 0, 0)


def _luminance(colour: tuple[int, int, int]) -> float:
    return (0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]) / 255


def _resolve_ink(
    plan: TextPlan,
    font_color: str,
    surface: tuple[int, int, int] | None,
) -> str:
    """The colour the words are drawn in, made legible on their surface.

    The model picks the ink; the compositor has the final say when the choice
    would disappear into the carrier, because a design whose text the OCR gate
    cannot read is a design that never ships.
    """
    if plan.ink == "accent":
        ink = accent_for(font_color)
    elif plan.ink == "light":
        ink = "#F7F3E8"
    elif plan.ink == "dark":
        ink = "#111111"
    else:
        ink = plan.ink

    if surface is not None and abs(_luminance(_rgb(ink)) - _luminance(surface)) < 0.32:
        return "#F7F3E8" if _luminance(surface) < 0.5 else "#111111"
    return ink


def _tinted(
    size: tuple[int, int],
    colour: tuple[int, int, int],
    mask: Image.Image,
    opacity: int = 255,
) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    solid = Image.new("RGBA", size, (*colour, opacity))
    layer.paste(solid, (0, 0), mask)
    return layer


def _offset(layer: Image.Image, dx: int, dy: int) -> Image.Image:
    result = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    result.paste(layer, (dx, dy), layer)
    return result


_STICKER_THRESHOLD = [0 if value <= 30 else 255 for value in range(256)]
_INSET_ALPHA = [int(value * 0.9) for value in range(256)]
_PAINT_GRAIN = [200 + value * 55 // 255 for value in range(256)]


def apply_material(
    overlay: Image.Image,
    method: str,
    ink: str,
    surface: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Apply the material of a text plan to a drawn text layer.

    Barriers are deliberately light: the OCR gate reads the raw render, so a
    texture may not eat the letters. Each material changes the edge, the
    shadow or the surround, not the glyphs themselves.
    """
    alpha = overlay.getchannel("A")
    if method == "sticker":
        mask = alpha.point(_STICKER_THRESHOLD)
        border = mask.filter(ImageFilter.MaxFilter(7))
        shadow = _offset(
            _tinted(overlay.size, (0, 0, 0), border.filter(ImageFilter.GaussianBlur(1.6)), 90),
            3,
            3,
        )
        white = _tinted(overlay.size, (255, 255, 255), border, 255)
        return Image.alpha_composite(Image.alpha_composite(shadow, white), overlay)

    if method == "neon":
        colour = _rgb(ink)
        wide = _tinted(overlay.size, colour, alpha.filter(ImageFilter.GaussianBlur(18)), 200)
        tight = _tinted(overlay.size, colour, alpha.filter(ImageFilter.GaussianBlur(6)), 240)
        return Image.alpha_composite(Image.alpha_composite(wide, tight), overlay)

    if method == "embroidery":
        # The bevel is the raised edge of the thread; the diagonal lines over
        # the fill are the stitches themselves. Both are light enough that the
        # OCR gate still reads the raw render.
        dark = _offset(_tinted(overlay.size, (0, 0, 0), alpha, 110), 1, 1)
        light = _offset(_tinted(overlay.size, (255, 255, 255), alpha, 110), -1, -1)
        width, height = overlay.size
        pattern = Image.new("L", overlay.size, 0)
        pattern_draw = ImageDraw.Draw(pattern)
        step = max(5, min(width, height) // 55)
        for x in range(-height, width + height, step):
            pattern_draw.line(
                [(x, 0), (x + height, height)],
                fill=190,
                width=max(2, step // 2),
            )
        stitches = _tinted(
            overlay.size,
            (255, 255, 255) if _luminance(_rgb(ink)) < 0.5 else (0, 0, 0),
            ImageChops.multiply(alpha, pattern),
            80,
        )
        base = Image.alpha_composite(dark, light)
        return Image.alpha_composite(Image.alpha_composite(base, overlay), stitches)

    if method == "carved":
        light = _offset(_tinted(overlay.size, (255, 255, 255), alpha, 150), -1, -1)
        dark = _offset(_tinted(overlay.size, (0, 0, 0), alpha, 160), 1, 1)
        inset = overlay.copy()
        inset.putalpha(alpha.point(_INSET_ALPHA))
        return Image.alpha_composite(Image.alpha_composite(light, dark), inset)

    if method == "chalk":
        rough = alpha.filter(ImageFilter.GaussianBlur(0.9))
        dust = _offset(_tinted(overlay.size, _rgb(ink), rough, 55), 1, 1)
        return Image.alpha_composite(dust, overlay)

    if method == "printed":
        shadow = _offset(_tinted(overlay.size, (0, 0, 0), alpha, 50), 1, 1)
        return Image.alpha_composite(shadow, overlay)

    if method == "painted":
        # Paint sinks into the surface: a darker bleed under the edge, and a
        # grain in the fill. The grain never drops below 200/255, so the ink
        # stays readable.
        colour = _rgb(ink)
        bleed_colour = (
            max(0, colour[0] - 40),
            max(0, colour[1] - 40),
            max(0, colour[2] - 40),
        )
        bleed = _tinted(
            overlay.size, bleed_colour, alpha.filter(ImageFilter.GaussianBlur(1.2)), 130
        )
        shadow = _offset(_tinted(overlay.size, (0, 0, 0), alpha, 60), 2, 2)
        grain = Image.effect_noise(overlay.size, 40).convert("L").point(_PAINT_GRAIN)
        grain_rgb = Image.merge("RGB", (grain, grain, grain))
        textured = ImageChops.multiply(overlay.convert("RGB"), grain_rgb).convert("RGBA")
        textured.putalpha(alpha)
        return Image.alpha_composite(Image.alpha_composite(shadow, bleed), textured)

    return overlay


def _draw_text_only(
    draw: ImageDraw.ImageDraw,
    layout: LayoutSpec,
    *,
    margin_x: int,
    text_width: int,
    height: int,
    force_outline: bool = False,
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
            force_outline=force_outline,
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
            force_outline=force_outline,
        )
