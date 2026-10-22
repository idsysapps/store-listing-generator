"""The type of a design.

Every design was drawn in one font (DejaVu Sans Bold), one weight, centred,
with no tracking and a uniform outline around it. That outline is the "meme
font" look, and it is the biggest reason the artwork reads as amateur
(issue #226).

The type is decided here, in one place, so the renderer and the brief
generator agree on what a design looks like:

- the family comes from the art direction (and, until there is one, from the
  voice of the brief), never from the model - the model picks a tone, not a
  font file,
- the treatment comes from the voice: the case, the tracking, the number of
  colours and the shadow.

The families are OFL 1.1, so they can be printed on a shirt. The licence of
each one sits next to the font file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from store_listing.orchestration.headline_voice import normalise_voice

__all__ = [
    "DEFAULT_FONT",
    "FONTS",
    "FontSpec",
    "TextTreatment",
    "accent_for",
    "family_for_voice",
    "font_path",
    "treatment_for_voice",
]

FONT_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "assets" / "fonts"


@dataclass(frozen=True)
class FontSpec:
    """One family of type that a design can be drawn in."""

    family: str
    label: str
    filename: str
    licence: str
    # The family has no lower case letters.
    all_caps: bool = False
    # The family has long ascenders and descenders, so it needs more room.
    script: bool = False


# The families we print with.
FONTS: Final[dict[str, FontSpec]] = {
    "bebas": FontSpec(
        "bebas", "Bebas Neue", "BebasNeue-Regular.ttf", "LICENSE-BebasNeue.txt", all_caps=True
    ),
    "anton": FontSpec("anton", "Anton", "Anton-Regular.ttf", "LICENSE-Anton.txt"),
    "alfa_slab": FontSpec(
        "alfa_slab", "Alfa Slab One", "AlfaSlabOne-Regular.ttf", "LICENSE-AlfaSlabOne.txt"
    ),
    "pacifico": FontSpec(
        "pacifico", "Pacifico", "Pacifico-Regular.ttf", "LICENSE-Pacifico.txt", script=True
    ),
    "bungee": FontSpec(
        "bungee", "Bungee", "Bungee-Regular.ttf", "LICENSE-Bungee.txt", all_caps=True
    ),
    "rye": FontSpec("rye", "Rye", "Rye-Regular.ttf", "LICENSE-Rye.txt"),
    "abril": FontSpec(
        "abril", "Abril Fatface", "AbrilFatface-Regular.ttf", "LICENSE-AbrilFatface.txt"
    ),
    "dejavu": FontSpec("dejavu", "DejaVu Sans Bold", "DejaVuSans-Bold.ttf", "LICENSE-DejaVu.txt"),
}

# The family a design falls back to.
DEFAULT_FONT: Final[str] = "bebas"


@dataclass(frozen=True)
class TextTreatment:
    """How the text of a design is drawn.

    ``tracking`` is in em: the space that is added between two letters, as a
    share of the size of the font. PIL has no tracking, so it is drawn here.
    """

    uppercase: bool = False
    tracking: float = 0.0
    # "shadow" (an offset copy behind the text), "block" (a bar behind the
    # text) or "halo" (an outline around the text).
    style: str = "shadow"
    # The word that is drawn in the accent colour.
    accent_word: int | None = None
    accent_color: str | None = None


# The type of each voice: the family of the headline and of the tagline.
VOICE_FAMILIES: Final[dict[str, tuple[str, str]]] = {
    "deadpan": ("bebas", "bebas"),
    "loud": ("anton", "bungee"),
    "cozy": ("alfa_slab", "pacifico"),
}

# The treatment of each voice.
VOICE_TREATMENTS: Final[dict[str, TextTreatment]] = {
    "deadpan": TextTreatment(uppercase=True, tracking=0.04, style="shadow"),
    "loud": TextTreatment(uppercase=True, tracking=0.08, style="shadow", accent_word=0),
    "cozy": TextTreatment(uppercase=False, tracking=0.01, style="shadow"),
}


def font_path(family: str) -> Path:
    """The file of a family. An unknown family falls back to the house font."""
    spec = FONTS.get(family)
    if spec is None:
        spec = FONTS[DEFAULT_FONT]
    return FONT_DIR / spec.filename


def family_for_voice(voice: str | None) -> str:
    """The family of the headline of a brief in that voice."""
    families = VOICE_FAMILIES.get(normalise_voice(voice))
    if families is None:
        families = VOICE_FAMILIES["deadpan"]
    return families[0]


def tagline_family_for_voice(voice: str | None) -> str:
    """The family of the tagline of a brief in that voice."""
    families = VOICE_FAMILIES.get(normalise_voice(voice))
    if families is None:
        families = VOICE_FAMILIES["deadpan"]
    return families[1]


def treatment_for_voice(voice: str | None) -> TextTreatment:
    """The treatment of the text of a brief in that voice."""
    return VOICE_TREATMENTS.get(normalise_voice(voice), VOICE_TREATMENTS["deadpan"])


def accent_for(color: str) -> str:
    """The accent colour that stands out against the colour of the text.

    A dark text gets a warm dark accent and a light text a deep one, so the
    two are always different and the accent word stays readable on the white
    paper and on the magenta of a DTF render.
    """
    digits = (color or "").lstrip("#")
    if len(digits) < 6:
        return "#C1502E"
    try:
        red = int(digits[0:2], 16)
        green = int(digits[2:4], 16)
        blue = int(digits[4:6], 16)
    except ValueError:
        return "#C1502E"

    brightness = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    return "#1F3050" if brightness > 0.5 else "#C1502E"
