"""The art direction of a design.

Every DTF brief came out as the same flat vector clip-art, because the prompt
bolted a fixed tail onto whatever the model wrote in ``visual_style``. The art
direction is a closed vocabulary now: the model picks one of the presets, and
the preset brings its own prompt, its own palette, its own negative prompt and
its own type.

Two rules hold for every preset:

- a preset may not describe a surface. A paper texture would survive the DTF
  cutout as a grey rectangle and would wreck the despill, so the grain, the
  halftone and the cracked ink are applied as an overlay after the keying, on
  the artwork only.
- the background instruction comes last in the prompt, so a look that asks for
  something else cannot win over the white paper of a print-through product
  (issue #188).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from store_listing.orchestration.typography import family_for_voice, tagline_family_for_voice

__all__ = [
    "PRESETS",
    "PRESET_IDS",
    "Preset",
    "family_for",
    "negative_for",
    "preset_for",
    "prompt_for",
    "tagline_family_for",
]


@dataclass(frozen=True)
class Preset:
    """One look a design can be drawn in."""

    id: str
    label: str
    # What the image model is told to draw.
    prompt: str
    # What the image model is told not to draw.
    negative: str
    # The colours of the look.
    palette: tuple[str, ...]
    # The type of the look: the family of the headline and of the tagline.
    headline_family: str
    tagline_family: str
    # The texture that is applied to the artwork after the keying: 'grain',
    # 'halftone', 'crackle' or nothing.
    texture: str | None = None


# The looks. The type of a look stays in the same period as the art, and the
# expressive families go on the tagline, which the OCR gate only reports on.
PRESETS: Final[dict[str, Preset]] = {
    "vintage_screenprint": Preset(
        id="vintage_screenprint",
        label="Vintage screenprint",
        prompt=(
            "vintage screenprint, three flat spot colours, faded ink, slight "
            "registration offset, bold simple shapes, no gradients"
        ),
        negative="photorealistic, 3d render, glossy, smooth gradient, neon glow",
        palette=("#F2E8D5", "#C1502E", "#1F3050", "#E0A526"),
        headline_family="alfa_slab",
        tagline_family="bebas",
        texture="grain",
    ),
    "seventies_retro": Preset(
        id="seventies_retro",
        label="Seventies retro",
        prompt=(
            "1970s retro graphic, groovy wavy shapes, warm sunset palette, thick "
            "rounded linework, flat colour"
        ),
        negative="modern, minimal, corporate, 3d render, photorealistic",
        palette=("#E2703A", "#F2B233", "#7A8B4A", "#4A2C1A"),
        headline_family="bungee",
        tagline_family="pacifico",
    ),
    "halftone_comic": Preset(
        id="halftone_comic",
        label="Halftone comic",
        prompt=(
            "comic book illustration, heavy black ink, halftone dot shading, bold "
            "linework, high contrast"
        ),
        negative="soft shading, pastel, photorealistic, gradient mesh, 3d render",
        palette=("#111111", "#FDF6E3", "#D6342C", "#FFC72C"),
        headline_family="anton",
        tagline_family="bebas",
        texture="halftone",
    ),
    "riso_two_color": Preset(
        id="riso_two_color",
        label="Riso two colour",
        prompt=(
            "risograph print, two flat ink colours, slight offset misregistration, "
            "bold simple shapes"
        ),
        negative="full colour, photorealistic, smooth gradient, 3d render",
        palette=("#FF48B0", "#0078BF", "#FFF8E7"),
        headline_family="bebas",
        tagline_family="abril",
        texture="grain",
    ),
    "hand_inked_woodcut": Preset(
        id="hand_inked_woodcut",
        label="Hand inked woodcut",
        prompt=(
            "hand-inked linocut, bold black linework, cross hatching, carved shapes, flat colour"
        ),
        negative="digital vector, clean geometric, smooth gradient, 3d render",
        palette=("#1A1A1A", "#EFE6D5", "#2F4F3E"),
        headline_family="rye",
        tagline_family="bebas",
    ),
    "kawaii_flat": Preset(
        id="kawaii_flat",
        label="Kawaii flat",
        prompt=(
            "kawaii illustration, cute rounded shapes, thick uniform outline, pastel "
            "colours, simple flat"
        ),
        negative="gritty, realistic, detailed shading, dark, 3d render",
        palette=("#FF9EC4", "#7FD8BE", "#FFE29A", "#2B2B2B"),
        headline_family="bungee",
        tagline_family="pacifico",
    ),
    "art_nouveau": Preset(
        id="art_nouveau",
        label="Art nouveau",
        prompt=(
            "art nouveau illustration, flowing organic linework, ornamental border, "
            "muted jewel tones, flat colour"
        ),
        negative="modern, geometric, neon, 3d render, photorealistic",
        palette=("#9CAF88", "#6B2737", "#C9A227", "#F5EFE0"),
        headline_family="abril",
        tagline_family="bebas",
    ),
    "distressed_band_tee": Preset(
        id="distressed_band_tee",
        label="Distressed band tee",
        prompt=(
            "distressed band tee graphic, cracked ink, heavy wear, high contrast, "
            "bold simple shapes"
        ),
        negative="clean vector, glossy, pastel, cute, 3d render",
        palette=("#2A2A2A", "#E8E1D5", "#8C1C13"),
        headline_family="anton",
        tagline_family="bebas",
        texture="crackle",
    ),
}

# The looks, in the order they are offered to the model.
PRESET_IDS: Final[tuple[str, ...]] = tuple(PRESETS)


def preset_for(name: str | None) -> Preset | None:
    """The look a brief named, or None when it named none we sell."""
    if not name:
        return None
    return PRESETS.get(name.strip().lower())


def prompt_for(preset: Preset | None) -> str:
    """What the image model is told about the look of the design."""
    return preset.prompt if preset else ""


def negative_for(preset: Preset | None, *, allow_surface: bool = False) -> str:
    """What the image model is told not to draw.

    The background is not on this list: the background instruction comes last
    in the prompt, so the magenta of a DTF render and the paper of a
    print-through product are never painted over (issue #188).

    ``allow_surface`` is for a design whose text lives on a carrier: the
    carrier is a surface in the picture, so the look must not forbid one
    (issue #246). The lettering is still forbidden: the carrier is blank.
    """
    parts = ["no paper texture", "no background texture"]
    if not allow_surface:
        parts.append("no surface")
    parts.append("no lettering")
    if allow_surface:
        # The carrier itself may be a sign; it has to stay blank. Naming the
        # lettered objects around it keeps a diner from growing a "DINER"
        # sign next to the blank one.
        parts.append("no lettering on signs, no menus with text, no shop signs with letters")
    if preset:
        parts.append(preset.negative)
    return ", ".join(parts)


def family_for(preset: Preset | None, voice: str | None) -> str:
    """The family the headline is drawn in.

    The look decides the family - the type has to be in the same period as
    the art - and the voice decides it when there is no look.
    """
    if preset:
        return preset.headline_family
    return family_for_voice(voice)


def tagline_family_for(preset: Preset | None, voice: str | None) -> str:
    """The family the tagline is drawn in."""
    if preset:
        return preset.tagline_family
    return tagline_family_for_voice(voice)
