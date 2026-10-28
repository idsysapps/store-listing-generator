"""The templates a design can be built on.

Every rendered design used to be one of three silhouettes: art in the middle
with a headline at the top, optionally a tagline at the bottom. The extra
layouts that existed were never picked, ``text_top`` was silently upgraded to
``text_top_bottom`` whenever a tagline was present, and the artwork prompt
asked for a compact centred composition every time (issue #238).

A template fixes the three things that make a design look like itself:

- how the artwork is composed (a full scene, a pattern, a tower, a roundel),
- where the headline and the tagline sit on the canvas,
- the shape of the canvas the model renders.

The batch assignment happens in code, not in the brief prompt: the model does
not get to pick the same safe geometry every day. A batch rotates through the
templates, so a batch of eight carries eight different silhouettes.

The ``layout_type`` column stores the template id, so a design that has been
rendered can be rendered again the same way. The old ids keep working and map
to a template, which means the briefs written before the templates existed
still render.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from store_listing.orchestration.copy_modes import NO_TEXT

__all__ = [
    "ART_TEMPLATE_IDS",
    "FULL_BLEED_COMPOSITIONS",
    "LEGACY_LAYOUTS",
    "TEMPLATES",
    "TEMPLATE_IDS",
    "DesignTemplate",
    "assign_templates",
    "canvas_size",
    "composition_for",
    "resolve_template",
    "template_for",
]


# The shape of a placement: (x, y, width, height) as a share of the canvas.
Placement = tuple[float, float, float, float]


@dataclass(frozen=True)
class DesignTemplate:
    """One way a design can be built."""

    id: str
    label: str
    # What the image model is told to draw, including the space to leave clear
    # for the text.
    composition: str
    # Where the headline and the tagline go, in shares of the canvas.
    headline: Placement | None = None
    tagline: Placement | None = None
    # The size the model renders, and the compositor draws on.
    size: tuple[int, int] = (1024, 1024)
    # Whether a band is drawn behind the text, so it stays readable on a busy
    # full-bleed composition.
    bar: bool = False
    # "center" or "left".
    align: str = "center"
    # A tagline that follows the curve of a roundel. The tagline is secondary,
    # so a vision model that cannot read a curve does not fail the render.
    arc_tagline: bool = False


TEMPLATES: Final[dict[str, DesignTemplate]] = {
    "emblem": DesignTemplate(
        id="emblem",
        label="Emblem",
        composition=(
            "compact centred emblem inside the middle of the canvas, with a clear "
            "empty band across the top and a clear empty band across the bottom: "
            "plain background only in those bands"
        ),
        headline=(0.06, 0.02, 0.88, 0.13),
        tagline=(0.08, 0.85, 0.84, 0.12),
    ),
    "hero_top": DesignTemplate(
        id="hero_top",
        label="Hero on top",
        composition=(
            "the artwork fills the upper two thirds of the canvas and is cropped by "
            "the left, right and top edges, and a clear empty band is left across the "
            "bottom third: plain background only in that band"
        ),
        headline=(0.07, 0.66, 0.86, 0.16),
        tagline=(0.07, 0.83, 0.86, 0.11),
        align="left",
    ),
    "hero_bottom": DesignTemplate(
        id="hero_bottom",
        label="Hero below the type",
        composition=(
            "the artwork fills the lower two thirds of the canvas and is cropped by "
            "the left, right and bottom edges, and a clear empty band is left across "
            "the top third: plain background only in that band"
        ),
        headline=(0.06, 0.03, 0.88, 0.14),
        tagline=(0.06, 0.17, 0.88, 0.10),
    ),
    "side_column": DesignTemplate(
        id="side_column",
        label="Art and side column",
        composition=(
            "the artwork fills the left half of the canvas and is cropped by the left "
            "edge, and the right side of the canvas is clear and empty: plain "
            "background only on the right side"
        ),
        headline=(0.58, 0.28, 0.37, 0.26),
        tagline=(0.58, 0.56, 0.37, 0.18),
        align="left",
    ),
    "wide_scene": DesignTemplate(
        id="wide_scene",
        label="Wide scene",
        composition=(
            "a wide full-bleed scene that fills the canvas edge to edge, with the "
            "horizon low and the subject off centre, and the bottom-left corner clear "
            "and empty: plain background only there"
        ),
        headline=(0.06, 0.72, 0.56, 0.14),
        tagline=(0.06, 0.87, 0.56, 0.09),
        size=(1216, 832),
        align="left",
    ),
    "tall_arch": DesignTemplate(
        id="tall_arch",
        label="Tall arch",
        composition=(
            "the artwork is a tall arch shape that fills most of the canvas, and a "
            "clear empty band is left at the top and at the bottom: plain background "
            "only in those bands"
        ),
        headline=(0.22, 0.04, 0.56, 0.12),
        tagline=(0.10, 0.88, 0.80, 0.09),
        size=(832, 1216),
    ),
    "mid_ribbon": DesignTemplate(
        id="mid_ribbon",
        label="Ribbon across the middle",
        composition=(
            "the artwork fills the whole canvas edge to edge and a clear horizontal "
            "band is left across the middle: plain background only in that band"
        ),
        headline=(0.05, 0.40, 0.90, 0.18),
        tagline=(0.10, 0.87, 0.80, 0.10),
        bar=True,
    ),
    "pattern_band": DesignTemplate(
        id="pattern_band",
        label="All-over pattern",
        composition=(
            "an all-over repeating pattern of the motif that covers the whole canvas "
            "edge to edge, and a clear band is left across the lower third: plain "
            "background only in that band"
        ),
        headline=(0.05, 0.70, 0.90, 0.16),
        tagline=(0.10, 0.87, 0.80, 0.10),
        bar=True,
    ),
    "tower": DesignTemplate(
        id="tower",
        label="Tower",
        composition=(
            "the motif stacked vertically as a tall column on the right of the canvas "
            "that is cropped by the top and bottom edges, and the left third of the "
            "canvas is clear and empty: plain background only on the left"
        ),
        headline=(0.03, 0.12, 0.30, 0.58),
        tagline=(0.03, 0.74, 0.30, 0.14),
        size=(832, 1216),
        align="left",
    ),
    "roundel": DesignTemplate(
        id="roundel",
        label="Roundel",
        composition=(
            "the artwork inside a bold circle in the centre of the canvas, with the "
            "four corners clear and empty: plain background only in the corners"
        ),
        headline=(0.10, 0.03, 0.80, 0.12),
        tagline=(0.12, 0.85, 0.76, 0.11),
        arc_tagline=True,
    ),
    "text_only": DesignTemplate(
        id="text_only",
        label="Typography only",
        composition="",
        headline=(0.05, 0.25, 0.90, 0.30),
        tagline=(0.05, 0.55, 0.90, 0.20),
    ),
    "full_bleed": DesignTemplate(
        id="full_bleed",
        label="Art only",
        composition="",
    ),
}

TEMPLATE_IDS: Final[tuple[str, ...]] = tuple(TEMPLATES)

# The templates that carry text and art. ``full_bleed`` has no copy and
# ``text_only`` has no art, so neither of them is part of the rotation.
ART_TEMPLATE_IDS: Final[tuple[str, ...]] = tuple(
    template_id for template_id in TEMPLATE_IDS if template_id not in ("full_bleed", "text_only")
)

# The ids the briefs and the pipeline used before the templates existed.
LEGACY_LAYOUTS: Final[dict[str, str]] = {
    "text_top": "hero_bottom",
    "text_top_bottom": "emblem",
    "badge": "roundel",
    "banner": "mid_ribbon",
    "arch": "tall_arch",
    "stack_side": "side_column",
}

# The compositions an art-only design can be drawn in. The concept decides
# which one, so two art-only briefs of a batch do not come back as the same
# centred object.
FULL_BLEED_COMPOSITIONS: Final[tuple[str, ...]] = (
    "a wide full-bleed scene that fills the canvas edge to edge, no border and no empty margin",
    "the subject cropped by the frame and filling the whole canvas, no border and no empty margin",
    (
        "an all-over repeating pattern that covers the whole canvas edge to edge, no border "
        "and no empty margin"
    ),
    (
        "the motif repeated and scattered across the whole canvas edge to edge, no border and "
        "no empty margin"
    ),
)


def resolve_template(
    layout_type: str | None,
    headline_text: str | None = None,
    tagline_text: str | None = None,
) -> str:
    """The template a layout id means.

    A template id is returned unchanged, a legacy id is mapped, and anything
    unknown falls back to a text template rather than dropping the text.
    """
    if not layout_type:
        return "emblem"
    if layout_type in TEMPLATES:
        return layout_type
    if layout_type in LEGACY_LAYOUTS:
        return LEGACY_LAYOUTS[layout_type]
    return "emblem"


def template_for(layout_type: str | None) -> DesignTemplate:
    """The template a layout id means."""
    return TEMPLATES[resolve_template(layout_type)]


def canvas_size(template_id: str | None) -> tuple[int, int]:
    """The size the template is rendered at."""
    return template_for(template_id).size


def composition_for(template_id: str | None, concept: str = "") -> str:
    """What the image model is told to draw for a template.

    An art-only design has no text placing it, so its composition is picked
    from the full-bleed set by the concept: the same concept always gets the
    same composition, and a batch of art-only designs does not all come back
    as the same centred object.
    """
    template = template_for(template_id)
    if template.id == "full_bleed":
        digest = hashlib.sha1(concept.strip().lower().encode()).digest()
        return FULL_BLEED_COMPOSITIONS[digest[0] % len(FULL_BLEED_COMPOSITIONS)]
    return template.composition


class _HasCopy(Protocol):
    @property
    def concept(self) -> str: ...

    @property
    def headline_text(self) -> str | None: ...

    @property
    def tagline_text(self) -> str | None: ...

    @property
    def layout_type(self) -> str: ...


def _has_copy(brief: _HasCopy) -> bool:
    if getattr(brief, "copy_mode", "caption") == NO_TEXT:
        return False
    return bool((brief.headline_text or "").strip() or (brief.tagline_text or "").strip())


def _template_cycle(count: int, batch_id: str) -> list[str]:
    """``count`` template ids, seeded by the batch, no template more than twice.

    A full pass over the templates comes first, so a batch of eight gets eight
    different silhouettes. Only a batch larger than two full passes repeats a
    template a third time.
    """
    rng = random.Random(f"{batch_id}|{count}")
    picks: list[str] = []
    while len(picks) < count:
        pass_ = list(ART_TEMPLATE_IDS)
        rng.shuffle(pass_)
        picks.extend(pass_)
    return picks[:count]


def assign_templates(briefs: Sequence[_HasCopy], batch_id: str = "") -> list[str]:
    """The template id for every brief of a batch, in the briefs' order.

    The model decides whether a design has copy, no copy at all, or only copy
    (``text_only``). It does not decide the geometry: that is rotated by the
    batch, so the shop does not look like one design repeated.
    """
    copy_count = sum(1 for brief in briefs if brief.layout_type != "text_only" and _has_copy(brief))
    cycle = _template_cycle(copy_count, batch_id)

    assigned: list[str] = []
    index = 0
    for brief in briefs:
        if brief.layout_type == "text_only":
            assigned.append("text_only")
            continue
        if not _has_copy(brief):
            assigned.append("full_bleed")
            continue
        assigned.append(cycle[index])
        index += 1
    return assigned
