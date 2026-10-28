"""How a design carries its words.

The geometry of a design is one decision and the words are another. A brief
used to have only ``layout_type``, so the only options were "art with a
caption", "art with no words" or "words with no art". The reference design is
none of those: it is a stacked emblem whose hero word *is* the illustration,
with a small kicker above it and a descriptor below. Qwen-Image can draw that
hero word, Flux cannot, and a self-explaining illustration should not have a
caption bolted on at all (issue #258).

``copy_mode`` names the third decision:

- ``type_hero``: Qwen draws the kicker and the hero word, the subject weaves
  through the letters, and the descriptor is composited afterwards.
- ``caption``: Flux draws the artwork and the compositor adds the headline and
  the tagline. This is the path every copied design used before.
- ``no_text``: Flux draws an illustration that explains itself. There is no
  compositor and no text gate; the inverted lettering gate is the only check.

The mode is kept separate from ``layout_type`` because the template is still
geometry: a hero brief may carry a normal template so it can fall back to a
caption render when the Qwen path is off.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Final, Protocol, cast

__all__ = [
    "CAPTION",
    "DEFAULT_MAX_HERO_FRACTION",
    "HERO_LAYOUT_TYPES",
    "MAX_HERO_WORDS",
    "MAX_KICKER_WORDS",
    "NO_TEXT",
    "TYPE_HERO",
    "VALID_COPY_MODES",
    "assign_batch_copy_modes",
    "copy_mode_from_row",
    "normalise_copy_mode",
    "normalise_kicker",
]

logger = logging.getLogger(__name__)

TYPE_HERO: Final[str] = "type_hero"
CAPTION: Final[str] = "caption"
NO_TEXT: Final[str] = "no_text"

VALID_COPY_MODES: Final[tuple[str, ...]] = (TYPE_HERO, CAPTION, NO_TEXT)

# The layout ids that meant "stacked wordmark" before copy_mode existed. They
# stay valid so briefs written before the column existed still render.
HERO_LAYOUT_TYPES: Final[frozenset[str]] = frozenset({"type_hero", "wordmark_stack"})

# A hero word becomes the illustration, so it has to be short enough to draw
# giant. Longer lines stay captions.
MAX_HERO_WORDS: Final[int] = 3
# A kicker is a small line above the hero word.
MAX_KICKER_WORDS: Final[int] = 6

# The batch keeps at least one hero brief and never lets hero copy take over.
DEFAULT_MAX_HERO_FRACTION: Final[float] = 0.5


def _has_text(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


def normalise_copy_mode(
    raw: str | None,
    *,
    layout_type: str | None = None,
    headline_text: str | None = None,
    tagline_text: str | None = None,
) -> str:
    """The one mode a brief actually has, from what the model answered.

    A model can name a mode that contradicts its copy. A hero with no headline
    cannot be a hero, and a caption with no copy has nothing to caption, so the
    answer is corrected here instead of producing an unrenderable brief.
    """
    layout = (layout_type or "").strip()
    has_headline = _has_text(headline_text)
    has_tagline = _has_text(tagline_text)

    if raw in VALID_COPY_MODES:
        mode = raw
    elif layout in HERO_LAYOUT_TYPES:
        mode = TYPE_HERO
    elif layout == "text_only":
        # A text-only design always has words, even when the model forgot the
        # headline; the parser falls back to the concept.
        mode = CAPTION
    elif layout == "full_bleed" or (not has_headline and not has_tagline):
        # A legacy art-only row may still carry a stale headline from an
        # earlier plan. The row is art-only, so the stale copy goes.
        mode = NO_TEXT
    else:
        mode = CAPTION

    if mode == TYPE_HERO:
        if has_headline:
            return TYPE_HERO
        return CAPTION if has_tagline else NO_TEXT
    if mode == NO_TEXT:
        return NO_TEXT
    if layout == "text_only":
        return CAPTION
    return CAPTION if (has_headline or has_tagline) else NO_TEXT


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def copy_mode_from_row(row: Mapping[str, Any]) -> str:
    """The copy mode of a stored brief, including rows written before it existed.

    Legacy art-only rows become ``no_text``; legacy wordmark layouts become
    ``type_hero``; every other row with copy is a caption. This is why the
    column can be added without re-rendering the catalogue.
    """
    raw = row.get("copy_mode")
    if isinstance(raw, str) and raw in VALID_COPY_MODES:
        return raw
    layout = _text_or_none(row.get("layout_type"))
    if layout not in HERO_LAYOUT_TYPES and row.get("text_plan"):
        # A legacy row with a text plan is not art-only: the plan was written
        # to put words in the picture, so it is a caption.
        return CAPTION
    return normalise_copy_mode(
        None,
        layout_type=layout,
        headline_text=_text_or_none(row.get("headline_text")),
        tagline_text=_text_or_none(row.get("tagline_text")),
    )


def normalise_kicker(raw: Any, *, copy_mode: str) -> str | None:
    """The kicker of a hero brief, or None when it is not one.

    Only the hero path renders a kicker. A caption or no-text brief that came
    back with one drops it rather than storing copy it will never draw.
    """
    if copy_mode != TYPE_HERO or not isinstance(raw, str):
        return None
    text = " ".join(raw.split())
    if not text or len(text.split()) > MAX_KICKER_WORDS:
        return None
    return text


class CopyModeBrief(Protocol):
    @property
    def concept(self) -> str: ...

    @property
    def confidence(self) -> int: ...

    @property
    def copy_mode(self) -> str: ...

    @property
    def headline_text(self) -> str | None: ...

    @property
    def scene_description(self) -> str | None: ...

    @property
    def kicker_text(self) -> str | None: ...


def _with_mode[BriefT: CopyModeBrief](
    brief: BriefT,
    mode: str,
    *,
    layout_type: str | None = None,
) -> BriefT:
    if layout_type is None:
        return cast(BriefT, replace(cast(Any, brief), copy_mode=mode, kicker_text=None))
    return cast(
        BriefT,
        replace(
            cast(Any, brief),
            copy_mode=mode,
            kicker_text=None,
            layout_type=layout_type,
        ),
    )


def _fallback_layout(brief: CopyModeBrief) -> str | None:
    """The layout a brief needs when its copy mode changes.

    A caption can keep a normal template as its fallback when the hero path is
    off. A hero layout is not a caption template, and an art-only layout is not
    a caption either, so those become 'auto' and the batch assigns one.
    """
    layout = (getattr(brief, "layout_type", "") or "").strip()
    if layout in HERO_LAYOUT_TYPES or layout in ("full_bleed", "text_only"):
        return "auto"
    return None


def _can_be_hero(brief: CopyModeBrief) -> bool:
    headline = (brief.headline_text or "").strip()
    scene = (brief.scene_description or "").strip()
    return bool(headline and scene and len(headline.split()) <= MAX_HERO_WORDS)


def assign_batch_copy_modes[BriefT: CopyModeBrief](
    briefs: Sequence[BriefT],
    *,
    max_hero_fraction: float = DEFAULT_MAX_HERO_FRACTION,
) -> list[BriefT]:
    """Keep a batch from being one copy mode, without forcing a no-text design.

    A batch with no hero brief gets one of its shortest captions promoted, so
    the reference style is exercised; a batch that came back all hero is cut
    back to half. No-text is never forced: it only works when the illustration
    explains itself, and the prompt is the place that decides that.
    """
    result = list(briefs)
    if not result:
        return result

    fraction = min(1.0, max(0.0, max_hero_fraction))
    max_hero = max(1, int(len(result) * fraction))

    heroes = [index for index, brief in enumerate(result) if brief.copy_mode == TYPE_HERO]
    if len(heroes) > max_hero:
        ranked = sorted(
            heroes, key=lambda index: (-result[index].confidence, result[index].concept)
        )
        for index in ranked[max_hero:]:
            logger.info(
                "Demoting brief %r from type_hero: the batch already has %d hero briefs",
                result[index].concept,
                max_hero,
            )
            result[index] = _with_mode(
                result[index],
                CAPTION,
                layout_type=_fallback_layout(result[index]),
            )

    if len(result) >= 2 and not any(brief.copy_mode == TYPE_HERO for brief in result):
        candidates = [
            index
            for index, brief in enumerate(result)
            if brief.copy_mode == CAPTION and _can_be_hero(brief)
        ]
        if candidates:
            best = min(
                candidates, key=lambda index: (-result[index].confidence, result[index].concept)
            )
            logger.info(
                "Promoting brief %r to type_hero: the batch had no typographic hero",
                result[best].concept,
            )
            result[best] = _with_mode(
                result[best],
                TYPE_HERO,
                layout_type=_fallback_layout(result[best]),
            )

    return result
