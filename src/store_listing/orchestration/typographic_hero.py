"""Single-pass typographic hero designs on Qwen-Image.

The reference design (``32_transparent.png``) is not a caption under artwork.
It is a stacked emblem: a small kicker above, a giant hero word that *is* the
illustration, and the subject weaving through the letters. Flux cannot spell,
so it draws empty bands for the compositor to fill. Qwen-Image can draw the
kicker and the hero word itself, and the Phase 0 bake-off showed it spells
them correctly (issue #254).

The descriptor stays out of this path. Qwen could not render three blocks in
one pass without garbling the third, so the descriptor is composited
deterministically afterwards. This module builds the two-block prompt, runs a
candidate or two, and refuses anything that does not pass a strict OCR gate.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from store_listing.orchestration.comfyui_client import (
    QWEN_SIZE,
    ComfyUIError,
    ComfyUIResult,
)
from store_listing.orchestration.copy_modes import HERO_LAYOUT_TYPES, NO_TEXT, TYPE_HERO
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.text_validation import StrictTextResult

__all__ = [
    "HeroCopy",
    "HeroResult",
    "TypographicHeroGenerator",
    "build_typographic_hero_prompt",
    "hero_candidate_count",
    "hero_copy_from_brief",
    "is_typographic_hero_brief",
    "typographic_hero_enabled",
]

DEFAULT_STYLE = "retro 1970s bubble lettering"
DEFAULT_PALETTE: tuple[str, ...] = ("lavender", "orange", "black", "white")
DEFAULT_CANDIDATES = 2
ENABLED_ENV = "TYPOGRAPHIC_HERO_ENABLED"
BRIEF_IDS_ENV = "TYPOGRAPHIC_HERO_BRIEF_IDS"
CANDIDATES_ENV = "TYPOGRAPHIC_HERO_CANDIDATES"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class QwenImageClient(Protocol):
    def generate_qwen_image(
        self,
        *,
        prompt: str,
        seed: int | None = None,
        width: int = QWEN_SIZE,
        height: int = QWEN_SIZE,
    ) -> ComfyUIResult: ...


class HeroTextValidator(Protocol):
    def validate(self, image_bytes: bytes, *, expected: list[str]) -> StrictTextResult: ...


def typographic_hero_enabled() -> bool:
    """Whether the pipeline may route briefs through the Qwen hero path."""
    value = os.environ.get(ENABLED_ENV, "").strip().lower()
    return value in _TRUTHY


def hero_brief_ids() -> frozenset[int]:
    """Briefs explicitly opted into the hero path for a trial batch."""
    raw = os.environ.get(BRIEF_IDS_ENV, "")
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            continue
    return frozenset(ids)


def hero_candidate_count() -> int:
    """How many Qwen renders to try before giving up on a brief."""
    raw = os.environ.get(CANDIDATES_ENV, str(DEFAULT_CANDIDATES))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_CANDIDATES


def is_typographic_hero_brief(brief_row: Mapping[str, Any], brief: DesignBrief) -> bool:
    """Whether this brief should be drawn by the typographic hero path.

    A hero copy mode is the normal route now. A dedicated layout type still
    works for the briefs written before the mode existed, and a trial batch can
    opt specific briefs in by id. A no-text design is never a hero: it has no
    words to draw.
    """
    if brief.copy_mode == NO_TEXT:
        return False
    if not (brief.headline_text or "").strip():
        return False
    if brief.copy_mode == TYPE_HERO:
        return True
    if (brief.layout_type or "").strip() in HERO_LAYOUT_TYPES:
        return True
    return brief_row.get("id") in hero_brief_ids()


@dataclass(frozen=True)
class HeroCopy:
    """The copy blocks the hero image renders.

    ``descriptor`` is deliberately not rendered by Qwen. It is kept on the
    object so the compositor can add it to the same artwork later.
    """

    hero: str
    kicker: str | None = None
    descriptor: str | None = None
    subject: str = ""
    style: str = DEFAULT_STYLE
    palette: tuple[str, ...] = DEFAULT_PALETTE

    def __post_init__(self) -> None:
        if not self.hero or not self.hero.strip():
            raise ValueError("A typographic hero needs a hero word")
        if not self.palette:
            raise ValueError("A typographic hero needs at least one palette colour")

    @property
    def text_blocks(self) -> tuple[str, ...]:
        """The blocks the image model draws, in reading order."""
        blocks = (self.kicker, self.hero)
        return tuple(block.strip() for block in blocks if block and block.strip())


@dataclass(frozen=True)
class HeroResult:
    """Outcome of one hero generation run."""

    image_bytes: bytes | None
    prompt: str
    copy: HeroCopy
    seed: int | None
    attempts: int
    validation: StrictTextResult | None = None
    error: str | None = None


def build_typographic_hero_prompt(copy: HeroCopy) -> str:
    """Describe a kicker + hero emblem, and nothing else.

    The prompt asks for one pass with the subject interlocked with the hero
    letters, which is the trait that makes the reference look designed rather
    than captioned. It never mentions the descriptor: asking Qwen for a third
    block garbled it in the Phase 0 tests.
    """
    hero = _clean_copy(copy.hero)
    kicker = _clean_copy(copy.kicker) if copy.kicker else ""
    subject = _clean_copy(copy.subject)
    palette = [colour.strip() for colour in copy.palette if colour.strip()]
    first, second = palette[0], palette[1]

    if subject:
        opening = f"A die-cut sticker design of {subject}."
    else:
        opening = "A die-cut sticker wordmark."

    parts = [opening]
    if kicker:
        parts.append(f'At the top, the words "{kicker}" in small {first} bubble letters.')
    if subject:
        parts.append(
            f'Across the middle, the word "{hero}" in giant {second} bubble '
            "letters, with the subject woven around and between the letters. "
            "Keep every letter fully visible and readable: the subject must not "
            "cover any letterform."
        )
    else:
        parts.append(f'Across the middle, the word "{hero}" in giant {second} bubble letters.')

    parts.append(
        "Thick black outline, hard offset shadow, flat "
        f"{' '.join(palette)} palette, plain white background, generous margin, "
        "no other text."
    )
    return " ".join(parts)


def hero_copy_from_brief(brief: DesignBrief) -> HeroCopy:
    """Map the brief's copy onto the two blocks Qwen renders.

    ``kicker_text`` is part of the brief now. The scene is stripped of any
    quoted lettering, because this path spells its own words and must not
    inherit a caption from the scene.
    """
    hero = (brief.headline_text or "").strip()
    if not hero:
        raise ValueError("The brief has no headline_text to use as the hero word")

    kicker = brief.kicker_text
    kicker = kicker.strip() if isinstance(kicker, str) and kicker.strip() else None
    descriptor = (brief.tagline_text or "").strip() or None
    subject = strip_lettering((brief.scene_description or "").strip())

    return HeroCopy(
        hero=hero,
        kicker=kicker,
        descriptor=descriptor,
        subject=subject,
    )


class TypographicHeroGenerator:
    """Generate hero emblems until one passes the strict text gate."""

    def __init__(
        self,
        *,
        comfyui_client: QwenImageClient,
        validator: HeroTextValidator,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        self._client = comfyui_client
        self._validator = validator
        self._candidates = candidates

    def generate(self, copy: HeroCopy, *, seed: int | None = None) -> HeroResult:
        prompt = build_typographic_hero_prompt(copy)
        base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        errors: list[str] = []
        last_validation: StrictTextResult | None = None

        for attempt in range(self._candidates):
            attempt_seed = base_seed + attempt
            try:
                result = self._client.generate_qwen_image(
                    prompt=prompt,
                    seed=attempt_seed,
                )
            except ComfyUIError as error:
                errors.append(f"attempt {attempt + 1}: {error}")
                continue

            if result.image_bytes is None:
                errors.append(f"attempt {attempt + 1}: {result.error or 'no image returned'}")
                continue

            validation = self._validator.validate(
                result.image_bytes, expected=list(copy.text_blocks)
            )
            last_validation = validation
            if validation.matched:
                return HeroResult(
                    image_bytes=result.image_bytes,
                    prompt=prompt,
                    copy=copy,
                    seed=attempt_seed,
                    attempts=attempt + 1,
                    validation=validation,
                )

            errors.append(
                f"attempt {attempt + 1}: text gate rejected the image "
                f"(missing={validation.missing}, duplicated={validation.duplicated})"
            )

        return HeroResult(
            image_bytes=None,
            prompt=prompt,
            copy=copy,
            seed=base_seed,
            attempts=self._candidates,
            validation=last_validation,
            error="No candidate passed the text gate: " + "; ".join(errors),
        )


def _clean_copy(text: str | None) -> str:
    """Make copy safe to quote inside the prompt."""
    if not text:
        return ""
    without_quotes = text.replace('"', "")
    return " ".join(without_quotes.split())
