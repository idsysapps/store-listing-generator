"""Model-drawn text for plans the compositor cannot execute (#133).

A "painted" text plan asks for the words to live inside the artwork: painted
into the smoke, in the same hand as the brushwork. The compositor can only
overlay a font on top of the illustration, which is why a memorial design came
out with a slab line across the dog and a detached tagline.

Qwen-Image can draw the scene and the line in one pass. This module builds
that prompt, runs a candidate or two, and refuses anything that does not pass
the same strict OCR gate the typographic hero uses. The descriptor (tagline)
stays out of the prompt: Qwen garbles a third block, so the pipeline
composites it afterwards as it does for a hero.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Protocol

from store_listing.orchestration.comfyui_client import (
    QWEN_SIZE,
    ComfyUIError,
    ComfyUIResult,
)
from store_listing.orchestration.copy_modes import NO_TEXT
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.text_validation import StrictTextResult

__all__ = [
    "ModelTextCopy",
    "ModelTextGenerator",
    "ModelTextResult",
    "build_model_text_prompt",
    "is_model_text_brief",
    "model_text_copy_from_brief",
    "model_text_enabled",
]

ENABLED_ENV = "MODEL_TEXT_ENABLED"
DEFAULT_CANDIDATES = 2
DEFAULT_BACKGROUND = "plain white background"
DTF_BACKGROUND = "solid pure magenta background, chroma key backdrop"

# The methods that ask the image model to make the words part of the art.
# The compositor can draw a printed label or a sticker well; a painted line is
# the one that reads as an overlay no matter which font it gets.
MODEL_TEXT_METHODS: frozenset[str] = frozenset({"painted"})

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_VOICE_STYLES: dict[str, str] = {
    "cozy": "soft, warm, hand-painted lettering",
    "loud": "bold, energetic brush lettering",
    "deadpan": "clean, dry hand-lettered lettering",
}


class QwenImageClient(Protocol):
    def generate_qwen_image(
        self,
        *,
        prompt: str,
        seed: int | None = None,
        width: int = QWEN_SIZE,
        height: int = QWEN_SIZE,
    ) -> ComfyUIResult: ...


class TextValidator(Protocol):
    def validate(self, image_bytes: bytes, *, expected: list[str]) -> StrictTextResult: ...


def model_text_enabled() -> bool:
    """Whether the pipeline may route a painted brief through the image model."""
    value = os.environ.get(ENABLED_ENV, "").strip().lower()
    return value in _TRUTHY


def is_model_text_brief(brief: DesignBrief) -> bool:
    """Whether the words of this brief belong in the artwork, not on top of it.

    A carrier is the compositor's home ground: a blank sign or label is a flat
    shape it can fill. The model path is for a line the scene has to absorb.
    """
    if brief.copy_mode == NO_TEXT:
        return False
    plan = brief.text_plan
    if plan is None or plan.method not in MODEL_TEXT_METHODS:
        return False
    if plan.needs_carrier:
        return False
    return bool((brief.headline_text or "").strip())


@dataclass(frozen=True)
class ModelTextCopy:
    """The line the image model draws, and the style it draws it in."""

    headline: str
    descriptor: str | None = None
    subject: str = ""
    style: str = "hand-painted lettering"
    integration: str = "painted into the artwork"
    ink: str = "dark lettering"
    background: str = DEFAULT_BACKGROUND

    def __post_init__(self) -> None:
        if not self.headline or not self.headline.strip():
            raise ValueError("A model-drawn text design needs a headline")

    @property
    def text_blocks(self) -> tuple[str, ...]:
        """The blocks the image model draws. The descriptor is composited."""
        return (self.headline.strip(),)


@dataclass(frozen=True)
class ModelTextResult:
    """Outcome of one model-text generation run."""

    image_bytes: bytes | None
    prompt: str
    copy: ModelTextCopy
    seed: int | None
    attempts: int
    validation: StrictTextResult | None = None
    error: str | None = None


def _voice_style(brief: DesignBrief) -> str:
    return _VOICE_STYLES.get((brief.voice or "").strip().lower(), "hand-painted lettering")


def _look_style(brief: DesignBrief) -> str:
    look = f"{brief.visual_style or ''} {brief.style_preset or ''}".lower()
    if "screenprint" in look or "screen print" in look:
        return " with a vintage screenprint texture"
    if "riso" in look:
        return " with a riso print texture"
    if "kawaii" in look:
        return " with rounded, playful shapes"
    return ""


def _ink_phrase(ink: str) -> str:
    value = (ink or "").strip().lower()
    if value == "light":
        return "light-colored lettering"
    if value == "dark":
        return "dark lettering"
    if value.startswith("#"):
        return f"lettering in {ink}"
    return "lettering"


def model_text_copy_from_brief(brief: DesignBrief) -> ModelTextCopy:
    """Map the brief and its text plan onto the one line Qwen draws."""
    headline = (brief.headline_text or "").strip()
    if not headline:
        raise ValueError("The brief has no headline_text to draw")

    plan = brief.text_plan
    integration = (
        (plan.reason or "").strip() if plan is not None else ""
    ) or "painted into the artwork"
    ink = _ink_phrase(plan.ink if plan is not None else "dark")
    subject = strip_lettering((brief.scene_description or "").strip()) or "the design"
    background = DTF_BACKGROUND if brief.product_type == "dtf_apparel" else DEFAULT_BACKGROUND
    descriptor = (brief.tagline_text or "").strip() or None

    return ModelTextCopy(
        headline=headline,
        descriptor=descriptor,
        subject=subject,
        style=_voice_style(brief) + _look_style(brief),
        integration=integration,
        ink=ink,
        background=background,
    )


def build_model_text_prompt(copy: ModelTextCopy) -> str:
    """Describe one scene with its line painted into it, and nothing else."""
    headline = _clean_copy(copy.headline)
    subject = _clean_copy(copy.subject) or "the design"
    style = _clean_copy(copy.style) or "hand-painted lettering"
    integration = _clean_copy(copy.integration) or "painted into the artwork"

    return (
        f"A die-cut sticker design of {subject}, with the words "
        f'"{headline}" {style}, {integration}. '
        "The art flows around the letters without covering any letterform; "
        "keep every letter fully visible and readable. "
        f"{copy.ink.capitalize()}, {copy.background}, generous margin, the "
        "words are printed plain with no quotation marks and no extra "
        "punctuation, no other text."
    )


class ModelTextGenerator:
    """Generate model-drawn text until one candidate passes the strict gate."""

    def __init__(
        self,
        *,
        comfyui_client: QwenImageClient,
        validator: TextValidator,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        self._client = comfyui_client
        self._validator = validator
        self._candidates = candidates

    def generate(self, copy: ModelTextCopy, *, seed: int | None = None) -> ModelTextResult:
        prompt = build_model_text_prompt(copy)
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
                return ModelTextResult(
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

        return ModelTextResult(
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
