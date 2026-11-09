"""Check that a generated image is flat print artwork and not a product.

The text gate reads the words back out of an image. Nothing looked at the
picture itself, so a brief whose scene named a bag produced an image of a
bag, and that bag was uploaded as the print file (issue #195).

The verdict is deliberately coarse - the image is either a flat graphic or it
is a product. An answer that cannot be read is ``unknown`` and the image is
kept: a vision model that is down must not throw away good designs.

The artwork must not carry words either. The image model cannot spell, so a
lettering that it drew is a lettering that is wrong: the bubble on the image
of brief #100 read "To get to the onthe sid" (issue #208). The text of the
design is drawn on afterwards with a font, so the artwork has to be free of it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from store_listing.orchestration.text_validation import encode_image_for_vision

logger = logging.getLogger(__name__)

__all__ = [
    "ARTWORK",
    "ARTWORK_CLASSIFICATION_PROMPT",
    "ARTWORK_TEXT_PROMPT",
    "HAS_TEXT",
    "NO_LETTERING",
    "PRODUCT",
    "UNKNOWN",
    "ArtworkTextResult",
    "ArtworkTextValidator",
    "ArtworkValidationResult",
    "ArtworkValidator",
    "NoTextResult",
    "parse_text_verdict",
    "parse_verdict",
]

ARTWORK = "artwork"
PRODUCT = "product"
UNKNOWN = "unknown"
HAS_TEXT = "text"
NO_LETTERING = "nolettering"

ARTWORK_CLASSIFICATION_PROMPT = (
    "Look at this image. Is it a flat print graphic - a design that could be printed "
    "on a product - or is it a product itself or a photo/mockup of a product (a bag, "
    "a tote bag, a pouch, a shirt, a hoodie, a mug, a tumbler, a phone case, a pillow, "
    "a blanket, a sticker sheet, or a photo of a printed item)? A flat print "
    "illustration that depicts an object, such as a coffee cup, is still ARTWORK. "
    "It is PRODUCT only when the image is a product photo or mockup with the object "
    "as the product, or a photo of a printed item. Answer with exactly one word: "
    "ARTWORK or PRODUCT."
)

# Alphabetic marks that belong on a phone key, spelled the way the key is printed.
CORRECT_KEY_LABELS = frozenset({"OK", "ABC", "DEF", "GHI", "JKL", "MNO", "PQRS", "TUV", "WXYZ"})

ARTWORK_TEXT_PROMPT = (
    "Look at this image. Does it contain a sign, a title, a sentence, or "
    "handwriting that is not printed on the object itself? Digits on a keypad, "
    "dial, or keys are part of the object, not lettering. A short label printed "
    "on a key is part of the keypad only when it is spelled correctly, such as "
    "OK. A misspelled key label, such as 0K or OKK, is lettering. Copy each "
    "alphabetic key label exactly as printed, after the verdict. Answer TEXT "
    "if it has lettering, otherwise NOTEXT. Example: NOTEXT OK."
)


@dataclass
class ArtworkValidationResult:
    """Verdict for a single generated image."""

    is_product: bool
    verdict: str
    error: str | None = None


@dataclass
class ArtworkTextResult:
    """Whether the artwork carries lettering of its own."""

    has_text: bool
    verdict: str
    error: str | None = None


@dataclass
class NoTextResult:
    """Whether a no-text design really carries no words.

    ``matched`` is true only for an explicit ``NOTEXT`` verdict. The ordinary
    lettering gate keeps an unreadable answer because the compositor will add
    the copy anyway; a no-text design has no compositor, so an unreadable
    answer is not proof that the artwork is safe to print.
    """

    matched: bool
    verdict: str
    error: str | None = None


def parse_verdict(answer: str) -> str:
    """Read the verdict out of the model's answer.

    The model is asked for a single word, so the first word is the answer.
    A longer answer is scanned for the keyword, and anything that cannot be
    read is ``unknown``.
    """
    cleaned = answer.strip().upper()
    if not cleaned:
        return UNKNOWN

    words = re.findall(r"[A-Z]+", cleaned)
    if words:
        if words[0] == "ARTWORK":
            return ARTWORK
        if words[0] == "PRODUCT":
            return PRODUCT

    if "ARTWORK" in cleaned:
        return ARTWORK
    if "PRODUCT" in cleaned:
        return PRODUCT
    return UNKNOWN


def misspelled_key_label(answer: str) -> bool:
    """Whether a quoted key label is not one of the real keypad spellings."""
    tokens = re.findall(r"[A-Z0-9]+", answer.strip().upper())
    labels = [
        token for token in tokens if token not in {"TEXT", "NOTEXT", "NO"} and not token.isdigit()
    ]
    return any(label not in CORRECT_KEY_LABELS for label in labels)


def parse_text_verdict(answer: str) -> str:
    """Read the lettering verdict out of the model's answer."""
    cleaned = answer.strip().upper()
    if not cleaned:
        return UNKNOWN

    words = re.findall(r"[A-Z]+", cleaned)
    if words:
        if words[0] == "TEXT":
            return HAS_TEXT
        if words[0] in ("NOTEXT", "NO"):
            return NO_LETTERING

    if "NOTEXT" in cleaned or "NO TEXT" in cleaned:
        return NO_LETTERING
    if "TEXT" in cleaned:
        return HAS_TEXT
    return UNKNOWN


class ArtworkValidator:
    """Reads a generated image back out of the vision model to judge the artwork."""

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def validate(self, image_bytes: bytes) -> ArtworkValidationResult:
        """Classify the image as artwork or as a product.

        A failure is reported as ``unknown`` and never as a product: the
        artwork is only rejected when the model says so.
        """
        try:
            data_url = encode_image_for_vision(image_bytes)
            answer = self._classify(data_url)
        except Exception as e:  # noqa: BLE001
            logger.warning("Artwork validation failed: %s", e)
            return ArtworkValidationResult(is_product=False, verdict=UNKNOWN, error=str(e))

        verdict = parse_verdict(answer)
        if verdict == UNKNOWN:
            logger.warning("The artwork verdict could not be read: %r", answer)
        return ArtworkValidationResult(is_product=verdict == PRODUCT, verdict=verdict)

    def _classify(self, data_url: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ARTWORK_CLASSIFICATION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""


class ArtworkTextValidator:
    """Reads a generated image back out to see whether the artwork has words in it.

    The design text is drawn on the artwork afterwards, with a font, so the
    artwork itself has to be free of lettering: the image model cannot spell
    and a letter it draws is a letter that is wrong (issue #208).

    An answer that cannot be read is ``unknown`` and the image is kept: a
    vision model that is down must not throw away good designs.
    """

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def validate(self, image_bytes: bytes, *, scene: str | None = None) -> ArtworkTextResult:
        """Report whether the artwork carries lettering of its own.

        Digits on a keypad or dial of the depicted object are part of that
        object. Words, signs, and labels are lettering.
        """
        try:
            data_url = encode_image_for_vision(image_bytes)
            answer = self._classify(data_url, scene=scene)
        except Exception as e:  # noqa: BLE001
            logger.warning("Artwork lettering validation failed: %s", e)
            return ArtworkTextResult(has_text=False, verdict=UNKNOWN, error=str(e))

        verdict = parse_text_verdict(answer)
        if verdict == NO_LETTERING and misspelled_key_label(answer):
            verdict = HAS_TEXT
        if verdict == UNKNOWN:
            logger.warning("The artwork lettering verdict could not be read: %r", answer)
        return ArtworkTextResult(has_text=verdict == HAS_TEXT, verdict=verdict)

    def validate_no_text(self, image_bytes: bytes, *, scene: str | None = None) -> NoTextResult:
        """The inverted gate: accept only an explicit "no lettering" verdict.

        Unknown and failed readings fail closed. A no-text design has no
        compositor pass to correct a letter that the image model drew, so the
        absence of words has to be verified rather than assumed.
        """
        result = self.validate(image_bytes, scene=scene)
        return NoTextResult(
            matched=result.verdict == NO_LETTERING,
            verdict=result.verdict,
            error=result.error,
        )

    def _classify(self, data_url: str, *, scene: str | None = None) -> str:
        prompt = ARTWORK_TEXT_PROMPT
        described = (scene or "").strip()
        if described:
            prompt = f"{prompt} The picture was asked to show: {described}."
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""
