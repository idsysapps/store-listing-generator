"""Check that a generated image is flat print artwork and not a product.

The text gate reads the words back out of an image. Nothing looked at the
picture itself, so a brief whose scene named a bag produced an image of a
bag, and that bag was uploaded as the print file (issue #195).

The verdict is deliberately coarse - the image is either a flat graphic or it
is a product. An answer that cannot be read is ``unknown`` and the image is
kept: a vision model that is down must not throw away good designs.
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
    "PRODUCT",
    "UNKNOWN",
    "ArtworkValidationResult",
    "ArtworkValidator",
    "parse_verdict",
]

ARTWORK = "artwork"
PRODUCT = "product"
UNKNOWN = "unknown"

ARTWORK_CLASSIFICATION_PROMPT = (
    "Look at this image. Is it a flat print graphic - a design that could be printed "
    "on a product - or is it a product itself (a bag, a tote bag, a pouch, a shirt, a "
    "hoodie, a mug, a tumbler, a phone case, a pillow, a blanket, a sticker sheet, or a "
    "photo of a printed item)? Answer with exactly one word: ARTWORK or PRODUCT."
)


@dataclass
class ArtworkValidationResult:
    """Verdict for a single generated image."""

    is_product: bool
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
