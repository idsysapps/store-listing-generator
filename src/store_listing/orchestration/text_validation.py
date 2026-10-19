"""Validate that the text rendered by Flux matches the text requested by the brief.

A vision LLM on the DGX (routed through the openai-proxy) reads the text back out
of the generated image, which is then compared with the expected headline and
tagline. The gate fails closed: if the text cannot be verified, the image is not
allowed through.
"""

from __future__ import annotations

import base64
import difflib
import io
import string
from dataclasses import dataclass
from typing import Any

from PIL import Image

__all__ = [
    "OCR_PROMPT",
    "TextValidationResult",
    "TextValidator",
    "encode_image_for_vision",
    "normalize_text",
    "text_matches",
]

OCR_PROMPT = (
    "Transcribe every word of text visible in this image, verbatim. "
    "Preserve capitalisation and reading order. "
    "If there is no text at all, reply with exactly: NO_TEXT"
)

NO_TEXT_MARKER = "NO_TEXT"

DEFAULT_THRESHOLD = 0.85
DEFAULT_MAX_SIDE = 1024
JPEG_QUALITY = 80


@dataclass
class TextValidationResult:
    """Verdict for a single generated image."""

    matched: bool
    expected: list[str]
    observed: str
    score: float
    error: str | None = None


def encode_image_for_vision(image_bytes: bytes, *, max_side: int = DEFAULT_MAX_SIDE) -> str:
    """Downscale and JPEG-encode an image into a base64 data URL for the vision model.

    A 1024x1024 PNG is around 1.5 MB and base64 adds a third again. The vision
    model does not need full resolution to read a headline.
    """
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_text(text: str) -> str:
    """Casefold, collapse whitespace and drop punctuation for tolerant comparison."""
    lowered = text.casefold()
    without_punctuation = "".join(
        " " if character in string.punctuation else character for character in lowered
    )
    return " ".join(without_punctuation.split())


def _best_window_ratio(expected: str, observed: str) -> float:
    """Best similarity between ``expected`` and any window of ``observed``.

    Sliding a window over the observation means a stray extra word in the
    rendered image does not drag the score down, while a genuinely wrong word
    still fails.
    """
    if expected in observed:
        return 1.0

    matcher = difflib.SequenceMatcher(None, expected, observed)
    best = matcher.ratio()

    expected_length = len(expected)
    for window_length in range(max(1, expected_length - 2), expected_length + 3):
        if window_length > len(observed):
            continue
        for start in range(len(observed) - window_length + 1):
            window = observed[start : start + window_length]
            ratio = difflib.SequenceMatcher(None, expected, window).ratio()
            best = max(best, ratio)
    return best


def text_matches(
    expected: str,
    observed: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, float]:
    """Return ``(matched, similarity)`` for the expected text inside the observation."""
    normalized_expected = normalize_text(expected)
    normalized_observed = normalize_text(observed)

    if not normalized_expected:
        return True, 1.0
    if not normalized_observed:
        return False, 0.0

    score = _best_window_ratio(normalized_expected, normalized_observed)
    return score >= threshold, score


class TextValidator:
    """Reads text back out of a generated image using a vision LLM on the DGX."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._client = client
        self._model = model
        self._threshold = threshold

    def validate(
        self,
        image_bytes: bytes,
        *,
        headline: str | None,
        tagline: str | None,
    ) -> TextValidationResult:
        """Check the rendered text against the expected headline and tagline."""
        expected = [text for text in (headline, tagline) if text and text.strip()]
        if not expected:
            return TextValidationResult(matched=True, expected=[], observed="", score=1.0)

        try:
            data_url = encode_image_for_vision(image_bytes)
            observed = self._transcribe(data_url)
        except Exception as e:  # noqa: BLE001
            return TextValidationResult(
                matched=False,
                expected=expected,
                observed="",
                score=0.0,
                error=f"Vision validation failed: {e}",
            )

        if NO_TEXT_MARKER in observed.strip().upper():
            return TextValidationResult(
                matched=False,
                expected=expected,
                observed=observed,
                score=0.0,
                error="No text was found in the generated image",
            )

        scores: list[float] = []
        matched = True
        for text in expected:
            text_matched, score = text_matches(text, observed, threshold=self._threshold)
            scores.append(score)
            matched = matched and text_matched

        return TextValidationResult(
            matched=matched,
            expected=expected,
            observed=observed,
            score=min(scores) if scores else 0.0,
        )

    def _transcribe(self, data_url: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""
