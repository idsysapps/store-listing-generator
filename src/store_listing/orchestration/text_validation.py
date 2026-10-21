"""Validate that the text rendered by Flux matches the text requested by the brief.

A vision LLM on the DGX (routed through the openai-proxy) reads the text back out
of the generated image, which is then compared with the expected headline and
tagline. The gate fails closed for the headline: if the message cannot be
verified, the image is not allowed through. The tagline is secondary - diffusion
renders it unreliably - so a missing tagline is reported but does not reject the
image.
"""

from __future__ import annotations

import base64
import difflib
import io
import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

__all__ = [
    "OCR_PROMPT",
    "TextValidationResult",
    "TextValidator",
    "encode_image_for_vision",
    "is_verifiable",
    "missing_words",
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

# A word of this length or longer may be one character off. The text is drawn
# with a font, so a long word that is nearly right is the vision model
# misreading it, not a typo in the design. A short word has to be right: the
# bubble on the image of brief #100 read "To get to the onthe sid", and a
# character score of 0.94 let it through (issue #208).
TOLERATED_WORD_LENGTH = 6

# Words this short are noise: the font can drop them and the text still reads.
IGNORED_WORD_LENGTH = 2


@dataclass
class TextValidationResult:
    """Verdict for a single generated image."""

    matched: bool
    expected: list[str]
    observed: str
    score: float
    error: str | None = None
    missing: list[str] = field(default_factory=list)
    unverifiable: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    wrong_words: list[str] = field(default_factory=list)


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


def _edit_distance(first: str, second: str) -> int:
    """How many single-character edits turn ``first`` into ``second``."""
    if first == second:
        return 0
    if not first:
        return len(second)
    if not second:
        return len(first)

    previous = list(range(len(second) + 1))
    for index, left in enumerate(first, start=1):
        current = [index]
        for offset, right in enumerate(second, start=1):
            current.append(
                min(
                    previous[offset] + 1,
                    current[offset - 1] + 1,
                    previous[offset - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _word_is_covered(expected_word: str, observed_words: Sequence[str]) -> bool:
    """Whether the observation has the word the design asked for.

    A long word may be one character off: the text is drawn with a font, so
    that is the vision model misreading it. A short word has to be right - the
    bubble on the image of brief #100 read "To get to the onthe sid" and a
    character score of 0.94 let it through (issue #208).
    """
    if expected_word in observed_words:
        return True
    if len(expected_word) < TOLERATED_WORD_LENGTH:
        return False
    return any(_edit_distance(expected_word, word) <= 1 for word in observed_words)


def missing_words(expected: str, observed: str) -> list[str]:
    """The words of the expected text that the observation does not have."""
    expected_words = normalize_text(expected).split()
    observed_words = normalize_text(observed).split()
    if not expected_words or not observed_words:
        return []

    missing: list[str] = []
    for word in expected_words:
        if len(word) <= IGNORED_WORD_LENGTH:
            # A short word is noise: the font can drop it and the text still reads.
            continue
        if not _word_is_covered(word, observed_words):
            missing.append(word)
    return missing


def text_matches(
    expected: str,
    observed: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, float]:
    """Return ``(matched, similarity)`` for the expected text inside the observation.

    The words have to be there: a character score on its own let a misspelled
    word through (issue #208).
    """
    normalized_expected = normalize_text(expected)
    normalized_observed = normalize_text(observed)

    if not normalized_expected:
        return True, 1.0
    if not normalized_observed:
        return False, 0.0

    score = _best_window_ratio(normalized_expected, normalized_observed)
    if missing_words(normalized_expected, normalized_observed):
        return False, score
    return score >= threshold, score


def is_verifiable(text: str) -> bool:
    """Whether the text can be read back out of an image at all.

    A headline that is only a symbol ("🎃") is never transcribed verbatim, so
    gating on it would reject every image that carries one.
    """
    return any(character.isalnum() for character in normalize_text(text))


def _observed_lines(observed: str) -> list[str]:
    """The lines of the transcription."""
    return [line.strip() for line in observed.splitlines() if line.strip()]


def _find_text(text: str, lines: list[str], *, threshold: float) -> tuple[bool, float, list[str]]:
    """Find the text in the transcription.

    A line that carries the text is the strongest evidence, so the lines are
    tried first. Text the model wrapped over two lines is still found in the
    joined observation - a headline that is too long for one line is drawn on
    two by the compositor, and the vision model reports it that way.
    """
    hits: list[str] = []
    best = 0.0
    for line in lines:
        matched, score = text_matches(text, line, threshold=threshold)
        if matched:
            hits.append(line)
            best = max(best, score)
    if hits:
        return True, best, hits

    joined = " ".join(lines)
    matched, score = text_matches(text, joined, threshold=threshold)
    return matched, score, [joined] if matched else []


class TextValidator:
    """Reads text back out of a generated image using a vision LLM on the DGX."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        threshold: float = DEFAULT_THRESHOLD,
        require_tagline: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._threshold = threshold
        self._require_tagline = require_tagline

    def validate(
        self,
        image_bytes: bytes,
        *,
        headline: str | None,
        tagline: str | None,
    ) -> TextValidationResult:
        """Check the rendered text against the expected headline and tagline.

        The headline gates the image: a design whose message came out garbled is
        not sellable. The tagline is secondary and diffusion drops it often, so
        a missing tagline is reported but does not reject the image - unless
        ``require_tagline`` says otherwise.
        """
        headline_text = headline.strip() if headline and headline.strip() else None
        tagline_text = tagline.strip() if tagline and tagline.strip() else None
        expected = [text for text in (headline_text, tagline_text) if text]
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
        missing: list[str] = []
        unverifiable: list[str] = []
        duplicated: list[str] = []
        wrong_words: list[str] = []
        lines = _observed_lines(observed)
        for text in expected:
            if not is_verifiable(text):
                unverifiable.append(text)
                continue
            found, score, hits = _find_text(text, lines, threshold=self._threshold)
            scores.append(score)
            if not found:
                missing.append(text)
                for word in missing_words(text, observed):
                    if word not in wrong_words:
                        wrong_words.append(word)
            elif len(hits) > 1:
                # The text is on the image more than once - the model drew it
                # twice, which is a defect of its own.
                duplicated.append(text)

        # The headline is what the design sells, so it has to be there. The
        # tagline is only required in strict mode.
        required = (
            list(missing) if self._require_tagline else [t for t in missing if t == headline_text]
        )

        return TextValidationResult(
            matched=not required,
            expected=expected,
            observed=observed,
            score=min(scores) if scores else 1.0,
            missing=missing,
            unverifiable=unverifiable,
            duplicated=duplicated,
            wrong_words=wrong_words,
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
