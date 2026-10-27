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
    "STRICT_OCR_PROMPTS",
    "StrictTextResult",
    "StrictTextValidator",
    "TextValidationResult",
    "TextValidator",
    "count_text_blocks",
    "encode_image_for_vision",
    "is_verifiable",
    "missing_words",
    "normalize_strict_text",
    "normalize_text",
    "strict_text_matches",
    "text_matches",
]

OCR_PROMPT = (
    "Transcribe every word of text visible in this image, verbatim. "
    "Preserve capitalisation and reading order. "
    "If there is no text at all, reply with exactly: NO_TEXT"
)

# Model-drawn bubble lettering needs a different question from the flat
# compositor type. The Phase 0 bake-off found "transcribe the stylised text"
# could misread a correct GROOVY as CROOVY, while asking what text is shown
# was stable. Three readings with a majority vote keep OCR noise from
# rejecting a good candidate without letting a genuine typo through.
STRICT_OCR_PROMPTS: tuple[str, ...] = (
    "What text is shown in this image? Output only the words.",
    (
        "Read the large lettering in this image exactly as written. "
        "Output only the words, one block per line."
    ),
    (
        "Transcribe the text in this image. Do not correct spelling or add words. "
        "Output only the words."
    ),
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


@dataclass
class StrictTextResult:
    """Verdict for model-drawn lettering, where every letter has to be right."""

    matched: bool
    expected: list[str]
    observed: str
    score: float
    error: str | None = None
    missing: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    votes: int = 0
    observations: list[str] = field(default_factory=list)


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


# Typographic lookalikes are the same character for spelling purposes. The
# rest of the punctuation is decoration: GROOVY! and GROOVY are the same word.
_TYPOGRAPHIC_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)

# Punctuation that changes the message and must survive strict normalisation.
_STRICT_PUNCTUATION = frozenset("&'-")


def normalize_strict_text(text: str) -> str:
    """Casefold and normalise, keeping the characters that change the words.

    The tolerant :func:`normalize_text` drops every punctuation mark, so
    "SALT & PEPPER" and "SALT PEPPER" compare equal. For model-drawn lettering
    that is exactly the defect the gate exists to catch, so strict
    normalisation keeps ``&``, ``'`` and ``-`` and drops only decoration.
    """
    translated = text.translate(_TYPOGRAPHIC_REPLACEMENTS)
    kept = "".join(
        character
        if character.isalnum() or character in _STRICT_PUNCTUATION or character.isspace()
        else " "
        for character in translated.casefold()
    )
    return " ".join(kept.split())


def _strict_tokens(text: str) -> list[str]:
    return normalize_strict_text(text).split()


def count_text_blocks(expected: str, observed: str) -> int:
    """How many times ``expected`` appears as whole tokens in ``observed``.

    A block may be split over lines by the transcription, so the match is done
    on the token sequence rather than on a line. Whole tokens matter: GROOVY
    must not match GROOVYX.
    """
    expected_tokens = _strict_tokens(expected)
    observed_tokens = _strict_tokens(observed)
    if not expected_tokens or not observed_tokens:
        return 0
    width = len(expected_tokens)
    return sum(
        1
        for start in range(len(observed_tokens) - width + 1)
        if observed_tokens[start : start + width] == expected_tokens
    )


def strict_text_matches(expected: str, observed: str) -> tuple[bool, float]:
    """Return ``(matched, 1.0)`` only when the text is there exactly once.

    This is the gate for a hero word: one wrong letter, a missing ampersand,
    or the word drawn twice all fail. It is deliberately stricter than
    :func:`text_matches`, which tolerates an OCR misread on a long word.
    """
    if not normalize_strict_text(expected):
        return True, 1.0
    if count_text_blocks(expected, observed) == 1:
        return True, 1.0
    return False, 0.0


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


class StrictTextValidator:
    """Fail-closed OCR gate for lettering the image model drew itself.

    The compositor's type is predictable, so a single transcription is enough.
    Qwen-Image draws stylised bubble letters, and a vision model can misread a
    correct word (the bake-off saw GROOVY read back as CROOVY). Three readings
    with a majority vote separate OCR noise from a real typo: a genuine error
    is read the same wrong way every time, while a correct word survives one
    noisy reading.
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        prompts: Sequence[str] = STRICT_OCR_PROMPTS,
        min_votes: int = 2,
    ) -> None:
        if min_votes < 1:
            raise ValueError("min_votes must be at least 1")
        self._client = client
        self._model = model
        self._prompts = tuple(prompts)
        self._min_votes = min_votes

    def validate(self, image_bytes: bytes, *, expected: Sequence[str]) -> StrictTextResult:
        blocks = [block.strip() for block in expected if block and block.strip()]
        verifiable = [block for block in blocks if is_verifiable(block)]
        if not verifiable:
            return StrictTextResult(
                matched=True,
                expected=blocks,
                observed="",
                score=1.0,
            )

        try:
            data_url = encode_image_for_vision(image_bytes)
        except Exception as error:  # noqa: BLE001 - fail closed on any image error
            return StrictTextResult(
                matched=False,
                expected=blocks,
                observed="",
                score=0.0,
                error=f"Vision validation failed: {error}",
            )

        observations: list[str] = []
        errors: list[str] = []
        for prompt in self._prompts:
            try:
                observations.append(self._transcribe(data_url, prompt))
            except Exception as error:  # noqa: BLE001 - one bad reading is not fatal
                errors.append(str(error))

        if not observations:
            return StrictTextResult(
                matched=False,
                expected=blocks,
                observed="",
                score=0.0,
                error="Vision validation failed: " + "; ".join(errors),
            )

        votes = 0
        missing: list[str] = []
        duplicated: list[str] = []
        for observation in observations:
            if NO_TEXT_MARKER in observation.strip().upper():
                _extend_unique(missing, verifiable)
                continue

            observation_missing: list[str] = []
            observation_duplicated: list[str] = []
            for block in verifiable:
                count = count_text_blocks(block, observation)
                if count == 0:
                    observation_missing.append(block)
                elif count > 1:
                    observation_duplicated.append(block)

            _extend_unique(missing, observation_missing)
            _extend_unique(duplicated, observation_duplicated)
            if not observation_missing and not observation_duplicated:
                votes += 1

        matched = votes >= self._min_votes
        error = None
        if errors and not matched:
            error = "Vision validation failed: " + "; ".join(errors)
        return StrictTextResult(
            matched=matched,
            expected=blocks,
            observed="\n---\n".join(observations),
            score=votes / len(observations),
            error=error,
            missing=missing,
            duplicated=duplicated,
            votes=votes,
            observations=observations,
        )

    def _transcribe(self, data_url: str, prompt: str) -> str:
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


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)
