"""Keep words out of the scene description.

The image model cannot spell. A scene that says "a grinning pumpkin with a
speech bubble saying 'To get to the other side!'" asks it to draw words, and
it draws them wrong: the image of brief #100 carried a bubble that read "To
get to the onthe sid!" (issue #208).

The scene is sanitised before it reaches the image model, so a scene that
describes lettering loses the lettering and keeps the picture.
"""

from __future__ import annotations

import re

__all__ = [
    "has_lettering",
    "lettering_cues",
    "strip_lettering",
]

# A quoted span is a word the image model has to spell.
_QUOTE_CHARS = r"[‘’“”\"']"

_QUOTED_SOURCE = rf"{_QUOTE_CHARS}[^‘’“”\"']{{0,120}}{_QUOTE_CHARS}"

_QUOTED = re.compile(_QUOTED_SOURCE)

# The things in a picture that hold words. They always do.
_STRONG_NOUNS = r"speech\s+bubbles?|bubbles?|lettering|slogans?|captions?|handwriting|watermarks?"

# The things that hold words only when they are said to hold words. A book
# title is a word in a sentence, not a word in the picture.
_WEAK_NOUNS = r"signboards?|signs?|banners?|placards?|labels?|titles?|text|words?"

# The verb that says the thing holds words.
_CUE = (
    r"(?:that\s+|which\s+|who\s+)?"
    r"(?:says?|say|saying|reads?|reading|states?|state|inscribes?|inscribing|"
    r"emblazoned|spells?|spelling)"
)

# The connector that belongs to the thing that is removed with it.
_CONNECTOR = r"(?:(?:with|and|or|plus|featuring|holding)\s+(?:the|a|an)\s+)?"

# A weak noun with words in it: "a sign that says 'OPEN'".
_WEAK_WITH_WORDS = re.compile(
    rf"{_CONNECTOR}(?:\b{_WEAK_NOUNS}\b)(?:\s+{_CUE}\b)?\s*[:\-]?\s*{_QUOTED_SOURCE}",
    re.IGNORECASE,
)

# A weak noun with a verb: "a sign that says".
_WEAK_WITH_CUE = re.compile(rf"(?:\b{_WEAK_NOUNS}\b)\s+{_CUE}\b", re.IGNORECASE)

# A strong noun, with its verb and its words: "a speech bubble saying 'hi'".
_STRONG_WITH_WORDS = re.compile(
    rf"{_CONNECTOR}(?:\b{_STRONG_NOUNS}\b)(?:\s+{_CUE}\b)?(?:\s*{_QUOTED_SOURCE})?",
    re.IGNORECASE,
)

# The words that are left over when the picture is taken away.
_FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "read",
        "reads",
        "saying",
        "says",
        "that",
        "the",
        "to",
        "which",
        "with",
    }
)


def _tidy(text: str) -> str:
    """Tidy up what is left after the lettering is taken out."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", cleaned)
    # A connector that lost its noun: "with with".
    cleaned = re.sub(r"\b(\w+)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" ,;:.!?-")

    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    if words and all(word.lower() in _FILLER_WORDS for word in words):
        # Nothing is left to draw.
        return ""
    return cleaned


def lettering_cues(text: str) -> list[str]:
    """The parts of the scene that ask for words in the picture."""
    if not text:
        return []

    cues: list[str] = []
    for pattern in (_WEAK_WITH_WORDS, _WEAK_WITH_CUE, _STRONG_WITH_WORDS, _QUOTED):
        cues.extend(match.group(0) for match in pattern.finditer(text))
    return cues


def has_lettering(text: str) -> bool:
    """Whether the scene asks for words in the picture."""
    return bool(lettering_cues(text))


def strip_lettering(text: str) -> str:
    """The scene without the words it asks the image model to spell."""
    if not text or not text.strip():
        return text

    cleaned = text
    for pattern in (_WEAK_WITH_WORDS, _WEAK_WITH_CUE, _STRONG_WITH_WORDS, _QUOTED):
        cleaned = pattern.sub(" ", cleaned)
    return _tidy(cleaned)
