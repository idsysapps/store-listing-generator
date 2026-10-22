"""The voice of the text a brief carries.

The text is what the buyer reads, so it is the part of a design that has to be
good. The model writes it, and the model writes the same flat line every time:
a caption of the picture ("Thanksgiving in the Magic Kingdom") or a list of
nouns ("Feast, fun, and fireworks"). Neither is a line anybody wears
(issue #225).

The gate here is mechanical on purpose. A word list cannot tell a good line
from a bad one, so the rules are about the shape of the text:

- a list of items is not a line,
- a headline that only restates the picture is not a line,
- a tagline that repeats the headline is not a second line,
- a name that carries a licence does not go on a shirt.

The voice itself - dry, loud, warm - is taught by the prompt and shown by the
examples there. The one rule that depends on the voice is the deadpan one: a
dry line has to say something, so a headline made of nothing but sentiment is
dropped.
"""

from __future__ import annotations

import re
from typing import Final

from store_listing.orchestration.brand_guard import licensed_brand

__all__ = [
    "DEFAULT_VOICE",
    "MAX_HEADLINE_WORDS",
    "SENTIMENT_WORDS",
    "VOICES",
    "normalise_voice",
    "screen_text",
]

# The voices the store sells. Every brief picks one.
VOICES: Final[tuple[str, ...]] = ("deadpan", "loud", "cozy")

# The voice a brief gets when the model did not name one.
DEFAULT_VOICE: Final[str] = "deadpan"

# The most words a headline may have. A longer one is a sentence.
MAX_HEADLINE_WORDS: Final[int] = 6

# The words that carry a feeling and nothing else. A deadpan headline made of
# these says nothing at all.
SENTIMENT_WORDS: Final[frozenset[str]] = frozenset(
    {
        "love",
        "loved",
        "loves",
        "joy",
        "happy",
        "happiness",
        "warm",
        "warmth",
        "cozy",
        "cosy",
        "grateful",
        "gratitude",
        "blessed",
        "blessing",
        "heart",
        "hearts",
        "smile",
        "smiles",
        "hug",
        "hugs",
        "magic",
        "wonderful",
        "amazing",
        "beautiful",
        "sweet",
        "memories",
        "forever",
        "together",
        "family",
        "friend",
        "friends",
        "joyful",
    }
)

# The words that carry no meaning for the comparison between two texts.
STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "there",
        "they",
        "this",
        "to",
        "up",
        "was",
        "we",
        "were",
        "with",
        "you",
        "your",
    }
)

# The things that separate the items of a list.
LIST_SEPARATOR: Final[re.Pattern[str]] = re.compile(r",|;|\band\b|&|\bplus\b", re.IGNORECASE)

# A word of a text, with the punctuation taken off.
WORD: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")


def normalise_voice(voice: str | None) -> str:
    """The voice of a brief, or the default when it named none we sell."""
    if voice is None:
        return DEFAULT_VOICE
    candidate = voice.strip().lower()
    if candidate in VOICES:
        return candidate
    return DEFAULT_VOICE


def _content_words(text: str | None) -> set[str]:
    """The words of a text that carry meaning."""
    if not text:
        return set()
    return {word for word in WORD.findall(text.lower()) if len(word) > 2 and word not in STOP_WORDS}


def _is_a_list(text: str) -> bool:
    """Whether the text is a list of items rather than a line.

    One separator is a line that joins two things ("Rock and Roll"). Two of
    them are a list ("Feast, fun, and fireworks").
    """
    return len(LIST_SEPARATOR.findall(text)) >= 2


def screen_text(
    headline: str | None,
    *,
    tagline: str | None = None,
    scene: str | None = None,
    voice: str | None = None,
    allow_brands: bool = False,
) -> str | None:
    """Why the text of a brief is not good enough, or None when it is.

    The reason is written for the log and for the model: it is the feedback
    the batch asks again with.

    A brief for personal use may name a franchise on purpose, so it is
    screened with ``allow_brands``.
    """
    if not headline or not headline.strip():
        return None

    text = headline.strip()

    if not allow_brands:
        brand = licensed_brand(text)
        if brand:
            return f"the headline names a licensed brand: {brand}"

        tagline_brand = licensed_brand(tagline)
        if tagline_brand:
            return f"the tagline names a licensed brand: {tagline_brand}"

    words = text.split()
    if len(words) > MAX_HEADLINE_WORDS:
        return f"the headline is {len(words)} words long, the most is {MAX_HEADLINE_WORDS}"

    if _is_a_list(text):
        return "the headline reads as a list of items, not as a line"

    headline_words = _content_words(text)
    scene_words = _content_words(scene)
    if headline_words and scene_words and headline_words <= scene_words:
        return "the headline only restates the picture"

    if tagline and tagline.strip():
        tagline_words = _content_words(tagline)
        shared = headline_words & tagline_words
        if headline_words and len(shared) >= len(headline_words) * 0.8:
            return "the tagline repeats the headline"

    if normalise_voice(voice) == "deadpan":
        meaningful = headline_words - SENTIMENT_WORDS
        if headline_words and not meaningful:
            return "the headline is all sentiment and says nothing"

    return None
