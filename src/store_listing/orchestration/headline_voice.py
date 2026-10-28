"""The mechanical safety checks for the text a brief carries.

The editorial decision about whether a line is worth wearing belongs to the
LLM copy editor (``copy_review``). A word list cannot tell a good line from a
bad one, and a code heuristic must not pretend to.

This module keeps only the hard constraints that are not a matter of taste:

- a name that carries a licence does not go on a shirt,
- a headline longer than the printer can lay out is not a headline,
- a list of items is not a line.

The voice itself - dry, loud, warm - is taught by the prompt and shown by the
examples there.
"""

from __future__ import annotations

import re
from typing import Final

from store_listing.orchestration.brand_guard import licensed_brand

__all__ = [
    "DEFAULT_VOICE",
    "MAX_HEADLINE_WORDS",
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

# The things that separate the items of a list.
LIST_SEPARATOR: Final[re.Pattern[str]] = re.compile(r",|;|\band\b|&|\bplus\b", re.IGNORECASE)


def normalise_voice(voice: str | None) -> str:
    """The voice of a brief, or the default when it named none we sell."""
    if voice is None:
        return DEFAULT_VOICE
    candidate = voice.strip().lower()
    if candidate in VOICES:
        return candidate
    return DEFAULT_VOICE


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
    """Why the text breaks a hard constraint, or None when it does not.

    The editorial question - is this line worth wearing? - belongs to the LLM
    copy editor. This function only checks the things a printer or a lawyer
    can decide: a licensed brand, the maximum line length and a list where a
    line was expected.

    ``scene`` and ``voice`` are kept in the signature because the callers
    already pass them; they are no longer used to judge the copy.
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

    return None
