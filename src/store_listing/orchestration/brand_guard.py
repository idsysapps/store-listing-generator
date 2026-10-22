"""The names a design may not carry.

The list lives here because it is used in two places: the trend queries that
are harvested, and the text the brief model wrote. A trend query that names a
brand is noise, and a headline that names one is a trademark we would print on
a shirt - the design that started this named a theme park (issue #225).

The names are matched as whole words, so "Legacy" is not "Lego" and
"Marvellous" is not "Marvel".
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["LICENSED_NAMES", "licensed_brand"]

# The names that carry a licence or a trademark.
LICENSED_NAMES: Final[tuple[str, ...]] = (
    # Artists and sports.
    "taylor swift",
    "beyonce",
    "drake",
    "nike",
    "nfl",
    "nba",
    "mlb",
    "super bowl",
    # Film, television and games.
    "disney",
    "marvel",
    "pokemon",
    "star wars",
    "harry potter",
    "hogwarts",
    "spider-man",
    "spider man",
    "superman",
    "batman",
    "mario",
    "luigi",
    "bowser",
    "pikachu",
    "fortnite",
    "the sims",
    # Theme parks and characters.
    "magic kingdom",
    "walt disney world",
    "disneyland",
    "happiest place",
    "cinderella castle",
    "mickey mouse",
    "minnie mouse",
    "hello kitty",
    "winnie the pooh",
    # Brands that end up on a shirt.
    "pepsi",
    "coca-cola",
    "coca cola",
    "mcdonald's",
    "mcdonalds",
    "starbucks",
    "barbie",
    "lego",
)

# The names are matched on their own, not as a part of a longer word.
LICENSED_IP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:" + "|".join(re.escape(name) for name in LICENSED_NAMES) + r")\b",
    re.IGNORECASE,
)


def licensed_brand(text: str | None) -> str | None:
    """The licensed name in the text, or None when the text carries none.

    The name is returned as it was written in the text, so the log says which
    name it was: "Magic Kingdom" reads better than a pattern dump.
    """
    if not text:
        return None
    match = LICENSED_IP_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0).strip()
