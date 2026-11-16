"""The merchandise themes a design may not carry.

The brief prompts tell the model the library. This screen is the part that
still runs when the model ignores the prompt. It is used on the way in (a
trend query), on the way through (a brief, a rewrite, a listing), and on the
way out (an image prompt).

Halloween costumes are allow-listed. The calendar already seeds witch,
skeleton, spooky, and pumpkin designs, and those subjects stay in the
library. The allow-list is applied only to the occult scan, so a Halloween
shirt still cannot carry a separate excluded theme. Tarot, a seance, and
witchcraft stay out.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "EXCLUDED_IMAGE_TERMS",
    "HALLOWEEN_THEMES",
    "PERSON_CONSTRAINT",
    "content_rejection",
    "excluded_theme",
]

# How a person is drawn when the artwork includes one. The sentence is a
# constraint on people who appear. It does not ask the model to add a person.
PERSON_CONSTRAINT: Final[str] = (
    "If a person appears, draw a man, a woman, a boy, or a girl in modest "
    "everyday clothes. A couple is a husband and wife. A family is parents "
    "and their children."
)

# A backstop for an image model that honors a negative prompt. The scene is
# still refused before it is sent.
EXCLUDED_IMAGE_TERMS: Final[str] = "no nudity, no explicit content, no occult symbols"

# The occult label. Halloween phrases are removed before this scan only.
_OCCULT_THEME: Final[str] = "an occult or esoteric practice"

# Costume subjects the store already prints for the observance. Longer phrases
# come first so "jack o lantern" is not left as a bare "lantern".
HALLOWEEN_THEMES: Final[tuple[str, ...]] = (
    "jack-o'-lantern",
    "jack o'lantern",
    "jack-o-lantern",
    "jack o lantern",
    "trick-or-treat",
    "trick or treat",
    "all hallows",
    "hocus pocus",
    "candy corn",
    "black cat",
    "broomstick",
    "graveyard",
    "tombstone",
    "halloween",
    "skeletons",
    "pumpkin",
    "cauldron",
    "costume",
    "haunted",
    "skeleton",
    "vampires",
    "vampire",
    "zombies",
    "zombie",
    "witches",
    "ghosts",
    "skulls",
    "mummy",
    "mummies",
    "witch",
    "ghost",
    "spooky",
    "skull",
)

# Longer phrases are listed before the short words they contain, and the
# identity phrases are checked before the short word "sex", so "same-sex"
# is the theme it names.
_THEMES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "an LGBTQ+ theme",
        (
            "lgbtqia+",
            "lgbtqia",
            "lgbtq+",
            "lgbtq",
            "lgbt",
            "pride month",
            "pride flag",
            "rainbow flag",
            "gay pride",
            "transgender",
            "non-binary",
            "nonbinary",
            "genderqueer",
            "gender identity",
            "drag queen",
            "drag king",
            "same-sex",
            "same sex",
            "they/them",
            "lesbian",
            "bisexual",
            "pansexual",
            "queer",
            "gay",
        ),
    ),
    (
        "abortion or contraception advocacy",
        (
            "my body my choice",
            "planned parenthood",
            "reproductive rights",
            "birth control",
            "pro-choice",
            "pro choice",
            "contraceptive",
            "contraception",
            "abortion",
        ),
    ),
    (
        _OCCULT_THEME,
        (
            "crystal healing",
            "palm reading",
            "black magic",
            "spell book",
            "spellbook",
            "demonology",
            "necromancy",
            "palmistry",
            "witchcraft",
            "astrology",
            "astrologer",
            "horoscope",
            "pentagram",
            "satanism",
            "satanist",
            "satanic",
            "esoteric",
            "séance",
            "seance",
            "new age",
            "occult",
            "coven",
            "voodoo",
            "zodiac",
            "pagan",
            "wiccan",
            "wicca",
            "tarot",
            "ouija",
            "satan",
        ),
    ),
    (
        "blasphemy",
        (
            "anti-christian",
            "anti christian",
            "anti-catholic",
            "anti catholic",
            "god is dead",
            "hail satan",
            "blasphemous",
            "blaspheme",
            "blasphemy",
        ),
    ),
    (
        "sexual or explicit content",
        (
            "intercourse",
            "pornography",
            "only fans",
            "onlyfans",
            "sexuality",
            "sex toy",
            "sexiest",
            "sexier",
            "erotica",
            "sensual",
            "stripper",
            "playboy",
            "hentai",
            "erotic",
            "orgasm",
            "nudity",
            "fetish",
            "sexual",
            "topless",
            "nudes",
            "naked",
            "porn",
            "nsfw",
            "bdsm",
            "nude",
            "sexy",
            "xxx",
            "sex",
        ),
    ),
)


def _pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    body = "|".join(re.escape(phrase) for phrase in phrases)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (label, _pattern(phrases)) for label, phrases in _THEMES
)
_HALLOWEEN_PATTERN: Final[re.Pattern[str]] = _pattern(HALLOWEEN_THEMES)


def excluded_theme(text: str | None) -> str | None:
    """The excluded merchandise theme in the text, or None when it has none.

    The label is the theme, not the word that matched, so a log says what was
    wrong without quoting a list.

    Halloween phrases are blanked before the occult scan. They are not blanked
    for any other theme, and a practice that is not on the allow-list is still
    found in what remains.
    """
    if not text or not text.strip():
        return None
    occult_text = _HALLOWEEN_PATTERN.sub(" ", text)
    for label, pattern in _PATTERNS:
        haystack = occult_text if label == _OCCULT_THEME else text
        if pattern.search(haystack):
            return label
    return None


def content_rejection(*parts: tuple[str, str | None]) -> str | None:
    """Why one of these texts is outside the library, or None when all of them are in it.

    Each part is a label and the text it names, so the reason says which
    field was dropped: "the scene depicts an occult or esoteric practice".
    """
    for label, text in parts:
        theme = excluded_theme(text)
        if theme is not None:
            return f"the {label} depicts {theme}"
    return None
