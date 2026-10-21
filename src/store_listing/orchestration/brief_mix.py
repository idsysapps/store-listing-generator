"""The seasonal/evergreen mix of a batch of design briefs.

The brief pool is fed by a calendar of holidays, so a batch can end up being
all seasonal without anybody noticing. The mix is a property of the briefs and
the seeds they were built from, so it is computed here with pure functions:
no database, no LLM, easy to test.

The theme of a seed is guessed from its query. Once `active_seeds.theme`
exists (see issue #198) the database answer is used instead; the guess is the
fallback for seeds that were never tagged.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

from store_listing.orchestration.context_seeds import SEASONAL_EVENT_THEME, SEASONAL_SEED_RULES

__all__ = [
    "BriefMix",
    "MixableBrief",
    "classify_brief_mix",
    "classify_themes",
    "event_theme_for_query",
    "known_event_names",
    "seed_themes_for_queries",
    "theme_of_queries",
]


class MixableBrief(Protocol):
    """What the mix needs to know about a brief."""

    @property
    def concept(self) -> str: ...

    @property
    def confidence(self) -> int: ...

    @property
    def source_seed_ids(self) -> list[int]: ...


@dataclass(frozen=True)
class BriefMix:
    """How many briefs of a batch are seasonal and how many are evergreen."""

    total: int
    seasonal: int
    evergreen: int
    by_theme: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"total={self.total} seasonal={self.seasonal} evergreen={self.evergreen} "
            f"themes={self.by_theme}"
        )


# Words that only make sense inside a calendar event. A bare "gift" or "mom"
# is deliberately absent: those words show up in evergreen seeds too.
_EVENT_TOKENS: Final[dict[str, tuple[str, ...]]] = {
    "halloween": ("halloween", "spooky", "pumpkin", "witch", "skeleton", "trick or treat"),
    "christmas": ("christmas", "xmas", "santa", "elf", "stocking", "ugly sweater"),
    "valentines_day": ("valentine", "galentine", "couples matching"),
    "mothers_day": ("mother's day", "mothers day", "mama bear", "plant mom", "best mom"),
    "fathers_day": ("father's day", "fathers day", "dad humor", "dad bod", "best dad", "grill dad"),
    "back_to_school": ("teacher", "back to school", "first day of school", "school bus"),
    "super_bowl": ("super bowl", "game day", "football", "tailgate"),
}

# The starter seeds are evergreen by definition, so they never carry a theme
# even when a calendar rule contains a seed that starts the same way.
_EVERGEN_QUERIES: Final[frozenset[str]] = frozenset(
    {
        "funny t-shirt",
        "hoodie",
        "gift",
        "socks",
        "leggings",
        "custom mugs",
        "room decor",
        "wall art",
        "mom humor",
        "dad jokes",
        "gym fitness",
        "running",
        "quirky gifts",
        "novelty socks",
        "personalized gifts",
    }
)


def known_event_names() -> set[str]:
    """Every event the calendar knows about."""
    return set(SEASONAL_SEED_RULES) | {SEASONAL_EVENT_THEME}


def _rule_seed_match(query: str) -> str | None:
    """The event whose seed list the query belongs to."""
    for name, rule in SEASONAL_SEED_RULES.items():
        for seed in rule.seeds:
            normalized_seed = seed.lower()
            if query == normalized_seed:
                return name
            if query.startswith(normalized_seed + " ") or normalized_seed.startswith(query + " "):
                return name
    return None


def _token_match(query: str) -> str | None:
    """The event a distinctive word of the query points at."""
    for name, tokens in _EVENT_TOKENS.items():
        for token in tokens:
            if token in query:
                return name
    return None


def event_theme_for_query(query: str) -> str | None:
    """The calendar event a seed query belongs to, or None when it is evergreen."""
    normalized = query.strip().lower()
    if not normalized or normalized in _EVERGEN_QUERIES:
        return None
    return _rule_seed_match(normalized) or _token_match(normalized)


def seed_themes_for_queries(
    seeds: Iterable[tuple[int, str]],
) -> dict[int, str | None]:
    """Map seed ids to the event they belong to."""
    return {seed_id: event_theme_for_query(query) for seed_id, query in seeds}


def _brief_theme(
    brief: MixableBrief,
    seed_themes: Mapping[int, str | None],
    seasonal_themes: Collection[str],
) -> str | None:
    for seed_id in brief.source_seed_ids:
        theme = seed_themes.get(seed_id)
        if theme is not None and theme in seasonal_themes:
            return theme
    return None


def classify_brief_mix(
    briefs: Iterable[MixableBrief],
    seed_themes: Mapping[int, str | None],
    seasonal_themes: Collection[str] | None = None,
) -> BriefMix:
    """Count the seasonal and evergreen briefs of a batch.

    A brief is seasonal when at least one of the seeds it was built from
    belongs to a calendar event.
    """
    themes = set(seasonal_themes) if seasonal_themes is not None else known_event_names()

    return classify_themes(
        (_brief_theme(brief, seed_themes, themes) for brief in briefs),
        seasonal_themes=themes,
    )


def classify_themes(
    themes: Iterable[str | None],
    seasonal_themes: Collection[str] | None = None,
) -> BriefMix:
    """Count the briefs of a batch by the theme they belong to.

    A brief whose theme is None is evergreen. A theme that is not a calendar
    event (an evergreen category) is evergreen as well.
    """
    seasonal_names = set(seasonal_themes) if seasonal_themes is not None else known_event_names()

    total = 0
    seasonal = 0
    by_theme: dict[str, int] = {}

    for theme in themes:
        total += 1
        if theme is None or theme not in seasonal_names:
            continue
        seasonal += 1
        by_theme[theme] = by_theme.get(theme, 0) + 1

    return BriefMix(
        total=total,
        seasonal=seasonal,
        evergreen=total - seasonal,
        by_theme=by_theme,
    )


def theme_of_queries(queries: Iterable[str]) -> str | None:
    """The event a brief's seed queries point at, or None when it is evergreen."""
    for query in queries:
        theme = event_theme_for_query(query)
        if theme is not None:
            return theme
    return None
