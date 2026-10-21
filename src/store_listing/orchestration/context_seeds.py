"""Rule-based calendar and evergreen seed injector for predictable trends.

Runs before harvesters to pre-load active_seeds with holiday/event seeds at
the right lead time, and with standing theme seeds all year round. No LLM
cost — pure date arithmetic.

The calendar used to be the only source of seeds, so a season took the whole
pool with it and the briefs that came out were all seasonal (issue #198).
The evergreen themes are the counterpart: they are always in season.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Final, Protocol

from store_listing.orchestration.promotion import (
    CALENDAR_SEEDS_PER_EVENT_PER_DAY,
    EVERGREEN_SEEDS_PER_DAY,
    ORIGIN_CALENDAR,
    ORIGIN_EVERGREEN,
)

logger = logging.getLogger(__name__)

# The theme of an event seed the curation model added: it is seasonal, but the
# event behind it is not named.
SEASONAL_EVENT_THEME: Final[str] = "seasonal_event"


@dataclass(frozen=True)
class SeasonalRule:
    date: tuple[int, int]  # (month, day)
    inject_days_before: int
    seeds: list[str]


@dataclass(frozen=True)
class EvergreenRule:
    theme: str
    seeds: list[str]


SEASONAL_SEED_RULES: Final[dict[str, SeasonalRule]] = {
    "halloween": SeasonalRule(
        date=(10, 31),
        inject_days_before=45,
        seeds=[
            "halloween costume shirt",
            "spooky mug",
            "trick or treat bag",
            "halloween sweatshirt",
            "witch humor shirt",
            "skeleton design",
            "pumpkin everything",
            "halloween gift",
        ],
    ),
    "christmas": SeasonalRule(
        date=(12, 25),
        inject_days_before=60,
        seeds=[
            "ugly christmas sweater",
            "christmas mug",
            "stocking stuffer",
            "holiday gift shirt",
            "santa humor",
            "elf shirt",
            "christmas family matching",
            "gift for mom christmas",
        ],
    ),
    "valentines_day": SeasonalRule(
        date=(2, 14),
        inject_days_before=30,
        seeds=[
            "valentines gift for him",
            "valentines gift for her",
            "funny valentine shirt",
            "couples matching shirt",
            "anti valentines day mug",
            "galentines day",
        ],
    ),
    "mothers_day": SeasonalRule(
        date=(5, 11),
        inject_days_before=30,
        seeds=[
            "mothers day gift",
            "best mom shirt",
            "mom humor mug",
            "funny mom shirt",
            "mama bear",
            "plant mom",
        ],
    ),
    "fathers_day": SeasonalRule(
        date=(6, 15),
        inject_days_before=30,
        seeds=[
            "fathers day gift",
            "best dad shirt",
            "dad humor mug",
            "funny dad shirt",
            "grill dad",
            "dad bod shirt",
        ],
    ),
    "back_to_school": SeasonalRule(
        date=(8, 1),
        inject_days_before=30,
        seeds=[
            "teacher shirt",
            "first day of school",
            "school bus design",
            "teacher appreciation",
            "funny teacher mug",
        ],
    ),
    "super_bowl": SeasonalRule(
        date=(2, 9),
        inject_days_before=21,
        seeds=[
            "game day shirt",
            "football party",
            "super bowl snacks humor",
            "football sunday",
            "tailgate shirt",
        ],
    ),
}

# Standing themes. They are not tied to a date, so they are always in
# season and they keep the pool from being nothing but a holiday.
EVERGREEN_SEED_RULES: Final[dict[str, EvergreenRule]] = {
    "pets": EvergreenRule(
        theme="pets",
        seeds=[
            "dog mom shirt",
            "cat lover mug",
            "pet parent gift",
            "cute pet sticker",
        ],
    ),
    "careers": EvergreenRule(
        theme="careers",
        seeds=[
            "nurse shirt",
            "office humor mug",
            "work from home hoodie",
            "first responder gift",
        ],
    ),
    "hobbies": EvergreenRule(
        theme="hobbies",
        seeds=[
            "crochet hobby shirt",
            "pickleball shirt",
            "fishing gift",
            "golf lover mug",
        ],
    ),
    "family": EvergreenRule(
        theme="family",
        seeds=[
            "family matching shirt",
            "grandparent gift",
            "sibling reunion shirt",
            "new parent gift",
        ],
    ),
    "wellness": EvergreenRule(
        theme="wellness",
        seeds=[
            "yoga shirt",
            "run club shirt",
            "mental health sticker",
            "water bottle gift",
        ],
    ),
    "food": EvergreenRule(
        theme="food",
        seeds=[
            "coffee lover mug",
            "baking shirt",
            "spice lover sticker",
            "meal prep shirt",
        ],
    ),
    "outdoors": EvergreenRule(
        theme="outdoors",
        seeds=[
            "hiking shirt",
            "camping sticker",
            "national park mug",
            "fishing rod shirt",
        ],
    ),
    "sarcasm": EvergreenRule(
        theme="sarcasm",
        seeds=[
            "sarcastic shirt",
            "introvert mug",
            "chatty shirt",
            "hangover shirt",
        ],
    ),
    "sports": EvergreenRule(
        theme="sports",
        seeds=[
            "bowling night shirt",
            "5k finisher shirt",
            "softball league shirt",
            "golf gift",
        ],
    ),
    "home": EvergreenRule(
        theme="home",
        seeds=[
            "houseplant lover mug",
            "kitchen gift",
            "organized home sticker",
            "cozy blanket slogan",
        ],
    ),
}


def _event_date(today: date, month: int, day: int) -> date:
    """The date of an event, next year when it has just passed."""
    target = date(today.year, month, day)
    if (target - today).days < -30:
        target = date(today.year + 1, month, day)
    return target


def _days_until(today: date, month: int, day: int) -> int:
    return (_event_date(today, month, day) - today).days


def get_upcoming_events(today: date) -> list[dict[str, Any]]:
    events = []
    for name, rule in SEASONAL_SEED_RULES.items():
        days = _days_until(today, rule.date[0], rule.date[1])
        if 0 <= days <= rule.inject_days_before:
            events.append(
                {
                    "name": name,
                    "days_until": days,
                    "seeds": rule.seeds,
                    "event_date": _event_date(today, rule.date[0], rule.date[1]).isoformat(),
                }
            )
    return events


def rotate_seeds(seeds: list[str], offset: int, count: int) -> list[str]:
    """A rotating window of a seed list.

    The same event injects a different slice of its seeds every day, so the
    whole list is used over the season instead of all of it on the first day.
    """
    if not seeds or count <= 0:
        return []
    if count >= len(seeds):
        return list(seeds)
    start = offset % len(seeds)
    return [seeds[(start + index) % len(seeds)] for index in range(count)]


def get_seeds_to_inject(
    today: date,
    per_event: int = CALENDAR_SEEDS_PER_EVENT_PER_DAY,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for event in get_upcoming_events(today):
        for seed in rotate_seeds(event["seeds"], today.toordinal(), per_event):
            pairs.append((seed, event["name"]))
    return pairs


def get_evergreen_seeds_to_inject(
    today: date,
    per_day: int = EVERGREEN_SEEDS_PER_DAY,
) -> list[tuple[str, str, str]]:
    """The standing theme seeds to inject today: (seed, theme, rule name)."""
    pairs: list[tuple[str, str, str]] = []
    for name, rule in EVERGREEN_SEED_RULES.items():
        for seed in rule.seeds:
            pairs.append((seed, rule.theme, name))

    chosen = set(rotate_seeds([pair[0] for pair in pairs], today.toordinal(), per_day))
    return [pair for pair in pairs if pair[0] in chosen]


class CurationDBClient(Protocol):
    def insert_active_seed(
        self,
        query: str,
        promotion_score: int,
        candidate_id: int | None,
        origin: str = ...,
        theme: str | None = ...,
        seasonal_window_end: date | None = ...,
    ) -> int | None: ...

    def insert_curation_log(
        self,
        candidate_id: int | None,
        action: str,
        reasoning: str | None,
        event_context: str | None,
        llm_model: str | None,
    ) -> int: ...


def inject_seasonal_seeds(
    db_client: CurationDBClient,
    today: date | None = None,
    per_event: int = CALENDAR_SEEDS_PER_EVENT_PER_DAY,
) -> dict[str, Any]:
    today = today or datetime.now(tz=UTC).date()
    pairs = get_seeds_to_inject(today, per_event=per_event)
    if not pairs:
        return {"status": "success", "injected": 0, "events": []}

    injected = 0
    events_seen: set[str] = set()
    for seed, event_name in pairs:
        rule = SEASONAL_SEED_RULES[event_name]
        result = db_client.insert_active_seed(
            query=seed,
            promotion_score=0,
            candidate_id=None,
            origin=ORIGIN_CALENDAR,
            theme=event_name,
            seasonal_window_end=_event_date(today, rule.date[0], rule.date[1]),
        )
        if result is not None:
            injected += 1
            db_client.insert_curation_log(
                candidate_id=None,
                action="calendar_inject",
                reasoning=f"Seasonal seed for {event_name}",
                event_context=event_name,
                llm_model=None,
            )
        events_seen.add(event_name)

    logger.info("Injected %d seasonal seeds for events: %s", injected, sorted(events_seen))
    return {
        "status": "success",
        "injected": injected,
        "events": sorted(events_seen),
    }


def inject_evergreen_seeds(
    db_client: CurationDBClient,
    today: date | None = None,
    per_day: int = EVERGREEN_SEEDS_PER_DAY,
) -> dict[str, Any]:
    """Inject the standing theme seeds of the day."""
    today = today or datetime.now(tz=UTC).date()
    pairs = get_evergreen_seeds_to_inject(today, per_day=per_day)
    if not pairs:
        return {"status": "success", "injected": 0, "themes": []}

    injected = 0
    themes_seen: set[str] = set()
    for seed, theme, rule_name in pairs:
        result = db_client.insert_active_seed(
            query=seed,
            promotion_score=0,
            candidate_id=None,
            origin=ORIGIN_EVERGREEN,
            theme=theme,
        )
        if result is not None:
            injected += 1
            db_client.insert_curation_log(
                candidate_id=None,
                action="evergreen_inject",
                reasoning=f"Evergreen theme seed for {theme}",
                event_context=rule_name,
                llm_model=None,
            )
        themes_seen.add(theme)

    logger.info("Injected %d evergreen seeds for themes: %s", injected, sorted(themes_seen))
    return {
        "status": "success",
        "injected": injected,
        "themes": sorted(themes_seen),
    }


def inject_context_seeds(
    db_client: CurationDBClient,
    today: date | None = None,
) -> dict[str, Any]:
    """Inject the calendar seeds and the evergreen theme seeds."""
    today = today or datetime.now(tz=UTC).date()
    seasonal = inject_seasonal_seeds(db_client, today=today)
    evergreen = inject_evergreen_seeds(db_client, today=today)
    return {
        "status": "success",
        "injected": seasonal["injected"] + evergreen["injected"],
        "events": seasonal["events"],
        "themes": evergreen["themes"],
    }
