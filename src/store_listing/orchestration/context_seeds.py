"""Rule-based calendar seed injector for predictable seasonal trends.

Runs before harvesters to pre-load active_seeds with holiday/event seeds
at the right lead time. No LLM cost — pure date arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeasonalRule:
    date: tuple[int, int]  # (month, day)
    inject_days_before: int
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


def _days_until(today: date, month: int, day: int) -> int:
    target = date(today.year, month, day)
    delta = (target - today).days
    if delta < -30:
        target = date(today.year + 1, month, day)
        delta = (target - today).days
    return delta


def get_upcoming_events(today: date) -> list[dict[str, Any]]:
    events = []
    for name, rule in SEASONAL_SEED_RULES.items():
        days = _days_until(today, rule.date[0], rule.date[1])
        if 0 <= days <= rule.inject_days_before:
            events.append({"name": name, "days_until": days, "seeds": rule.seeds})
    return events


def get_seeds_to_inject(today: date) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for event in get_upcoming_events(today):
        for seed in event["seeds"]:
            pairs.append((seed, event["name"]))
    return pairs


class CurationDBClient(Protocol):
    def insert_active_seed(
        self, query: str, promotion_score: int, candidate_id: int | None
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
) -> dict[str, Any]:
    today = today or date.today()
    pairs = get_seeds_to_inject(today)
    if not pairs:
        return {"status": "success", "injected": 0, "events": []}

    injected = 0
    events_seen: set[str] = set()
    for seed, event_name in pairs:
        result = db_client.insert_active_seed(query=seed, promotion_score=0, candidate_id=None)
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
