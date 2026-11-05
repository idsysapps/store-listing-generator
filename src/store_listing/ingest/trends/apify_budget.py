"""Rough Apify cost of one capped harvest.

Free-plan rates from the spend review: Pinterest $0.00345 per pin, X $0.00015
per tweet, TikTok $0.005 per video, Instagram $2.60 per 1,000 posts. Pinterest
and X run twice a week. TikTok and Instagram run weekly. A month is 52/12 weeks.

A harvest asks for at most 100 results. Pinterest at that cap is about $0.35 a
run. All four sources together are about $6.42 a month. The earlier $8 figure
counted X at the uncapped invoice rate of about $0.18 a run, which was roughly
1,200 tweets rather than 100.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

HARVEST_RESULT_CAP: int = 100
PAID_SEED_LIMIT: int = 20
CHARGE_CAP_USD: Decimal = Decimal("0.50")

_WEEKS_PER_MONTH: Decimal = Decimal(52) / Decimal(12)
_CENT: Decimal = Decimal("0.01")
_TENTH_CENT: Decimal = Decimal("0.001")

# Order is the order the all-in line adds them up.
SOURCES: tuple[str, ...] = ("pinterest", "x", "tiktok", "instagram")

_UNIT_PRICE_USD: dict[str, Decimal] = {
    "pinterest": Decimal("0.00345"),
    "x": Decimal("0.00015"),
    "tiktok": Decimal("0.005"),
    "instagram": Decimal("0.0026"),
}

_HARVESTS_PER_WEEK: dict[str, int] = {
    "pinterest": 2,
    "x": 2,
    "tiktok": 1,
    "instagram": 1,
}


def _money(value: Decimal) -> Decimal:
    """Cents, or tenths of a cent when the charge is under ten cents."""
    cents = value.quantize(_CENT, rounding=ROUND_HALF_UP)
    if cents < Decimal("0.10"):
        return value.quantize(_TENTH_CENT, rounding=ROUND_HALF_UP)
    return cents


def per_run_usd(source: str, results: int = HARVEST_RESULT_CAP) -> Decimal:
    """What one capped harvest of this source costs."""
    return _money(_UNIT_PRICE_USD[source] * results)


def per_month_usd(source: str, results: int = HARVEST_RESULT_CAP) -> Decimal:
    """What this source costs across its scheduled harvests."""
    runs = _UNIT_PRICE_USD[source] * results * _HARVESTS_PER_WEEK[source] * _WEEKS_PER_MONTH
    return _money(runs)


def all_in_monthly_usd(results: int = HARVEST_RESULT_CAP) -> Decimal:
    """The four sources added from their displayed monthly amounts."""
    return sum((per_month_usd(source, results) for source in SOURCES), Decimal(0))


def format_budget(source: str) -> str:
    """One line a harvest task can log before it spends anything."""
    return (
        f"Apify {source} harvest: up to {HARVEST_RESULT_CAP} results, "
        f"about ${per_run_usd(source)} this run, "
        f"about ${per_month_usd(source)} per month"
    )


def format_all_in() -> str:
    """One line the scheduler can log when it loads."""
    return (
        f"Apify all-in at {HARVEST_RESULT_CAP} results per harvest: "
        f"about ${all_in_monthly_usd()} per month"
    )


def actor_call_kwargs(run_input: dict[str, Any]) -> dict[str, Any]:
    """Arguments every paid actor call shares.

    ``max_items`` is Apify's own billing cap. ``max_total_charge_usd`` stops a
    run that ignores the result cap before it can spend another monthly credit.
    """
    return {
        "run_input": run_input,
        "run_timeout": timedelta(minutes=4),
        "max_items": HARVEST_RESULT_CAP,
        "max_total_charge_usd": CHARGE_CAP_USD,
    }


def select_paid_seeds(seeds: list[str]) -> list[str]:
    """The seeds one paid run is allowed to ask about.

    The brief pool can hold 200. Paying to re-scrape all of them is what burned
    the monthly credit in a single harvest.
    """
    chosen: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        text = seed.strip()
        folded = text.lower()
        if not text or folded in seen:
            continue
        seen.add(folded)
        chosen.append(text)
        if len(chosen) >= PAID_SEED_LIMIT:
            break
    return chosen


def per_target_limit(target_count: int, env_var: str | None = None) -> int:
    """How many results each target may return so the harvest stays at the cap.

    An env value can tighten that share. It cannot raise it.
    """
    count = max(target_count, 1)
    share = max(1, HARVEST_RESULT_CAP // count)
    if env_var is None:
        return share
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return share
    try:
        tighter = int(raw)
    except ValueError:
        return share
    return min(share, max(1, tighter))


def _match_key(label: str, keys: list[str]) -> str | None:
    folded = label.strip().lstrip("#").lower()
    for key in keys:
        if key.strip().lstrip("#").lower() == folded:
            return key
    return None


def bucket_items[Item](
    keys: list[str],
    items: list[dict[str, Any]],
    build: Callable[[dict[str, Any]], Item],
    label_of: Callable[[dict[str, Any]], str | None],
) -> dict[str, list[Item]]:
    """Split actor rows back onto the seeds that asked for them.

    A row with no seed label is kept once, on the first seed, so a batched run
    still stores what it paid for.
    """
    grouped: dict[str, list[Item]] = {key: [] for key in keys}
    if not keys:
        return grouped
    for item in items:
        label = label_of(item)
        match = _match_key(label, keys) if label else None
        grouped[match or keys[0]].append(build(item))
    return grouped
