"""Seed promotion rules and pool sizing (framework-agnostic, TEA-style `update`).

The pool of curated seeds is the raw material the design brief task is made
of, so its size, its reserve and the reuse cooldown are policy, not detail
(issue #192). The limits are configurable per deployment.

The pool also has to stay balanced: a calendar of holidays competes with
standing themes for the same slots, and without a rule the season wins
every time (issue #198). A seed therefore says where it came from
(`origin`) and, for a holiday, which event it belongs to (`theme`).
"""

import os
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final

DEFAULT_MAX_ACTIVE_SEEDS: Final[int] = 200
DEFAULT_MIN_BRIEFABLE_SEEDS: Final[int] = 30
DEFAULT_BRIEF_SEED_COOLDOWN_DAYS: Final[int] = 14


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read an int knob from the environment, falling back to the default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float) -> float:
    """Read a float knob from the environment, falling back to the default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, 0.0), 1.0)


# How many calendar seeds one event may inject per day.
DEFAULT_CALENDAR_SEEDS_PER_EVENT_PER_DAY: Final[int] = 3
# How many evergreen theme seeds are injected per day.
DEFAULT_EVERGREEN_SEEDS_PER_DAY: Final[int] = 4
# How many evergreen seeds the pool always keeps.
DEFAULT_EVERGREEN_RESERVE: Final[int] = 10
# How many seeds of each kind the brief task may see in one batch.
DEFAULT_BRIEF_SEED_SEASONAL_QUOTA: Final[int] = 6
DEFAULT_BRIEF_SEED_EVERGREEN_QUOTA: Final[int] = 8
DEFAULT_BRIEF_SEED_TREND_QUOTA: Final[int] = 6

# Where a seed came from.
ORIGIN_TREND: Final[str] = "trend"
ORIGIN_CALENDAR: Final[str] = "calendar"
ORIGIN_EVERGREEN: Final[str] = "evergreen"
ORIGIN_PIVOT: Final[str] = "pivot"
ORIGIN_EVENT: Final[str] = "event"
ORIGIN_DISCOVERY: Final[str] = "discovery"

SEASONAL_ORIGINS: Final[frozenset[str]] = frozenset({ORIGIN_CALENDAR, ORIGIN_EVENT})

# The source of a trend that was found without a seed to start from.
DISCOVERY_SOURCE: Final = "discovery"

# What a discovery is worth on top of the score of its source.
DEFAULT_DISCOVERY_BONUS: Final[int] = 1000
# What is left of the score of a candidate that only repeats its seed.
DEFAULT_ECHO_FACTOR: Final[float] = 0.5

# What a discovery is worth on top of the score of its source.
DISCOVERY_BONUS: Final[int] = _env_int("LLM_DISCOVERY_BONUS", DEFAULT_DISCOVERY_BONUS, minimum=0)
# What is left of the score of a candidate that only repeats its seed.
ECHO_FACTOR: Final[float] = _env_float("LLM_ECHO_FACTOR", DEFAULT_ECHO_FACTOR)


# How many seeds the pool may hold before the curation archives the surplus.
MAX_ACTIVE_SEEDS: Final[int] = _env_int("LLM_MAX_ACTIVE_SEEDS", DEFAULT_MAX_ACTIVE_SEEDS)
# How many briefable seeds the curation must never archive below.
MIN_BRIEFABLE_SEEDS: Final[int] = _env_int("LLM_MIN_BRIEFABLE_SEEDS", DEFAULT_MIN_BRIEFABLE_SEEDS)
# How long a seed stays out of the brief pool after it produced a brief.
BRIEF_SEED_COOLDOWN_DAYS: Final[int] = _env_int(
    "LLM_BRIEF_SEED_COOLDOWN_DAYS", DEFAULT_BRIEF_SEED_COOLDOWN_DAYS, minimum=0
)
# How many calendar seeds one event may inject per day.
CALENDAR_SEEDS_PER_EVENT_PER_DAY: Final[int] = _env_int(
    "LLM_CALENDAR_SEEDS_PER_EVENT_PER_DAY", DEFAULT_CALENDAR_SEEDS_PER_EVENT_PER_DAY
)
# How many evergreen theme seeds are injected per day.
EVERGREEN_SEEDS_PER_DAY: Final[int] = _env_int(
    "LLM_EVERGREEN_SEEDS_PER_DAY", DEFAULT_EVERGREEN_SEEDS_PER_DAY
)
# How many evergreen seeds the pool always keeps.
EVERGREEN_RESERVE: Final[int] = _env_int("LLM_EVERGREEN_RESERVE", DEFAULT_EVERGREEN_RESERVE)
# How many seeds of each kind the brief task may see in one batch.
BRIEF_SEED_SEASONAL_QUOTA: Final[int] = _env_int(
    "LLM_BRIEF_SEED_SEASONAL_QUOTA", DEFAULT_BRIEF_SEED_SEASONAL_QUOTA
)
BRIEF_SEED_EVERGREEN_QUOTA: Final[int] = _env_int(
    "LLM_BRIEF_SEED_EVERGREEN_QUOTA", DEFAULT_BRIEF_SEED_EVERGREEN_QUOTA
)
BRIEF_SEED_TREND_QUOTA: Final[int] = _env_int(
    "LLM_BRIEF_SEED_TREND_QUOTA", DEFAULT_BRIEF_SEED_TREND_QUOTA
)

STARTER_SEEDS: Final[list[str]] = [
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
]


@dataclass(frozen=True)
class SeedCandidate:
    id: int
    query: str
    source: str
    query_type: str
    score: int
    delta: int
    promotion_score: int
    status: str = "pending"


@dataclass(frozen=True)
class ActiveSeed:
    id: int
    query: str
    promotion_score: int
    origin: str = ORIGIN_TREND
    theme: str | None = None
    seasonal_window_end: date | None = None

    @property
    def is_seasonal(self) -> bool:
        return self.origin in SEASONAL_ORIGINS

    @property
    def is_evergreen(self) -> bool:
        return self.origin == ORIGIN_EVERGREEN

    def is_expired(self, today: date) -> bool:
        """Whether the window of a seasonal seed has passed."""
        return self.seasonal_window_end is not None and self.seasonal_window_end < today


def compute_promotion_score(source: str, query_type: str, score: int, delta: int) -> int:
    """Normalize each source's native score range into 0–10000.

    Without normalization YouTube view counts (billions) dominate Google
    interest (0–100) and Amazon presence (1). Each branch maps its range
    so "very popular in source X" ≈ 10 000 regardless of source.
    """
    if source == "google":
        if query_type == "rising":
            return min(max(delta, 0), 10000)
        if query_type == "top":
            return max(score, 0) * 100
    if source == "tiktok" and query_type in ("hashtag", "sound"):
        return min(max(score, 0) // 1000, 10000)
    if source == "pinterest" and query_type in ("search", "board"):
        return min(max(score, 0), 10000)
    if source in ("amazon", "etsy") and query_type == "search":
        return max(score, 0) * 500
    if source == "x" and query_type in ("hashtag", "search"):
        return min(max(score, 0) // 100, 10000)
    if source == "reddit" and query_type in ("search", "subreddit"):
        return min(max(score, 0), 10000)
    if source == "youtube" and query_type in ("video", "hashtag"):
        return min(max(score, 0) // 100_000, 10000)
    if source == "instagram" and query_type == "hashtag":
        return min(max(score, 0) // 1000, 10000)
    if source == DISCOVERY_SOURCE:
        return min(max(score, 0) * 100, 10000)
    return 0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def echoes_seed(query: str, seed: str) -> bool:
    """Whether a candidate only says back what its seed already said.

    A candidate that adds a word is a new idea; a candidate that repeats the
    seed is the pool talking to itself, and it should not push the good
    candidates down (issue #201).
    """
    query_tokens = _tokens(query)
    seed_tokens = _tokens(seed)
    if not query_tokens or not seed_tokens:
        return False
    return query_tokens <= seed_tokens


def compute_novelty_score(
    source: str,
    query_type: str,
    score: int,
    delta: int,
    query: str = "",
    seed: str = "",
) -> int:
    """The promotion score of a candidate, with novelty taken into account.

    A discovery is worth more than the same signal harvested from a seed,
    because it did not come out of the pool. A candidate that only repeats
    its seed is worth less, because it did not come from anywhere.
    """
    value = compute_promotion_score(source, query_type, score, delta)

    if source == DISCOVERY_SOURCE:
        value += DISCOVERY_BONUS
    if seed and echoes_seed(query, seed):
        value = int(value * ECHO_FACTOR)

    return max(0, min(10000, value))


def _archive_tier(seed: ActiveSeed, today: date) -> int:
    """How willing we are to archive a seed. Lower goes first.

    A seed that already produced a brief is useless until its cooldown ends,
    so it goes first. After that the season goes: a holiday whose day has
    passed, then a holiday that is still ahead. Standing themes are archived
    last, and never below the evergreen reserve.
    """
    if seed.is_seasonal:
        return 1 if seed.is_expired(today) else 2
    if seed.is_evergreen:
        return 4
    return 3


def enforce_cap(
    active_seeds: list[ActiveSeed],
    incoming: int,
    max_active: int = MAX_ACTIVE_SEEDS,
    briefable_ids: Collection[int] = frozenset(),
    reserve: int = 0,
    evergreen_reserve: int = EVERGREEN_RESERVE,
    today: date | None = None,
) -> list[str]:
    """Queries to archive (lowest promotion_score first) so the seed pool stays <= max.

    Seeds the brief task cannot use right now - the ones that already produced
    a brief and are still inside their cooldown - are archived first. Fresh
    seeds are only archived down to `reserve`, so the pool is allowed to stay
    above the cap rather than starve the brief task (issue #192).

    Evergreen seeds are archived last and never below `evergreen_reserve`:
    a holiday season must not leave the pool without standing themes
    (issue #198).
    """
    excess = len(active_seeds) + incoming - max_active
    if excess <= 0:
        return []

    today = today or datetime.now(tz=UTC).date()

    briefable = {seed.id for seed in active_seeds if seed.id in briefable_ids}
    retired = sorted(
        (seed for seed in active_seeds if seed.id not in briefable),
        key=lambda seed: (seed.promotion_score, seed.query),
    )
    fresh = sorted(
        (seed for seed in active_seeds if seed.id in briefable),
        key=lambda seed: (_archive_tier(seed, today), seed.promotion_score, seed.query),
    )

    archives: list[str] = []
    for seed in retired:
        if len(archives) >= excess:
            break
        archives.append(seed.query)

    remaining_briefable = len(fresh)
    evergreen_left = sum(1 for seed in fresh if seed.is_evergreen)
    for seed in fresh:
        if len(archives) >= excess:
            break
        if remaining_briefable - 1 < reserve:
            break
        if seed.is_evergreen and evergreen_left - 1 < evergreen_reserve:
            continue
        archives.append(seed.query)
        remaining_briefable -= 1
        if seed.is_evergreen:
            evergreen_left -= 1

    return archives


def _scale_quotas(quotas: dict[str, int], limit: int) -> dict[str, int]:
    """Fit the bucket quotas into the limit, keeping every bucket represented."""
    total = sum(quotas.values())
    if total <= limit or total == 0:
        return dict(quotas)

    scaled = {name: max(1, value * limit // total) for name, value in quotas.items()}
    while sum(scaled.values()) > limit:
        biggest = max(scaled, key=lambda name: (scaled[name], name))
        scaled[biggest] -= 1
    return scaled


def _bucket_of(seed: ActiveSeed) -> str:
    if seed.is_seasonal:
        return "seasonal"
    if seed.is_evergreen:
        return "evergreen"
    return "trend"


def select_briefable_seeds(
    seeds: Sequence[ActiveSeed],
    limit: int,
    seasonal_quota: int = BRIEF_SEED_SEASONAL_QUOTA,
    evergreen_quota: int = BRIEF_SEED_EVERGREEN_QUOTA,
    trend_quota: int = BRIEF_SEED_TREND_QUOTA,
    today: date | None = None,
) -> list[ActiveSeed]:
    """The seeds the brief task may see, as a mix instead of a top list.

    A flat "highest score first" selection hands the whole batch to the
    season whenever a holiday is trending, because holiday queries are
    exactly the ones that score high in season (issue #198). Each kind of
    seed gets a quota and the rest of the slots go to the best of whatever
    is left.
    """
    if limit <= 0:
        return []

    today = today or datetime.now(tz=UTC).date()
    live = [seed for seed in seeds if not seed.is_expired(today)]

    quotas = _scale_quotas(
        {"seasonal": seasonal_quota, "evergreen": evergreen_quota, "trend": trend_quota},
        limit,
    )

    buckets: dict[str, list[ActiveSeed]] = {"seasonal": [], "evergreen": [], "trend": []}
    for seed in live:
        buckets[_bucket_of(seed)].append(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda seed: (-seed.promotion_score, seed.query))

    chosen: list[ActiveSeed] = []
    taken: set[int] = set()
    for name in ("seasonal", "evergreen", "trend"):
        for seed in buckets[name][: quotas[name]]:
            chosen.append(seed)
            taken.add(seed.id)

    if len(chosen) < limit:
        rest = [seed for seed in live if seed.id not in taken]
        rest.sort(key=lambda seed: (-seed.promotion_score, seed.query))
        chosen.extend(rest[: limit - len(chosen)])

    chosen.sort(key=lambda seed: (-seed.promotion_score, seed.query))
    return chosen
