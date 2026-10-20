"""Seed promotion rules and pool sizing (framework-agnostic, TEA-style `update`).

The pool of curated seeds is the raw material the design brief task is made
of, so its size, its reserve and the reuse cooldown are policy, not detail
(issue #192). The limits are configurable per deployment.
"""

import os
from collections.abc import Collection
from dataclasses import dataclass
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


# How many seeds the pool may hold before the curation archives the surplus.
MAX_ACTIVE_SEEDS: Final[int] = _env_int("LLM_MAX_ACTIVE_SEEDS", DEFAULT_MAX_ACTIVE_SEEDS)
# How many briefable seeds the curation must never archive below.
MIN_BRIEFABLE_SEEDS: Final[int] = _env_int("LLM_MIN_BRIEFABLE_SEEDS", DEFAULT_MIN_BRIEFABLE_SEEDS)
# How long a seed stays out of the brief pool after it produced a brief.
BRIEF_SEED_COOLDOWN_DAYS: Final[int] = _env_int(
    "LLM_BRIEF_SEED_COOLDOWN_DAYS", DEFAULT_BRIEF_SEED_COOLDOWN_DAYS, minimum=0
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
    return 0


def enforce_cap(
    active_seeds: list[ActiveSeed],
    incoming: int,
    max_active: int = MAX_ACTIVE_SEEDS,
    briefable_ids: Collection[int] = frozenset(),
    reserve: int = 0,
) -> list[str]:
    """Queries to archive (lowest promotion_score first) so the seed pool stays <= max.

    Seeds the brief task cannot use right now - the ones that already produced
    a brief and are still inside their cooldown - are archived first. Fresh
    seeds are only archived down to `reserve`, so the pool is allowed to stay
    above the cap rather than starve the brief task (issue #192).
    """
    excess = len(active_seeds) + incoming - max_active
    if excess <= 0:
        return []

    briefable = {seed.id for seed in active_seeds if seed.id in briefable_ids}
    retired = sorted(
        (seed for seed in active_seeds if seed.id not in briefable),
        key=lambda seed: (seed.promotion_score, seed.query),
    )
    fresh = sorted(
        (seed for seed in active_seeds if seed.id in briefable),
        key=lambda seed: (seed.promotion_score, seed.query),
    )

    archives: list[str] = []
    for seed in retired:
        if len(archives) >= excess:
            break
        archives.append(seed.query)

    remaining_briefable = len(fresh)
    for seed in fresh:
        if len(archives) >= excess:
            break
        if remaining_briefable - 1 < reserve:
            break
        archives.append(seed.query)
        remaining_briefable -= 1

    return archives
