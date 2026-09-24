"""Pure, framework-agnostic seed promotion rules (TEA-style `update` shared by tasks)."""

from dataclasses import dataclass
from typing import Final

MAX_ACTIVE_SEEDS: Final[int] = 50

PROMOTION_THRESHOLDS: Final[dict[str, dict[str, int]]] = {
    "google": {"rising_min_delta": 5000, "top_min_cross_seeds": 2},
    "tiktok": {},
    "pinterest": {},
    "amazon": {},
}

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
    """Per-source promotion score at candidate creation.

    Only google rules are live today; tiktok/pinterest/amazon score 0 until their
    ingesters (#2/#3) define thresholds.
    """
    if source == "google":
        if query_type == "rising":
            return max(delta, 0)
        if query_type == "top":
            return max(score, 0)
    return 0


def candidates_to_promote(
    candidates: list[SeedCandidate],
    cross_seed_counts: dict[str, int],
) -> list[int]:
    """Return ids of pending candidates that meet their source's promotion rules."""
    promoted: list[int] = []
    for candidate in candidates:
        if candidate.status != "pending":
            continue
        if _rule_met(candidate, cross_seed_counts.get(candidate.query, 0)):
            promoted.append(candidate.id)
    return promoted


def _rule_met(candidate: SeedCandidate, cross_seed_count: int) -> bool:
    rules = PROMOTION_THRESHOLDS.get(candidate.source, {})
    if candidate.source == "google" and candidate.query_type == "rising":
        return candidate.delta >= rules.get("rising_min_delta", 0)
    if candidate.source == "google" and candidate.query_type == "top":
        return cross_seed_count >= rules.get("top_min_cross_seeds", 0)
    return False


def enforce_cap(
    active_seeds: list[ActiveSeed],
    incoming: int,
    max_active: int = MAX_ACTIVE_SEEDS,
) -> list[str]:
    """Queries to archive (lowest promotion_score first) so the seed pool stays <= max."""
    excess = len(active_seeds) + incoming - max_active
    if excess <= 0:
        return []
    ranked = sorted(active_seeds, key=lambda seed: (seed.promotion_score, seed.query))
    return [seed.query for seed in ranked[:excess]]
