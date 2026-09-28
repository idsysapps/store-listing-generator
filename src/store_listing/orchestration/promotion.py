"""Pure, framework-agnostic seed promotion rules (TEA-style `update` shared by tasks)."""

from dataclasses import dataclass
from typing import Final

MAX_ACTIVE_SEEDS: Final[int] = 50

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
) -> list[str]:
    """Queries to archive (lowest promotion_score first) so the seed pool stays <= max."""
    excess = len(active_seeds) + incoming - max_active
    if excess <= 0:
        return []
    ranked = sorted(active_seeds, key=lambda seed: (seed.promotion_score, seed.query))
    return [seed.query for seed in ranked[:excess]]
