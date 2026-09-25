"""Pure, framework-agnostic seed promotion rules (TEA-style `update` shared by tasks)."""

from dataclasses import dataclass
from typing import Final

MAX_ACTIVE_SEEDS: Final[int] = 50

PROMOTION_THRESHOLDS: Final[dict[str, dict[str, int]]] = {
    "google": {"rising_min_delta": 5000, "top_min_cross_seeds": 2},
    "tiktok": {
        "hashtag_min_virality": 100_000,
        "hashtag_min_videos": 5,
        "sound_min_usage": 10,
        "min_cross_seeds": 2,
    },
    "pinterest": {
        "search_min_repins": 500,
        "search_min_pins": 3,
        "board_min_repins": 1_000,
    },
    "amazon": {"min_cross_seeds": 2},
    "etsy": {"min_cross_seeds": 2},
    "x": {
        "hashtag_min_virality": 100_000,
        "hashtag_min_tweets": 5,
        "search_min_tweets": 5,
        "min_cross_seeds": 2,
    },
    "reddit": {
        "search_min_upvotes": 500,
        "subreddit_min_posts": 3,
        "min_cross_seeds": 2,
    },
    "youtube": {
        "video_min_views": 50_000,
        "hashtag_min_videos": 5,
        "min_cross_seeds": 2,
    },
    "instagram": {
        "hashtag_min_virality": 100_000,
        "hashtag_min_posts": 5,
        "min_cross_seeds": 2,
    },
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
    if candidate.source == "google":
        if candidate.query_type == "rising":
            return candidate.delta >= rules.get("rising_min_delta", 0)
        if candidate.query_type == "top":
            return cross_seed_count >= rules.get("top_min_cross_seeds", 0)
    if candidate.source == "tiktok":
        if candidate.query_type == "hashtag":
            virality_met = candidate.score >= rules.get(
                "hashtag_min_virality", 0
            ) and candidate.delta >= rules.get("hashtag_min_videos", 0)
        elif candidate.query_type == "sound":
            virality_met = candidate.score >= rules.get(
                "hashtag_min_virality", 0
            ) and candidate.delta >= rules.get("sound_min_usage", 0)
        else:
            return False
        return virality_met or cross_seed_count >= rules.get("min_cross_seeds", 0)
    if candidate.source == "pinterest":
        if candidate.query_type == "search":
            return candidate.score >= rules.get(
                "search_min_repins", 0
            ) and candidate.delta >= rules.get("search_min_pins", 0)
        if candidate.query_type == "board":
            return candidate.score >= rules.get("board_min_repins", 0)
        return False
    if candidate.source in ("amazon", "etsy") and candidate.query_type == "search":
        return cross_seed_count >= rules.get("min_cross_seeds", 0)
    if candidate.source == "x":
        if candidate.query_type == "hashtag":
            virality_met = candidate.score >= rules.get(
                "hashtag_min_virality", 0
            ) and candidate.delta >= rules.get("hashtag_min_tweets", 0)
        elif candidate.query_type == "search":
            virality_met = candidate.delta >= rules.get("search_min_tweets", 0)
        else:
            return False
        return virality_met or cross_seed_count >= rules.get("min_cross_seeds", 0)
    if candidate.source == "reddit":
        if candidate.query_type == "search":
            return candidate.score >= rules.get(
                "search_min_upvotes", 0
            ) or cross_seed_count >= rules.get("min_cross_seeds", 0)
        if candidate.query_type == "subreddit":
            return candidate.delta >= rules.get("subreddit_min_posts", 0)
        return False
    if candidate.source == "youtube":
        if candidate.query_type == "video":
            return candidate.score >= rules.get(
                "video_min_views", 0
            ) or cross_seed_count >= rules.get("min_cross_seeds", 0)
        if candidate.query_type == "hashtag":
            return candidate.delta >= rules.get(
                "hashtag_min_videos", 0
            ) or cross_seed_count >= rules.get("min_cross_seeds", 0)
        return False
    if candidate.source == "instagram":
        if candidate.query_type == "hashtag":
            virality_met = candidate.score >= rules.get(
                "hashtag_min_virality", 0
            ) and candidate.delta >= rules.get("hashtag_min_posts", 0)
            return virality_met or cross_seed_count >= rules.get("min_cross_seeds", 0)
        return False
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
