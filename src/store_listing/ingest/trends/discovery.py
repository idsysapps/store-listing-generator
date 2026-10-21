"""Trend discovery that does not start from the seed pool.

Every harvester used to be fed the active seeds, so a new trend could only
ever be an expansion of a seed that was itself derived from an earlier seed.
A holiday therefore reproduced itself, and a non-seasonal idea could only
enter the pool through the calendar (issue #201).

This module finds trends in inputs that stand on their own:

- the real-time trending list, which is not derived from anything we did;
- the theme subreddits, which are about a standing theme instead of a season;
- the marketplace search boxes, asked for a theme instead of for a seed.

A discovery is stored like any other trend, but it is marked as a discovery
and it is never used as a harvest input, so it cannot feed the loop it came
from.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol
from xml.etree import ElementTree

import httpx

from store_listing.orchestration.promotion import (
    DISCOVERY_SOURCE,
    compute_novelty_score,
)

from .google_trends import DatabaseClient
from .schemas import QueryType, TrendResult

logger = logging.getLogger(__name__)

GOOGLE_TRENDING_RSS_URL: Final[str] = "https://trends.google.com/trending/rss"

DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# The standing themes the pool is missing when the calendar is in charge.
# The prefixes are what the marketplace search boxes are asked for.
DISCOVERY_CATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "pets": ("dog ", "cat ", "pet "),
    "gardening": ("plant ", "garden ", "succulent "),
    "hobbies": ("crochet ", "knitting ", "woodworking ", "fishing "),
    "food": ("coffee ", "baking ", "spice "),
    "home": ("houseplant ", "kitchen ", "organization "),
    "sports": ("pickleball ", "bowling ", "golf "),
    "wellness": ("yoga ", "running ", "meditation "),
    "trades": ("nurse ", "electrician ", "carpenter "),
    "travel": ("camper ", "road trip ", "beach "),
    "music": ("guitar ", "vinyl ", "band "),
}

# How many of the categories are asked for in one run.
DEFAULT_DISCOVERY_CATEGORIES_PER_RUN: Final[int] = 4


@dataclass(frozen=True)
class TrendingTopic:
    """A topic from the real-time trending list."""

    title: str
    traffic: str = ""


@dataclass(frozen=True)
class DiscoveryRequest:
    """What to look for in one discovery run."""

    categories: tuple[str, ...] = tuple(DISCOVERY_CATEGORIES)
    geo: str = "US"
    trending: bool = True


class TrendingGateway(Protocol):
    def trending(self, geo: str = "US") -> list[TrendingTopic]: ...


class SuggestionGateway(Protocol):
    def suggest(self, prefix: str) -> list[str]: ...


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_trending_rss(payload: str) -> list[TrendingTopic]:
    """The topics of a trending feed. An unreadable feed reads as no topics."""
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        logger.warning("The trending feed could not be read")
        return []

    topics: list[TrendingTopic] = []
    for item in root.iter():
        if _local_name(item.tag) != "item":
            continue
        title = ""
        traffic = ""
        for child in item:
            name = _local_name(child.tag)
            if name == "title":
                title = (child.text or "").strip()
            elif name == "approx_traffic":
                traffic = (child.text or "").strip()
        if title:
            topics.append(TrendingTopic(title=title, traffic=traffic))
    return topics


class GoogleTrendingGateway:
    """The public real-time trending feed of Google Trends."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._http = client or httpx.Client(
            headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=15, follow_redirects=True
        )

    def trending(self, geo: str = "US") -> list[TrendingTopic]:
        response = self._http.get(GOOGLE_TRENDING_RSS_URL, params={"geo": geo})
        response.raise_for_status()
        return parse_trending_rss(response.text)


class DiscoveryHarvester:
    """Finds trends that do not come from the seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        trending: TrendingGateway | None = None,
        amazon: SuggestionGateway | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.trending = trending or GoogleTrendingGateway()
        self.amazon = amazon

    def harvest_and_store(
        self,
        request: DiscoveryRequest,
    ) -> tuple[list[TrendResult], list[str]]:
        """Find trends and store them. Returns the results and the failures."""
        results: list[TrendResult] = []
        failed: list[str] = []

        if request.trending:
            results.extend(self._harvest_trending(request, failed))
        results.extend(self._harvest_marketplace(request, failed))

        return results, failed

    def _harvest_trending(
        self,
        request: DiscoveryRequest,
        failed: list[str],
    ) -> list[TrendResult]:
        results: list[TrendResult] = []
        try:
            topics = self.trending.trending(request.geo)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch the trending list: %s", e)
            failed.append("trending")
            return results

        for position, topic in enumerate(topics):
            result = self._store(
                category="trending",
                query=topic.title.strip().lower(),
                score=max(1, len(topics) - position),
                delta=0,
                query_type="top",
            )
            results.append(result)
        return results

    def _harvest_marketplace(
        self,
        request: DiscoveryRequest,
        failed: list[str],
    ) -> list[TrendResult]:
        if self.amazon is None:
            return []

        results: list[TrendResult] = []
        for category in request.categories:
            for prefix in DISCOVERY_CATEGORIES.get(category, ()):
                try:
                    suggestions = self.amazon.suggest(prefix)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to fetch %s suggestions for %s: %s", prefix, category, e)
                    failed.append(f"{category}:{prefix}")
                    continue

                for position, suggestion in enumerate(suggestions):
                    query = suggestion.strip().lower()
                    if not query or query == prefix.strip().lower():
                        continue
                    results.append(
                        self._store(
                            category=category,
                            query=query,
                            score=max(1, 11 - position),
                            delta=0,
                            query_type="search",
                        )
                    )
        return results

    def _store(
        self,
        category: str,
        query: str,
        score: int,
        delta: int,
        query_type: QueryType,
    ) -> TrendResult:
        """Write a discovery to the trend store and to the seed pool."""
        result = TrendResult(
            query=query,
            score=score,
            delta=delta,
            region="US",
            query_type=query_type,
            source=DISCOVERY_SOURCE,
            fetched_at=datetime.now(UTC),
        )
        query_id = self.db_client.insert_trend_query(
            seed_keyword=category, query=result.query, source=DISCOVERY_SOURCE
        )
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source=DISCOVERY_SOURCE,
        )
        promotion_score = compute_novelty_score(
            DISCOVERY_SOURCE,
            result.query_type,
            result.score,
            result.delta,
            query=result.query,
            seed=category,
        )
        self.db_client.upsert_seed_candidate(
            source_seed=category,
            query=result.query,
            source=DISCOVERY_SOURCE,
            query_type=result.query_type,
            score=result.score,
            delta=result.delta,
            promotion_score=promotion_score,
        )
        return result


def discovery_categories_per_run() -> int:
    """How many categories one run asks for."""
    raw = os.environ.get("DISCOVERY_CATEGORIES_PER_RUN", "")
    if raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_DISCOVERY_CATEGORIES_PER_RUN


def rotate_categories(
    categories: tuple[str, ...],
    offset: int,
    count: int,
) -> tuple[str, ...]:
    """A rotating window of the categories, so every category gets its turn."""
    if not categories or count <= 0:
        return ()
    if count >= len(categories):
        return categories
    start = offset % len(categories)
    return tuple(categories[(start + index) % len(categories)] for index in range(count))
