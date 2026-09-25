"""X (Twitter) micro-trend harvester.

Scrapes X search timelines for the active seed keywords. Uses an Apify actor
(such as ``bernardo/x-scraper``) when APIFY_API_TOKEN is set; otherwise falls
back to a best-effort parse of the public search page (login-walled, so it
typically yields nothing without a token). Co-occurring hashtags and top
keywords are aggregated into the unified trend store (trend_queries +
trend_scores with ``source='x'``) and fed to the seed pool via
``seed_candidates``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol, runtime_checkable

import httpx

from store_listing.orchestration.promotion import compute_promotion_score

from .google_trends import DatabaseClient
from .schemas import TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

DEFAULT_X_ACTOR: Final[str] = "xquik/x-tweet-scraper"

_ENGAGEMENT_KEYS: Final[tuple[str, ...]] = (
    "likeCount",
    "retweetCount",
    "replyCount",
    "viewCount",
    "favoriteCount",
    "views",
)

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "because",
        "been",
        "before",
        "being",
        "between",
        "could",
        "did",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "from",
        "have",
        "having",
        "here",
        "just",
        "like",
        "more",
        "most",
        "much",
        "must",
        "only",
        "other",
        "our",
        "over",
        "some",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
        "dont",
        "can",
        "the",
        "and",
        "for",
        "you",
        "not",
        "are",
        "out",
        "its",
    }
)

_SCRIPT_RE = re.compile(
    r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',
    re.DOTALL,
)

_DEFAULT_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


@dataclass(frozen=True)
class XTweet:
    tweet_id: str
    text: str
    handle: str
    like_count: int
    retweet_count: int
    reply_count: int
    view_count: int
    hashtags: tuple[str, ...]

    def engagements(self) -> int:
        return self.like_count + self.retweet_count + self.reply_count + self.view_count


@runtime_checkable
class XScraperGateway(Protocol):
    def scrape_query(self, query: str) -> list[XTweet]: ...


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _hashtags(item: dict[str, Any]) -> tuple[str, ...]:
    raw = item.get("hashtags") or []
    if not isinstance(raw, list):
        raw = []
    names: list[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            name = entry.get("text") or entry.get("name")
        elif isinstance(entry, str):
            name = entry
        else:
            name = None
        if name:
            names.append(str(name).lstrip("#").lower())
    if names:
        return tuple(dict.fromkeys(names))
    text = item.get("text") or item.get("fullText") or ""
    return tuple(dict.fromkeys(re.findall(r"#([a-z0-9_]+)", str(text).lower())))


def _x_tweet(item: dict[str, Any]) -> XTweet:
    user: dict[str, Any] = {}
    if isinstance(item.get("user"), dict):
        user = item["user"]
    return XTweet(
        tweet_id=str(item.get("id") or item.get("tweet_id") or ""),
        text=str(item.get("text") or item.get("fullText") or item.get("tweet") or ""),
        handle=str(user.get("username") or item.get("username") or item.get("author") or ""),
        like_count=_to_int(
            item.get("likeCount")
            or item.get("favoriteCount")
            or item.get("likes")
            or item.get("favorite_count")
        ),
        retweet_count=_to_int(
            item.get("retweetCount") or item.get("retweet_count") or item.get("retweets")
        ),
        reply_count=_to_int(
            item.get("replyCount") or item.get("reply_count") or item.get("replies")
        ),
        view_count=_to_int(
            item.get("viewCount")
            or item.get("views")
            or item.get("impressionCount")
            or item.get("impressions")
        ),
        hashtags=_hashtags(item),
    )


class ApifyXGateway:
    """Apify actor wrapper (default bernardo/x-scraper)."""

    def __init__(
        self,
        token: str,
        actor_id: str = DEFAULT_X_ACTOR,
        apify_client_cls: Callable[[str], Any] | None = None,
    ) -> None:
        self.token = token
        self.actor_id = actor_id
        self._apify_client_cls = apify_client_cls
        self._client: Any | None = None

    def _apify(self) -> Any:
        if self._client is None:
            self._client = self._resolve_client_class()(self.token)
        return self._client

    def _resolve_client_class(self) -> Any:
        if self._apify_client_cls is not None:
            return self._apify_client_cls
        from apify_client import ApifyClient  # type: ignore[import-not-found]

        return ApifyClient

    def _api_actor_ref(self) -> str:
        return self.actor_id.replace("/", "~")

    def scrape_query(self, query: str) -> list[XTweet]:
        run = (
            self._apify()
            .actor(self._api_actor_ref())
            .call(
                run_input={"searchTerms": [query], "maxItems": 30},
                run_timeout=timedelta(minutes=4),
            )
        )
        if run is None:
            return []
        dataset_id = (
            run.get("defaultDatasetId") if isinstance(run, dict) else run.default_dataset_id
        )
        raw_items = self._apify().dataset(dataset_id).list_items().items
        return [_x_tweet(item) for item in raw_items]


class XWebGateway:
    """Free fallback: parse the public search page's embedded state (best-effort)."""

    BASE_URL = "https://x.com/search"

    def scrape_query(self, query: str) -> list[XTweet]:
        response = httpx.get(
            self.BASE_URL,
            params={"q": query, "f": "live"},
            headers=_DEFAULT_HEADERS,
            timeout=20,
            follow_redirects=True,
        )
        response.raise_for_status()
        return self._parse_html(response.text)

    @staticmethod
    def _parse_html(html: str) -> list[XTweet]:
        tweets: list[XTweet] = []
        for match in _SCRIPT_RE.finditer(html):
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            tweets.extend(_extract_tweets(data))
        return tweets


def _extract_tweets(data: Any) -> list[XTweet]:
    entries = [
        item
        for item in _iter_dicts(data)
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and any(
            isinstance(item.get(key), (int, float)) if key in item else False
            for key in _ENGAGEMENT_KEYS
        )
    ]
    return [_x_tweet(item) for item in entries]


class XClient:
    """Two-step contract: raw rows to trend store, discoveries to seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        gateway: XScraperGateway | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.gateway = gateway or self._build_gateway()

    @staticmethod
    def _build_gateway() -> XScraperGateway:
        token = os.environ.get("APIFY_API_TOKEN")
        if token:
            actor_id = os.environ.get("APIFY_ACTOR_X") or DEFAULT_X_ACTOR
            return ApifyXGateway(token=token, actor_id=actor_id)
        return XWebGateway()

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        all_results: list[TrendResult] = []
        failed: list[str] = []

        for seed in request.seed_keywords:
            try:
                tweets = self.gateway.scrape_query(seed)
                for result in self._derive_results(seed, tweets):
                    self._store(seed, result)
                    all_results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to scrape X for %s: %s", seed, e)
                failed.append(seed)

        return all_results, failed

    @staticmethod
    def _seed_tokens(seed: str) -> set[str]:
        return set(re.findall(r"[a-z]{3,}", seed.lower()))

    def _derive_results(self, seed: str, tweets: list[XTweet]) -> list[TrendResult]:
        hash_score: dict[str, int] = {}
        hash_count: dict[str, int] = {}
        kw_score: dict[str, int] = {}
        kw_count: dict[str, int] = {}
        seed_tokens = self._seed_tokens(seed)
        normalized_seed = re.sub(r"[^a-z0-9]", "", seed.lower())

        for tweet in tweets:
            engagement = tweet.engagements()
            for tag in tweet.hashtags:
                if tag == normalized_seed:
                    continue
                key = f"#{tag}"
                hash_score[key] = hash_score.get(key, 0) + engagement
                hash_count[key] = hash_count.get(key, 0) + 1
            plain = re.sub(r"#\S+", "", tweet.text)
            for word in re.findall(r"[a-z]{3,}", plain.lower()):
                if word in _STOPWORDS or word in seed_tokens:
                    continue
                kw_score[word] = kw_score.get(word, 0) + engagement
                kw_count[word] = kw_count.get(word, 0) + 1

        timestamp = datetime.now(UTC)
        results: list[TrendResult] = []
        for tag in sorted(hash_score):
            results.append(
                TrendResult(
                    query=tag,
                    score=hash_score[tag],
                    delta=hash_count[tag],
                    region="US",
                    query_type="hashtag",
                    source="x",
                    fetched_at=timestamp,
                )
            )
        for word in sorted(kw_score):
            results.append(
                TrendResult(
                    query=word,
                    score=kw_score[word],
                    delta=kw_count[word],
                    region="US",
                    query_type="search",
                    source="x",
                    fetched_at=timestamp,
                )
            )
        return results

    def _store(self, seed: str, result: TrendResult) -> None:
        query_id = self.db_client.insert_trend_query(seed, result.query, source="x")
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source="x",
            trend_direction=result.trend_direction,
        )
        self._upsert_candidate(seed, result)

    def _upsert_candidate(self, seed: str, result: TrendResult) -> None:
        if result.query_type not in ("hashtag", "search"):
            return
        if result.query.lower() == seed.lower():
            return
        promotion_score = compute_promotion_score(
            result.source, result.query_type, result.score, result.delta
        )
        self.db_client.upsert_seed_candidate(
            source_seed=seed,
            query=result.query,
            source=result.source,
            query_type=result.query_type,
            score=result.score,
            delta=result.delta,
            promotion_score=promotion_score,
        )
