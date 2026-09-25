"""Reddit micro-trend harvester.

Uses Reddit's OAuth2 API (oauth.reddit.com) with client_credentials grant
for server-to-server access. Requires REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET from https://www.reddit.com/prefs/apps (script type).

Rate limit: ~100 requests/minute with OAuth. Tokens expire after 1 hour
and are refreshed automatically.

Discoveries are written to the unified trend store (trend_queries +
trend_scores with ``source='reddit'``) and fed to the seed pool via
``seed_candidates``.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

import httpx

from store_listing.orchestration.promotion import compute_promotion_score

from .google_trends import DatabaseClient
from .schemas import TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

_USER_AGENT: Final[str] = "script:store-listing-bot:v1.0 (by /u/nene-store-generation)"

_TOKEN_URL: Final[str] = "https://www.reddit.com/api/v1/access_token"

SEARCH_URL: Final[str] = "https://oauth.reddit.com/search"

SUBREDDIT_HOT_URL: Final[str] = "https://oauth.reddit.com/r/{sub}/hot"

POD_SUBREDDITS: Final[tuple[str, ...]] = (
    "funny",
    "memes",
    "oddlyspecific",
    "TargetedShirts",
    "starterpacks",
    "shutupandtakemymoney",
    "ATBGE",
)

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "about",
        "after",
        "been",
        "before",
        "could",
        "does",
        "from",
        "have",
        "just",
        "like",
        "more",
        "only",
        "other",
        "some",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "your",
        "the",
        "and",
        "for",
        "you",
        "not",
        "are",
        "was",
        "can",
        "its",
        "has",
    }
)


@dataclass(frozen=True)
class RedditPost:
    post_id: str
    title: str
    subreddit: str
    score: int
    num_comments: int
    upvote_ratio: float


@runtime_checkable
class RedditGateway(Protocol):
    def search(self, query: str) -> list[RedditPost]: ...

    def subreddit_hot(self, subreddit: str) -> list[RedditPost]: ...


def _post(item: dict[str, Any]) -> RedditPost:
    data = item.get("data") or item
    return RedditPost(
        post_id=str(data.get("id") or data.get("name") or ""),
        title=str(data.get("title") or ""),
        subreddit=str(data.get("subreddit") or ""),
        score=int(data.get("score") or data.get("ups") or 0),
        num_comments=int(data.get("num_comments") or 0),
        upvote_ratio=float(data.get("upvote_ratio") or 0.0),
    )


def _extract_posts(payload: dict[str, Any]) -> list[RedditPost]:
    data = payload.get("data") or {}
    children = data.get("children") or []
    return [_post(child) for child in children if isinstance(child, dict)]


class RedditOAuthGateway:
    """OAuth2 client_credentials gateway for server-to-server Reddit API access."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._client_id = client_id or os.environ.get("REDDIT_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
        self._http = client or httpx.Client(
            headers={"User-Agent": _USER_AGENT}, timeout=15, follow_redirects=True
        )
        self._token: str = ""
        self._token_expires_at: float = 0

    @property
    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def _ensure_token(self) -> None:
        if self._token and time.monotonic() < self._token_expires_at:
            return
        if not self.configured:
            raise RuntimeError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are required")
        response = self._http.post(
            _TOKEN_URL,
            auth=(self._client_id, self._client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expires_at = time.monotonic() + data.get("expires_in", 3600) - 60

    def _auth_headers(self) -> dict[str, str]:
        self._ensure_token()
        return {"Authorization": f"bearer {self._token}", "User-Agent": _USER_AGENT}

    def search(self, query: str) -> list[RedditPost]:
        response = self._http.get(
            SEARCH_URL,
            params={"q": query, "sort": "hot", "t": "day", "limit": 25},
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        return _extract_posts(response.json())

    def subreddit_hot(self, subreddit: str) -> list[RedditPost]:
        response = self._http.get(
            SUBREDDIT_HOT_URL.format(sub=subreddit),
            params={"limit": 25},
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        return _extract_posts(response.json())


class RedditClient:
    """Two-step contract: raw rows to trend store, discoveries to seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        gateway: RedditGateway | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.gateway = gateway or RedditOAuthGateway()

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        if isinstance(self.gateway, RedditOAuthGateway) and not self.gateway.configured:
            logger.warning("REDDIT_CLIENT_ID/SECRET not set — skipping Reddit harvest")
            return [], []

        all_results: list[TrendResult] = []
        failed: list[str] = []

        for seed in request.seed_keywords:
            try:
                posts = self.gateway.search(seed)
                for result in self._derive_results(seed, posts):
                    self._store(seed, result)
                    all_results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to search Reddit for %s: %s", seed, e)
                failed.append(seed)

        for sub in POD_SUBREDDITS:
            try:
                posts = self.gateway.subreddit_hot(sub)
                for result in self._derive_subreddit_results(sub, posts):
                    self._store(sub, result)
                    all_results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to fetch r/%s hot posts: %s", sub, e)
                failed.append(f"r/{sub}")

        return all_results, failed

    @staticmethod
    def _seed_tokens(seed: str) -> set[str]:
        return set(re.findall(r"[a-z]{3,}", seed.lower()))

    def _derive_results(self, seed: str, posts: list[RedditPost]) -> list[TrendResult]:
        timestamp = datetime.now(UTC)
        results: list[TrendResult] = []
        seen_titles: set[str] = set()
        sub_score: dict[str, int] = {}
        sub_count: dict[str, int] = {}
        seed_tokens = self._seed_tokens(seed)

        for post in posts:
            title = post.title.strip()
            if not title or title.lower() in seen_titles:
                continue
            title_tokens = set(re.findall(r"[a-z]{3,}", title.lower()))
            if title_tokens == seed_tokens:
                continue
            seen_titles.add(title.lower())
            results.append(
                TrendResult(
                    query=title,
                    score=post.score,
                    delta=post.num_comments,
                    region="US",
                    query_type="search",
                    source="reddit",
                    fetched_at=timestamp,
                )
            )
            if post.subreddit:
                sub_score[post.subreddit] = sub_score.get(post.subreddit, 0) + post.score
                sub_count[post.subreddit] = sub_count.get(post.subreddit, 0) + 1

        for sub in sorted(sub_score):
            results.append(
                TrendResult(
                    query=f"r/{sub}",
                    score=sub_score[sub],
                    delta=sub_count[sub],
                    region="US",
                    query_type="subreddit",
                    source="reddit",
                    fetched_at=timestamp,
                )
            )
        return results

    def _derive_subreddit_results(
        self, subreddit: str, posts: list[RedditPost]
    ) -> list[TrendResult]:
        timestamp = datetime.now(UTC)
        results: list[TrendResult] = []
        seen: set[str] = set()
        for post in posts:
            title = post.title.strip()
            if not title or title.lower() in seen:
                continue
            seen.add(title.lower())
            results.append(
                TrendResult(
                    query=title,
                    score=post.score,
                    delta=post.num_comments,
                    region="US",
                    query_type="search",
                    source="reddit",
                    fetched_at=timestamp,
                )
            )
        return results

    def _store(self, seed: str, result: TrendResult) -> None:
        query_id = self.db_client.insert_trend_query(seed, result.query, source="reddit")
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source="reddit",
            trend_direction=result.trend_direction,
        )
        self._upsert_candidate(seed, result)

    def _upsert_candidate(self, seed: str, result: TrendResult) -> None:
        if result.query_type not in ("search", "subreddit"):
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
