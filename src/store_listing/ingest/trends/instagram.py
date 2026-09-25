"""Instagram hashtag trend harvester (gated — requires Apify).

Instagram is fully login-walled as of 2024: no public JSON endpoints, no free
web-scrape fallback. This module is disabled by default (``INSTAGRAM_ENABLED``
env var). When enabled, it uses an Apify actor to extract hashtag post volume
and engagement. Cost: ~$0.40–2.50 per 1K posts.

See GitHub issue for cost strategy and rollout plan.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol, runtime_checkable

from store_listing.orchestration.promotion import compute_promotion_score

from .google_trends import DatabaseClient
from .schemas import TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

DEFAULT_INSTAGRAM_ACTOR: Final[str] = "apify/instagram-hashtag-scraper"
DEFAULT_INSTAGRAM_RESULTS: Final[int] = 30


@dataclass(frozen=True)
class InstagramPost:
    post_id: str
    caption: str
    like_count: int
    comment_count: int
    hashtags: tuple[str, ...]


@runtime_checkable
class InstagramScraperGateway(Protocol):
    def scrape_hashtag(self, hashtag: str) -> list[InstagramPost]: ...


def _post(item: dict[str, Any]) -> InstagramPost:
    caption = str(item.get("caption") or item.get("text") or "")
    hashtags_raw = item.get("hashtags") or []
    if not isinstance(hashtags_raw, list):
        hashtags_raw = []
    hashtags = tuple(str(h).lstrip("#").lower() for h in hashtags_raw if h)
    return InstagramPost(
        post_id=str(item.get("id") or item.get("shortcode") or ""),
        caption=caption,
        like_count=int(item.get("likesCount") or item.get("like_count") or 0),
        comment_count=int(item.get("commentsCount") or item.get("comment_count") or 0),
        hashtags=hashtags,
    )


class ApifyInstagramGateway:
    """Apify actor wrapper for Instagram hashtag scraping."""

    def __init__(
        self,
        token: str,
        actor_id: str = DEFAULT_INSTAGRAM_ACTOR,
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

    @staticmethod
    def _results_limit() -> int:
        return int(os.environ.get("INSTAGRAM_RESULTS_PER_TAG") or DEFAULT_INSTAGRAM_RESULTS)

    def scrape_hashtag(self, hashtag: str) -> list[InstagramPost]:
        run = (
            self._apify()
            .actor(self._api_actor_ref())
            .call(
                run_input={
                    "hashtags": [hashtag.lstrip("#")],
                    "resultsLimit": self._results_limit(),
                },
                run_timeout=timedelta(minutes=4),
            )
        )
        if run is None:
            return []
        dataset_id = (
            run.get("defaultDatasetId") if isinstance(run, dict) else run.default_dataset_id
        )
        raw_items = self._apify().dataset(dataset_id).list_items().items
        return [_post(item) for item in raw_items]


class InstagramClient:
    """Two-step contract: raw rows to trend store, discoveries to seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        gateway: InstagramScraperGateway | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.gateway = gateway or self._build_gateway()

    @staticmethod
    def _build_gateway() -> InstagramScraperGateway:
        token = os.environ.get("APIFY_API_TOKEN")
        if not token:
            msg = (
                "Instagram harvesting requires APIFY_API_TOKEN "
                "(no free web-scrape fallback available)"
            )
            raise RuntimeError(msg)
        actor_id = os.environ.get("APIFY_ACTOR_INSTAGRAM") or DEFAULT_INSTAGRAM_ACTOR
        return ApifyInstagramGateway(token=token, actor_id=actor_id)

    @staticmethod
    def _normalize_hashtag(seed: str) -> str:
        import re

        return f"#{re.sub(r'[^a-z0-9]', '', seed.lower())}"

    def target_hashtags(self, seed_keywords: list[str]) -> list[str]:
        tags = [self._normalize_hashtag(seed) for seed in seed_keywords]
        return [tag for tag in dict.fromkeys(tags) if tag != "#"]

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        all_results: list[TrendResult] = []
        failed: list[str] = []

        for tag in self.target_hashtags(request.seed_keywords):
            try:
                posts = self.gateway.scrape_hashtag(tag)
                results = self._derive_results(tag, posts)
                for result in results:
                    self._store(tag, result)
                    all_results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to scrape Instagram tag %s: %s", tag, e)
                failed.append(tag)

        return all_results, failed

    @staticmethod
    def _derive_results(seed_tag: str, posts: list[InstagramPost]) -> list[TrendResult]:
        timestamp = datetime.now(UTC)
        results: list[TrendResult] = []
        hashtag_likes: dict[str, int] = {}
        hashtag_count: dict[str, int] = {}

        for post in posts:
            for raw_tag in post.hashtags:
                tag = f"#{raw_tag}"
                if tag.lower() == seed_tag.lower() or tag == "#":
                    continue
                hashtag_likes[tag] = hashtag_likes.get(tag, 0) + post.like_count
                hashtag_count[tag] = hashtag_count.get(tag, 0) + 1

        for tag in sorted(hashtag_likes):
            results.append(
                TrendResult(
                    query=tag,
                    score=hashtag_likes[tag],
                    delta=hashtag_count[tag],
                    region="US",
                    query_type="hashtag",
                    source="instagram",
                    fetched_at=timestamp,
                )
            )
        return results

    def _store(self, seed_tag: str, result: TrendResult) -> None:
        query_id = self.db_client.insert_trend_query(seed_tag, result.query, source="instagram")
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source="instagram",
            trend_direction=result.trend_direction,
        )
        self._upsert_candidate(seed_tag, result)

    def _upsert_candidate(self, seed_tag: str, result: TrendResult) -> None:
        if result.query_type != "hashtag":
            return
        if result.query.lower() == seed_tag.lower():
            return
        promotion_score = compute_promotion_score(
            result.source, result.query_type, result.score, result.delta
        )
        self.db_client.upsert_seed_candidate(
            source_seed=seed_tag,
            query=result.query,
            source=result.source,
            query_type=result.query_type,
            score=result.score,
            delta=result.delta,
            promotion_score=promotion_score,
        )
