"""YouTube Shorts micro-trend harvester.

Searches for trending short-form videos via the YouTube Data API v3.
Requires a ``YOUTUBE_API_KEY`` environment variable (free tier: 10,000
units/day). Each seed keyword uses 1 search call + 1 videos.list call
(~2 units total), so 20 seeds = ~40 units/day — well under the free cap.

Discoveries are written to the unified trend store (trend_queries +
trend_scores with ``source='youtube'``) and fed to the seed pool via
``seed_candidates``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

import httpx

from store_listing.orchestration.promotion import compute_promotion_score

from .google_trends import DatabaseClient
from .schemas import TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

SEARCH_URL: Final[str] = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL: Final[str] = "https://www.googleapis.com/youtube/v3/videos"


@dataclass(frozen=True)
class YouTubeShort:
    video_id: str
    title: str
    tags: tuple[str, ...]
    view_count: int
    like_count: int
    comment_count: int
    channel_title: str


@runtime_checkable
class YouTubeGateway(Protocol):
    def search_shorts(self, query: str) -> list[YouTubeShort]: ...


def _parse_short(item: dict[str, Any], stats: dict[str, Any]) -> YouTubeShort:
    snippet = item.get("snippet") or {}
    statistics = stats.get("statistics") or {}
    tags = snippet.get("tags") or []
    return YouTubeShort(
        video_id=str(item.get("id") or ""),
        title=str(snippet.get("title") or ""),
        tags=tuple(str(t).lower() for t in tags if t),
        view_count=int(statistics.get("viewCount") or 0),
        like_count=int(statistics.get("likeCount") or 0),
        comment_count=int(statistics.get("commentCount") or 0),
        channel_title=str(snippet.get("channelTitle") or ""),
    )


class YouTubeAPIGateway:
    """YouTube Data API v3 wrapper (API key only, no OAuth)."""

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self._http = client or httpx.Client(timeout=15)

    def search_shorts(self, query: str) -> list[YouTubeShort]:
        search_resp = self._http.get(
            SEARCH_URL,
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "videoDuration": "short",
                "maxResults": 25,
                "order": "viewCount",
                "key": self.api_key,
            },
        )
        search_resp.raise_for_status()
        items = search_resp.json().get("items") or []

        video_ids = []
        search_items = []
        for item in items:
            vid_id = item.get("id")
            if isinstance(vid_id, dict):
                vid_id = vid_id.get("videoId")
            if vid_id:
                video_ids.append(str(vid_id))
                search_items.append(item)

        if not video_ids:
            return []

        stats_resp = self._http.get(
            VIDEOS_URL,
            params={
                "part": "snippet,statistics",
                "id": ",".join(video_ids),
                "key": self.api_key,
            },
        )
        stats_resp.raise_for_status()
        stats_items = stats_resp.json().get("items") or []
        stats_by_id: dict[str, dict[str, Any]] = {str(s.get("id") or ""): s for s in stats_items}

        results: list[YouTubeShort] = []
        for search_item in search_items:
            vid_id = search_item.get("id")
            if isinstance(vid_id, dict):
                vid_id = vid_id.get("videoId")
            vid_id = str(vid_id or "")
            stats = stats_by_id.get(vid_id, {})
            merged = {
                "id": vid_id,
                "snippet": stats.get("snippet") or search_item.get("snippet") or {},
            }
            results.append(_parse_short(merged, stats))
        return results


class YouTubeClient:
    """Two-step contract: raw rows to trend store, discoveries to seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        gateway: YouTubeGateway | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.gateway = gateway or self._build_gateway()

    @staticmethod
    def _build_gateway() -> YouTubeGateway:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            msg = "YOUTUBE_API_KEY environment variable is required for YouTube harvesting"
            raise RuntimeError(msg)
        return YouTubeAPIGateway(api_key=api_key)

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        all_results: list[TrendResult] = []
        failed: list[str] = []

        for seed in request.seed_keywords:
            try:
                shorts = self.gateway.search_shorts(seed)
                for result in self._derive_results(seed, shorts):
                    self._store(seed, result)
                    all_results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to search YouTube Shorts for %s: %s", seed, e)
                failed.append(seed)

        return all_results, failed

    @staticmethod
    def _derive_results(seed: str, shorts: list[YouTubeShort]) -> list[TrendResult]:
        timestamp = datetime.now(UTC)
        results: list[TrendResult] = []
        seen_titles: set[str] = set()
        tag_views: dict[str, int] = {}
        tag_count: dict[str, int] = {}

        for short in shorts:
            title = short.title.strip()
            if title and title.lower() not in seen_titles and title.lower() != seed.lower():
                seen_titles.add(title.lower())
                results.append(
                    TrendResult(
                        query=title,
                        score=short.view_count,
                        delta=short.like_count,
                        region="US",
                        query_type="video",
                        source="youtube",
                        fetched_at=timestamp,
                    )
                )
            for tag in short.tags:
                if tag and tag.lower() != seed.lower():
                    tag_views[tag] = tag_views.get(tag, 0) + short.view_count
                    tag_count[tag] = tag_count.get(tag, 0) + 1

        for tag in sorted(tag_views):
            results.append(
                TrendResult(
                    query=tag,
                    score=tag_views[tag],
                    delta=tag_count[tag],
                    region="US",
                    query_type="hashtag",
                    source="youtube",
                    fetched_at=timestamp,
                )
            )
        return results

    def _store(self, seed: str, result: TrendResult) -> None:
        query_id = self.db_client.insert_trend_query(seed, result.query, source="youtube")
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source="youtube",
            trend_direction=result.trend_direction,
        )
        self._upsert_candidate(seed, result)

    def _upsert_candidate(self, seed: str, result: TrendResult) -> None:
        if result.query_type not in ("video", "hashtag"):
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
