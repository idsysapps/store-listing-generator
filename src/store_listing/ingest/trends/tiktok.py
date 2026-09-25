"""TikTok micro-trend harvester.

Scrapes video captions, co-occurring hashtags, and sounds under target seeds
(#shirttok, #momhumor, #pickleball, #gymhumor). Uses an Apify actor when
APIFY_API_TOKEN is set; otherwise falls back to a free web scrape of the public
tag pages (best-effort). Discoveries are written to the unified trend store
(trend_queries + trend_scores with ``source='tiktok'``) and fed to the seed
pool via ``seed_candidates``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

import httpx

from store_listing.orchestration.promotion import compute_promotion_score

from .google_trends import DatabaseClient
from .schemas import TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

DEFAULT_TIKTOK_ACTOR: Final[str] = "clockworks/tiktok-hashtag-scraper"

TARGET_HASHTAGS: Final[tuple[str, ...]] = (
    "#shirttok",
    "#momhumor",
    "#pickleball",
    "#gymhumor",
)

_SCRIPT_RE = re.compile(
    r'<script[^>]*id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>',
    re.DOTALL,
)

_DEFAULT_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


@dataclass(frozen=True)
class TikTokItem:
    video_id: str
    text: str
    music: str
    music_author: str
    play_count: int
    digg_count: int
    share_count: int
    comment_count: int
    hashtags: tuple[str, ...]


@runtime_checkable
class TikTokScraperGateway(Protocol):
    def scrape_hashtag(self, hashtag: str) -> list[TikTokItem]: ...


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _tiktok_item(item: dict[str, Any]) -> TikTokItem:
    stats = item.get("stats") or {}
    music = item.get("musicMeta") or {}
    hashtags_raw = item.get("hashtags") or item.get("challenges") or []

    def _name(entry: Any) -> str:
        if isinstance(entry, dict):
            return str(entry.get("name") or entry.get("title") or "")
        return str(entry)

    hashtags = list(dict.fromkeys(_name(h) for h in hashtags_raw if _name(h)))
    return TikTokItem(
        video_id=str(item.get("id") or ""),
        text=str(item.get("text") or item.get("desc") or ""),
        music=str(music.get("musicName") or ""),
        music_author=str(music.get("musicAuthor") or ""),
        play_count=int(stats.get("playCount") or 0),
        digg_count=int(stats.get("diggCount") or 0),
        share_count=int(stats.get("shareCount") or 0),
        comment_count=int(stats.get("commentCount") or 0),
        hashtags=tuple(hashtags),
    )


class ApifyTikTokGateway:
    """Apify actor wrapper (clockworks/tiktok-hashtag-scraper)."""

    def __init__(
        self,
        token: str,
        actor_id: str = DEFAULT_TIKTOK_ACTOR,
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

    def scrape_hashtag(self, hashtag: str) -> list[TikTokItem]:
        run = self._apify().actor(self.actor_id).call(run_input={"hashtags": [hashtag.lstrip("#")]})
        raw_items = self._apify().dataset(run["defaultDatasetId"]).list_items().items
        return [_tiktok_item(item) for item in raw_items]


class TikTokWebGateway:
    """Free fallback: parse the public tag page's rehydrated JSON (best-effort)."""

    BASE_URL = "https://www.tiktok.com/tag/{tag}"

    def scrape_hashtag(self, hashtag: str) -> list[TikTokItem]:
        tag = hashtag.lstrip("#")
        response = httpx.get(
            self.BASE_URL.format(tag=tag),
            headers=_DEFAULT_HEADERS,
            timeout=20,
            follow_redirects=True,
        )
        response.raise_for_status()
        return self._parse_html(response.text)

    @staticmethod
    def _parse_html(html: str) -> list[TikTokItem]:
        match = _SCRIPT_RE.search(html)
        if match is None:
            return []
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        entries = [
            item
            for item in _iter_dicts(data)
            if isinstance(item, dict)
            and isinstance(item.get("stats"), dict)
            and (isinstance(item.get("desc"), str) or isinstance(item.get("text"), str))
        ]
        return [_tiktok_item(item) for item in entries]


class TikTokClient:
    """Two-step contract: raw rows to trend store, discoveries to seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        gateway: TikTokScraperGateway | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.gateway = gateway or self._build_gateway()

    @staticmethod
    def _build_gateway() -> TikTokScraperGateway:
        token = os.environ.get("APIFY_API_TOKEN")
        if token:
            actor_id = os.environ.get("APIFY_ACTOR_TIKTOK_HASHTAG") or DEFAULT_TIKTOK_ACTOR
            return ApifyTikTokGateway(token=token, actor_id=actor_id)
        return TikTokWebGateway()

    @staticmethod
    def _normalize_hashtag(seed: str) -> str:
        return f"#{re.sub(r'[^a-z0-9]', '', seed.lower())}"

    def target_hashtags(self, seed_keywords: list[str]) -> list[str]:
        tags = [self._normalize_hashtag(seed) for seed in seed_keywords]
        tags = [tag for tag in tags if tag != "#"]
        return list(dict.fromkeys([*tags, *TARGET_HASHTAGS]))

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        all_results: list[TrendResult] = []
        failed: list[str] = []

        for tag in self.target_hashtags(request.seed_keywords):
            try:
                items = self.gateway.scrape_hashtag(tag)
                results = self._derive_results(tag, items)
                for result in results:
                    self._store(tag, result)
                    all_results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to scrape TikTok tag %s: %s", tag, e)
                failed.append(tag)

        return all_results, failed

    def _store(self, seed_tag: str, result: TrendResult) -> None:
        query_id = self.db_client.insert_trend_query(seed_tag, result.query, source="tiktok")
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source="tiktok",
            trend_direction=result.trend_direction,
        )
        self._upsert_candidate(seed_tag, result)

    def _derive_results(self, seed_tag: str, items: list[TikTokItem]) -> list[TrendResult]:
        hashtag_plays: dict[str, int] = {}
        hashtag_count: dict[str, int] = {}
        sound_plays: dict[str, int] = {}
        sound_usage: dict[str, int] = {}

        for item in items:
            for raw_tag in item.hashtags:
                tag = f"#{raw_tag.lstrip('#').lower()}"
                if tag == seed_tag.lower() or tag == "#":
                    continue
                hashtag_plays[tag] = hashtag_plays.get(tag, 0) + item.play_count
                hashtag_count[tag] = hashtag_count.get(tag, 0) + 1
            if item.music:
                sound_plays[item.music] = sound_plays.get(item.music, 0) + item.play_count
                sound_usage[item.music] = sound_usage.get(item.music, 0) + 1

        timestamp = datetime.now(UTC)
        results: list[TrendResult] = []
        for tag in sorted(hashtag_plays):
            results.append(
                TrendResult(
                    query=tag,
                    score=hashtag_plays[tag],
                    delta=hashtag_count[tag],
                    region="US",
                    query_type="hashtag",
                    source="tiktok",
                    fetched_at=timestamp,
                )
            )
        for music in sorted(sound_plays):
            results.append(
                TrendResult(
                    query=music,
                    score=sound_plays[music],
                    delta=sound_usage[music],
                    region="US",
                    query_type="sound",
                    source="tiktok",
                    fetched_at=timestamp,
                )
            )
        return results

    def _upsert_candidate(self, seed_tag: str, result: TrendResult) -> None:
        if result.query_type not in ("hashtag", "sound"):
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
