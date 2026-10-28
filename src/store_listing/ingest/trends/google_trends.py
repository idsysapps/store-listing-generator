import logging
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, ClassVar, Protocol, TypeVar, cast, runtime_checkable

import pandas as pd
import psycopg2
import requests
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq

from store_listing.orchestration.brand_guard import licensed_brand
from store_listing.orchestration.brief_reranking import RerankableBrief
from store_listing.orchestration.design_briefs import BriefableCandidate
from store_listing.orchestration.headline_voice import DEFAULT_VOICE
from store_listing.orchestration.promotion import (
    BRIEF_SEED_COOLDOWN_DAYS,
    ORIGIN_TREND,
    ActiveSeed,
    SeedCandidate,
    compute_novelty_score,
    select_briefable_seeds,
)

from .schemas import QueryType, TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

# The names a seed may not carry. The guard is shared with the text the brief
# model writes, so a brand cannot slip in through the text (issue #225).
_CANDIDATE_NOISE_RE = re.compile(
    r"^(how to\b|tutorial\b|diy\b|recipe\b|review\b|unboxing\b)",
    re.IGNORECASE,
)


def _is_candidate_noise(query: str) -> bool:
    """Return True if the query is too noisy to be a useful seed candidate."""
    if " " not in query:
        return True
    if len(query) <= 2 or len(query) > 80:
        return True
    if "&#" in query or "http" in query or "www." in query or "@" in query:
        return True
    if licensed_brand(query):
        return True
    return bool(_CANDIDATE_NOISE_RE.search(query))


T = TypeVar("T")

MAX_SEED_LENGTH = 60
CIRCUIT_BREAKER_THRESHOLD = 3
THROTTLE_SECONDS = 2.0

_NOISE_RE = re.compile(
    r"[#@!]"
    r"|https?://"
    r"|\.com\b"
    r"|\(official\s+(video|audio)\)"
    r"|#shorts\b"
    r"|\bfull\s+video\b",
    re.IGNORECASE,
)


def is_valid_trends_query(seed: str) -> bool:
    if len(seed) > MAX_SEED_LENGTH:
        return False
    if _NOISE_RE.search(seed):
        return False
    word_count = len(seed.split())
    return not word_count > 7


def _is_rate_limited(exc: Exception) -> bool:
    if isinstance(exc, ResponseError):
        status = getattr(exc.response, "status_code", None)
        if status == 429:
            return True
        body = getattr(exc.response, "text", "") or ""
        if "/sorry/" in body or "too many" in body.lower():
            return True
    return False


@runtime_checkable
class TrendRequestGateway(Protocol):
    cookies: dict[str, str]

    def build_payload(self, keywords: list[str], timeframe: str, geo: str) -> None: ...

    def interest_by_region(self) -> pd.DataFrame: ...

    def related_queries(self) -> dict[str, Any]: ...

    def interest_over_time(self) -> pd.DataFrame: ...

    def GetGoogleCookie(self) -> dict[str, str]: ...


class DatabaseClient:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.connection_params = {
            "host": host or os.environ.get("DATABASE_HOST") or "localhost",
            "port": port if port is not None else int(os.environ.get("DATABASE_PORT") or 5432),
            "database": database or os.environ.get("DATABASE_NAME") or "store_listing",
            "user": user if user is not None else os.environ.get("DATABASE_USER"),
            "password": password if password is not None else os.environ.get("DATABASE_PASSWORD"),
        }

    def connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(**self.connection_params)

    def insert_trend_query(self, seed_keyword: str, query: str, source: str = "google") -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trend_queries (seed_keyword, query, source)
                VALUES (%s, %s, %s)
                ON CONFLICT (seed_keyword, query) DO UPDATE SET created_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (seed_keyword, query, source),
            )
            result = cur.fetchone()
            if result is None:
                msg = "Failed to insert trend query"
                raise RuntimeError(msg)
            return result[0]

    def insert_trend_score(
        self,
        query_id: int,
        score: int,
        delta: int,
        region: str,
        query_type: QueryType = "interest",
        source: str = "google",
        trend_direction: int | None = None,
    ) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trend_scores (query_id, score, delta, region, query_type, source, trend_direction)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (query_id, score, delta, region, query_type, source, trend_direction),
            )
            result = cur.fetchone()
            if result is None:
                msg = "Failed to insert trend score"
                raise RuntimeError(msg)
            return result[0]

    def upsert_seed_candidate(
        self,
        source_seed: str,
        query: str,
        source: str,
        query_type: str,
        score: int,
        delta: int,
        promotion_score: int,
    ) -> int | None:
        """Insert or refresh a pending candidate; leave promoted/rejected rows untouched."""
        if _is_candidate_noise(query):
            return None
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO seed_candidates
                    (query, source_seed, source, query_type, score, delta, promotion_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (query) DO UPDATE SET
                    source_seed = EXCLUDED.source_seed,
                    score = EXCLUDED.score,
                    delta = EXCLUDED.delta,
                    promotion_score = EXCLUDED.promotion_score
                WHERE seed_candidates.status = 'pending'
                RETURNING id
                """,
                (query, source_seed, source, query_type, score, delta, promotion_score),
            )
            result = cur.fetchone()
            if result is not None:
                return result[0]
            cur.execute("SELECT id FROM seed_candidates WHERE query = %s", (query,))
            row = cur.fetchone()
            return None if row is None else row[0]

    def list_pending_candidates(self) -> list[SeedCandidate]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, query, source, query_type, score, delta, promotion_score, status
                FROM seed_candidates
                WHERE status = 'pending'
                ORDER BY promotion_score DESC
                """
            )
            rows = cur.fetchall()
        return [
            SeedCandidate(
                id=row[0],
                query=row[1],
                source=row[2],
                query_type=row[3],
                score=row[4],
                delta=row[5],
                promotion_score=row[6],
                status=row[7],
            )
            for row in rows
        ]

    def cross_seed_counts(self) -> dict[str, int]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT query, COUNT(DISTINCT source_seed) AS seed_count
                FROM seed_candidates
                WHERE status = 'pending'
                GROUP BY query
                """
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def list_active_seeds(self) -> list[ActiveSeed]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, query, promotion_score, origin, theme, seasonal_window_end
                FROM active_seeds
                WHERE archived_at IS NULL
                ORDER BY promotion_score DESC
                """
            )
            return [
                ActiveSeed(
                    id=row[0],
                    query=row[1],
                    promotion_score=row[2],
                    origin=row[3] or ORIGIN_TREND,
                    theme=row[4],
                    seasonal_window_end=row[5],
                )
                for row in cur.fetchall()
            ]

    def count_active_seeds(self) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM active_seeds WHERE archived_at IS NULL")
            row = cur.fetchone()
            return 0 if row is None else int(row[0])

    def list_historic_rising_top(
        self,
    ) -> list[tuple[str, str, str, str, int, int]]:
        """Pre-population source: existing trend_scores with rising/top signals."""
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT tq.seed_keyword, tq.query, ts.query_type, ts.source, ts.score, ts.delta
                FROM trend_scores ts
                JOIN trend_queries tq ON tq.id = ts.query_id
                WHERE ts.query_type IN ('rising', 'top')
                """
            )
            return [(row[0], row[1], row[2], row[3], row[4], row[5]) for row in cur.fetchall()]

    def insert_active_seed(
        self,
        query: str,
        promotion_score: int,
        candidate_id: int | None,
        origin: str = ORIGIN_TREND,
        theme: str | None = None,
        seasonal_window_end: date | None = None,
    ) -> int | None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO active_seeds
                    (query, promotion_score, promoted_from_candidate_id, origin, theme, seasonal_window_end)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (query) DO NOTHING
                RETURNING id
                """,
                (query, promotion_score, candidate_id, origin, theme, seasonal_window_end),
            )
            row = cur.fetchone()
            return None if row is None else row[0]

    def promote_candidate(self, candidate_id: int) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE seed_candidates
                SET status = 'promoted', promoted_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (candidate_id,),
            )
            return cur.rowcount

    def archive_active_seed(self, query: str) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE active_seeds
                SET archived_at = CURRENT_TIMESTAMP
                WHERE query = %s AND archived_at IS NULL
                """,
                (query,),
            )
            return cur.rowcount

    def reject_candidate(self, candidate_id: int) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE seed_candidates
                SET status = 'rejected', archived_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (candidate_id,),
            )
            return cur.rowcount

    def insert_curation_log(
        self,
        candidate_id: int | None,
        action: str,
        reasoning: str | None,
        event_context: str | None,
        llm_model: str | None,
    ) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO seed_curation_log
                    (candidate_id, action, reasoning, event_context, llm_model)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (candidate_id, action, reasoning, event_context, llm_model),
            )
            result = cur.fetchone()
            if result is None:
                msg = "Failed to insert curation log"
                raise RuntimeError(msg)
            return result[0]

    def insert_product_tags(self, active_seed_id: int, tags: list[str]) -> int:
        if not tags:
            return 0
        with self.connect() as conn, conn.cursor() as cur:
            for tag in tags:
                cur.execute(
                    """
                    INSERT INTO seed_product_tags (active_seed_id, tag)
                    VALUES (%s, %s)
                    ON CONFLICT (active_seed_id, tag) DO NOTHING
                    """,
                    (active_seed_id, tag),
                )
            return len(tags)

    def list_active_seeds_by_tag(self, tag: str) -> list[ActiveSeed]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, a.query, a.promotion_score
                FROM active_seeds a
                JOIN seed_product_tags t ON t.active_seed_id = a.id
                WHERE a.archived_at IS NULL AND t.tag = %s
                ORDER BY a.promotion_score DESC
                """,
                (tag,),
            )
            return [
                ActiveSeed(id=row[0], query=row[1], promotion_score=row[2])
                for row in cur.fetchall()
            ]

    def list_briefable_seeds(
        self,
        cooldown_days: int | None = None,
        limit: int | None = None,
    ) -> list[BriefableCandidate]:
        """Seeds the brief task may use right now.

        A seed that already produced a brief stays out of the pool for
        `cooldown_days` days, then becomes available again. Excluding every
        seed that ever produced a brief drained the pool to nothing
        (issue #192).

        With a `limit` the seeds are picked as a mix of seasonal, evergreen
        and trend seeds instead of the highest scoring ones, so a busy
        holiday cannot take the whole batch (issue #198).
        """
        days = BRIEF_SEED_COOLDOWN_DAYS if cooldown_days is None else cooldown_days
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    a.id AS active_seed_id,
                    a.query,
                    COALESCE(sc.source, 'calendar') AS source,
                    COALESCE(sc.query_type, 'inject') AS query_type,
                    COALESCE(sc.score, 0) AS score,
                    COALESCE(sc.delta, 0) AS delta,
                    a.promotion_score,
                    COALESCE(
                        ARRAY_AGG(spt.tag) FILTER (WHERE spt.tag IS NOT NULL),
                        '{}'
                    ) AS product_tags,
                    a.origin,
                    a.theme,
                    a.seasonal_window_end
                FROM active_seeds a
                LEFT JOIN seed_candidates sc
                    ON sc.id = a.promoted_from_candidate_id
                LEFT JOIN seed_product_tags spt
                    ON spt.active_seed_id = a.id
                WHERE a.archived_at IS NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM design_brief_sources dbs
                      WHERE dbs.active_seed_id = a.id
                        AND dbs.created_at > now() - make_interval(days => %s)
                  )
                GROUP BY a.id, a.query, a.promotion_score,
                         sc.source, sc.query_type, sc.score, sc.delta,
                         a.origin, a.theme, a.seasonal_window_end
                ORDER BY a.promotion_score DESC
                """,
                (days,),
            )
            rows = cur.fetchall()

        candidates = [
            BriefableCandidate(
                active_seed_id=row[0],
                query=row[1],
                source=row[2],
                query_type=row[3],
                score=row[4],
                delta=row[5],
                promotion_score=row[6],
                product_tags=list(row[7]) if row[7] else [],
                origin=row[8] or ORIGIN_TREND,
                theme=row[9],
                seasonal_window_end=row[10],
            )
            for row in rows
        ]
        if limit is None:
            return candidates
        return self._as_brief_mix(candidates, limit)

    def _as_brief_mix(
        self,
        candidates: list[BriefableCandidate],
        limit: int,
    ) -> list[BriefableCandidate]:
        """Reduce the candidate list to the mix the brief task should see."""
        seeds = [
            ActiveSeed(
                id=candidate.active_seed_id,
                query=candidate.query,
                promotion_score=candidate.promotion_score,
                origin=candidate.origin,
                theme=candidate.theme,
                seasonal_window_end=candidate.seasonal_window_end,
            )
            for candidate in candidates
        ]
        chosen = select_briefable_seeds(seeds, limit=limit)
        keep = {seed.id for seed in chosen}
        return [candidate for candidate in candidates if candidate.active_seed_id in keep]

    def list_briefable_seed_ids(self, cooldown_days: int | None = None) -> list[int]:
        """Ids of the seeds the brief task may use right now."""
        return [candidate.active_seed_id for candidate in self.list_briefable_seeds(cooldown_days)]

    def list_recent_brief_seed_links(
        self, days: int = 14
    ) -> list[tuple[str, str, str, str | None]]:
        """The briefs of the last `days` days with the seed each one came from.

        Read-only: (batch_id, concept, seed query, seed theme) per brief/seed
        pair.
        """
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT db.batch_id, db.concept, a.query, a.theme
                FROM design_briefs db
                JOIN design_brief_sources dbs ON dbs.brief_id = db.id
                JOIN active_seeds a ON a.id = dbs.active_seed_id
                WHERE db.created_at > now() - make_interval(days => %s)
                ORDER BY db.created_at DESC
                """,
                (days,),
            )
            return [(row[0] or "", row[1] or "", row[2] or "", row[3]) for row in cur.fetchall()]

    def insert_design_brief(
        self,
        concept: str,
        product_type: str,
        specific_products: list[str],
        audience: str,
        visual_style: str,
        confidence: int,
        reasoning: str,
        llm_model: str,
        batch_id: str,
        layout_type: str = "full_bleed",
        headline_text: str | None = None,
        tagline_text: str | None = None,
        copy_mode: str = "caption",
        kicker_text: str | None = None,
        font_color: str | None = None,
        scene_description: str | None = None,
        personal_use: bool = False,
        recommended_materials: list[dict[str, Any]] | None = None,
        voice: str = DEFAULT_VOICE,
        font_family: str | None = None,
        style_preset: str | None = None,
        text_plan: dict[str, Any] | None = None,
    ) -> int:
        import json as _json

        materials_json = _json.dumps(recommended_materials) if recommended_materials else None
        text_plan_json = _json.dumps(text_plan) if text_plan else None
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO design_briefs
                    (concept, product_type, specific_products, audience,
                     visual_style, confidence, reasoning, llm_model, batch_id,
                     layout_type, headline_text, tagline_text, copy_mode, kicker_text,
                     font_color, scene_description, personal_use, recommended_materials, voice,
                     font_family, style_preset, text_plan)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    concept,
                    product_type,
                    specific_products,
                    audience,
                    visual_style,
                    confidence,
                    reasoning,
                    llm_model,
                    batch_id,
                    layout_type,
                    headline_text,
                    tagline_text,
                    copy_mode,
                    kicker_text,
                    font_color,
                    scene_description,
                    personal_use,
                    materials_json,
                    voice,
                    font_family,
                    style_preset,
                    text_plan_json,
                ),
            )
            result = cur.fetchone()
            if result is None:
                msg = "Failed to insert design brief"
                raise RuntimeError(msg)
            return result[0]

    def insert_brief_source(self, brief_id: int, active_seed_id: int) -> int | None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO design_brief_sources (brief_id, active_seed_id)
                VALUES (%s, %s)
                ON CONFLICT (brief_id, active_seed_id) DO NOTHING
                RETURNING id
                """,
                (brief_id, active_seed_id),
            )
            row = cur.fetchone()
            return None if row is None else row[0]

    def list_briefs_without_images(
        self,
        limit: int = 20,
        product_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        import json as _json

        with self.connect() as conn, conn.cursor() as cur:
            if product_types:
                cur.execute(
                    """
                    SELECT id, concept, product_type, audience, visual_style,
                           layout_type, headline_text, tagline_text, font_color,
                           regeneration_feedback, scene_description, recommended_materials,
                           voice, font_family, style_preset, text_plan,
                           copy_mode, kicker_text
                    FROM design_briefs
                    WHERE image_key_raw IS NULL
                      AND product_type = ANY(%s)
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (product_types, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, concept, product_type, audience, visual_style,
                           layout_type, headline_text, tagline_text, font_color,
                           regeneration_feedback, scene_description, recommended_materials,
                           voice, font_family, style_preset, text_plan,
                           copy_mode, kicker_text
                    FROM design_briefs
                    WHERE image_key_raw IS NULL
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            return [
                {
                    "id": row[0],
                    "concept": row[1],
                    "product_type": row[2],
                    "audience": row[3],
                    "visual_style": row[4],
                    "layout_type": row[5] or "full_bleed",
                    "headline_text": row[6],
                    "tagline_text": row[7],
                    "font_color": row[8],
                    "regeneration_feedback": row[9],
                    "scene_description": row[10],
                    "recommended_materials": (
                        _json.loads(row[11]) if isinstance(row[11], str) else row[11]
                    ),
                    "voice": row[12] or DEFAULT_VOICE,
                    "font_family": row[13],
                    "style_preset": row[14],
                    "text_plan": (_json.loads(row[15]) if isinstance(row[15], str) else row[15]),
                    "copy_mode": row[16] or "caption",
                    "kicker_text": row[17],
                }
                for row in cur.fetchall()
            ]

    def get_brief_by_id(self, brief_id: int) -> dict[str, Any] | None:
        import json as _json

        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, concept, product_type, audience, visual_style,
                       layout_type, headline_text, tagline_text, font_color,
                       regeneration_feedback, scene_description, recommended_materials,
                       voice, font_family, style_preset, text_plan,
                       copy_mode, kicker_text
                FROM design_briefs
                WHERE id = %s
                """,
                (brief_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "concept": row[1],
                "product_type": row[2],
                "audience": row[3],
                "visual_style": row[4],
                "layout_type": row[5] or "full_bleed",
                "headline_text": row[6],
                "tagline_text": row[7],
                "font_color": row[8],
                "regeneration_feedback": row[9],
                "scene_description": row[10],
                "recommended_materials": (
                    _json.loads(row[11]) if isinstance(row[11], str) else row[11]
                ),
                "voice": row[12] or DEFAULT_VOICE,
                "font_family": row[13],
                "style_preset": row[14],
                "text_plan": (_json.loads(row[15]) if isinstance(row[15], str) else row[15]),
                "copy_mode": row[16] or "caption",
                "kicker_text": row[17],
            }

    def list_briefs_by_batch(self, batch_id: str) -> list[dict[str, Any]]:
        """Every brief in a batch, with the plan it already has."""
        import json as _json

        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, concept, product_type, audience, visual_style,
                       layout_type, headline_text, tagline_text, font_color,
                       regeneration_feedback, scene_description, recommended_materials,
                       voice, font_family, style_preset, text_plan,
                       copy_mode, kicker_text
                FROM design_briefs
                WHERE batch_id = %s
                ORDER BY id
                """,
                (batch_id,),
            )
            return [
                {
                    "id": row[0],
                    "concept": row[1],
                    "product_type": row[2],
                    "audience": row[3],
                    "visual_style": row[4],
                    "layout_type": row[5] or "full_bleed",
                    "headline_text": row[6],
                    "tagline_text": row[7],
                    "font_color": row[8],
                    "regeneration_feedback": row[9],
                    "scene_description": row[10],
                    "recommended_materials": (
                        _json.loads(row[11]) if isinstance(row[11], str) else row[11]
                    ),
                    "voice": row[12] or DEFAULT_VOICE,
                    "font_family": row[13],
                    "style_preset": row[14],
                    "text_plan": (_json.loads(row[15]) if isinstance(row[15], str) else row[15]),
                    "copy_mode": row[16] or "caption",
                    "kicker_text": row[17],
                }
                for row in cur.fetchall()
            ]

    def update_brief_text_plan(
        self,
        brief_id: int,
        text_plan: dict[str, Any] | None,
        scene_description: str | None,
    ) -> None:
        """Store a replanned text treatment and the scene that carries it."""
        import json as _json

        plan_json = _json.dumps(text_plan) if text_plan else None
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE design_briefs
                SET text_plan = %s, scene_description = %s
                WHERE id = %s
                """,
                (plan_json, scene_description, brief_id),
            )

    def list_active_materials(
        self,
        product_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """The materials the cost book offers.

        A material can serve several product types, so the product types are
        stored as an array and the filter matches any of them. An equality
        comparison never matched, because the stored value was the whole
        array (issue #186).
        """
        with self.connect() as conn, conn.cursor() as cur:
            if product_type:
                cur.execute(
                    """
                    SELECT external_id, name, brand, substrate_type, color,
                           sizes, material_composition, product_type, sku, unit_cost
                    FROM fabrication_materials
                    WHERE active = TRUE AND %s = ANY(product_type)
                    ORDER BY name, color
                    """,
                    (product_type,),
                )
            else:
                cur.execute(
                    """
                    SELECT external_id, name, brand, substrate_type, color,
                           sizes, material_composition, product_type, sku, unit_cost
                    FROM fabrication_materials
                    WHERE active = TRUE
                    ORDER BY product_type, name, color
                    """
                )
            return [
                {
                    "external_id": row[0],
                    "name": row[1],
                    "brand": row[2],
                    "substrate_type": row[3],
                    "color": row[4],
                    "sizes": row[5],
                    "material_composition": row[6],
                    "product_type": row[7],
                    "sku": row[8],
                    "unit_cost": float(row[9]) if row[9] is not None else None,
                }
                for row in cur.fetchall()
            ]

    def get_app_setting(self, key: str) -> str | None:
        """The value of a setting, or None when it was never stored."""
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set_app_setting(self, key: str, value: str) -> None:
        """Store a setting, replacing the value it had."""
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (key, value),
            )

    def list_briefs_for_reranking(self) -> list[RerankableBrief]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, concept, product_type, audience, confidence,
                       COALESCE(original_confidence, confidence) AS original_confidence,
                       reasoning, batch_id
                FROM design_briefs
                WHERE created_at > NOW() - INTERVAL '30 days'
                  AND personal_use = FALSE
                  AND (last_reranked_at IS NULL
                       OR last_reranked_at < NOW() - INTERVAL '24 hours')
                ORDER BY created_at DESC
                """
            )
            return [
                RerankableBrief(
                    id=row[0],
                    concept=row[1],
                    product_type=row[2],
                    audience=row[3],
                    confidence=row[4],
                    original_confidence=row[5],
                    reasoning=row[6] or "",
                    batch_id=row[7] or "",
                )
                for row in cur.fetchall()
            ]

    def list_top_active_seed_signals(self) -> list[dict[str, Any]]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.query, ts.score, ts.delta, ts.source
                FROM active_seeds a
                JOIN trend_queries tq ON tq.seed_keyword = a.query
                JOIN trend_scores ts ON ts.query_id = tq.id
                WHERE a.archived_at IS NULL
                  AND ts.fetched_at > NOW() - INTERVAL '7 days'
                ORDER BY ts.score DESC
                LIMIT 50
                """
            )
            return [
                {"query": row[0], "score": row[1], "delta": row[2], "source": row[3]}
                for row in cur.fetchall()
            ]

    def update_brief_confidence(self, *, brief_id: int, confidence: int, reasoning: str) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE design_briefs
                SET confidence = %s,
                    reasoning = %s,
                    original_confidence = COALESCE(original_confidence, confidence),
                    last_reranked_at = NOW()
                WHERE id = %s
                """,
                (confidence, reasoning, brief_id),
            )

    def update_brief_image_keys(
        self,
        brief_id: int,
        image_key_raw: str,
        image_key_transparent: str | None,
    ) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE design_briefs
                SET image_key_raw = %s, image_key_transparent = %s
                WHERE id = %s
                """,
                (image_key_raw, image_key_transparent, brief_id),
            )


class GoogleTrendsClient:
    DEFAULT_SEEDS: ClassVar[list[str]] = [
        "funny t-shirt",
        "hoodie",
        "gift",
        "mom humor",
        "gym fitness",
    ]

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        trend_req: TrendRequestGateway | None = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        request_timeout: tuple[int, int] = (5, 30),
        consent_max_retries: int = 2,
        throttle_seconds: float = THROTTLE_SECONDS,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.request_timeout = request_timeout
        self.consent_max_retries = consent_max_retries
        self._throttle_seconds = throttle_seconds
        self.pytrends: TrendRequestGateway = trend_req or cast(
            TrendRequestGateway,
            TrendReq(
                hl="en-US",
                tz=360,
                timeout=request_timeout,
                retries=max_retries,
                backoff_factor=int(backoff_factor),
            ),
        )
        self.db_client = db_client or DatabaseClient()

    def refresh_cookies(self) -> None:
        """Google's consent/NID cookie expires; fetch a fresh one and retry."""
        self.pytrends.cookies.update(self.pytrends.GetGoogleCookie())

    def _call_with_retry(self, func: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(self.consent_max_retries + 1):
            try:
                return func()
            except ResponseError as e:
                status = getattr(e.response, "status_code", None)
                if status == 404 and attempt < self.consent_max_retries:
                    self.refresh_cookies()
                    continue
                raise
            except requests.RequestException as e:
                if attempt >= self.consent_max_retries:
                    raise
                self.refresh_cookies()
                last_error = e
        assert last_error is not None
        raise last_error

    def _build_payload(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> dict[str, Any]:
        self._call_with_retry(
            lambda: self.pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
        )
        return {
            "interest_by_region": self._call_with_retry(self.pytrends.interest_by_region),
            "related_queries": self._fetch_related_queries(),
            "interest_over_time": self._fetch_interest_over_time(),
        }

    def _fetch_related_queries(self) -> dict[str, Any]:
        """Best-effort: never abort the seed if Google fails a related-queries call."""
        try:
            related = self._call_with_retry(self.pytrends.related_queries)
            return related if isinstance(related, dict) else {}
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch related queries (ignored): %s", e)
            return {}

    def _fetch_interest_over_time(self) -> pd.DataFrame:
        """Best-effort: never abort the seed if Google fails an interest-over-time call."""
        try:
            data = self._call_with_retry(self.pytrends.interest_over_time)
            return data if isinstance(data, pd.DataFrame) else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch interest over time (ignored): %s", e)
            return pd.DataFrame()

    def _calculate_delta(self, current_data: pd.DataFrame, keyword: str) -> int:
        if current_data.empty or keyword not in current_data.columns:
            return 0
        values = current_data[keyword].dropna()
        if len(values) < 2:
            return 0
        latest = values.iloc[-1]
        previous = values.iloc[-2]
        if previous == 0:
            return 100 if latest > 0 else 0
        return int(((latest - previous) / previous) * 100)

    def fetch_trending(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> list[TrendResult]:
        return self._regional_results(self._build_payload(keyword, timeframe, geo), keyword)

    def _regional_results(self, payload: dict[str, Any], keyword: str) -> list[TrendResult]:
        region_data = payload.get("interest_by_region")
        if not isinstance(region_data, pd.DataFrame):
            region_data = pd.DataFrame()

        results = []
        for region, row in region_data.iterrows():
            value = row.get(keyword)
            if value is not None and pd.notna(value) and value > 0:
                score = int(value)
                delta = self._calculate_delta(region_data, keyword)
                results.append(
                    TrendResult(
                        query=keyword,
                        score=score,
                        delta=delta,
                        region=region if isinstance(region, str) else str(region),
                        query_type="interest",
                        fetched_at=datetime.now(UTC),
                    )
                )
        return results

    @staticmethod
    def _parse_rise_pct(value: object) -> int:
        """Parse pytrends rising value strings like "+250000%"/"Breakout" into an int delta."""
        if value is None:
            return 0
        text = str(value).strip()
        if text.lower() == "breakout":
            return 5000
        digits = re.sub(r"[^0-9.]", "", text)
        try:
            return int(float(digits or 0))
        except ValueError:
            return 0

    def fetch_related_queries(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> list[TrendResult]:
        """Rising + top related queries for a seed, tagged by type/region."""
        return self._related_results(self._build_payload(keyword, timeframe, geo), keyword, geo)

    def _related_results(
        self, payload: dict[str, Any], keyword: str, geo: str
    ) -> list[TrendResult]:
        related = payload.get("related_queries") or {}
        entry = related.get(keyword)
        if not isinstance(entry, dict):
            return []

        rising = entry.get("rising")
        top = entry.get("top")

        results: list[TrendResult] = []
        if isinstance(rising, pd.DataFrame) and not rising.empty:
            for _, row in rising.iterrows():
                results.append(
                    TrendResult(
                        query=str(row.get("query", keyword)),
                        score=0,
                        delta=self._parse_rise_pct(row.get("value")),
                        region=geo,
                        query_type="rising",
                        fetched_at=datetime.now(UTC),
                    )
                )
        if isinstance(top, pd.DataFrame) and not top.empty:
            for _, row in top.iterrows():
                results.append(
                    TrendResult(
                        query=str(row.get("query", keyword)),
                        score=int(self._parse_rise_pct(row.get("value"))),
                        delta=0,
                        region="GLOBAL",
                        query_type="top",
                        fetched_at=datetime.now(UTC),
                    )
                )
        return results

    def fetch_global_trend(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> list[TrendResult]:
        """Interest-over-time trajectory for a seed; one GLOBAL result with direction."""
        return self._global_results(self._build_payload(keyword, timeframe, geo), keyword)

    def _global_results(self, payload: dict[str, Any], keyword: str) -> list[TrendResult]:
        over_time = payload.get("interest_over_time")
        if not isinstance(over_time, pd.DataFrame):
            over_time = pd.DataFrame()
        if over_time.empty or keyword not in over_time.columns:
            return []

        values = over_time[keyword].dropna()
        if values.empty:
            return []

        latest = int(values.iloc[-1])
        delta = self._calculate_delta(over_time, keyword)
        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
        return [
            TrendResult(
                query=keyword,
                score=latest,
                delta=delta,
                region="GLOBAL",
                query_type="top",
                trend_direction=direction,
                fetched_at=datetime.now(UTC),
            )
        ]

    def _harvest_seed(self, keyword: str, timeframe: str, geo: str) -> list[TrendResult]:
        """Regional interest + related rising/top + global trend for one seed."""
        payload = self._build_payload(keyword, timeframe, geo)
        return (
            self._regional_results(payload, keyword)
            + self._related_results(payload, keyword, geo)
            + self._global_results(payload, keyword)
        )

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        """Harvest requested seeds; return (stored results, failed seed keywords)."""
        all_results: list[TrendResult] = []
        failed_seeds: list[str] = []
        consecutive_rate_limits = 0

        seeds = [s for s in request.seed_keywords if is_valid_trends_query(s)]
        skipped = set(request.seed_keywords) - set(seeds)
        if skipped:
            logger.info(
                "Filtered %d noisy seeds from Google Trends batch: %s",
                len(skipped),
                list(skipped)[:5],
            )

        for i, seed in enumerate(seeds):
            if consecutive_rate_limits >= CIRCUIT_BREAKER_THRESHOLD:
                remaining = seeds[i:]
                logger.warning(
                    "Circuit breaker: %d consecutive 429s, skipping %d remaining seeds",
                    consecutive_rate_limits,
                    len(remaining),
                )
                failed_seeds.extend(remaining)
                break

            if i > 0:
                time.sleep(self._throttle_seconds)

            try:
                results = self._harvest_seed(seed, request.timeframe, request.region)
                consecutive_rate_limits = 0

                for result in results:
                    query_id = self.db_client.insert_trend_query(seed, result.query)
                    self.db_client.insert_trend_score(
                        query_id=query_id,
                        score=result.score,
                        delta=result.delta,
                        region=result.region,
                        query_type=result.query_type,
                        source=result.source,
                        trend_direction=result.trend_direction,
                    )
                    self._upsert_candidate(seed, result)
                    all_results.append(result)

            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to fetch trends for seed %s: %s", seed, e)
                failed_seeds.append(seed)
                if _is_rate_limited(e):
                    consecutive_rate_limits += 1
                else:
                    consecutive_rate_limits = 0

        return all_results, failed_seeds

    def _upsert_candidate(self, seed: str, result: TrendResult) -> None:
        """Feed the unified seed pool: rising/top discoveries (not the seed itself)."""
        if result.query_type not in ("rising", "top") or result.query == seed:
            return
        promotion_score = compute_novelty_score(
            result.source,
            result.query_type,
            result.score,
            result.delta,
            query=result.query,
            seed=seed,
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
