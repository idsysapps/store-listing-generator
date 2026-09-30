"""LLM-driven design brief generation from curated trend signals.

Synthesizes active seeds and their trend data into actionable design
briefs that a graphic designer can immediately execute.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Final, Protocol

from store_listing.orchestration.context_seeds import get_upcoming_events
from store_listing.orchestration.seed_curation import PRODUCT_TAGS

logger = logging.getLogger(__name__)

MAX_SEEDS_PER_BRIEF = int(os.environ.get("LLM_MAX_BRIEF_SEEDS", "20"))
MAX_BRIEFS = int(os.environ.get("LLM_MAX_BRIEFS", "8"))

SYSTEM_PROMPT: Final[str] = (
    "You are a print-on-demand design strategist. Your job is to synthesize trend "
    "signals into actionable design briefs that a graphic designer can immediately "
    "execute.\n\n"
    "We produce designs for these products ONLY:\n"
    "- DTF printing (product_type='dtf_apparel'): t-shirts, hoodies, sweatshirts, "
    "tank tops, hats/caps, tote bags, baby onesies\n"
    "- Sublimation (product_type='sublimation'): mugs, tumblers, phone cases, "
    "mouse pads, coasters, jigsaw puzzles, ornaments, canvas/metal/poster wall art, "
    "polyester pillows, blankets, socks\n"
    "- Stickers and vinyl decals (product_type='sticker_vinyl')\n\n"
    "For each brief:\n"
    "- concept: A specific, design-ready phrase or visual idea (not a category)\n"
    "- product_type: exactly one of 'dtf_apparel', 'sublimation', 'sticker_vinyl'\n"
    "- specific_products: array of 1-3 specific products from the catalog above\n"
    "- audience: who would buy this (demographics, interests)\n"
    "- visual_style: art direction notes (color palette, illustration style, "
    "typography). Do NOT include any text/slogan in this field — text is rendered "
    "separately via the layout fields below.\n"
    "- confidence: 0-100, how confident you are this will sell\n"
    "- reasoning: why this concept is timely and marketable (cite the trend signals)\n"
    "- source_seed_ids: array of seed IDs from the input that inspired this brief\n\n"
    "TEXT LAYOUT — controls how text is composited onto the design:\n"
    "- layout_type: one of 'full_bleed' (no text, illustration fills canvas), "
    "'text_top' (headline at top, art below), 'text_top_bottom' (headline top, "
    "tagline bottom, art in center). Use 'full_bleed' for purely visual designs "
    "with no slogan. Use text layouts when a catchphrase or slogan strengthens the product.\n"
    "- headline_text: (required if layout_type is not 'full_bleed') the main text "
    "line, short and punchy (1-6 words)\n"
    "- tagline_text: (optional, only for 'text_top_bottom') secondary text line\n"
    "- font_color: hex color for text (e.g. '#FFFFFF' for white, '#000000' for black), "
    "choose to contrast with the design background\n\n"
    "IMPORTANT: Do not suggest designs requiring licensed IP (movie characters, "
    "team logos, brand names). Create original concepts inspired by cultural moments.\n\n"
    "Return ONLY valid JSON with this exact structure:\n"
    '{"briefs": [{"concept": "...", "product_type": "...", "specific_products": [...], '
    '"audience": "...", "visual_style": "...", "confidence": 0-100, '
    '"reasoning": "...", "source_seed_ids": [id1, id2], '
    '"layout_type": "text_top", "headline_text": "...", '
    '"tagline_text": "...", "font_color": "#000000"}]}'
)


@dataclass(frozen=True)
class BriefableCandidate:
    active_seed_id: int
    query: str
    source: str
    query_type: str
    score: int
    delta: int
    promotion_score: int
    product_tags: list[str] = field(default_factory=list)


VALID_LAYOUT_TYPES: set[str] = {"full_bleed", "text_top", "text_top_bottom"}


@dataclass(frozen=True)
class DesignBrief:
    concept: str
    product_type: str
    specific_products: list[str]
    audience: str
    visual_style: str
    confidence: int
    reasoning: str
    source_seed_ids: list[int]
    layout_type: str = "full_bleed"
    headline_text: str | None = None
    tagline_text: str | None = None
    font_color: str | None = None


class DesignBriefDBClient(Protocol):
    def list_briefable_seeds(self) -> list[BriefableCandidate]: ...

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
        font_color: str | None = None,
    ) -> int: ...

    def insert_brief_source(self, brief_id: int, active_seed_id: int) -> int | None: ...


def build_brief_prompt(
    candidates: list[BriefableCandidate],
    context: list[dict[str, Any]],
    today: date,
) -> list[dict[str, str]]:
    truncated = candidates[:MAX_SEEDS_PER_BRIEF]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in truncated:
        by_source[c.source].append(
            {
                "id": c.active_seed_id,
                "query": c.query,
                "query_type": c.query_type,
                "score": c.score,
                "delta": c.delta,
                "promotion_score": c.promotion_score,
                "product_tags": c.product_tags,
            }
        )

    user_content = json.dumps(
        {
            "date": today.isoformat(),
            "upcoming_events": context,
            "max_briefs": MAX_BRIEFS,
            "signals_by_source": dict(by_source),
        },
        indent=2,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_brief_response(text: str) -> list[DesignBrief]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning("No JSON object found in LLM brief response")
            return []
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM brief response: %s", e)
        return []

    briefs_raw = data.get("briefs", [])
    if not isinstance(briefs_raw, list):
        return []

    result: list[DesignBrief] = []
    for b in briefs_raw:
        if not isinstance(b, dict):
            continue

        concept = b.get("concept", "")
        if not concept:
            continue

        product_type = b.get("product_type", "")
        if product_type not in PRODUCT_TAGS:
            continue

        raw_products = b.get("specific_products", [])
        specific_products = (
            [p for p in raw_products if isinstance(p, str)]
            if isinstance(raw_products, list)
            else []
        )

        confidence = b.get("confidence", 0)
        if not isinstance(confidence, int):
            confidence = 0
        confidence = max(0, min(100, confidence))

        source_seed_ids = [i for i in b.get("source_seed_ids", []) if isinstance(i, int)]

        layout_type = b.get("layout_type", "full_bleed")
        if layout_type not in VALID_LAYOUT_TYPES:
            layout_type = "full_bleed"

        headline_text = b.get("headline_text")
        if headline_text is not None and not isinstance(headline_text, str):
            headline_text = None

        tagline_text = b.get("tagline_text")
        if tagline_text is not None and not isinstance(tagline_text, str):
            tagline_text = None

        font_color = b.get("font_color")
        if font_color is not None and not isinstance(font_color, str):
            font_color = None

        result.append(
            DesignBrief(
                concept=concept,
                product_type=product_type,
                specific_products=specific_products,
                audience=str(b.get("audience", "")),
                visual_style=str(b.get("visual_style", "")),
                confidence=confidence,
                reasoning=str(b.get("reasoning", "")),
                source_seed_ids=source_seed_ids,
                layout_type=layout_type,
                headline_text=headline_text,
                tagline_text=tagline_text,
                font_color=font_color,
            )
        )

    return result


def generate_design_briefs(
    db_client: DesignBriefDBClient,
    llm_completions: Any,
    model: str,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or datetime.now(tz=UTC).date()

    candidates = db_client.list_briefable_seeds()
    if not candidates:
        return {"status": "success", "briefs_created": 0}

    context = get_upcoming_events(today)
    messages = build_brief_prompt(candidates, context, today)

    try:
        response = llm_completions.create(model=model, messages=messages)
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM brief generation call failed: %s", e)
        return {"status": "error", "error": str(e)}

    briefs = parse_brief_response(content)
    valid_ids = {c.active_seed_id for c in candidates}
    batch_id = today.isoformat()

    created = 0
    for brief in briefs:
        brief_id = db_client.insert_design_brief(
            concept=brief.concept,
            product_type=brief.product_type,
            specific_products=brief.specific_products,
            audience=brief.audience,
            visual_style=brief.visual_style,
            confidence=brief.confidence,
            reasoning=brief.reasoning,
            llm_model=model,
            batch_id=batch_id,
            layout_type=brief.layout_type,
            headline_text=brief.headline_text,
            tagline_text=brief.tagline_text,
            font_color=brief.font_color,
        )
        for seed_id in brief.source_seed_ids:
            if seed_id in valid_ids:
                db_client.insert_brief_source(brief_id=brief_id, active_seed_id=seed_id)
        created += 1

    logger.info("Design brief generation: created=%d from %d candidates", created, len(candidates))

    return {"status": "success", "briefs_created": created}
