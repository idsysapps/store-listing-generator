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

from store_listing.orchestration.brief_mix import (
    BRIEF_MIX_REGENERATE_ONCE,
    MAX_BRIEFS_PER_THEME,
    MIN_EVERGREEN_BRIEFS,
    classify_brief_mix,
    enforce_brief_mix,
    event_theme_for_query,
    evergreen_count,
)
from store_listing.orchestration.brief_prompts import (
    DESIGN_ONLY_RULES,
    DIVERSITY_RULES,
    LAYOUT_RULES,
    PRODUCT_CATALOG,
)
from store_listing.orchestration.context_seeds import get_upcoming_events
from store_listing.orchestration.promotion import ORIGIN_TREND
from store_listing.orchestration.seed_curation import PRODUCT_TAGS, _log_llm_response

logger = logging.getLogger(__name__)

MAX_SEEDS_PER_BRIEF = int(os.environ.get("LLM_MAX_BRIEF_SEEDS", "20"))
MAX_BRIEFS = int(os.environ.get("LLM_MAX_BRIEFS", "8"))

SYSTEM_PROMPT: Final[str] = (
    "You are a print-on-demand design strategist. Your job is to synthesize trend "
    "signals into actionable design briefs that a graphic designer can immediately "
    "execute.\n\n"
    f"{PRODUCT_CATALOG}\n"
    "For each brief:\n"
    "- concept: A specific, design-ready phrase or visual idea (not a category)\n"
    "- product_type: exactly one of 'dtf_apparel', 'sublimation', 'sticker_vinyl'\n"
    "- specific_products: array of 1-3 specific products from the catalog above\n"
    "- audience: who would buy this (demographics, interests)\n"
    "- visual_style: art direction notes (color palette, illustration style, "
    "typography). Do NOT include any text/slogan in this field — text is rendered "
    "separately via the layout fields below.\n"
    "- confidence: 0-100, how confident you are this will sell\n"
    "- reasoning: why this concept is timely or evergreen and marketable (cite the "
    "trend signals)\n"
    "- source_seed_ids: array of seed IDs from the input that inspired this brief\n\n"
    f"{LAYOUT_RULES}\n"
    f"{DESIGN_ONLY_RULES}\n"
    "MATERIALS — picking the right substrate:\n"
    "- recommended_materials: array of objects from the available_materials list, each with "
    "{external_id, name, color, substrate_type, unit_cost}. Pick 1-3 materials that best "
    "match this design's aesthetic and target audience. Consider color contrast with the "
    "design, substrate suitability, and unit cost for margin.\n"
    "- If no materials are available for the product_type, omit recommended_materials.\n\n"
    "IMPORTANT: Do not suggest designs requiring licensed IP (movie characters, "
    "team logos, brand names). Create original concepts inspired by cultural moments.\n\n"
    "Return ONLY valid JSON with this exact structure:\n"
    '{"briefs": [{"concept": "...", "product_type": "...", "specific_products": [...], '
    '"audience": "...", "visual_style": "...", "confidence": 0-100, '
    '"reasoning": "...", "source_seed_ids": [id1, id2], '
    '"layout_type": "text_top", "headline_text": "...", '
    '"tagline_text": "...", "font_color": "#000000", '
    '"scene_description": "...", '
    '"recommended_materials": [{"external_id": "...", "name": "...", '
    '"color": "...", "substrate_type": "...", "unit_cost": 0.00}]}]}'
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
    origin: str = ORIGIN_TREND
    theme: str | None = None
    seasonal_window_end: date | None = None


VALID_LAYOUT_TYPES: set[str] = {"full_bleed", "text_top", "text_top_bottom", "text_only"}

# The most words a headline may have. A typography-only design without a
# headline falls back to its concept, and a long concept is not a headline.
MAX_HEADLINE_WORDS = 6


def headline_for_text_only(concept: str, headline_text: str | None) -> str | None:
    """The headline of a typography-only design.

    A typography-only design is nothing but its text, so it needs a headline.
    A short concept makes a fine headline, a long one does not.
    """
    if headline_text and headline_text.strip():
        return headline_text
    words = concept.split()
    if 1 <= len(words) <= MAX_HEADLINE_WORDS:
        return concept
    return None


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
    regeneration_feedback: str | None = None
    scene_description: str | None = None
    recommended_materials: list[dict[str, Any]] | None = None


class DesignBriefDBClient(Protocol):
    def list_briefable_seeds(
        self, cooldown_days: int | None = None, limit: int | None = None
    ) -> list[BriefableCandidate]: ...

    def list_active_materials(
        self,
        product_type: str | None = None,
    ) -> list[dict[str, Any]]: ...

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
        scene_description: str | None = None,
        personal_use: bool = False,
        recommended_materials: list[dict[str, Any]] | None = None,
    ) -> int: ...

    def insert_brief_source(self, brief_id: int, active_seed_id: int) -> int | None: ...


def diversity_rules(
    max_per_theme: int = MAX_BRIEFS_PER_THEME,
    min_evergreen: int = MIN_EVERGREEN_BRIEFS,
) -> str:
    """The diversity rules with the quotas filled in."""
    return DIVERSITY_RULES.format(max_per_theme=max_per_theme, min_evergreen=min_evergreen)


def build_brief_prompt(
    candidates: list[BriefableCandidate],
    context: list[dict[str, Any]],
    today: date,
    materials: list[dict[str, Any]] | None = None,
    extra_instructions: list[str] | None = None,
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

    payload: dict[str, Any] = {
        "date": today.isoformat(),
        # The events are named, but their seed lists are not: the seed lists
        # sat next to the signals and read as the instruction (issue #200).
        "upcoming_events": [
            {
                "name": event.get("name"),
                "days_until": event.get("days_until"),
                "theme": event.get("theme") or event.get("name"),
            }
            for event in context
        ],
        "max_briefs": MAX_BRIEFS,
        "signals_by_source": dict(by_source),
    }

    if materials:
        payload["available_materials"] = materials

    user_content = json.dumps(payload, indent=2)

    system = SYSTEM_PROMPT
    if extra_instructions:
        system = f"{system}\n" + "\n".join(extra_instructions)
    system = f"{system}\n{diversity_rules()}"

    return [
        {"role": "system", "content": system},
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

        scene_description = b.get("scene_description")
        if scene_description is not None and not isinstance(scene_description, str):
            scene_description = None

        if layout_type == "text_only":
            headline_text = headline_for_text_only(concept, headline_text)
            if headline_text is None:
                logger.warning("Skipping brief %r: a text_only design needs a headline", concept)
                continue

        recommended_materials = b.get("recommended_materials")
        if recommended_materials is not None and not isinstance(recommended_materials, list):
            recommended_materials = None

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
                scene_description=scene_description,
                recommended_materials=recommended_materials,
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

    candidates = db_client.list_briefable_seeds(limit=MAX_SEEDS_PER_BRIEF)
    if not candidates:
        logger.warning("Design brief generation skipped: no briefable seeds")
        return {
            "status": "success",
            "briefs_created": 0,
            "briefable": 0,
            "reason": "no briefable seeds",
        }

    materials = db_client.list_active_materials()
    context = get_upcoming_events(today)
    messages = build_brief_prompt(candidates, context, today, materials=materials)

    briefs, error = _request_briefs(llm_completions, model, messages)
    if briefs is None:
        return {"status": "error", "error": error or "LLM brief generation failed"}

    valid_ids = {c.active_seed_id for c in candidates}
    batch_id = today.isoformat()
    seed_themes = {
        c.active_seed_id: (c.theme or event_theme_for_query(c.query)) for c in candidates
    }

    kept, dropped = enforce_brief_mix(briefs, seed_themes)
    evergreen = evergreen_count(kept, seed_themes)
    logger.info(
        "Design brief batch: %d briefs, %d evergreen, %d dropped over the event quota",
        len(kept),
        evergreen,
        len(dropped),
    )

    if evergreen < MIN_EVERGREEN_BRIEFS and BRIEF_MIX_REGENERATE_ONCE:
        # The model answered with a season. Ask once more, and say what went
        # wrong, before shipping a batch that is not a batch (issue #200).
        retry_messages = build_brief_prompt(
            candidates,
            context,
            today,
            materials=materials,
            extra_instructions=[
                (
                    f"The previous answer was {evergreen} evergreen briefs out of "
                    f"{len(kept)}. Write at least {MIN_EVERGREEN_BRIEFS} briefs that are "
                    "not tied to a calendar event."
                ),
            ],
        )
        retry_briefs, retry_error = _request_briefs(llm_completions, model, retry_messages)
        if retry_error:
            logger.warning("The second brief attempt failed: %s", retry_error)
        if retry_briefs:
            retry_kept, retry_dropped = enforce_brief_mix(retry_briefs, seed_themes)
            retry_evergreen = evergreen_count(retry_kept, seed_themes)
            if (len(retry_kept), retry_evergreen) > (len(kept), evergreen):
                kept, dropped = retry_kept, retry_dropped
                briefs = retry_briefs
                evergreen = retry_evergreen

    if dropped:
        logger.info(
            "Dropped %d briefs over the event quota: %s",
            len(dropped),
            ", ".join(brief.concept for brief in dropped),
        )

    mix = classify_brief_mix(kept, seed_themes)
    logger.info("Design brief mix: %s", mix.summary())

    created = 0
    for brief in kept:
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
            scene_description=brief.scene_description,
            recommended_materials=brief.recommended_materials,
        )
        linked = 0
        for seed_id in brief.source_seed_ids:
            if seed_id in valid_ids:
                db_client.insert_brief_source(brief_id=brief_id, active_seed_id=seed_id)
                linked += 1
        if not linked:
            logger.warning(
                "Design brief '%s' has no usable source seed (asked for %s)",
                brief.concept,
                brief.source_seed_ids,
            )
        created += 1

    logger.info("Design brief generation: created=%d from %d candidates", created, len(candidates))

    return {
        "status": "success",
        "briefs_created": created,
        "briefable": len(candidates),
        "seasonal": mix.seasonal,
        "evergreen": mix.evergreen,
        "dropped": len(dropped),
    }


def _request_briefs(
    llm_completions: Any,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[list[DesignBrief] | None, str | None]:
    """Ask the model for a batch of briefs: (briefs, error)."""
    try:
        raw = llm_completions.with_raw_response.create(model=model, messages=messages)
        headers = raw.headers
        response = raw.parse()
        _log_llm_response(response, headers, "briefs")
        if not response.choices:
            logger.warning(
                "LLM brief generation returned empty choices: model=%s, id=%s, body=%s",
                getattr(response, "model", None),
                getattr(response, "id", None),
                getattr(raw, "text", None),
            )
            return None, "LLM returned empty choices"
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM brief generation call failed: %s", e)
        return None, str(e)

    return parse_brief_response(content), None
