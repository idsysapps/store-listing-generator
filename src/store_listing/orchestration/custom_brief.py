"""Generate a design brief from a free-text user description via LLM."""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from store_listing.orchestration.design_briefs import (
    VALID_LAYOUT_TYPES,
    DesignBrief,
    DesignBriefDBClient,
)
from store_listing.orchestration.seed_curation import PRODUCT_TAGS, _log_llm_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: Final[str] = (
    "You are a print-on-demand design strategist. A customer has described a design "
    "idea. Turn it into one actionable design brief that a graphic designer can "
    "immediately execute.\n\n"
    "Products we offer:\n"
    "- DTF printing (product_type='dtf_apparel'): t-shirts, hoodies, sweatshirts, "
    "tank tops, hats/caps, tote bags, baby onesies\n"
    "- Sublimation (product_type='sublimation'): mugs, tumblers, phone cases, "
    "mouse pads, coasters, jigsaw puzzles, ornaments, wall art, pillows, blankets, "
    "socks\n"
    "- Stickers and vinyl decals (product_type='sticker_vinyl')\n\n"
    "Return a single JSON object with this structure:\n"
    '{"brief": {\n'
    '  "concept": "design-ready phrase or visual idea",\n'
    '  "product_type": "dtf_apparel | sublimation | sticker_vinyl",\n'
    '  "specific_products": ["t-shirts"],\n'
    '  "audience": "who would buy this",\n'
    '  "visual_style": "art direction (colors, style, typography). No text/slogans here.",\n'
    '  "confidence": 0-100,\n'
    '  "reasoning": "why this design works",\n'
    '  "layout_type": "full_bleed | text_top | text_top_bottom",\n'
    '  "headline_text": "short punchy text (1-6 words) or null if full_bleed",\n'
    '  "tagline_text": "secondary text or null",\n'
    '  "font_color": "#FFFFFF",\n'
    '  "scene_description": "concise visual description of the illustration only, '
    'no text. Under 20 words."\n'
    "}}\n\n"
    "IMPORTANT: Do not use licensed IP (movie characters, team logos, brand names). "
    "Create original concepts. Respond ONLY with valid JSON."
)


def build_custom_brief_prompt(description: str, product_type: str) -> list[dict[str, str]]:
    user_content = (
        f"Design idea: {description}\n"
        f"Requested product type: {product_type}\n\n"
        "Create one design brief for this idea."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_custom_brief_response(text: str) -> DesignBrief | None:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    b = data.get("brief", {})
    if not isinstance(b, dict):
        return None

    concept = b.get("concept", "")
    if not concept:
        return None

    product_type = b.get("product_type", "")
    if product_type not in PRODUCT_TAGS:
        return None

    raw_products = b.get("specific_products", [])
    specific_products = (
        [p for p in raw_products if isinstance(p, str)] if isinstance(raw_products, list) else []
    )

    confidence = b.get("confidence", 0)
    if not isinstance(confidence, int):
        confidence = 0
    confidence = max(0, min(100, confidence))

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

    return DesignBrief(
        concept=concept,
        product_type=product_type,
        specific_products=specific_products,
        audience=str(b.get("audience", "")),
        visual_style=str(b.get("visual_style", "")),
        confidence=confidence,
        reasoning=str(b.get("reasoning", "")),
        source_seed_ids=[],
        layout_type=layout_type,
        headline_text=headline_text,
        tagline_text=tagline_text,
        font_color=font_color,
        scene_description=scene_description,
    )


def create_custom_brief(
    db_client: DesignBriefDBClient,
    llm_completions: Any,
    model: str,
    description: str,
    product_type: str,
) -> dict[str, Any]:
    messages = build_custom_brief_prompt(description, product_type)

    try:
        raw = llm_completions.with_raw_response.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        headers = raw.headers
        response = raw.parse()
        _log_llm_response(response, headers, "custom_brief")
        if not response.choices:
            return {"status": "error", "error": "LLM returned empty choices"}
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("Custom brief LLM call failed: %s", e)
        return {"status": "error", "error": str(e)}

    brief = parse_custom_brief_response(content)
    if brief is None:
        return {"status": "error", "error": "Failed to parse LLM response"}

    brief_id = db_client.insert_design_brief(
        concept=brief.concept,
        product_type=brief.product_type,
        specific_products=brief.specific_products,
        audience=brief.audience,
        visual_style=brief.visual_style,
        confidence=brief.confidence,
        reasoning=brief.reasoning,
        llm_model=model,
        batch_id="custom",
        layout_type=brief.layout_type,
        headline_text=brief.headline_text,
        tagline_text=brief.tagline_text,
        font_color=brief.font_color,
        scene_description=brief.scene_description,
    )

    return {"status": "success", "brief_id": brief_id}
