"""LLM-driven SVG design generation from design briefs.

Takes approved design briefs (concept + visual_style + product_type) and
generates vector SVG artwork via the OpenRouter LLM, then validates and
stores the results for mockup generation and production export.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final, Protocol
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

UNSAFE_TAGS = frozenset(
    {
        "script",
        "foreignObject",
        "iframe",
        "embed",
        "object",
    }
)

UNSAFE_ATTR_PREFIXES = ("on",)

CANVAS_SIZES: Final[dict[str, dict[str, int]]] = {
    "dtf_apparel": {"width": 4500, "height": 5400},
    "sublimation": {"width": 3000, "height": 3000},
    "sticker_vinyl": {"width": 2400, "height": 2400},
}

SYSTEM_PROMPT: Final[str] = (
    "You are a print-on-demand graphic designer. Your job is to create SVG "
    "artwork for apparel and merchandise based on design briefs.\n\n"
    "RULES:\n"
    "- Output ONLY a single valid SVG element — no prose, no explanation before "
    "or after (a markdown code fence is OK)\n"
    '- The SVG MUST use a viewBox (e.g. viewBox="0 0 800 1000") so it scales '
    "to any print size\n"
    '- Include the xmlns attribute: xmlns="http://www.w3.org/2000/svg"\n'
    "- Use text elements for typography — this is primarily text-based design\n"
    "- Use font-family with common web-safe or Google Fonts names\n"
    "- Use a transparent background (no background rect) unless the design "
    "requires one\n"
    "- Keep the design clean, bold, and print-ready\n"
    "- NO script tags, NO event handlers, NO foreignObject elements\n"
    "- NO embedded raster images (no <image> with base64 data)\n"
    "- Colors should be high-contrast and suitable for printing\n"
    "- Center the design in the viewBox\n"
    "- The design should be immediately recognizable and readable at t-shirt "
    "scale\n\n"
    "STYLE GUIDANCE:\n"
    "- For apparel (dtf_apparel): bold, graphic, often humorous text-based designs\n"
    "- For sublimation: can be more detailed, wrapping designs for mugs/tumblers\n"
    "- For stickers/vinyl (sticker_vinyl): clean outlines, die-cut friendly shapes\n"
)


@dataclass(frozen=True)
class DesignBriefInput:
    brief_id: int
    concept: str
    product_type: str
    specific_products: list[str] = field(default_factory=list)
    visual_style: str = ""


@dataclass(frozen=True)
class DesignRender:
    brief_id: int
    svg_content: str
    llm_model: str


class DesignRenderDBClient(Protocol):
    def list_renderable_briefs(self) -> list[DesignBriefInput]: ...

    def insert_design_render(
        self,
        brief_id: int,
        svg_content: str,
        llm_model: str,
    ) -> int: ...


def build_svg_prompt(brief: DesignBriefInput) -> list[dict[str, str]]:
    canvas = CANVAS_SIZES.get(brief.product_type, CANVAS_SIZES["dtf_apparel"])
    products_str = ", ".join(brief.specific_products) if brief.specific_products else "general"

    user_content = (
        f"Design concept: {brief.concept}\n"
        f"Product type: {brief.product_type}\n"
        f"Target products: {products_str}\n"
        f"Visual style: {brief.visual_style}\n"
        f"Recommended viewBox: 0 0 {canvas['width']} {canvas['height']}\n"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def validate_svg(svg_string: str) -> bool:
    if not svg_string or not svg_string.strip():
        return False

    try:
        root = ElementTree.fromstring(svg_string)
    except ElementTree.ParseError:
        return False

    tag = root.tag
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if tag != "svg":
        return False

    for elem in root.iter():
        elem_tag = elem.tag
        if "}" in elem_tag:
            elem_tag = elem_tag.split("}", 1)[1]

        if elem_tag in UNSAFE_TAGS:
            return False

        for attr_name in elem.attrib:
            if any(attr_name.lower().startswith(p) for p in UNSAFE_ATTR_PREFIXES):
                return False

    return True


def parse_svg_response(text: str) -> str | None:
    if not text:
        return None

    code_block_pattern = re.compile(r"```(?:svg|xml)?\s*\n(.*?)\n```", re.DOTALL)
    match = code_block_pattern.search(text)
    if match:
        svg_candidate = match.group(1).strip()
    elif "<svg" in text:
        start = text.index("<svg")
        end = text.rfind("</svg>")
        if end < 0:
            return None
        svg_candidate = text[start : end + len("</svg>")]
    else:
        return None

    if not validate_svg(svg_candidate):
        return None

    return svg_candidate


def generate_design_svgs(
    db_client: DesignRenderDBClient,
    llm_completions: Any,
    model: str,
) -> dict[str, Any]:
    briefs = db_client.list_renderable_briefs()
    if not briefs:
        return {"status": "success", "renders_created": 0, "renders_failed": 0}

    created = 0
    failed = 0

    for brief in briefs:
        messages = build_svg_prompt(brief)

        try:
            response = llm_completions.create(model=model, messages=messages)
            content = response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            logger.warning("LLM SVG generation failed for brief_id=%d", brief.brief_id)
            failed += 1
            continue

        svg = parse_svg_response(content)
        if svg is None:
            logger.warning("Invalid SVG response for brief_id=%d, skipping", brief.brief_id)
            failed += 1
            continue

        db_client.insert_design_render(
            brief_id=brief.brief_id,
            svg_content=svg,
            llm_model=model,
        )
        created += 1

    logger.info(
        "SVG generation: created=%d, failed=%d from %d briefs", created, failed, len(briefs)
    )

    return {"status": "success", "renders_created": created, "renders_failed": failed}
