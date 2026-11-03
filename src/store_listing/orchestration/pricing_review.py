"""The optional LLM pricing pass.

The model chooses a price positioning for each DTF brief, using the design's
confidence, the recommended material's blank cost, the calculated print cost
and the substrate's price ceiling. It never supplies the final price: the
pricing module computes that and downgrades a tier that does not fit.

The pass is opt-in (``LLM_PRICING_REVIEW``). When it is off or fails, the
pipeline falls back to the confidence gate and standard pricing, so a broken
pricing call never blocks a design batch (issue #295).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.design_templates import canvas_size
from store_listing.orchestration.json_response import extract_json_objects, merge_items
from store_listing.orchestration.llm_client import create_answer
from store_listing.orchestration.pricing import (
    POSITIONING_ORDER,
    PricingPolicy,
    allowed_positionings,
    estimated_prices,
    fallback_positioning,
    policy_from_env,
    print_spec_for,
    suggest_print_size,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PositioningChoice",
    "build_pricing_entries",
    "build_pricing_prompt",
    "choose_batch_positionings",
    "parse_pricing_response",
    "pricing_review_enabled",
    "request_pricing_choices",
]

_SYSTEM_PROMPT: Final[
    str
] = """You are the pricing strategist for a print-on-demand DTF apparel store
selling on Amazon.

You choose a price positioning for each design. The code will calculate the
final price from the blank cost, the DTF print cost, the Amazon fee and the
target margin. You never write the price.

POSITIONINGS
- budget: 30% target margin, $14.99 minimum price
- standard: 40% target margin, $19.99 minimum price
- premium: 50% target margin, $24.99 minimum price

SELLABILITY GATE
- confidence below 40: budget only
- confidence 40 to 69: budget or standard
- confidence 70 and above: budget, standard or premium

RULES
- A high-confidence, broad-audience design can carry a premium price.
- A niche, seasonal or weak design should use budget or standard so it stays
  competitive.
- A higher-cost blank needs a higher price. If premium would push the design
  over its price ceiling, choose standard or budget instead.
- The code will downgrade a positioning that does not fit the ceiling; pick the
  best fit rather than the most expensive tier.
- Do not choose a positioning outside the allowed list for the confidence.
- Give one short reason per design.

Return ONLY valid JSON with this shape:
{"positionings":[{"index":1,"positioning":"budget | standard | premium","reason":"one sentence"}]}
"""


def pricing_review_enabled() -> bool:
    """Whether the brief pipeline asks the model to choose a positioning."""
    value = os.environ.get("LLM_PRICING_REVIEW", "false").strip().lower()
    return value not in ("0", "false", "no")


@dataclass(frozen=True)
class PositioningChoice:
    """The pricing model's choice for one brief."""

    index: int
    positioning: str
    reason: str = ""


def _pricing_entry(index: int, brief: DesignBrief, policy: PricingPolicy) -> dict[str, Any] | None:
    materials = [
        material for material in (brief.recommended_materials or []) if isinstance(material, dict)
    ]
    if not materials:
        return None
    primary = materials[0]
    spec = print_spec_for(brief.product_type, primary, brief.specific_products)
    if spec is None:
        return None
    unit_cost = primary.get("unit_cost")
    if unit_cost is None:
        return None

    canvas = canvas_size(brief.layout_type)
    print_size = suggest_print_size(canvas, spec, policy=policy)
    print_cost = policy.print_rate_per_sq_in * print_size.width_in * print_size.height_in
    return {
        "index": index,
        "concept": brief.concept,
        "audience": brief.audience,
        "confidence": brief.confidence,
        "product_type": brief.product_type,
        "primary_material": {
            "name": primary.get("name"),
            "substrate_type": primary.get("substrate_type"),
            "unit_cost": unit_cost,
        },
        "print_size": {
            "width_in": float(print_size.width_in),
            "height_in": float(print_size.height_in),
            "placement": print_size.placement,
        },
        "print_cost": float(print_cost),
        "estimated_retail": estimated_prices(
            primary,
            brief.product_type,
            brief.specific_products,
            canvas,
            policy,
        ),
        "price_ceiling": float(spec.max_price),
        "allowed_positionings": list(allowed_positionings(brief.confidence)),
    }


def build_pricing_entries(
    briefs: Sequence[DesignBrief],
    *,
    policy: PricingPolicy | None = None,
) -> list[dict[str, Any]]:
    """The DTF briefs the pricing model can price, with their cost context."""
    active = policy or policy_from_env()
    entries: list[dict[str, Any]] = []
    for index, brief in enumerate(briefs, start=1):
        entry = _pricing_entry(index, brief, active)
        if entry is not None:
            entries.append(entry)
    return entries


def build_pricing_prompt(entries: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """The messages that ask the model to choose a positioning per brief."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"briefs": list(entries)}, indent=2)},
    ]


def _choice_from(value: Any, expected_indexes: set[int]) -> PositioningChoice | None:
    if not isinstance(value, dict):
        return None
    index = value.get("index")
    if not isinstance(index, int) or index not in expected_indexes:
        return None
    raw_positioning = value.get("positioning")
    if not isinstance(raw_positioning, str):
        return None
    positioning = raw_positioning.strip().lower()
    if positioning not in POSITIONING_ORDER:
        return None
    raw_reason = value.get("reason")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    return PositioningChoice(index=index, positioning=positioning, reason=reason)


def parse_pricing_response(
    text: str,
    *,
    expected_indexes: set[int],
) -> tuple[dict[int, PositioningChoice], str | None]:
    """The choices in an answer, and why the answer was incomplete."""
    objects = extract_json_objects(text)
    items = merge_items(objects, "positionings")
    if not items:
        return {}, "the answer has no 'positionings' array"

    choices: dict[int, PositioningChoice] = {}
    for item in items:
        choice = _choice_from(item, expected_indexes)
        if choice is not None:
            choices[choice.index] = choice

    missing = expected_indexes - set(choices)
    if missing:
        return choices, f"missing positionings for indexes {sorted(missing)}"
    return {index: choices[index] for index in sorted(choices)}, None


def _answer_content(raw: Any) -> tuple[str | None, str | None]:
    try:
        response = raw.parse()
        if not response.choices:
            return None, "the model returned no choices"
        return response.choices[0].message.content or "", None
    except Exception as error:  # noqa: BLE001 - report the failure, do not hide it
        return None, str(error)


def request_pricing_choices(
    llm_completions: Any,
    model: str,
    entries: Sequence[dict[str, Any]],
    *,
    attempts: int = 2,
) -> tuple[dict[int, PositioningChoice], list[str]]:
    """One pricing call: the choices and any reason the answer was unusable."""
    if not entries:
        return {}, []
    messages = build_pricing_prompt(entries)
    expected = {int(entry["index"]) for entry in entries}
    last: list[str] = []

    for _ in range(max(1, attempts)):
        raw, error = create_answer(
            llm_completions,
            model,
            messages,
            "pricing_review",
            temperature=0,
        )
        if error is not None:
            last = [f"pricing review failed: {error}"]
            continue
        content, read_error = _answer_content(raw)
        if read_error is not None or content is None:
            last = [f"pricing review failed: {read_error or 'empty answer'}"]
            continue
        choices, parse_error = parse_pricing_response(content, expected_indexes=expected)
        if parse_error is None:
            return choices, []
        last = [f"pricing review was incomplete: {parse_error}"]

    return {}, last


def choose_batch_positionings(
    llm_completions: Any,
    model: str,
    briefs: Sequence[DesignBrief],
    *,
    policy: PricingPolicy | None = None,
    enabled: bool | None = None,
) -> tuple[dict[int, PositioningChoice], list[str]]:
    """Choose a positioning per DTF brief, with a deterministic fallback.

    The fallback is not an error: a disabled or unavailable pricing model still
    leaves the pipeline with a grounded cost-plus price.
    """
    active = policy or policy_from_env()
    entries = build_pricing_entries(briefs, policy=active)
    if not entries:
        return {}, []

    fallback = {
        int(entry["index"]): PositioningChoice(
            index=int(entry["index"]),
            positioning=fallback_positioning(entry.get("confidence")),
            reason="deterministic fallback",
        )
        for entry in entries
    }

    if enabled is None:
        enabled = pricing_review_enabled()
    if not enabled:
        return fallback, []

    choices, issues = request_pricing_choices(llm_completions, model, entries)
    if issues:
        return fallback, issues

    for index, choice in fallback.items():
        choices.setdefault(index, choice)
    return choices, []
