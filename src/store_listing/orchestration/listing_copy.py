"""Write the marketplace description a buyer reads on the product page.

The scene description is an image-generation prompt. It describes the picture,
not the product, and it cannot be pasted into an Amazon listing. This pass
writes the listing copy after the design, headline and material are final, so
it cannot quote copy that the copy editor removed.

The pass is optional. A deterministic fallback keeps the field populated when
the model is disabled or unavailable (issue #297).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.json_response import extract_json_objects, merge_items
from store_listing.orchestration.llm_client import create_answer

logger = logging.getLogger(__name__)

__all__ = [
    "ListingCopy",
    "build_listing_copy_entries",
    "build_listing_copy_prompt",
    "fallback_listing_description",
    "listing_copy_enabled",
    "parse_listing_copy_response",
    "write_batch_listing_copy",
]

MIN_DESCRIPTION_CHARS: Final[int] = 40
MAX_DESCRIPTION_CHARS: Final[int] = 1200

# A description that uses these phrases is describing the picture again.
_SCENE_PHRASES: Final[tuple[str, ...]] = (
    "this image",
    "the image",
    "this illustration",
    "the illustration",
    "illustration shows",
    "shows a",
    "depicts",
    "mockup",
    "design file",
    "prompt",
    "the scene",
    "this scene",
)

# Claims an Amazon listing must not make without evidence.
_BANNED_CLAIMS: Final[tuple[str, ...]] = (
    "best seller",
    "best-selling",
    "perfect gift",
    "guaranteed",
    "lifetime warranty",
    "free shipping",
)

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

_SYSTEM_PROMPT: Final[str] = """You write Amazon product descriptions for a print-on-demand store.
Each description is what a buyer reads on the product page. It must sell the
product, not describe the picture that will be printed.

RULES
- Write 2-4 sentences of plain text. No HTML, markdown, bullet points or emojis.
- Start with the product and the design or headline.
- Say who the design is for and when to wear or give it.
- Do not describe the scene, illustration, artwork, mockup, design file or prompt.
- Do not invent material, size, price, shipping, discounts, warranties or claims
  such as "best seller" or "perfect gift".
- Use the primary material's colour and substrate when it is given.
- Do not name a brand, franchise, team or trademark.
- Do not wrap the whole description in quotation marks.

BAD: "A skeleton doing yoga on a beach towel in a retro palette." This describes
the picture.
GOOD: "A white t-shirt for yoga lovers, featuring a skeleton in a beachside yoga
pose. Wear it to class or give it to the friend who never misses a flow."

Return ONLY valid JSON with this shape:
{"listings":[{"index":1,"description":"..."}]}
"""


def listing_copy_enabled() -> bool:
    """Whether the brief pipeline asks the model to write the listing copy."""
    value = os.environ.get("LLM_LISTING_COPY", "false").strip().lower()
    return value not in ("0", "false", "no")


@dataclass(frozen=True)
class ListingCopy:
    """The model's description for one brief."""

    index: int
    description: str


def _primary_material(brief: DesignBrief) -> dict[str, Any] | None:
    for material in brief.recommended_materials or []:
        if isinstance(material, dict) and material.get("external_id"):
            return material
    return None


def _product_name(brief: DesignBrief) -> str:
    if brief.specific_products:
        return brief.specific_products[0].strip()
    names = {
        "dtf_apparel": "apparel",
        "sublimation": "sublimation product",
        "sticker_vinyl": "sticker",
    }
    return names.get(brief.product_type, brief.product_type.replace("_", " "))


def _entry(index: int, brief: DesignBrief) -> dict[str, Any]:
    material = _primary_material(brief)
    return {
        "index": index,
        "concept": brief.concept,
        "product_type": brief.product_type,
        "specific_products": list(brief.specific_products),
        "audience": brief.audience,
        "visual_style": brief.visual_style,
        "headline": brief.headline_text,
        "tagline": brief.tagline_text,
        "kicker": brief.kicker_text,
        "primary_material": (
            {
                "name": material.get("name"),
                "color": material.get("color"),
                "substrate_type": material.get("substrate_type"),
            }
            if material
            else None
        ),
    }


def build_listing_copy_entries(briefs: Sequence[DesignBrief]) -> list[dict[str, Any]]:
    """The briefs the listing writer sees, without their scene prompts."""
    return [_entry(index, brief) for index, brief in enumerate(briefs, start=1)]


def build_listing_copy_prompt(entries: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """The messages that ask the model for marketplace descriptions."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"briefs": list(entries)}, indent=2)},
    ]


def _clean_description(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = _HTML_TAG.sub(" ", value)
    text = _WHITESPACE.sub(" ", text).strip()
    if not (MIN_DESCRIPTION_CHARS <= len(text) <= MAX_DESCRIPTION_CHARS):
        return None
    lowered = text.lower()
    if any(phrase in lowered for phrase in _SCENE_PHRASES):
        return None
    if any(claim in lowered for claim in _BANNED_CLAIMS):
        return None
    return text


def _listing_from(value: Any, expected_indexes: set[int]) -> ListingCopy | None:
    if not isinstance(value, dict):
        return None
    index = value.get("index")
    if not isinstance(index, int) or index not in expected_indexes:
        return None
    description = _clean_description(value.get("description"))
    if description is None:
        return None
    return ListingCopy(index=index, description=description)


def parse_listing_copy_response(
    text: str,
    *,
    expected_indexes: set[int],
) -> tuple[dict[int, ListingCopy], str | None]:
    """The descriptions in an answer, and why the answer was incomplete."""
    objects = extract_json_objects(text)
    items = merge_items(objects, "listings")
    if not items:
        return {}, "the answer has no 'listings' array"

    copies: dict[int, ListingCopy] = {}
    for item in items:
        copy = _listing_from(item, expected_indexes)
        if copy is not None:
            copies[copy.index] = copy

    missing = expected_indexes - set(copies)
    if missing:
        return copies, f"missing listings for indexes {sorted(missing)}"
    return {index: copies[index] for index in sorted(copies)}, None


def _answer_content(raw: Any) -> tuple[str | None, str | None]:
    try:
        response = raw.parse()
        if not response.choices:
            return None, "the model returned no choices"
        return response.choices[0].message.content or "", None
    except Exception as error:  # noqa: BLE001 - report the failure, do not hide it
        return None, str(error)


def _request_listing_copy(
    llm_completions: Any,
    model: str,
    entries: Sequence[dict[str, Any]],
    *,
    attempts: int = 2,
) -> tuple[dict[int, ListingCopy], list[str]]:
    if not entries:
        return {}, []
    messages = build_listing_copy_prompt(entries)
    expected = {int(entry["index"]) for entry in entries}
    last: list[str] = []

    for _ in range(max(1, attempts)):
        raw, error = create_answer(
            llm_completions,
            model,
            messages,
            "listing_copy",
            temperature=0.3,
        )
        if error is not None:
            last = [f"listing copy failed: {error}"]
            continue
        content, read_error = _answer_content(raw)
        if read_error is not None or content is None:
            last = [f"listing copy failed: {read_error or 'empty answer'}"]
            continue
        copies, parse_error = parse_listing_copy_response(content, expected_indexes=expected)
        if parse_error is None:
            return copies, []
        last = [f"listing copy was incomplete: {parse_error}"]

    return {}, last


def fallback_listing_description(brief: DesignBrief) -> str:
    """A plain, marketplace-safe description when the model is unavailable."""
    material = _primary_material(brief) or {}
    colour = str(material.get("color") or "").strip()
    product = _product_name(brief)
    headline = (brief.headline_text or "").strip().rstrip(".!? ")
    if not headline:
        # A no-text concept can still be a scene description. The fallback must
        # not turn it back into one, so it stays deliberately generic.
        headline = "An original design"

    printed_on = f"{colour + ' ' if colour else ''}{product}".strip()
    if printed_on:
        first = f"{headline}. Printed on {printed_on}."
    else:
        first = f"{headline}. A printed apparel design."

    audience = (brief.audience or "").strip()
    if audience:
        return f"{first} Designed for {audience}."
    return f"{first} A wearable design for everyday outfits and gifting."


def write_batch_listing_copy(
    llm_completions: Any,
    model: str,
    briefs: Sequence[DesignBrief],
    *,
    enabled: bool | None = None,
) -> tuple[list[DesignBrief], list[str]]:
    """Write a description for every brief, with a deterministic fallback."""
    original = list(briefs)
    if not original:
        return original, []

    fallback = {
        index: fallback_listing_description(brief) for index, brief in enumerate(original, start=1)
    }
    if enabled is None:
        enabled = listing_copy_enabled()
    if not enabled:
        return [
            replace(brief, listing_description=fallback[index])
            for index, brief in enumerate(original, start=1)
        ], []

    entries = build_listing_copy_entries(original)
    copies, issues = _request_listing_copy(llm_completions, model, entries)
    if issues:
        return [
            replace(brief, listing_description=fallback[index])
            for index, brief in enumerate(original, start=1)
        ], issues

    result: list[DesignBrief] = []
    for index, brief in enumerate(original, start=1):
        copy = copies.get(index)
        description = copy.description if copy is not None else fallback[index]
        result.append(replace(brief, listing_description=description))
    return result, []
