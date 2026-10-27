"""How the words of a design live in the artwork.

The text used to be drawn the same way on every design: a flat headline and
tagline in the template's text zone. It read as a caption laid over the art,
not as part of it.

The brief now authors a text-integration plan: the material the words are made
of (painted, printed, embroidered, carved, neon, ...), the carrier they sit on
(a sign, a label, a banner, a ticket), where that carrier is, and why. The
plan is written by the model, which can reason about the concept; the
compositor executes it and still draws every letter from a font, so spelling
stays exact.

The vocabulary is a grammar, not a menu: the model composes a method and a
carrier freely, and the code renders the combination.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from store_listing.orchestration.design_templates import resolve_template, template_for
from store_listing.orchestration.scene_text import strip_lettering

logger = logging.getLogger(__name__)

__all__ = [
    "CARRIERS",
    "METHODS",
    "REGIONS",
    "SURFACES",
    "PlannedText",
    "TextPlan",
    "build_text_plan_entries",
    "build_text_plan_prompt",
    "carrier_is_attached",
    "check_text_plan_fit",
    "check_text_plan_mix",
    "infer_surface",
    "read_text_plan_response",
    "text_integration_rules",
    "text_plan_from_value",
]

# The material the words are made of. ``typographic`` is the existing
# treatment: the type is the design and has no surface of its own.
METHODS: Final[tuple[str, ...]] = (
    "painted",
    "printed",
    "sticker",
    "embroidery",
    "carved",
    "chalk",
    "neon",
    "typographic",
)

# A method that cuts or stitches needs a surface. The compositor can draw the
# material on the art, but the plan has to name the object it belongs to or
# the words float in the template band (issue #246).
SURFACE_METHODS: Final[frozenset[str]] = frozenset({"carved", "embroidery"})

# Carriers that have to be attached to something in the scene. A label on a
# cup is wrapped around it, a ticket is pinned, a banner hangs; a floating
# card or a flag on a stick reads as a sign beside the design, not as part of
# it (issue #246).
ATTACHABLE_CARRIERS: Final[frozenset[str]] = frozenset({"label", "ticket", "banner"})
ATTACHMENT_WORDS: Final[tuple[str, ...]] = (
    "wrapped",
    "tied",
    "hanging",
    "hangs",
    "attached",
    "pinned",
    "taped",
    "stuck",
    "around",
    "handle",
    "string",
    "band",
    "between",
    "pole",
    "post",
    "on the front",
    "on the side",
    "on the surface",
    "on the cup",
    "on the bottle",
)

# The material a carrier is normally made of, and the words that name it in
# the plan. A plan that forgot the surface still gets a sensibly matching one
# instead of a label the compositor looks for in the paper palette.
CARRIER_SURFACES: Final[dict[str, str]] = {
    "label": "paper",
    "ticket": "paper",
    "banner": "fabric",
    "book": "paper",
    "packaging": "kraft",
    "sign": "wood",
    "headstone": "stone",
    "screen": "glass",
    "apparel": "fabric",
    "none": "paper",
}
SURFACE_WORDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("kraft", ("kraft",)),
    ("chalkboard", ("chalkboard", "chalk board")),
    ("metal", ("metal", "tin", "steel", "aluminium", "aluminum")),
    ("stone", ("stone", "granite", "slate", "marble")),
    ("fabric", ("fabric", "cloth", "canvas", "linen", "cotton")),
    ("glass", ("glass",)),
    ("wood", ("wood", "wooden")),
    ("paper", ("paper", "card", "cream", "white")),
)

# What the words sit on. A carrier is optional: the concept decides.
CARRIERS: Final[tuple[str, ...]] = (
    "none",
    "sign",
    "label",
    "banner",
    "ticket",
    "book",
    "packaging",
    "headstone",
    "apparel",
    "screen",
)

# Where the carrier is on the canvas. A nine-zone grid is coarse enough for a
# model to reason about and precise enough for the compositor to search.
REGIONS: Final[tuple[str, ...]] = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
    "any",
)

# The surface of the carrier. The compositor maps each to a colour it can
# look for; the model also uses the name in the scene description.
SURFACES: Final[tuple[str, ...]] = (
    "paper",
    "kraft",
    "wood",
    "chalkboard",
    "metal",
    "stone",
    "fabric",
    "glass",
    "none",
)

INKS: Final[tuple[str, ...]] = ("dark", "light", "accent")

_HEX_INK = re.compile(r"^#[0-9A-Fa-f]{6}$")

# The model may describe a zone as "lower centre" instead of the vocabulary's
# "bottom_center". A placement it clearly meant should not decay to "any".
_REGION_ALIASES: Final[dict[str, str]] = {
    "upper_left": "top_left",
    "upper_center": "top_center",
    "upper_right": "top_right",
    "top": "top_center",
    "upper": "top_center",
    "lower_left": "bottom_left",
    "lower_center": "bottom_center",
    "lower_right": "bottom_right",
    "bottom": "bottom_center",
    "lower": "bottom_center",
    "left": "middle_left",
    "right": "middle_right",
    "middle": "center",
    "centre": "center",
    "lower_centre": "bottom_center",
    "upper_centre": "top_center",
}

_OUTPUT_SCHEMA: Final[str] = (
    "Return ONLY valid JSON with this exact shape:\n"
    '{"plans": [{"brief": 1, "method": "painted | printed | sticker | embroidery | '
    'carved | chalk | neon | typographic", "carrier": "none | sign | label | banner | '
    'ticket | book | packaging | headstone | apparel | screen", "carrier_description": '
    '"a blank wooden sign, no writing", "region": "top_left ... bottom_right | any", '
    '"surface": "paper | kraft | wood | chalkboard | metal | stone | fabric | glass | '
    'none", "ink": "dark | light | accent | #RRGGBB", "reason": "one sentence", '
    '"alternatives": [{"method": "painted", "carrier": "none", "reason": "if the '
    'carrier is not drawn"}], "scene_description": "the scene with the blank carrier '
    'in it, or omitted when the carrier is none"}]}'
)


@dataclass(frozen=True)
class TextPlan:
    """One way the words of a design live in the artwork."""

    method: str
    carrier: str = "none"
    carrier_description: str | None = None
    region: str = "any"
    surface: str = "none"
    ink: str = "dark"
    reason: str = ""
    alternatives: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def needs_carrier(self) -> bool:
        return self.carrier != "none"

    @classmethod
    def from_dict(cls, data: Any) -> TextPlan | None:
        """The plan in an answer object, or None when there is no usable plan.

        An unknown method is not a plan: the compositor has no way to draw it.
        An unknown carrier is dropped to ``none`` instead, because the words
        still have to be drawn somewhere.
        """
        if not isinstance(data, dict):
            return None

        method = str(data.get("method") or "").strip().lower()
        if method not in METHODS:
            return None

        carrier = str(data.get("carrier") or "none").strip().lower()
        if carrier not in CARRIERS:
            carrier = "none"

        region = str(data.get("region") or "any").strip().lower()
        region = _REGION_ALIASES.get(region, region)
        if region not in REGIONS:
            region = "any"

        surface = str(data.get("surface") or "none").strip().lower()
        if surface not in SURFACES:
            surface = "none"

        ink = str(data.get("ink") or "dark").strip()
        if ink not in INKS and not _HEX_INK.match(ink):
            ink = "dark"

        description = data.get("carrier_description")
        if not isinstance(description, str) or not description.strip():
            description = None

        reason = data.get("reason")
        if not isinstance(reason, str):
            reason = ""

        alternatives: list[dict[str, Any]] = []
        raw_alternatives = data.get("alternatives")
        if isinstance(raw_alternatives, list):
            for entry in raw_alternatives:
                if not isinstance(entry, dict):
                    continue
                fallback_method = str(entry.get("method") or "").strip().lower()
                if fallback_method not in METHODS:
                    continue
                fallback_carrier = str(entry.get("carrier") or "none").strip().lower()
                if fallback_carrier not in CARRIERS:
                    fallback_carrier = "none"
                alternatives.append(
                    {
                        "method": fallback_method,
                        "carrier": fallback_carrier,
                        "reason": str(entry.get("reason") or ""),
                    }
                )

        # A carrier that the image model forgot to draw must not leave the
        # design without text. The caller can fall back to a method that
        # needs no carrier at all.
        if carrier != "none" and not any(
            alternative["carrier"] == "none" for alternative in alternatives
        ):
            alternatives.append(
                {
                    "method": "painted",
                    "carrier": "none",
                    "reason": "fallback when the carrier is not drawn",
                }
            )

        return cls(
            method=method,
            carrier=carrier,
            carrier_description=description,
            region=region,
            surface=surface,
            ink=ink,
            reason=reason,
            alternatives=tuple(alternatives),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "carrier": self.carrier,
            "carrier_description": self.carrier_description,
            "region": self.region,
            "surface": self.surface,
            "ink": self.ink,
            "reason": self.reason,
            "alternatives": [dict(alternative) for alternative in self.alternatives],
        }

    @property
    def non_carrier_method(self) -> str | None:
        """The first alternative that needs no carrier, if there is one."""
        for alternative in self.alternatives:
            if alternative.get("carrier") == "none":
                method = alternative.get("method")
                if isinstance(method, str) and method in METHODS:
                    return method
        return None


def text_plan_from_value(value: Any) -> TextPlan | None:
    """A plan from a JSONB value, a JSON string, or an existing plan."""
    if isinstance(value, TextPlan):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return TextPlan.from_dict(value)


def text_integration_rules() -> str:
    """The rules for the pass that decides how the words live in a design.

    The examples are evergreen and deliberately spread across the vocabulary:
    a rule that teaches one carrier is a rule that makes every design use that
    carrier.
    """
    return (
        "TEXT INTEGRATION - how the words live in the design:\n"
        "- Decide how the headline and tagline are part of the picture, not a caption "
        "laid over it. The method is the material; the carrier is the thing it sits on "
        "and is optional.\n"
        "- Do not default to a sign. A sign, a label or a banner is right only when the "
        "concept naturally contains one. If the message is the design, use typographic. "
        "If the scene is an action, paint or print the words into the art. If the scene "
        "is an object, make the words part of that object.\n"
        "- methods: "
        "painted (brush or ink, in the same hand as the art), "
        "printed (screenprint, letterpress, riso, clean ink on a surface), "
        "sticker (die-cut vinyl, a slap of colour), "
        "embroidery (thread, stitched into fabric), "
        "carved (cut or engraved into stone, wood or metal), "
        "chalk (rough, dusty, hand-written), "
        "neon (glowing tube, dark surround), "
        "typographic (the type is the design and needs no surface).\n"
        "- carriers: none, sign, label, banner, ticket, book, packaging, headstone, "
        "apparel, screen. Combine a method and a carrier freely: embroidered on a "
        "banner, letterpress on a label, carved into a headstone, painted on a sign. "
        "A carrier is an object in the illustration, not a mockup of the product for "
        "sale. A DTF design is print artwork: never use 'apparel' as a carrier, "
        "because a garment in the scene is a product mockup.\n"
        "- A label, ticket or banner must be attached in the scene: wrapped around an "
        "object, tied to a handle, pinned, taped, hanging from a wall or between "
        "poles. Never make it a floating sign, flag, placard or card.\n"
        "- carved and embroidery need a surface: give them a carrier. painted, printed, "
        "sticker, chalk, neon and typographic can work in the art without one.\n"
        "- typographic means the type is the design: use it with carrier 'none' only.\n"
        "- reason: one sentence that says why this integration fits this concept.\n"
        "- surface is the material of the carrier; use 'none' only when the carrier is "
        "'none'. The compositor uses it to find the carrier in the picture.\n"
        "- alternatives: when the plan uses a carrier, include at least one method with "
        "carrier 'none'. It is the fallback when the image model does not draw the "
        "carrier. The fallback still has to fit the concept.\n"
        "- Never write the headline or tagline into the scene. The words are drawn "
        "afterwards by a font, so their spelling is exact. You only describe the blank "
        "carrier: 'a blank weathered wooden sign, no writing'.\n"
        "- When the carrier is not 'none', rewrite scene_description to include the "
        "blank carrier, keep the original picture, and keep it under 35 words. The "
        "image model must be able to draw it: a sign, a label, a banner, a ticket, a "
        "book, packaging, a headstone, a screen. No lettering, words, numbers or symbols "
        "on it.\n"
        "- The template's text zones are where the design expects the words. Put the "
        "carrier in one of those zones and name that region.\n"
        "- A text-only brief has no scene and no carrier: choose a method with carrier "
        "'none' and leave its scene empty.\n"
        "- A scene that already contains lettering (a shop sign, a menu, a billboard, a "
        "newspaper) will make the image model letter the carrier too. Choose a "
        "non-carrier method, or rewrite the scene so no other object has writing.\n"
        "\n"
        "BATCH MIX - the model plans the whole batch, so it does not arrive as eight "
        "versions of one treatment:\n"
        "- Use at least four different methods in a batch of eight, or as many as the "
        "batch has briefs when it is smaller.\n"
        "- No method more than twice in a batch.\n"
        "- At most one third of the batch may use a carrier.\n"
        "- At least half of the batch must use a method with carrier 'none'.\n"
        "- Let at least one design use a carrier when its concept naturally contains an "
        "object to carry the words; never force a sign into a design that has no such "
        "object.\n"
    )


class _PlannableBrief(Protocol):
    @property
    def concept(self) -> str: ...

    @property
    def product_type(self) -> str: ...

    @property
    def headline_text(self) -> str | None: ...

    @property
    def tagline_text(self) -> str | None: ...

    @property
    def scene_description(self) -> str | None: ...

    @property
    def voice(self) -> str: ...

    @property
    def style_preset(self) -> str | None: ...

    @property
    def idea_device(self) -> str | None: ...

    @property
    def layout_type(self) -> str: ...

    @property
    def text_plan(self) -> TextPlan | None: ...


def build_text_plan_entries(briefs: Sequence[_PlannableBrief]) -> list[dict[str, Any]]:
    """The briefs the planner sees, with the template each one will be drawn on.

    A full-bleed brief has no words, so it is left out. A text-only brief is
    kept because its material can still change, but it has no scene to carry a
    carrier. The ``brief`` key is 1-based and is the position in this list,
    which is how the answer is matched back to the briefs even when the model
    skips one.
    """
    entries: list[dict[str, Any]] = []
    for brief in briefs:
        template_id = resolve_template(brief.layout_type, brief.headline_text, brief.tagline_text)
        if template_id == "full_bleed":
            continue
        template = template_for(template_id)
        entries.append(
            {
                "brief": len(entries) + 1,
                "concept": brief.concept,
                "product_type": brief.product_type,
                "headline": brief.headline_text,
                "tagline": brief.tagline_text,
                "scene": brief.scene_description,
                "voice": brief.voice,
                "style_preset": brief.style_preset,
                "idea_device": brief.idea_device,
                "previous_plan": brief.text_plan.to_dict() if brief.text_plan else None,
                "template": {
                    "id": template.id,
                    "label": template.label,
                    "composition": template.composition,
                    "headline_zone": list(template.headline) if template.headline else None,
                    "tagline_zone": list(template.tagline) if template.tagline else None,
                    "size": list(template.size),
                },
            }
        )
    return entries


def build_text_plan_prompt(
    entries: Sequence[dict[str, Any]],
    extra_instructions: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """The prompt for the text-plan pass: the briefs, their templates and the rules."""
    system = (
        "You are the art director of a print-on-demand studio. You decide how the words "
        "of each design live in the picture. The compositor will execute your plan "
        "exactly; it draws the letters with a font, so you never write the words into "
        "the scene.\n\n"
        f"{text_integration_rules()}\n"
        f"{_OUTPUT_SCHEMA}"
    )
    if extra_instructions:
        system = f"{system}\n" + "\n".join(extra_instructions)
    user = json.dumps({"briefs": list(entries)}, indent=2)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True)
class PlannedText:
    """A plan and the scene that carries its blank object."""

    plan: TextPlan
    scene_description: str | None = None


def read_text_plan_response(text: str, count: int) -> tuple[dict[int, PlannedText], str | None]:
    """The plans in an answer, keyed by the 1-based brief number.

    A plan that names a method the compositor cannot draw is left out; the
    brief keeps its current text treatment instead of losing its words.
    """
    # Imported here so this module does not depend on the response parser at
    # import time, and so the parser stays free to change.
    from store_listing.orchestration.json_response import extract_json_objects, merge_items

    objects = extract_json_objects(text)
    if not any(isinstance(obj.get("plans"), list) for obj in objects):
        return {}, "the LLM response has no 'plans' array"

    raw_plans = [entry for entry in merge_items(objects, "plans") if isinstance(entry, dict)]
    plans: dict[int, PlannedText] = {}
    for index, entry in enumerate(raw_plans, start=1):
        brief_number = entry.get("brief", index)
        if isinstance(brief_number, str) and brief_number.strip().isdigit():
            brief_number = int(brief_number.strip())
        if not isinstance(brief_number, int) or brief_number < 1 or brief_number > count:
            brief_number = index
        if brief_number < 1 or brief_number > count:
            continue
        plan = TextPlan.from_dict(entry)
        if plan is None:
            logger.warning("Ignoring an unusable text plan for brief %d: %r", brief_number, entry)
            continue
        scene = entry.get("scene_description")
        if isinstance(scene, str) and scene.strip():
            scene = strip_lettering(scene) or None
        else:
            scene = None
        plans[brief_number] = PlannedText(plan=plan, scene_description=scene)

    if not plans:
        return {}, "the LLM response has no usable text plans"
    return plans, None


def check_text_plan_mix(plans: Sequence[TextPlan]) -> list[str]:
    """The ways a batch's text treatments are too alike, if any.

    This is a check, not an assignment: the model is asked again with the
    reason. The code never replaces the model's choices.
    """
    if len(plans) < 4:
        return []

    issues: list[str] = []
    methods = Counter(plan.method for plan in plans)
    if len(methods) < 4:
        issues.append(
            f"the batch uses {len(methods)} methods; at least four are needed: "
            + ", ".join(sorted(methods))
        )
    for method, count in sorted(methods.items()):
        if count > 2:
            issues.append(f"{method} is used {count} times; no method may be used more than twice")

    carriers = sum(1 for plan in plans if plan.needs_carrier)
    if carriers > len(plans) // 3:
        issues.append(f"{carriers} of {len(plans)} designs use a carrier; at most one third may")
    if carriers > 0 and len(plans) - carriers < (len(plans) + 1) // 2:
        issues.append("at least half of the designs must use a method with no carrier")

    return issues


def carrier_is_attached(plan: TextPlan, scene: str | None) -> bool:
    """Whether the plan describes the carrier as part of the scene.

    A label, ticket or banner that is not wrapped, tied, pinned, hanging or
    stuck to something becomes a floating card or a flag on a stick. That is
    the difference between type on the teacup and a sign beside it.
    """
    if plan.carrier not in ATTACHABLE_CARRIERS:
        return True
    text = f"{plan.carrier_description or ''} {scene or ''}".lower()
    return any(word in text for word in ATTACHMENT_WORDS)


def infer_surface(carrier: str, description: str | None) -> str:
    """The surface a carrier is made of, from its description or its kind."""
    text = (description or "").lower()
    for surface, words in SURFACE_WORDS:
        if any(word in text for word in words):
            return surface
    return CARRIER_SURFACES.get(carrier, "paper")


def check_text_plan_fit(
    entries: Sequence[dict[str, Any]],
    plans: Mapping[int, PlannedText],
) -> list[str]:
    """The ways a plan contradicts its product or its own method, if any.

    These are not style preferences. A garment carrier on a DTF design asks
    the image model for a product mockup; a carved or embroidered plan with
    no carrier asks the compositor to engrave words into nothing. The model
    gets the same one feedback retry as a poor mix; the code only checks.
    """
    issues: list[str] = []
    for number, planned in sorted(plans.items()):
        if number < 1 or number > len(entries):
            continue
        entry = entries[number - 1]
        plan = planned.plan
        product = str(entry.get("product_type") or "")
        if product == "dtf_apparel" and plan.carrier == "apparel":
            issues.append(
                f"brief {number} uses 'apparel' as a carrier on a DTF design; a "
                "garment is a product mockup, not artwork. Choose a sign, label, "
                "banner, ticket, book, packaging, headstone or screen, or no carrier"
            )
        if plan.needs_carrier and plan.surface == "none":
            issues.append(
                f"brief {number} uses carrier '{plan.carrier}' with surface 'none'; "
                "name the carrier's material (paper, kraft, wood, chalkboard, metal, "
                "stone, fabric or glass)"
            )
        if plan.carrier in ATTACHABLE_CARRIERS and not carrier_is_attached(
            plan, str(entry.get("scene") or "")
        ):
            issues.append(
                f"brief {number} uses carrier '{plan.carrier}' as a floating object; "
                "describe it attached in the scene: wrapped around the object, tied "
                "to the handle, pinned, taped or hanging. No sign, flag, placard or card"
            )
        if plan.method in SURFACE_METHODS and plan.carrier == "none":
            issues.append(
                f"brief {number} uses {plan.method} with no carrier; that method "
                "needs a surface. Choose a carrier, or use painted, printed, chalk, "
                "sticker, neon or typographic"
            )
        if plan.method == "typographic" and plan.carrier != "none":
            issues.append(
                f"brief {number} uses typographic with carrier '{plan.carrier}'; "
                "typographic needs no surface"
            )
    return issues
