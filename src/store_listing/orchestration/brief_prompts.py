"""The rules the brief prompts share.

The daily trend briefs and the custom briefs are written by two different
prompts, and they drifted apart: the daily prompt learned that a scene has to
be print artwork and the custom prompt never did, so a custom brief for a bag
design produced an image of a bag (issue #195).

The rules live here so that both prompts say the same thing.

The rules are also free of any seasonal example. The rules used to teach the
design-vs-product rule with a holiday example, and the example anchored
every prompt that used the rules, so a brief batch kept coming back
seasonal no matter what the trends said (issue #200).

The product catalog is built from the types that are in production, so a
store that focuses on DTF apparel is never offered mugs (issue #215).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

__all__ = [
    "DESIGN_ONLY_RULES",
    "DIVERSITY_RULES",
    "LAYOUT_RULES",
    "PRODUCT_CATALOG",
    "PRODUCT_CATALOG_BY_TYPE",
    "SEASONAL_WORDS",
    "layout_rules",
    "product_belongs_to",
    "product_catalog",
]

PRODUCT_CATALOG_BY_TYPE: Final[dict[str, str]] = {
    "dtf_apparel": (
        "- DTF printing (product_type='dtf_apparel'): t-shirts, hoodies, sweatshirts, "
        "tank tops, hats/caps, tote bags, baby onesies"
    ),
    "sublimation": (
        "- Sublimation (product_type='sublimation'): mugs, tumblers, phone cases, "
        "mouse pads, coasters, jigsaw puzzles, ornaments, canvas/metal/poster wall art, "
        "polyester pillows, blankets, socks"
    ),
    "sticker_vinyl": "- Stickers and vinyl decals (product_type='sticker_vinyl')",
}

# The words that name a product of a production method. A brief that names a
# product of another method would be printed on something we do not make.
PRODUCT_KEYWORDS_BY_TYPE: Final[dict[str, tuple[str, ...]]] = {
    "dtf_apparel": (
        "shirt",
        "tee",
        "hoodie",
        "sweatshirt",
        "tank top",
        "long sleeve",
        "hat",
        "cap",
        "tote",
        "onesie",
    ),
    "sublimation": (
        "mug",
        "tumbler",
        "phone case",
        "mouse pad",
        "coaster",
        "puzzle",
        "ornament",
        "wall art",
        "poster",
        "pillow",
        "blanket",
        "sock",
    ),
    "sticker_vinyl": ("sticker", "vinyl", "decal"),
}


def product_catalog(enabled: Sequence[str] | None = None) -> str:
    """The catalog of the products that are in production right now."""
    types = list(enabled) if enabled else list(PRODUCT_CATALOG_BY_TYPE)
    lines = [PRODUCT_CATALOG_BY_TYPE[t] for t in types if t in PRODUCT_CATALOG_BY_TYPE]
    if not lines:
        lines = list(PRODUCT_CATALOG_BY_TYPE.values())
    return "We produce designs for these products ONLY:\n" + "\n".join(lines) + "\n"


def product_belongs_to(product: str, product_type: str) -> bool:
    """Whether a product can be made with that production method."""
    keywords = PRODUCT_KEYWORDS_BY_TYPE.get(product_type, ())
    normalized = product.strip().lower().replace("-", " ").replace("_", " ")
    return any(keyword in normalized for keyword in keywords)


PRODUCT_CATALOG: Final[str] = product_catalog()


def layout_rules(enabled: Sequence[str] | None = None) -> str:
    """The layout rules, for the product types that are in production.

    The background of a design depends on the production method, so the rule
    only talks about the types that are in production right now (issue #215).
    """
    types = list(enabled) if enabled else list(PRODUCT_CATALOG_BY_TYPE)
    background_rules: list[str] = []
    if "dtf_apparel" in types:
        background_rules.append(
            "For 'dtf_apparel' the design is rendered on a magenta background that is "
            "removed before printing, so white text is fine."
        )
    if any(t in types for t in ("sublimation", "sticker_vinyl")):
        background_rules.append(
            "For 'sublimation' and 'sticker_vinyl' the background is printed as white paper, "
            "so the text must not be white or near-white or it would be invisible - pick a "
            "saturated colour that stands out against the illustration."
        )

    return (
        "TEXT LAYOUT - controls how the design is built:\n"
        "- layout_type: one of 'full_bleed' (illustration only, no text), 'text_top' "
        "(headline at the top, art below), 'text_top_bottom' (headline top, tagline "
        "bottom, art in center), 'text_only' (typography only: the headline and the "
        "tagline, with no illustration at all). Use 'full_bleed' for a purely visual "
        "design with no slogan, 'text_only' for a pure slogan design with no "
        "illustration, and a text layout with art when a catchphrase strengthens the "
        "illustration.\n"
        "- headline_text: (required unless layout_type is 'full_bleed') the main text "
        "line, short and punchy (1-6 words). Words only - no emoji, no symbols, no "
        "punctuation-only text.\n"
        "- tagline_text: (optional) secondary text line\n"
        "- font_color: hex color for the text (e.g. '#000000' for black, '#FF8C42' for "
        f"orange). {' '.join(background_rules)}\n"
    )


LAYOUT_RULES: Final[str] = layout_rules()

DESIGN_ONLY_RULES: Final[str] = (
    "DESIGN vs PRODUCT - the design is the artwork, never the product it is printed on:\n"
    "- concept: the design idea, NOT the product. Bad: 'Weekend Tote with Coffee Beans'. "
    "Good: 'Coffee Beans' - the bag itself belongs in specific_products.\n"
    "- scene_description: a concise visual description of ONLY the illustration to draw "
    "(no text, no slogans, no words - just the visual scene). Under 20 words. It is what "
    "the image generator sees, so it has to be the print artwork itself, not a photo of a "
    "product. Bad: 'a tote bag full of coffee beans'. Good: 'scattered roasted coffee "
    "beans in a flat graphic'. Never mention a bag, tote, pouch, shirt, "
    "hoodie, mug, tumbler, phone case, pillow, blanket, sticker sheet, mockup or blank - "
    "if the design is for a bag, describe the graphic printed on it, not the bag. "
    "A mockup cannot be printed.\n"
    "- Never put words in the scene: no quoted text, no speech bubble, no sign, banner or "
    "label that carries writing, and no 'saying' or 'reading' clauses. The image model "
    "cannot spell - a scene that asks it to draw the punchline comes back misspelled "
    "(issue #208). The punchline belongs in headline_text and tagline_text, which are "
    "drawn on the artwork afterwards with a font.\n"
    "- For layout_type 'text_only' there is no illustration and no scene_description: the "
    "design is the text.\n"
)

DIVERSITY_RULES: Final[str] = (
    "DIVERSITY - a batch is not a season:\n"
    "- The calendar events above are an opportunity, not the default. Most briefs in a "
    "batch must be evergreen: a design someone buys in any month of the year.\n"
    "- At most {max_per_theme} briefs may be tied to the same event.\n"
    "- At least {min_evergreen} briefs in the batch must NOT be tied to any event.\n"
    "- If the signals are mostly seasonal, build the evergreen briefs out of the "
    "standing themes in the signals (occupations, hobbies, pets, food, home, sports).\n"
)

# The words that must not appear in the shared rules. A rule that mentions a
# holiday is a rule that pulls every brief towards that holiday.
SEASONAL_WORDS: Final[tuple[str, ...]] = (
    "halloween",
    "christmas",
    "xmas",
    "valentine",
    "easter",
    "thanksgiving",
    "mother",
    "father",
    "graduation",
    "trick or treat",
    "pumpkin",
    "spooky",
    "santa",
    "elf",
    "witch",
    "skeleton",
    "reindeer",
    "turkey",
    "pumpkin",
    "holiday",
    "super bowl",
    "football",
    "back to school",
    "teacher",
    "candy",
)
