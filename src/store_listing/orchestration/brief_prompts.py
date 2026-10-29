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
    "ART_DIRECTION_RULES",
    "CONCEPT_FRESHNESS_RULES",
    "COPY_MODE_RULES",
    "DESIGN_ONLY_RULES",
    "DIVERSITY_RULES",
    "LAYOUT_RULES",
    "PRODUCT_CATALOG",
    "PRODUCT_CATALOG_BY_TYPE",
    "SEASONAL_WORDS",
    "SELLABILITY_RULES",
    "VOICE_RULES",
    "art_direction_rules",
    "concept_freshness_rules",
    "copy_mode_rules",
    "layout_rules",
    "product_belongs_to",
    "product_catalog",
    "sellability_rules",
    "voice_rules",
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
        "TEXT LAYOUT - what the design carries:\n"
        "- copy_mode decides whether the design has words at all; layout_type is the "
        "geometry. Use 'auto' for a design with a slogan (the pipeline picks one of its "
        "templates and spreads the batch across them), 'full_bleed' for a design with "
        "no text at all, and 'text_only' for a pure slogan design with no illustration. "
        "Do not name a specific template - the pipeline owns the geometry.\n"
        "- headline_text: (required unless copy_mode is 'no_text') the main text "
        "line, short and punchy (1-6 words). Words only - no emoji, no symbols, no "
        "punctuation-only text.\n"
        "- tagline_text: (optional) a second line that adds a joke, identity or "
        "occasion; usually null, especially for type_hero. It never describes the "
        "scene.\n"
        "- kicker_text: (optional, copy_mode 'type_hero' only) a short line above the "
        "hero word, 1-6 words\n"
        "- font_color: hex color for the text (e.g. '#000000' for black, '#FF8C42' for "
        f"orange). {' '.join(background_rules)}\n"
    )


LAYOUT_RULES: Final[str] = layout_rules()


COPY_MODE_RULES: Final[str] = (
    "COPY MODE - how the design carries its words:\n"
    "- copy_mode: 'type_hero', 'caption' or 'no_text'. This is not layout_type: the "
    "layout is geometry, the copy mode is whether the design has words and who draws "
    "them.\n"
    "- 'type_hero': a stacked typographic emblem. headline_text is the hero word "
    "(1-3 words, at most 6) drawn giant as the illustration; scene_description is the "
    "subject that interacts with the letters without hiding them, not the product "
    "and not the words; kicker_text is an optional short line above it; tagline_text "
    "is optional and should usually be null. A tagline is allowed only when it adds "
    "a second joke, an occasion, or a brand-like line; it must never describe the "
    "scene. Pick it when the idea is a phrase that can become the picture.\n"
    "- 'caption': an illustration with a headline and optional tagline. The headline "
    "says something the picture does not. This is the ordinary path.\n"
    "- 'no_text': the illustration explains itself. headline_text, tagline_text and "
    "kicker_text are null, and layout_type is 'full_bleed'. Pick it only for a strong "
    "visual pun, transformation or character moment that words would weaken.\n"
    "- A batch is a mix: use type_hero for some strong short phrases, no_text for the "
    "self-explaining scenes, and caption for the rest. At most half the briefs may be "
    "type_hero and at most one third no_text.\n"
)


def copy_mode_rules() -> str:
    """The copy-mode rules, for the prompts that write a brief."""
    return COPY_MODE_RULES


SELLABILITY_RULES: Final[str] = (
    "SELLABILITY - what makes a design a shirt someone buys:\n"
    "- One idea, one joke, one identity. The buyer wears the line, not a "
    "description of the picture.\n"
    "- The illustration shows the scene. Never add a line that explains what "
    "the image already shows.\n"
    "- Every line must earn its place. One strong line beats two weak ones.\n"
    "- The headline is the main line a person wears. It must work at thumbnail "
    "size, from across the room, and up close.\n"
    "- A tagline is optional. For type_hero, leave it null unless it adds a "
    "second joke, an occasion, or a brand-like line. Drop it when the design "
    "stands alone.\n"
    "- A good tagline adds a new layer. BAD: 'Through A Kitchen' (describes the "
    "scene). GOOD: 'Kitchen Commute' (a second joke).\n"
    "- A tagline is a line, not a list of words, and it never names a brand, a "
    "franchise or a trademark.\n"
    "- Specificity beats a broad pun: write for a niche, an inside joke, a "
    "profession, or a subculture.\n"
    "- Ask: would a stranger wear this without the product description? If not, "
    "rewrite it or leave the design text-free.\n"
)


def sellability_rules() -> str:
    """The design principles that make a line worth wearing."""
    return SELLABILITY_RULES


CONCEPT_FRESHNESS_RULES: Final[str] = (
    "CONCEPT FRESHNESS - the store already has designs:\n"
    "- The concepts in 'concepts_to_avoid' are already in the store. Do not write "
    "them again. Write a different scene for the same seed, even when the seed is "
    "the same one.\n"
    "- A repeat is a duplicate product, not a second chance at the same idea.\n"
)


def concept_freshness_rules() -> str:
    """The rule that keeps a second copy of a design out of the store."""
    return CONCEPT_FRESHNESS_RULES


# The store sells three voices and every brief picks one. The operator curates
# afterwards, so a batch carries a spread of them rather than one house style.
#
# The examples are deliberately evergreen. A seasonal example anchors every
# brief in the batch, which is what happened when the rules taught the
# design-vs-product rule with a holiday example (issue #200), so nothing in
# here names a calendar event.
VOICE_RULES: Final[str] = (
    "VOICE - every brief picks one of the three voices the store sells:\n"
    "- voice: 'deadpan' (dry and understated - the joke is in what is not said), "
    "'loud' (big, declarative, all caps energy) or 'cozy' (warm, handmade, "
    "affectionate).\n"
    "- The headline is what the buyer would wear. It is never a caption of the "
    "picture and never a list of three nouns: it says something, in the voice of "
    "the brief.\n"
    "- The tagline names the occasion or the setting. The headline carries the "
    "joke and the tagline never repeats the headline.\n"
    "- Words only - no emoji, no symbols, no punctuation-only text. At most 6 "
    "words.\n"
    "\n"
    "EXAMPLES - what a line looks like in each voice:\n"
    "deadpan - GOOD: 'Sleep Is For Other People' | BAD: 'Sweet Treats To Brighten Your Day'\n"
    "deadpan - GOOD: 'I Brought Snacks' | BAD: 'Cozy Nights And Warm Memories'\n"
    "loud - GOOD: 'Crush The Day' | BAD: 'Make Every Moment Count'\n"
    "loud - GOOD: 'Loud And Proud' | BAD: 'Embrace Your Unique Journey'\n"
    "cozy - GOOD: 'Made With Soup And Patience' | BAD: 'Live Your Best Life Today'\n"
    "cozy - GOOD: 'Home Is Where The Kettle Is' | BAD: 'Success Is Coming Soon'\n"
)


def voice_rules() -> str:
    """The voice rules, for the prompts that write the text of a design."""
    return VOICE_RULES


# The looks the shop prints in. The model names one of them and never writes
# a style of its own: free text made every design come out as the same flat
# vector clip-art (issue #227).
ART_DIRECTION_RULES: Final[str] = (
    "ART DIRECTION - every brief is drawn in one of the looks the shop prints:\n"
    "- style_preset: one of 'vintage_screenprint', 'seventies_retro', "
    "'halftone_comic', 'riso_two_color', 'hand_inked_woodcut', 'kawaii_flat', "
    "'art_nouveau' or 'distressed_band_tee'.\n"
    "- The look decides the colours, the linework and the type of the design. "
    "Do not name a style or a look in visual_style - name the look in "
    "style_preset and let it decide.\n"
    "- The look has to suit the concept: a quiet design is not a comic book "
    "cover and a loud one is not a quiet vintage print.\n"
)


def art_direction_rules() -> str:
    """The art direction rules, for the prompts that write a brief."""
    return ART_DIRECTION_RULES


DESIGN_ONLY_RULES: Final[str] = (
    "DESIGN vs PRODUCT - the design is the artwork, never the product it is printed on:\n"
    "- concept: the design idea, NOT the product. Bad: 'Weekend Tote with Coffee Beans'. "
    "Good: 'Coffee Beans' - the bag itself belongs in specific_products.\n"
    "- scene_description: a concise visual description of ONLY the illustration to draw "
    "(no text, no slogans, no words - just the visual scene). Under 35 words. It is what "
    "the image generator sees, so it has to be the print artwork itself, not a photo of a "
    "product. Bad: 'a tote bag full of coffee beans'. Good: 'scattered roasted coffee "
    "beans in a flat graphic'. Never mention a bag, tote, pouch, shirt, "
    "hoodie, mug, tumbler, phone case, pillow, blanket, sticker sheet, mockup or blank - "
    "if the design is for a bag, describe the graphic printed on it, not the bag. "
    "A mockup cannot be printed.\n"
    "- The scene carries ONE idea, not a list of objects and not a single object "
    "sitting still on a plain field: something unexpected, something transformed or "
    "something happening. A raccoon where the gargoyle belongs, a teapot as a "
    "lighthouse, a dragon made of paper cranes. 'A castle with trees and leaves' is a "
    "list of objects, and 'a running figure with a medal' is one object - both read as "
    "clip art.\n"
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
