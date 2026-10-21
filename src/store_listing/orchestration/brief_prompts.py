"""The rules the brief prompts share.

The daily trend briefs and the custom briefs are written by two different
prompts, and they drifted apart: the daily prompt learned that a scene has to
be print artwork and the custom prompt never did, so a custom brief for a bag
design produced an image of a bag (issue #195).

The rules live here so that both prompts say the same thing.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "DESIGN_ONLY_RULES",
    "LAYOUT_RULES",
    "PRODUCT_CATALOG",
]

PRODUCT_CATALOG: Final[str] = (
    "We produce designs for these products ONLY:\n"
    "- DTF printing (product_type='dtf_apparel'): t-shirts, hoodies, sweatshirts, "
    "tank tops, hats/caps, tote bags, baby onesies\n"
    "- Sublimation (product_type='sublimation'): mugs, tumblers, phone cases, "
    "mouse pads, coasters, jigsaw puzzles, ornaments, canvas/metal/poster wall art, "
    "polyester pillows, blankets, socks\n"
    "- Stickers and vinyl decals (product_type='sticker_vinyl')\n"
)

LAYOUT_RULES: Final[str] = (
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
    "orange). For 'dtf_apparel' the design is rendered on a magenta background that is "
    "removed before printing, so white text is fine. For 'sublimation' and "
    "'sticker_vinyl' the background is printed as white paper, so the text must not be "
    "white or near-white or it would be invisible - pick a saturated colour that stands "
    "out against the illustration.\n"
)

DESIGN_ONLY_RULES: Final[str] = (
    "DESIGN vs PRODUCT - the design is the artwork, never the product it is printed on:\n"
    "- concept: the design idea, NOT the product. Bad: 'Trick or Treat Bag with Candy "
    "Scatter'. Good: 'Candy Scatter' - the bag itself belongs in specific_products.\n"
    "- scene_description: a concise visual description of ONLY the illustration to draw "
    "(no text, no slogans, no words - just the visual scene). Under 20 words. It is what "
    "the image generator sees, so it has to be the print artwork itself, not a photo of a "
    "product. Bad: 'a trick-or-treat bag full of candy'. Good: 'scattered wrapped "
    "Halloween candy in a flat graphic'. Never mention a bag, tote, pouch, shirt, "
    "hoodie, mug, tumbler, phone case, pillow, blanket, sticker sheet, mockup or blank - "
    "if the design is for a bag, describe the graphic printed on it, not the bag. "
    "A mockup cannot be printed.\n"
    "- For layout_type 'text_only' there is no illustration and no scene_description: the "
    "design is the text.\n"
)
