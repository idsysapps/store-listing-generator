"""Commercial shirt silhouettes.

A template says where a headline and a tagline sit. These references are a
different decision: a stack of mixed type, a badge, a crest, a flat icon, an
arched wordmark, or a row of icons. The model writes original lines for the
slot it is given. The batch stores the slot afterwards, so a brief cannot
skip the look.

Letters stay in fonts. A silhouette with a picture tells Flux what to draw
and still forbids letters and a backdrop.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from store_listing.orchestration.art_direction import family_for, preset_for
from store_listing.orchestration.copy_modes import CAPTION, NO_TEXT
from store_listing.orchestration.text_compositor import DrawnRole

__all__ = [
    "SILHOUETTES",
    "SILHOUETTE_IDS",
    "DesignSilhouette",
    "RoleSpec",
    "apply_silhouette",
    "assign_batch_silhouettes",
    "drawn_roles",
    "plan_silhouette_slots",
    "silhouette_for",
    "silhouette_rules",
]


@dataclass(frozen=True)
class RoleSpec:
    """One line of a silhouette, before a brief fills in the words."""

    field: str
    family: str
    uppercase: bool
    color_slot: int
    region: str
    arc: str | None = None
    rule: str | None = None
    tracking: float = 0.04
    max_words: int = 4
    outline: bool = False


@dataclass(frozen=True)
class DesignSilhouette:
    """One commercial layout."""

    id: str
    label: str
    copy_mode: str
    layout_type: str
    style_preset: str
    roles: tuple[RoleSpec, ...] = ()
    composition: str = ""
    # Space the picture must leave empty so the type has shirt to sit on.
    keep_out: str = ""
    # Applied to type-only ink after the letters are drawn. Picture looks
    # take the texture of their preset instead.
    textures: tuple[str, ...] = ()
    # How the model should shape the copy for this slot.
    copy_shape: str = ""


_ISOLATED = "isolated shapes, no letters, no backdrop"

SILHOUETTES: Final[dict[str, DesignSilhouette]] = {
    "vintage_stack": DesignSilhouette(
        id="vintage_stack",
        label="Vintage type stack",
        copy_mode=CAPTION,
        layout_type="text_only",
        style_preset="vintage_screenprint",
        textures=("grain", "crackle"),
        copy_shape=(
            "three original lines and no picture: kicker_text is 2-4 short tracked "
            "words, headline_text is one script word, tagline_text is 1-3 block words"
        ),
        roles=(
            RoleSpec(
                "kicker",
                "bebas",
                True,
                1,
                "stack_kicker",
                rule="straight",
                tracking=0.08,
                max_words=4,
            ),
            RoleSpec(
                "headline",
                "pacifico",
                False,
                3,
                "stack_script",
                tracking=0.0,
                max_words=2,
            ),
            RoleSpec(
                "tagline",
                "anton",
                True,
                0,
                "stack_closer",
                tracking=0.03,
                max_words=3,
            ),
        ),
    ),
    "retro_badge": DesignSilhouette(
        id="retro_badge",
        label="Retro badge",
        copy_mode=CAPTION,
        layout_type="emblem",
        style_preset="seventies_retro",
        composition=(
            "a small cluster of simple original characters under one thin rainbow "
            "arch and below a clear horizontal centre band, the arch is a stroke "
            f"and not a filled plate, {_ISOLATED}"
        ),
        keep_out="leave a clear horizontal band across the centre unprinted",
        copy_shape=(
            "kicker_text is a short script line of 2-4 words, headline_text is one "
            "block word, tagline_text is a short script line of 1-3 words, and the "
            "scene is the character cluster only"
        ),
        roles=(
            RoleSpec("kicker", "pacifico", False, 0, "top", tracking=0.0, max_words=4),
            RoleSpec(
                "headline",
                "bungee",
                True,
                3,
                "centre_band",
                tracking=0.04,
                max_words=2,
                outline=True,
            ),
            RoleSpec("tagline", "pacifico", False, 1, "bottom", tracking=0.0, max_words=3),
        ),
    ),
    "band_crest": DesignSilhouette(
        id="band_crest",
        label="Band crest",
        copy_mode=CAPTION,
        layout_type="emblem",
        style_preset="distressed_band_tee",
        composition=(f"one centered creature or object in the middle of the frame, {_ISOLATED}"),
        keep_out=(
            "leave the space above the subject unprinted and leave a clear "
            "horizontal gap across the lower third unprinted"
        ),
        copy_shape=(
            "headline_text is 1-2 words for an arch, kicker_text is 2-3 short words "
            "across the middle, tagline_text is a tiny line of 1-3 words"
        ),
        roles=(
            RoleSpec(
                "headline",
                "anton",
                True,
                1,
                "arc_top",
                arc="up",
                tracking=0.06,
                max_words=2,
            ),
            RoleSpec("kicker", "bebas", True, 2, "crest_banner", tracking=0.12, max_words=3),
            RoleSpec("tagline", "bebas", True, 0, "crest_tiny", tracking=0.08, max_words=3),
        ),
    ),
    "flat_icon": DesignSilhouette(
        id="flat_icon",
        label="Flat icon",
        copy_mode=NO_TEXT,
        layout_type="full_bleed",
        style_preset="vintage_screenprint",
        composition=(
            "one or two flat shapes in two or three spot colours, most of the "
            f"print left bare, {_ISOLATED}"
        ),
        keep_out="leave the ground around the shapes unprinted",
        copy_shape="no words at all: headline_text, tagline_text and kicker_text are null",
    ),
    "collegiate_arch": DesignSilhouette(
        id="collegiate_arch",
        label="Collegiate arch",
        copy_mode=CAPTION,
        layout_type="emblem",
        style_preset="distressed_band_tee",
        composition=(f"one small simple mark in the centre, {_ISOLATED}"),
        keep_out="leave the space above the mark and both sides of the mark unprinted",
        copy_shape=(
            "headline_text is one arched word, kicker_text and tagline_text are one "
            "short word each and sit on either side of the mark"
        ),
        roles=(
            RoleSpec(
                "headline",
                "bebas",
                True,
                1,
                "arc_top",
                arc="up",
                tracking=0.14,
                max_words=1,
            ),
            RoleSpec("kicker", "bebas", True, 2, "flank_left", tracking=0.08, max_words=1),
            RoleSpec("tagline", "bebas", True, 0, "flank_right", tracking=0.08, max_words=1),
        ),
    ),
    "icon_row": DesignSilhouette(
        id="icon_row",
        label="Icon row",
        copy_mode=CAPTION,
        layout_type="emblem",
        style_preset="seventies_retro",
        composition=(
            "three to five simple repeated shapes in a horizontal row, one of them "
            f"with a simple pattern, {_ISOLATED}"
        ),
        keep_out="leave the space below the row unprinted",
        copy_shape="tagline_text is one script line of 1-3 words and the other lines are null",
        roles=(
            RoleSpec(
                "tagline",
                "pacifico",
                False,
                0,
                "script_below",
                rule="straight",
                tracking=0.0,
                max_words=3,
            ),
        ),
    ),
    "market_label": DesignSilhouette(
        id="market_label",
        label="Market label",
        copy_mode=CAPTION,
        layout_type="emblem",
        style_preset="vintage_screenprint",
        composition=(
            "a small still-life cluster in the centre with bare ground on both "
            f"sides and under the cluster, {_ISOLATED}"
        ),
        keep_out=("leave the top, both sides of the cluster, and the area under it unprinted"),
        copy_shape=(
            "headline_text is one word for a wide arch, subtitle_text is a short "
            "script line under the arch, kicker_text and tagline_text are one short "
            "word each beside the still-life, footer_text is a short script line "
            "under the still-life"
        ),
        roles=(
            RoleSpec(
                "headline",
                "abril",
                True,
                2,
                "arc_top",
                arc="up",
                tracking=0.04,
                max_words=2,
            ),
            RoleSpec("subtitle", "pacifico", False, 2, "arch_belly", tracking=0.0, max_words=4),
            RoleSpec("kicker", "bebas", True, 1, "flank_left", tracking=0.12, max_words=1),
            RoleSpec("tagline", "bebas", True, 1, "flank_right", tracking=0.12, max_words=1),
            RoleSpec("footer", "pacifico", False, 2, "script_below", tracking=0.0, max_words=4),
        ),
    ),
    "line_scene": DesignSilhouette(
        id="line_scene",
        label="Open line scene",
        copy_mode=NO_TEXT,
        layout_type="full_bleed",
        style_preset="hand_inked_woodcut",
        composition=(
            "a large open line drawing of one place, filling the centre of the "
            "frame, in one ink colour with a small second-colour accent, open "
            "linework, the ground between the lines stays unprinted, "
            f"{_ISOLATED}"
        ),
        keep_out="open linework, no filled plate",
        copy_shape="no words at all: headline_text, tagline_text and kicker_text are null",
    ),
    "script_scene": DesignSilhouette(
        id="script_scene",
        label="Script over a scene",
        copy_mode=CAPTION,
        layout_type="emblem",
        style_preset="riso_two_color",
        composition=(
            "a two-ink spot illustration that stays in the lower two thirds, "
            f"open linework and flat shapes, {_ISOLATED}"
        ),
        keep_out="leave the open top unprinted for a script line",
        copy_shape=(
            "headline_text is a script line of 1-3 words in the open top and the "
            "other lines are null"
        ),
        roles=(
            RoleSpec("headline", "pacifico", False, 1, "script_above", tracking=0.0, max_words=3),
        ),
    ),
}

SILHOUETTE_IDS: Final[tuple[str, ...]] = tuple(SILHOUETTES)


def silhouette_for(name: str | None) -> DesignSilhouette | None:
    """The silhouette a brief named, or None when it named none we print."""
    if not name or not isinstance(name, str):
        return None
    return SILHOUETTES.get(name.strip().lower())


def _rotation(batch_id: str) -> int:
    """Where this batch starts in the silhouette list.

    A batch id that ends in a number walks forward by that number, so a run
    of batches covers the list. Anything else is a stable hash.
    """
    match = re.search(r"(\d+)$", batch_id)
    if match:
        return int(match.group(1))
    digest = hashlib.sha1(batch_id.encode()).digest()
    return digest[0]


def plan_silhouette_slots(count: int, batch_id: str = "") -> list[str | None]:
    """One slot per brief. None keeps the ordinary path.

    At most half the batch, at most one of each silhouette. The window
    rotates with the batch id so successive batches are not the same four.
    """
    slots: list[str | None] = [None] * count
    reserved = min(len(SILHOUETTE_IDS), count // 2)
    if reserved <= 0:
        return slots
    start = _rotation(batch_id) % len(SILHOUETTE_IDS)
    picks = [SILHOUETTE_IDS[(start + offset) % len(SILHOUETTE_IDS)] for offset in range(reserved)]
    rng = random.Random(f"{batch_id}|silhouettes|{count}")
    positions = list(range(count))
    rng.shuffle(positions)
    for silhouette_id, index in zip(picks, positions, strict=False):
        slots[index] = silhouette_id
    return slots


def silhouette_rules() -> str:
    """The silhouette rules, for the prompts that write a brief."""
    lines = [
        "COMMERCIAL SILHOUETTE - the layout of some briefs in the batch:",
        (
            "- A brief_slot names a silhouette or null. Null keeps the ordinary "
            "templates. A named silhouette is the layout: write original lines in "
            "that shape and do not reuse the wording of a reference shirt."
        ),
        (
            "- Put the lines in kicker_text, headline_text and tagline_text even "
            "when copy_mode is caption. A null line stays null."
        ),
        (
            "- The lines of one silhouette are one phrase, read from the top line "
            "to the bottom line. A middle word belongs to that phrase."
        ),
    ]
    for silhouette in SILHOUETTES.values():
        lines.append(f"- {silhouette.id}: {silhouette.copy_shape}.")
    return "\n".join(lines) + "\n"


def _line(brief: Any, field: str) -> str | None:
    if field == "kicker":
        text = getattr(brief, "kicker_text", None) or getattr(brief, "stacked_kicker", None)
    else:
        text = getattr(brief, f"{field}_text", None)
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split())
    return cleaned or None


def apply_silhouette(brief: Any, silhouette_id: str) -> Any:
    """The brief locked to one silhouette.

    The copy the model wrote stays. The mode, the layout and the look are
    replaced, because those are what make the shirt this shape.
    """
    spec = SILHOUETTES[silhouette_id]
    preset = preset_for(spec.style_preset)
    fields = {role.field for role in spec.roles}
    kicker = _line(brief, "kicker") if "kicker" in fields else None
    headline = _line(brief, "headline") if "headline" in fields else None
    tagline = _line(brief, "tagline") if "tagline" in fields else None
    subtitle = _line(brief, "subtitle") if "subtitle" in fields else None
    footer = _line(brief, "footer") if "footer" in fields else None
    if spec.copy_mode == NO_TEXT:
        written = getattr(brief, "scene_description", None)
        scene = (
            written.strip() if isinstance(written, str) and written.strip() else spec.composition
        )
        headline = None
        tagline = None
        kicker = None
        subtitle = None
        footer = None
    elif spec.layout_type == "text_only":
        scene = None
    else:
        scene = getattr(brief, "scene_description", None) or spec.composition
    if scene and spec.keep_out and spec.keep_out not in scene:
        scene = f"{scene}. {spec.keep_out}"
    font_family = None
    for role in spec.roles:
        if role.field == "headline":
            font_family = role.family
            break
    if font_family is None:
        font_family = family_for(preset, getattr(brief, "voice", None))
    return replace(
        brief,
        silhouette=silhouette_id,
        copy_mode=spec.copy_mode,
        layout_type=spec.layout_type,
        style_preset=spec.style_preset,
        font_family=font_family,
        headline_text=headline,
        tagline_text=tagline,
        kicker_text=kicker,
        subtitle_text=subtitle,
        footer_text=footer,
        stacked_kicker=None,
        scene_description=scene,
        text_plan=None,
    )


def assign_batch_silhouettes(briefs: Sequence[Any], batch_id: str = "") -> list[Any]:
    """The briefs with this batch's silhouette slots applied.

    A caption that is not a silhouette does not keep a kicker: nothing draws
    it. A hero keeps the kicker it already had.
    """
    from store_listing.orchestration.copy_modes import TYPE_HERO

    slots = plan_silhouette_slots(len(briefs), batch_id)
    assigned: list[Any] = []
    for brief, slot in zip(briefs, slots, strict=True):
        if slot:
            assigned.append(apply_silhouette(brief, slot))
            continue
        if getattr(brief, "copy_mode", None) != TYPE_HERO and getattr(brief, "kicker_text", None):
            assigned.append(replace(brief, kicker_text=None, stacked_kicker=None))
            continue
        if getattr(brief, "stacked_kicker", None):
            assigned.append(replace(brief, stacked_kicker=None))
            continue
        assigned.append(brief)
    return assigned


def drawn_roles(brief: Any) -> tuple[DrawnRole, ...]:
    """The lines to draw for a silhouette brief, in order."""
    spec = silhouette_for(getattr(brief, "silhouette", None))
    if spec is None:
        return ()
    preset = preset_for(spec.style_preset)
    palette = preset.palette if preset is not None else ("#111111",)
    drawn: list[DrawnRole] = []
    for role in spec.roles:
        text = _line(brief, role.field)
        if not text:
            continue
        drawn.append(
            DrawnRole(
                text=text,
                family=role.family,
                color=palette[role.color_slot % len(palette)],
                uppercase=role.uppercase,
                region=role.region,
                arc=role.arc,
                rule=role.rule,
                tracking=role.tracking,
                outline=role.outline,
            )
        )
    return tuple(drawn)
