"""Render the pipeline bake-off arms on the DGX.

This is a spike harness, not production code. It reads a JSONL export of
design briefs, renders the simplified hybrid arm and the text-first arm, and
writes a manifest that ``score_spike.py`` and ``make_contact_sheets.py``
consume. The current production renders are exported separately from the
running app.

Run it where ComfyUI is reachable (the Kubernetes app pod):

    python spike/bakeoff/run_spike.py \
        --briefs /tmp/spike/bakeoff/briefs.jsonl \
        --out /tmp/spike/out \
        --comfy-url http://192.168.2.123:8188
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from store_listing.orchestration.art_direction import (
    family_for,
    negative_for,
    palette_phrase,
    preset_for,
    prompt_for,
    tagline_family_for,
)
from store_listing.orchestration.chroma_key import remove_dark_ink, trim_sticker_border
from store_listing.orchestration.comfyui_client import (
    DEFAULT_BASE_URL,
    QWEN_SIZE,
    ComfyUIClient,
)
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.design_templates import (
    canvas_size,
    composition_for,
    resolve_template,
)
from store_listing.orchestration.hero_descriptor import compose_hero_descriptor
from store_listing.orchestration.image_generation import (
    BACKGROUND_INSTRUCTIONS,
    CLEAN_PRINT,
    DESIGN_ONLY,
    NO_MOCKUP,
    PRODUCT_TYPE_CONTEXT,
    is_dark_garment,
)
from store_listing.orchestration.image_pipeline import _brief_from_row
from store_listing.orchestration.model_text import (
    build_model_text_prompt,
    model_text_copy_from_brief,
)
from store_listing.orchestration.text_compositor import LayoutSpec, blank_canvas, composite_text
from store_listing.orchestration.text_integration import TextPlan
from store_listing.orchestration.typographic_hero import (
    build_typographic_hero_prompt,
    hero_copy_from_brief,
)
from store_listing.orchestration.typography import treatment_for_voice

logger = logging.getLogger("spike")

DEFAULT_SEED = 20261105
TEXT_ONLY_PROMPT = (
    "A typographic apparel design, the words are the illustration, "
    "no scene, no background, no backdrop, no sticker border."
)


def _load_briefs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _garment_colour(brief: DesignBrief) -> str:
    return brief.garment_color or "White"


def _simple_template(brief: DesignBrief) -> str:
    """The template the simplified arm draws on.

    The current template set is kept, but the arm does not use the text plan's
    carrier, surface, region or method. A no-text design is full bleed; the
    rest use the template the brief already names.
    """
    if brief.copy_mode == "no_text":
        return "full_bleed"
    template = resolve_template(brief.layout_type, brief.headline_text, brief.tagline_text)
    if template == "text_only":
        return "text_only"
    return template


def _composition_without_plate(template_id: str, concept: str) -> str:
    composition = composition_for(template_id, concept)
    return composition.replace(
        "the artwork fills the whole canvas edge to edge",
        "the subject is isolated on the garment, with the garment colour around it",
    ).replace(
        "covers the whole canvas edge to edge",
        "is isolated on the garment, with the garment colour around it",
    )


def hybrid_prompt(brief: DesignBrief, template_id: str) -> str:
    """The minimal Flux prompt: the subject, the look and a magenta key."""
    scene = (brief.scene_description or "").strip()
    if not scene:
        raise ValueError(f"brief {brief.concept!r} has no scene_description")
    preset = preset_for(brief.style_preset)
    look = ", ".join(text for text in (prompt_for(preset), palette_phrase(preset)) if text)
    parts = [
        f"{DESIGN_ONLY}: {scene}.",
        (
            f"{look}, {PRODUCT_TYPE_CONTEXT['dtf_apparel']}, {NO_MOCKUP}."
            if look
            else f"{PRODUCT_TYPE_CONTEXT['dtf_apparel']}, {NO_MOCKUP}."
        ),
        (
            "Isolated on the garment: no printed background, no backdrop, no scene "
            "behind the subject; the garment colour is the background."
        ),
        "No text, no lettering, no words, no numbers, no watermark, no mockup, no garment.",
        CLEAN_PRINT,
        _composition_without_plate(template_id, brief.concept),
        f"{BACKGROUND_INSTRUCTIONS['dtf_apparel']}.",
    ]
    return " ".join(parts)


def hybrid_negative() -> str:
    """The negative prompt for the simplified arm.

    It deliberately does not name the garment colour or ask the model to paint
    a black shirt. That instruction is what split earlier renders in half.
    """
    return (
        f"{negative_for(None)}, no text, no lettering, no words, no background, "
        "no backdrop, no printed background, no solid background, no sticker border, "
        "no white outline, no rectangular background, no mockup, no product"
    )


def _layout(brief: DesignBrief, template_id: str) -> LayoutSpec:
    dark = is_dark_garment(_garment_colour(brief))
    ink = "light" if dark else "dark"
    plan = TextPlan(method="typographic", carrier="none", ink=ink)
    preset = preset_for(brief.style_preset)
    return LayoutSpec(
        layout_type=template_id,
        headline_text=brief.headline_text,
        tagline_text=brief.tagline_text,
        font_color="#FFFFFF" if dark else "#000000",
        font_family=family_for(preset, brief.voice),
        tagline_font_family=tagline_family_for(preset, brief.voice),
        voice=brief.voice,
        # The accent word is off in the simplified arm: the first word was
        # being drawn in a dark accent on a dark garment (brief 213).
        treatment_override=replace(treatment_for_voice(brief.voice), accent_word=None),
        text_plan=plan,
    )


def _save_png(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _text_overlay(layout: LayoutSpec, size: tuple[int, int]) -> bytes | None:
    if not layout.needs_text:
        return None
    return composite_text(
        blank_canvas("dtf_apparel", transparent=True, size=size),
        layout,
        transparent=True,
    )


def _render_text_only_hybrid(
    brief_id: int,
    brief: DesignBrief,
    *,
    out_dir: Path,
    seed: int,
    size: tuple[int, int],
) -> dict[str, Any]:
    """A type_hero brief with no scene has no art to generate.

    The deterministic arm draws the type on a blank canvas instead of sending
    the brief to Qwen. This is the fair test of whether the compositor can
    replace the hero path when there is no scene.
    """
    layout = _layout(brief, "text_only")
    raw = composite_text(blank_canvas("dtf_apparel", size=size), layout)
    transparent = composite_text(
        blank_canvas("dtf_apparel", transparent=True, size=size), layout, transparent=True
    )
    overlay = _text_overlay(layout, size)
    stem = f"{brief_id}_hybrid_{seed}"
    _save_png(out_dir / "hybrid" / f"{stem}_raw_text.png", raw)
    _save_png(out_dir / "hybrid" / f"{stem}_transparent.png", transparent)
    _save_png(
        out_dir / "hybrid" / f"{stem}_subject.png",
        blank_canvas("dtf_apparel", transparent=True, size=size),
    )
    if overlay is not None:
        _save_png(out_dir / "hybrid" / f"{stem}_text.png", overlay)
    dark = is_dark_garment(_garment_colour(brief))
    return {
        "brief_id": brief_id,
        "arm": "hybrid",
        "seed": seed,
        "template": "text_only",
        "garment": _garment_colour(brief),
        "headline": brief.headline_text,
        "tagline": brief.tagline_text,
        "expected": [block for block in (brief.headline_text, brief.tagline_text) if block],
        "raw_path": str(out_dir / "hybrid" / f"{stem}_raw_text.png"),
        "transparent_path": str(out_dir / "hybrid" / f"{stem}_transparent.png"),
        "subject_path": str(out_dir / "hybrid" / f"{stem}_subject.png"),
        "text_path": str(out_dir / "hybrid" / f"{stem}_text.png") if overlay else None,
        "text_ink": "#F7F3E8" if dark else "#111111",
        "prompt": "text-only fallback (no scene_description)",
    }


def _render_hybrid(
    brief_id: int,
    brief: DesignBrief,
    *,
    client: ComfyUIClient,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    template_id = _simple_template(brief)
    width, height = canvas_size(template_id)
    if not (brief.scene_description or "").strip():
        return _render_text_only_hybrid(
            brief_id, brief, out_dir=out_dir, seed=seed, size=(width, height)
        )
    prompt = hybrid_prompt(brief, template_id)
    result = client.generate_image(
        prompt=prompt,
        negative_prompt=hybrid_negative(),
        width=width,
        height=height,
        seed=seed,
    )
    if result.image_bytes is None:
        raise RuntimeError(result.error or "Flux returned no image")

    raw = result.image_bytes
    keyed = client.remove_background(raw)
    if keyed.image_bytes is None:
        raise RuntimeError(keyed.error or "background removal returned no image")

    transparent = trim_sticker_border(keyed.image_bytes)
    dark = is_dark_garment(_garment_colour(brief))
    if dark:
        transparent = remove_dark_ink(transparent)

    layout = _layout(brief, template_id)
    raw_with_text = composite_text(raw, layout) if layout.needs_text else raw
    transparent_with_text = (
        composite_text(transparent, layout, transparent=True) if layout.needs_text else transparent
    )
    overlay = _text_overlay(layout, (width, height))

    stem = f"{brief_id}_hybrid_{seed}"
    _save_png(out_dir / "hybrid" / f"{stem}_raw_text.png", raw_with_text)
    _save_png(out_dir / "hybrid" / f"{stem}_transparent.png", transparent_with_text)
    _save_png(out_dir / "hybrid" / f"{stem}_subject.png", transparent)
    if overlay is not None:
        _save_png(out_dir / "hybrid" / f"{stem}_text.png", overlay)

    return {
        "brief_id": brief_id,
        "arm": "hybrid",
        "seed": seed,
        "template": template_id,
        "garment": _garment_colour(brief),
        "headline": brief.headline_text,
        "tagline": brief.tagline_text,
        "expected": [block for block in (brief.headline_text, brief.tagline_text) if block],
        "raw_path": str(out_dir / "hybrid" / f"{stem}_raw_text.png"),
        "transparent_path": str(out_dir / "hybrid" / f"{stem}_transparent.png"),
        "subject_path": str(out_dir / "hybrid" / f"{stem}_subject.png"),
        "text_path": str(out_dir / "hybrid" / f"{stem}_text.png") if overlay else None,
        "text_ink": "#F7F3E8" if dark else "#111111",
        "prompt": prompt,
    }


def _text_first_prompt(brief: DesignBrief) -> tuple[str, str, str | None]:
    if brief.copy_mode == "type_hero":
        copy = hero_copy_from_brief(brief)
        prompt = build_typographic_hero_prompt(copy)
        return prompt, "typographic_hero", copy.descriptor
    if brief.copy_mode == "no_text" or not (brief.headline_text or "").strip():
        scene = (brief.scene_description or "").strip()
        prompt = (
            f"{DESIGN_ONLY}: {scene}. {PRODUCT_TYPE_CONTEXT['dtf_apparel']}, "
            f"{NO_MOCKUP}. No text, no lettering, no words. {CLEAN_PRINT} "
            f"{BACKGROUND_INSTRUCTIONS['dtf_apparel']}."
        )
        return prompt, "art_only", None
    copy = model_text_copy_from_brief(brief)
    return build_model_text_prompt(copy), "model_text", copy.descriptor


def _render_text_first(
    brief_id: int,
    brief: DesignBrief,
    *,
    client: ComfyUIClient,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    prompt, variant, descriptor = _text_first_prompt(brief)
    result = client.generate_qwen_image(
        prompt=prompt,
        width=QWEN_SIZE,
        height=QWEN_SIZE,
        seed=seed,
        negative_prompt=(
            "white sticker border, die-cut outline, background, backdrop, "
            "offset shadow, extra text, watermark"
        ),
    )
    if result.image_bytes is None:
        raise RuntimeError(result.error or "Qwen returned no image")

    raw = result.image_bytes
    keyed = client.remove_background(raw)
    if keyed.image_bytes is None:
        raise RuntimeError(keyed.error or "background removal returned no image")
    transparent = trim_sticker_border(keyed.image_bytes)
    if is_dark_garment(_garment_colour(brief)):
        transparent = remove_dark_ink(transparent)

    if descriptor:
        raw = compose_hero_descriptor(raw, descriptor, transparent=False)
        transparent = compose_hero_descriptor(transparent, descriptor, transparent=True)

    stem = f"{brief_id}_textfirst_{seed}"
    _save_png(out_dir / "text_first" / f"{stem}_raw_text.png", raw)
    _save_png(out_dir / "text_first" / f"{stem}_transparent.png", transparent)

    expected = [block for block in (brief.headline_text, brief.tagline_text) if block]
    return {
        "brief_id": brief_id,
        "arm": "text_first",
        "seed": seed,
        "template": variant,
        "garment": _garment_colour(brief),
        "headline": brief.headline_text,
        "tagline": brief.tagline_text,
        "expected": expected,
        "raw_path": str(out_dir / "text_first" / f"{stem}_raw_text.png"),
        "transparent_path": str(out_dir / "text_first" / f"{stem}_transparent.png"),
        "subject_path": None,
        "text_path": None,
        "text_ink": None,
        "prompt": prompt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--arms", default="hybrid,text_first")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--text-first-seeds", type=int, default=1)
    parser.add_argument("--only-ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arms = {arm.strip() for arm in args.arms.split(",") if arm.strip()}
    only_ids = {int(value) for value in args.only_ids.split(",") if value.strip()}

    rows = _load_briefs(args.briefs)
    if only_ids:
        rows = [row for row in rows if int(row["id"]) in only_ids]
    if args.limit:
        rows = rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    entries: list[dict[str, Any]] = []
    if args.skip_existing and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
        done = {(entry["brief_id"], entry["arm"], entry["seed"]) for entry in entries}
    else:
        done = set()

    rng = random.Random(DEFAULT_SEED)
    with (
        ComfyUIClient(base_url=args.comfy_url, timeout_seconds=900.0) as client,
        manifest_path.open("a", encoding="utf-8") as manifest,
    ):
        for row in rows:
            brief_id = int(row["id"])
            brief = _brief_from_row(row)
            logger.info("Brief %s: %s (%s)", brief_id, brief.concept, brief.copy_mode)
            if "hybrid" in arms:
                for _index in range(args.seeds):
                    seed = rng.randint(0, 2**31 - 1)
                    if (brief_id, "hybrid", seed) in done:
                        continue
                    try:
                        entry = _render_hybrid(
                            brief_id, brief, client=client, out_dir=args.out, seed=seed
                        )
                    except Exception as error:  # noqa: BLE001 - spike must keep going
                        logger.warning("Brief %s hybrid failed: %s", brief_id, error)
                        continue
                    entries.append(entry)
                    manifest.write(json.dumps(entry) + "\n")
                    manifest.flush()
            if "text_first" in arms:
                for _index in range(args.text_first_seeds):
                    seed = rng.randint(0, 2**31 - 1)
                    if (brief_id, "text_first", seed) in done:
                        continue
                    try:
                        entry = _render_text_first(
                            brief_id, brief, client=client, out_dir=args.out, seed=seed
                        )
                    except Exception as error:  # noqa: BLE001 - spike must keep going
                        logger.warning("Brief %s text-first failed: %s", brief_id, error)
                        continue
                    entries.append(entry)
                    manifest.write(json.dumps(entry) + "\n")
                    manifest.flush()

    logger.info("Wrote %d manifest entries to %s", len(entries), manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
