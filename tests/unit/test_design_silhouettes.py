"""Commercial silhouettes: batch slots, the type stack, and the picture prompts."""

from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from store_listing.orchestration.art_direction import preset_for
from store_listing.orchestration.design_briefs import DesignBrief, build_brief_prompt
from store_listing.orchestration.design_silhouettes import (
    SILHOUETTE_IDS,
    apply_silhouette,
    assign_batch_silhouettes,
    plan_silhouette_slots,
    silhouette_for,
)
from store_listing.orchestration.image_generation import build_flux_prompt
from store_listing.orchestration.image_pipeline import generate_images_for_briefs
from store_listing.orchestration.text_compositor import (
    ROLE_BOXES,
    DrawnRole,
    LayoutSpec,
    blank_canvas,
    composite_text,
)
from store_listing.orchestration.text_validation import TextValidationResult

_PICTURES = (
    "retro_badge",
    "band_crest",
    "flat_icon",
    "collegiate_arch",
    "icon_row",
    "market_label",
    "line_scene",
    "script_scene",
)


def _brief(concept: str, **overrides: Any) -> DesignBrief:
    defaults: dict[str, Any] = {
        "concept": concept,
        "product_type": "dtf_apparel",
        "specific_products": ["t-shirt"],
        "audience": "everyone",
        "visual_style": "bold",
        "confidence": 50,
        "reasoning": "trending",
        "source_seed_ids": [1],
        "layout_type": "auto",
        "headline_text": "Sunday",
        "tagline_text": "Club",
        "copy_mode": "caption",
        "stacked_kicker": "Off duty",
    }
    defaults.update(overrides)
    return DesignBrief(**defaults)


def _candidate() -> Any:
    from store_listing.orchestration.design_briefs import BriefableCandidate

    return BriefableCandidate(
        active_seed_id=1,
        query="dad jokes shirt",
        source="google",
        query_type="seed",
        score=10,
        delta=1,
        promotion_score=10,
        product_tags=["dtf_apparel"],
    )


class TestSilhouetteSlots:
    def test_a_batch_uses_each_silhouette_once_and_at_most_half(self) -> None:
        slots = plan_silhouette_slots(8, "run-0")
        chosen = [slot for slot in slots if slot]

        assert len(chosen) == 4
        assert len(chosen) == len(set(chosen))
        assert len(chosen) <= 8 // 2

    def test_a_single_brief_is_not_given_a_silhouette(self) -> None:
        assert plan_silhouette_slots(1, "run-0") == [None]

    def test_a_run_of_batches_covers_every_silhouette(self) -> None:
        seen: set[str] = set()
        for index in range(6):
            seen.update(slot for slot in plan_silhouette_slots(8, f"run-{index}") if slot)

        assert seen == set(SILHOUETTE_IDS)

    def test_the_prompt_names_every_silhouette_and_slots_the_batch(self) -> None:
        messages = build_brief_prompt([_candidate()], [], date(2026, 10, 22))
        system = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        slots = [slot["silhouette"] for slot in payload["brief_slots"]]
        chosen = [slot for slot in slots if slot]

        for silhouette_id in SILHOUETTE_IDS:
            assert silhouette_id in system
        assert "one phrase" in system
        assert len(slots) == payload["max_briefs"]
        assert len(chosen) <= payload["max_briefs"] // 2
        assert len(chosen) == len(set(chosen))

    def test_a_vintage_stack_keeps_the_kicker_and_drops_the_picture(self) -> None:
        brief = apply_silhouette(_brief("Sunday club"), "vintage_stack")

        assert brief.silhouette == "vintage_stack"
        assert brief.layout_type == "text_only"
        assert brief.copy_mode == "caption"
        assert brief.style_preset == "vintage_screenprint"
        assert brief.kicker_text == "Off duty"
        assert brief.headline_text == "Sunday"
        assert brief.tagline_text == "Club"
        assert brief.scene_description is None
        assert brief.text_plan is None

    def test_a_flat_icon_has_no_words(self) -> None:
        brief = apply_silhouette(_brief("A palm"), "flat_icon")

        assert brief.copy_mode == "no_text"
        assert brief.layout_type == "full_bleed"
        assert brief.headline_text is None
        assert brief.tagline_text is None
        assert brief.kicker_text is None
        assert brief.scene_description is not None
        assert "isolated shapes" in brief.scene_description

    def test_assignment_does_not_repeat_a_silhouette(self) -> None:
        briefs = [_brief(f"concept {index}") for index in range(8)]

        assigned = assign_batch_silhouettes(briefs, "run-4")
        chosen = [brief.silhouette for brief in assigned if brief.silhouette]

        assert len(chosen) <= 4
        assert len(chosen) == len(set(chosen))


class TestPicturePrompts:
    def test_picture_silhouettes_ask_for_isolated_shapes_without_letters_or_a_backdrop(
        self,
    ) -> None:
        for silhouette_id in _PICTURES:
            brief = apply_silhouette(
                _brief("Original", scene_description="a small original subject"),
                silhouette_id,
            )

            prompt = build_flux_prompt(brief).lower()

            assert "isolated shapes" in prompt
            assert "no letters" in prompt
            assert "no backdrop" in prompt
            assert "unprinted" in prompt
            assert "a shirt" not in prompt
            assert "the shirt" not in prompt
            assert "no person" in prompt
            assert "solid pure magenta" in prompt
            assert "on the garment" not in prompt

    def test_a_no_text_silhouette_keeps_the_subject_it_was_given(self) -> None:
        brief = apply_silhouette(
            _brief(
                "A shore",
                scene_description="a path between two trees toward the water",
            ),
            "line_scene",
        )

        assert brief.scene_description is not None
        assert "path between two trees" in brief.scene_description
        assert "open line" in build_flux_prompt(brief).lower()

    def test_a_line_scene_asks_for_open_linework(self) -> None:
        brief = apply_silhouette(_brief("A quiet shore"), "line_scene")

        prompt = build_flux_prompt(brief).lower()

        assert "open line" in prompt
        assert brief.scene_description is not None
        assert "shirt" not in brief.scene_description.lower()


def _near(pixel: np.ndarray, colour: str, tolerance: int = 36) -> bool:
    digits = colour.lstrip("#")
    target = np.array([int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)])
    return bool(np.abs(pixel.astype(int) - target).sum() <= tolerance)


def _png(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGBA"))


class TestTheTypeStack:
    def test_a_vintage_stack_is_three_colours_with_a_thin_rule_and_no_plate(self) -> None:
        spec = silhouette_for("vintage_stack")
        assert spec is not None
        preset = preset_for(spec.style_preset)
        assert preset is not None
        colours = [preset.palette[role.color_slot] for role in spec.roles]
        brief = apply_silhouette(_brief("Sunday club"), "vintage_stack")
        from store_listing.orchestration.design_silhouettes import drawn_roles

        image_bytes = composite_text(
            blank_canvas("dtf_apparel", transparent=True),
            LayoutSpec(
                layout_type="text_only",
                headline_text=brief.headline_text,
                tagline_text=brief.tagline_text,
                roles=drawn_roles(brief),
            ),
            transparent=True,
        )
        pixels = _png(image_bytes)
        opaque = pixels[:, :, 3] > 16
        assert opaque.mean() < 0.5

        present = []
        for colour in colours:
            count = sum(1 for pixel in pixels[opaque][:, :3] if _near(pixel, colour))
            present.append(count > 40)
        assert present == [True, True, True]

        rule_colour = colours[0]
        rule_rows = []
        for y in range(pixels.shape[0]):
            row = pixels[y, :, :3]
            hits = sum(1 for pixel in row if _near(pixel, rule_colour))
            rule_rows.append(hits > 80)
        clusters: list[int] = []
        length = 0
        for hit in rule_rows:
            if hit:
                length += 1
            elif length:
                clusters.append(length)
                length = 0
        if length:
            clusters.append(length)
        assert any(cluster <= 8 for cluster in clusters)
        assert any(cluster > 8 for cluster in clusters)

        ys, xs = np.where(opaque)
        box = opaque[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        assert box.mean() < 0.65
        assert (ys.max() - ys.min()) / pixels.shape[0] < 0.62

        spans = [_colour_span(pixels, colour) for colour in colours]
        assert spans[2] > spans[0]
        assert spans[2] > spans[1]

    def test_an_upward_arc_stays_inside_its_box(self) -> None:
        image_bytes = composite_text(
            blank_canvas("dtf_apparel", transparent=True),
            LayoutSpec(
                layout_type="text_only",
                roles=(
                    DrawnRole(
                        text="NORTH",
                        family="bebas",
                        color="#1F3050",
                        uppercase=True,
                        region="arc_top",
                        arc="up",
                        tracking=0.08,
                    ),
                ),
            ),
            transparent=True,
        )
        pixels = _png(image_bytes)
        height, width = pixels.shape[:2]
        x, y, box_width, box_height = ROLE_BOXES["arc_top"]
        left = int(x * width) - 2
        top = int(y * height) - 2
        right = int((x + box_width) * width) + 2
        bottom = int((y + box_height) * height) + 2
        ys, xs = np.where(pixels[:, :, 3] > 16)

        assert len(xs) > 0
        assert xs.min() >= left
        assert xs.max() <= right
        assert ys.min() >= top
        assert ys.max() <= bottom

    def test_a_long_serif_arch_is_separated_and_rises_in_the_middle(self) -> None:
        image_bytes = composite_text(
            blank_canvas("dtf_apparel", transparent=True),
            LayoutSpec(
                layout_type="text_only",
                roles=(
                    DrawnRole(
                        text="HARBORLIGHT",
                        family="abril",
                        color="#1F3050",
                        uppercase=True,
                        region="arc_top",
                        arc="up",
                        tracking=0.06,
                    ),
                ),
            ),
            transparent=True,
        )
        pixels = _png(image_bytes)
        height, width = pixels.shape[:2]
        x, y, box_width, box_height = ROLE_BOXES["arc_top"]
        left = int(x * width) - 2
        top = int(y * height) - 2
        right = int((x + box_width) * width) + 2
        bottom = int((y + box_height) * height) + 2
        ink = pixels[:, :, 3] > 16
        ys, xs = np.where(ink)

        assert len(xs) > 0
        assert xs.min() >= left
        assert xs.max() <= right
        assert ys.min() >= top
        assert ys.max() <= bottom

        columns = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].any(axis=0)
        gaps = 0
        in_gap = False
        for filled in columns:
            if not filled and not in_gap:
                gaps += 1
                in_gap = True
            elif filled:
                in_gap = False
        assert gaps >= 4

        span = xs.max() - xs.min()
        left_top = ys[xs < xs.min() + span / 6].min()
        middle_top = ys[(xs > xs.min() + span * 0.4) & (xs < xs.min() + span * 0.6)].min()
        assert middle_top < left_top - 4

    def test_a_covered_line_moves_into_open_shirt(self) -> None:
        covered = _covered("middle")
        moved = _role_rows(
            covered,
            DrawnRole(
                text="BLOOM",
                family="bungee",
                color="#2244CC",
                uppercase=True,
                region="middle",
                tracking=0.04,
            ),
        )
        _assert_outside(moved, covered, "middle")

        flank = _covered("flank_left")
        flank_rows = _role_rows(
            flank,
            DrawnRole(
                text="NORTH",
                family="bebas",
                color="#2244CC",
                uppercase=True,
                region="flank_left",
                tracking=0.08,
            ),
        )
        _assert_outside(flank_rows, flank, "flank_left")

        sky = _covered("script_above")
        sky_rows = _role_rows(
            sky,
            DrawnRole(
                text="easy tide",
                family="pacifico",
                color="#2244CC",
                uppercase=False,
                region="script_above",
                tracking=0.0,
            ),
        )
        _assert_outside(sky_rows, sky, "script_above")


def _covered(region: str) -> bytes:
    canvas = Image.open(BytesIO(blank_canvas("dtf_apparel", transparent=True))).convert("RGBA")
    width, height = canvas.size
    x, y, box_width, box_height = ROLE_BOXES[region]
    left, top = int(x * width), int(y * height)
    right = int((x + box_width) * width)
    bottom = int((y + box_height) * height)
    paint = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    block = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)), (180, 40, 40, 255))
    paint.paste(block, (left, top))
    canvas.alpha_composite(paint)
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _role_rows(image_bytes: bytes, role: DrawnRole) -> np.ndarray:
    rendered = composite_text(
        image_bytes,
        LayoutSpec(layout_type="emblem", roles=(role,)),
        transparent=True,
    )
    pixels = _png(rendered)
    return np.where((pixels[:, :, 0] < 80) & (pixels[:, :, 2] > 140) & (pixels[:, :, 3] > 16))[0]


def _assert_outside(rows: np.ndarray, image_bytes: bytes, region: str) -> None:
    assert len(rows) > 0
    height = _png(image_bytes).shape[0]
    _, y, _, box_height = ROLE_BOXES[region]
    top = int(y * height)
    bottom = int((y + box_height) * height)
    median = float(np.median(rows))
    assert median < top or median > bottom


def _colour_span(pixels: np.ndarray, colour: str) -> int:
    opaque = pixels[:, :, 3] > 16
    ys = [
        y
        for y in range(pixels.shape[0])
        if any(_near(pixel, colour) for pixel in pixels[y, opaque[y], :3])
    ]
    if not ys:
        return 0
    return ys[-1] - ys[0] + 1


class TestTypeOnlyUpload:
    def test_a_vintage_stack_uploads_words_that_match_and_distresses_only_the_ink(self) -> None:
        from store_listing.orchestration.design_silhouettes import drawn_roles

        row = {
            "id": 7,
            "concept": "Sunday club",
            "product_type": "dtf_apparel",
            "audience": "everyone",
            "visual_style": "vintage",
            "layout_type": "text_only",
            "copy_mode": "caption",
            "headline_text": "Sunday",
            "tagline_text": "Club",
            "kicker_text": "Off duty",
            "font_color": "#C1502E",
            "scene_description": None,
            "style_preset": "vintage_screenprint",
            "silhouette": "vintage_stack",
            "voice": "deadpan",
        }
        validator = MagicMock()
        validator.validate.return_value = TextValidationResult(
            matched=True, expected=["Sunday", "Club"], observed="Sunday Club", score=1.0
        )
        db = MagicMock()
        db.list_briefs_without_images.return_value = [row]
        comfyui = MagicMock()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=s3,
            bucket="test-bucket",
            text_validator=validator,
        )

        assert result["images_generated"] == 1
        comfyui.generate_image.assert_not_called()
        assert validator.validate.call_args.kwargs["headline"] == "Sunday"
        assert validator.validate.call_args.kwargs["tagline"] == "Club"
        transparent = s3.put_object.call_args_list[1].kwargs["Body"]
        uploaded = _png(transparent)
        plain = _png(
            composite_text(
                blank_canvas("dtf_apparel", transparent=True),
                LayoutSpec(
                    layout_type="text_only",
                    headline_text="Sunday",
                    tagline_text="Club",
                    roles=drawn_roles(
                        _brief(
                            "Sunday club",
                            silhouette="vintage_stack",
                            layout_type="text_only",
                            kicker_text="Off duty",
                            stacked_kicker=None,
                        )
                    ),
                ),
                transparent=True,
            )
        )
        assert np.array_equal(uploaded[:, :, 3] == 0, plain[:, :, 3] == 0)
        clear = uploaded[:, :, 3] == 0
        assert np.all(uploaded[clear][:, :3] == 0)
        ink = uploaded[:, :, 3] > 16
        assert not np.array_equal(uploaded[ink][:, :3], plain[ink][:, :3])
