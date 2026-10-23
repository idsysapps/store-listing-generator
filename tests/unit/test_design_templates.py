"""Tests for the design templates.

Every rendered design used to be one of three silhouettes: art in the middle
with a headline at the top, optionally a tagline at the bottom. The layouts
that existed were never picked, and the artwork prompt asked for a compact
centred composition every time (issue #238).

A template fixes the art composition, the text geometry and the shape of the
canvas together. The assignment is done here, in code, so a batch is not at
the mercy of what the model likes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from store_listing.orchestration.design_templates import (
    ART_TEMPLATE_IDS,
    FULL_BLEED_COMPOSITIONS,
    LEGACY_LAYOUTS,
    TEMPLATE_IDS,
    TEMPLATES,
    assign_templates,
    canvas_size,
    composition_for,
    resolve_template,
    template_for,
)


@dataclass
class _Brief:
    concept: str = "A design"
    headline_text: str | None = "GOOD LINE"
    tagline_text: str | None = None
    layout_type: str = "auto"


class TestTheRegistry:
    def test_there_are_enough_templates_to_make_a_batch(self) -> None:
        assert len(ART_TEMPLATE_IDS) >= 8

    def test_every_art_template_has_a_composition_prompt(self) -> None:
        for template_id in ART_TEMPLATE_IDS:
            assert TEMPLATES[template_id].composition.strip(), template_id

    def test_every_template_names_its_own_art_composition(self) -> None:
        compositions = [TEMPLATES[template_id].composition for template_id in ART_TEMPLATE_IDS]

        assert len(set(compositions)) == len(compositions)

    def test_the_templates_are_not_all_the_same_shape(self) -> None:
        shapes = {canvas_size(template_id) for template_id in TEMPLATE_IDS}

        assert len(shapes) >= 3
        assert (1216, 832) in shapes
        assert (832, 1216) in shapes
        assert (1024, 1024) in shapes

    def test_every_text_template_has_a_place_for_the_headline(self) -> None:
        for template_id, template in TEMPLATES.items():
            if template_id in ("full_bleed", "text_only"):
                continue
            assert template.headline is not None, template_id

    def test_every_text_template_can_hold_text_without_a_tagline(self) -> None:
        # A tagline is optional, so the headline has to fit on its own.
        for template_id, template in TEMPLATES.items():
            if template_id in ("full_bleed", "text_only"):
                continue
            assert template.tagline is None or template.headline is not None

    def test_the_placements_stay_inside_the_canvas(self) -> None:
        for template_id, template in TEMPLATES.items():
            for placement in (template.headline, template.tagline):
                if placement is None:
                    continue
                x, y, width, height = placement
                assert 0 <= x < 1, template_id
                assert 0 <= y < 1, template_id
                assert x + width <= 1.001, template_id
                assert y + height <= 1.001, template_id


class TestResolveTemplate:
    @pytest.mark.parametrize(
        ("legacy", "expected"),
        [
            ("text_top", "hero_bottom"),
            ("text_top_bottom", "emblem"),
            ("badge", "roundel"),
            ("banner", "mid_ribbon"),
            ("arch", "tall_arch"),
            ("stack_side", "side_column"),
        ],
    )
    def test_a_legacy_layout_maps_to_one_template(self, legacy: str, expected: str) -> None:
        assert resolve_template(legacy) == expected

    def test_text_top_with_a_tagline_does_not_collapse_to_text_top_bottom(self) -> None:
        # The old code upgraded text_top to text_top_bottom whenever a tagline
        # existed, which is how every design became top-and-bottom.
        assert resolve_template("text_top", "GOOD", "TAG") == "hero_bottom"

    def test_a_template_id_is_returned_as_is(self) -> None:
        for template_id in TEMPLATE_IDS:
            assert resolve_template(template_id) == template_id

    def test_an_unknown_layout_falls_back_to_a_text_template(self) -> None:
        assert resolve_template("something_else", "GOOD", None) in TEMPLATE_IDS

    def test_full_bleed_stays_full_bleed(self) -> None:
        assert resolve_template("full_bleed", None, None) == "full_bleed"

    def test_text_only_stays_text_only(self) -> None:
        assert resolve_template("text_only", "GOOD", None) == "text_only"

    def test_every_legacy_layout_maps_to_a_known_template(self) -> None:
        for legacy, template_id in LEGACY_LAYOUTS.items():
            assert template_id in TEMPLATES, legacy

    def test_template_for_returns_the_template(self) -> None:
        assert template_for("badge").id == "roundel"


class TestComposition:
    def test_a_template_does_not_ask_for_a_centred_emblem(self) -> None:
        # Every prompt used to carry this phrase, which is why every design
        # was an object in the middle of the canvas.
        for template_id in ART_TEMPLATE_IDS:
            assert "centered composition" not in composition_for(template_id).lower()

    def test_a_full_bleed_design_gets_one_of_the_full_bleed_compositions(self) -> None:
        composition = composition_for("full_bleed", "a raccoon king")

        assert composition in FULL_BLEED_COMPOSITIONS

    def test_the_full_bleed_composition_is_stable_for_a_concept(self) -> None:
        assert composition_for("full_bleed", "a raccoon king") == composition_for(
            "full_bleed", "a raccoon king"
        )

    def test_different_concepts_can_get_different_full_bleed_compositions(self) -> None:
        compositions = {composition_for("full_bleed", f"concept {index}") for index in range(40)}

        assert len(compositions) > 1

    def test_a_layout_that_leaves_room_says_where(self) -> None:
        for template_id in ("hero_bottom", "side_column", "wide_scene"):
            composition = composition_for(template_id).lower()
            assert "clear" in composition or "plain background" in composition, template_id


class TestAssignment:
    def test_the_assignment_is_deterministic_for_a_batch(self) -> None:
        briefs = [_Brief(concept=f"concept {index}") for index in range(8)]

        assert assign_templates(briefs, "2026-10-22") == assign_templates(briefs, "2026-10-22")

    def test_a_batch_of_eight_gets_eight_different_templates(self) -> None:
        briefs = [_Brief(concept=f"concept {index}") for index in range(8)]

        assigned = assign_templates(briefs, "2026-10-22")

        assert len(set(assigned)) == 8

    def test_no_template_runs_away_with_a_big_batch(self) -> None:
        briefs = [_Brief(concept=f"concept {index}") for index in range(20)]

        assigned = assign_templates(briefs, "2026-10-22")

        assert max(assigned.count(template) for template in set(assigned)) <= 2

    def test_a_brief_without_copy_is_rendered_as_art_only(self) -> None:
        briefs = [_Brief(headline_text=None, tagline_text=None) for _ in range(4)]

        assert assign_templates(briefs, "2026-10-22") == ["full_bleed"] * 4

    def test_a_text_only_brief_keeps_its_layout(self) -> None:
        briefs = [_Brief(layout_type="text_only")]

        assert assign_templates(briefs, "2026-10-22") == ["text_only"]

    def test_a_copy_brief_never_gets_an_art_only_or_type_only_template(self) -> None:
        briefs = [_Brief(concept=f"concept {index}") for index in range(20)]

        assigned = assign_templates(briefs, "2026-10-22")

        assert "full_bleed" not in assigned
        assert "text_only" not in assigned

    def test_the_model_does_not_pick_the_geometry(self) -> None:
        # A brief that says text_top still gets a template from the batch
        # rotation, not the one the model named.
        briefs = [_Brief(layout_type="text_top") for _ in range(8)]

        assigned = assign_templates(briefs, "2026-10-22")

        assert len(set(assigned)) == 8

    def test_a_single_custom_brief_still_gets_a_text_template(self) -> None:
        assigned = assign_templates([_Brief(concept="a custom idea")], "custom")

        assert assigned[0] in ART_TEMPLATE_IDS
