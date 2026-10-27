"""Tests for the LLM-authored text-integration plan.

The words of a design used to be drawn the same way every time: a flat
headline and tagline in the template's text zone. The brief now authors how
the words live in the design - painted into the art, printed on a label,
carved into a stone - and the compositor executes that plan.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image, ImageDraw

from store_listing.orchestration.design_briefs import (
    BriefableCandidate,
    DesignBrief,
    DesignBriefDBClient,
    TextPlanDBClient,
    generate_design_briefs,
    plan_batch_text_integration,
    replan_batch,
)
from store_listing.orchestration.image_generation import (
    build_flux_prompt,
    build_negative_prompt,
)
from store_listing.orchestration.image_pipeline import _brief_from_row
from store_listing.orchestration.text_compositor import LayoutSpec, composite_text
from store_listing.orchestration.text_integration import (
    CARRIERS,
    METHODS,
    REGIONS,
    SURFACES,
    PlannedText,
    TextPlan,
    build_text_plan_prompt,
    check_text_plan_fit,
    check_text_plan_mix,
    read_text_plan_response,
    text_integration_rules,
    text_plan_from_value,
)


def _canvas(
    size: tuple[int, int] = (1024, 1024), color: tuple[int, int, int] = (255, 255, 255)
) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _non_white(image: Image.Image) -> int:
    return sum(1 for pixel in image.getdata() if pixel != (255, 255, 255))


def _brief(
    concept: str = "A quiet graveyard",
    *,
    headline: str | None = "GONE BUT NOT FORGOTTEN",
    tagline: str | None = None,
    layout_type: str = "hero_bottom",
    scene: str | None = "a quiet graveyard in the mist",
) -> DesignBrief:
    return DesignBrief(
        concept=concept,
        product_type="dtf_apparel",
        specific_products=[],
        audience="everyone",
        visual_style="ink",
        confidence=80,
        reasoning="",
        source_seed_ids=[],
        layout_type=layout_type,
        headline_text=headline,
        tagline_text=tagline,
        scene_description=scene,
        voice="deadpan",
        style_preset="hand_inked_woodcut",
        idea_device="narrative",
    )


class _Raw:
    def __init__(self, content: str) -> None:
        choice = _Choice(content)
        self.headers = {}
        self.parse = lambda: _Parsed(choice)


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Parsed:
    def __init__(self, choice: _Choice) -> None:
        self.choices = [choice]


class _LLM:
    """A fake completions client that answers with a fixed list of bodies."""

    def __init__(self, *bodies: dict[str, Any] | str) -> None:
        self.calls: list[dict[str, Any]] = []
        self._bodies = [body if isinstance(body, str) else json.dumps(body) for body in bodies]
        self.with_raw_response = self

    def create(self, **kwargs: Any) -> _Raw:
        self.calls.append(kwargs)
        if not self._bodies:
            raise RuntimeError("no answer left")
        return _Raw(self._bodies.pop(0))


class TestTheVocabularies:
    def test_the_methods_cover_materials_and_plain_type(self) -> None:
        assert {"painted", "printed", "sticker", "embroidery", "carved", "neon"} <= set(METHODS)
        assert "typographic" in METHODS

    def test_a_carrier_is_optional(self) -> None:
        assert "none" in CARRIERS
        assert {"sign", "label", "banner", "ticket"} <= set(CARRIERS)

    def test_the_regions_cover_the_canvas(self) -> None:
        assert {"top_left", "center", "bottom_right", "any"} <= set(REGIONS)

    def test_the_surfaces_name_what_a_text_can_sit_on(self) -> None:
        assert {"paper", "kraft", "wood", "chalkboard", "metal", "stone", "fabric"} <= set(SURFACES)

    def test_the_rules_do_not_make_a_sign_the_default(self) -> None:
        rules = text_integration_rules().lower()

        assert "sign" in rules
        assert "do not default to a sign" in rules


class TestThePlan:
    def test_a_valid_plan_is_read(self) -> None:
        plan = TextPlan.from_dict(
            {
                "method": "carved",
                "carrier": "headstone",
                "carrier_description": "a blank granite headstone",
                "region": "bottom_center",
                "surface": "stone",
                "ink": "light",
                "reason": "the words belong to the stone",
                "alternatives": [{"method": "painted", "carrier": "none"}],
            }
        )

        assert plan is not None
        assert plan.method == "carved"
        assert plan.carrier == "headstone"
        assert plan.needs_carrier
        assert plan.alternatives[0]["method"] == "painted"

    def test_an_unknown_method_is_not_a_plan(self) -> None:
        assert TextPlan.from_dict({"method": "hologram"}) is None

    def test_an_unknown_carrier_falls_back_to_no_carrier(self) -> None:
        # A carrier we cannot draw is not a reason to lose the text.
        plan = TextPlan.from_dict({"method": "painted", "carrier": "spaceship"})

        assert plan is not None
        assert plan.carrier == "none"

    def test_a_lower_centre_region_is_understood(self) -> None:
        plan = TextPlan.from_dict(
            {"method": "painted", "carrier": "sign", "region": "lower_center"}
        )

        assert plan is not None
        assert plan.region == "bottom_center"

    def test_a_hex_ink_is_kept(self) -> None:
        plan = TextPlan.from_dict({"method": "painted", "ink": "#C94A2B"})

        assert plan is not None
        assert plan.ink == "#C94A2B"

    def test_a_carrier_plan_always_has_a_non_carrier_fallback(self) -> None:
        plan = TextPlan.from_dict({"method": "printed", "carrier": "label"})

        assert plan is not None
        assert any(alternative["carrier"] == "none" for alternative in plan.alternatives), (
            "a carrier we cannot find must not leave the design without text"
        )

    def test_the_plan_round_trips_through_json(self) -> None:
        plan = TextPlan.from_dict({"method": "neon", "carrier": "sign", "ink": "light"})

        assert plan is not None
        again = text_plan_from_value(plan.to_dict())

        assert again == plan

    def test_a_plan_stored_as_json_text_is_read(self) -> None:
        assert text_plan_from_value('{"method": "sticker"}') is not None

    def test_a_missing_plan_is_none(self) -> None:
        assert text_plan_from_value(None) is None


class TestThePlanResponse:
    def _answer(self, *plans: dict[str, Any]) -> str:
        return json.dumps({"plans": list(plans)})

    def test_the_plans_are_matched_to_their_brief(self) -> None:
        text = self._answer(
            {"brief": 1, "method": "painted"},
            {"brief": 2, "method": "sticker", "carrier": "none"},
        )

        plans, error = read_text_plan_response(text, 2)

        assert error is None
        assert plans[1].plan.method == "painted"
        assert plans[2].plan.method == "sticker"

    def test_an_invalid_plan_is_left_out_not_fatal(self) -> None:
        text = self._answer({"brief": 1, "method": "hologram"}, {"brief": 2, "method": "neon"})

        plans, error = read_text_plan_response(text, 2)

        assert error is None
        assert 1 not in plans
        assert plans[2].plan.method == "neon"

    def test_a_revised_scene_is_read(self) -> None:
        text = self._answer(
            {
                "brief": 1,
                "method": "painted",
                "carrier": "sign",
                "scene_description": "a blank wooden sign in the mist",
            }
        )

        plans, _ = read_text_plan_response(text, 1)

        assert plans[1].scene_description == "a blank wooden sign in the mist"

    def test_an_answer_without_plans_is_an_error(self) -> None:
        plans, error = read_text_plan_response('{"briefs": []}', 2)

        assert plans == {}
        assert error is not None


class TestTheMix:
    def test_a_varied_batch_has_no_complaints(self) -> None:
        plans = [
            TextPlan(method="painted"),
            TextPlan(method="sticker"),
            TextPlan(method="printed"),
            TextPlan(method="embroidery"),
            TextPlan(method="carved"),
            TextPlan(method="typographic"),
            TextPlan(method="painted"),
            TextPlan(method="neon"),
        ]

        assert check_text_plan_mix(plans) == []

    def test_a_batch_of_one_is_always_fine(self) -> None:
        assert check_text_plan_mix([TextPlan(method="painted")]) == []

    def test_one_method_used_three_times_is_reported(self) -> None:
        plans = [
            TextPlan(method="painted"),
            TextPlan(method="painted"),
            TextPlan(method="painted"),
            TextPlan(method="sticker"),
        ]

        assert any("painted" in issue for issue in check_text_plan_mix(plans))

    def test_too_many_carriers_are_reported(self) -> None:
        plans = [
            TextPlan(method="painted", carrier="sign"),
            TextPlan(method="printed", carrier="label"),
            TextPlan(method="embroidery"),
            TextPlan(method="sticker"),
        ]

        assert any("carrier" in issue for issue in check_text_plan_mix(plans))

    def test_a_batch_with_too_few_methods_is_reported(self) -> None:
        plans = [
            TextPlan(method="painted"),
            TextPlan(method="painted"),
            TextPlan(method="sticker"),
            TextPlan(method="sticker"),
        ]

        assert any("methods" in issue for issue in check_text_plan_mix(plans))


class TestThePlanFit:
    def _entries(self, *product_types: str) -> list[dict[str, Any]]:
        return [
            {"brief": index + 1, "product_type": product_type}
            for index, product_type in enumerate(product_types)
        ]

    def test_a_garment_carrier_on_dtf_is_flagged(self) -> None:
        plans = {
            1: PlannedText(plan=TextPlan(method="embroidery", carrier="apparel", surface="fabric"))
        }

        issues = check_text_plan_fit(self._entries("dtf_apparel"), plans)

        assert any("apparel" in issue for issue in issues)

    def test_carved_without_a_carrier_is_flagged(self) -> None:
        plans = {1: PlannedText(plan=TextPlan(method="carved", carrier="none"))}

        issues = check_text_plan_fit(self._entries("dtf_apparel"), plans)

        assert any("carved" in issue for issue in issues)

    def test_embroidery_without_a_carrier_is_flagged(self) -> None:
        plans = {1: PlannedText(plan=TextPlan(method="embroidery", carrier="none"))}

        issues = check_text_plan_fit(self._entries("dtf_apparel"), plans)

        assert any("embroidery" in issue for issue in issues)

    def test_typographic_with_a_carrier_is_flagged(self) -> None:
        plans = {1: PlannedText(plan=TextPlan(method="typographic", carrier="sign"))}

        issues = check_text_plan_fit(self._entries("dtf_apparel"), plans)

        assert any("typographic" in issue for issue in issues)

    def test_a_valid_plan_has_no_issues(self) -> None:
        plans = {
            1: PlannedText(plan=TextPlan(method="painted", carrier="none")),
            2: PlannedText(plan=TextPlan(method="carved", carrier="headstone", surface="stone")),
            3: PlannedText(
                plan=TextPlan(
                    method="embroidery",
                    carrier="banner",
                    carrier_description="a blank fabric banner hanging from the wall",
                    surface="fabric",
                )
            ),
            4: PlannedText(plan=TextPlan(method="typographic", carrier="none")),
        }

        assert (
            check_text_plan_fit(
                self._entries("dtf_apparel", "dtf_apparel", "dtf_apparel", "dtf_apparel"),
                plans,
            )
            == []
        )

    def test_a_carrier_without_a_surface_is_flagged(self) -> None:
        plans = {
            1: PlannedText(
                plan=TextPlan(
                    method="printed",
                    carrier="label",
                    carrier_description="a blank paper label wrapped around the cup",
                )
            )
        }

        issues = check_text_plan_fit(self._entries("dtf_apparel"), plans)

        assert any("surface" in issue for issue in issues)

    def test_a_floating_label_is_flagged(self) -> None:
        entries = [
            {
                "brief": 1,
                "product_type": "dtf_apparel",
                "scene": "a teacup at the top right with a blank label on a teacup",
            }
        ]
        plans = {
            1: PlannedText(
                plan=TextPlan(
                    method="printed",
                    carrier="label",
                    carrier_description="a blank label on a teacup",
                    surface="paper",
                )
            )
        }

        issues = check_text_plan_fit(entries, plans)

        assert any("floating" in issue for issue in issues)

    def test_a_wrapped_label_is_fine(self) -> None:
        plans = {
            1: PlannedText(
                plan=TextPlan(
                    method="printed",
                    carrier="label",
                    carrier_description=(
                        "a blank cream paper label wrapped around the front of the cup"
                    ),
                    surface="paper",
                )
            )
        }

        assert check_text_plan_fit(self._entries("dtf_apparel"), plans) == []


class TestThePrompt:
    def test_the_prompt_shows_the_template_geometry(self) -> None:
        messages = build_text_plan_prompt(
            [
                {
                    "brief": 1,
                    "concept": "a quiet graveyard",
                    "headline": "GONE BUT NOT FORGOTTEN",
                    "template": {
                        "id": "hero_bottom",
                        "composition": "art below, a clear band above",
                        "headline_zone": [0.06, 0.03, 0.88, 0.14],
                    },
                }
            ]
        )

        prompt = messages[0]["content"] + messages[1]["content"]

        assert "hero_bottom" in prompt
        assert "art below, a clear band above" in prompt
        assert "GONE BUT NOT FORGOTTEN" in prompt

    def test_the_prompt_names_the_mix_and_the_spelling_rule(self) -> None:
        messages = build_text_plan_prompt([])
        prompt = " ".join(message["content"] for message in messages).lower()

        assert "at least four" in prompt
        assert "never write the headline" in prompt or "do not write the words" in prompt
        assert "blank" in prompt

    def test_a_feedback_note_is_appended(self) -> None:
        messages = build_text_plan_prompt([], extra_instructions=["use more methods"])

        assert "use more methods" in messages[0]["content"]


class TestThePlanner:
    def test_a_plan_is_added_to_a_brief(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        llm = _LLM(
            {
                "plans": [
                    {
                        "brief": 1,
                        "method": "carved",
                        "carrier": "headstone",
                        "scene_description": "a blank headstone in the mist",
                    }
                ]
            }
        )

        planned = plan_batch_text_integration(llm, "test-model", [_brief()])

        assert planned[0].text_plan is not None
        assert planned[0].text_plan.method == "carved"
        assert planned[0].scene_description == "a blank headstone in the mist"

    def test_the_planner_can_be_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "false")
        llm = _LLM({"plans": [{"brief": 1, "method": "carved"}]})

        planned = plan_batch_text_integration(llm, "test-model", [_brief()])

        assert planned[0].text_plan is None
        assert llm.calls == []

    def test_a_failed_planner_keeps_the_brief(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        llm = _LLM("not json")

        planned = plan_batch_text_integration(llm, "test-model", [_brief()])

        assert planned[0].concept == "A quiet graveyard"
        assert planned[0].text_plan is None

    def test_a_poor_mix_is_asked_for_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        monkeypatch.setenv("LLM_TEXT_PLAN_REGENERATE", "true")
        briefs = [_brief(concept=f"concept {index}") for index in range(4)]
        first = {
            "plans": [
                {"brief": 1, "method": "painted"},
                {"brief": 2, "method": "painted"},
                {"brief": 3, "method": "painted"},
                {"brief": 4, "method": "painted"},
            ]
        }
        second = {
            "plans": [
                {"brief": 1, "method": "painted"},
                {"brief": 2, "method": "sticker"},
                {
                    "brief": 3,
                    "method": "embroidery",
                    "carrier": "banner",
                    "carrier_description": "a blank fabric banner hanging from the wall",
                    "surface": "fabric",
                },
                {"brief": 4, "method": "typographic"},
            ]
        }
        llm = _LLM(first, second)

        planned = plan_batch_text_integration(llm, "test-model", briefs)

        assert len(llm.calls) == 2, "the model gets one more chance with feedback"
        assert {brief.text_plan.method for brief in planned if brief.text_plan} == {
            "painted",
            "sticker",
            "embroidery",
            "typographic",
        }

    def test_a_text_only_brief_can_choose_a_material(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        llm = _LLM({"plans": [{"brief": 1, "method": "neon", "carrier": "none"}]})

        planned = plan_batch_text_integration(
            llm, "test-model", [_brief(layout_type="text_only", scene=None)]
        )

        assert planned[0].text_plan is not None
        assert planned[0].text_plan.method == "neon"

    def test_an_unusable_plan_is_asked_for_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        monkeypatch.setenv("LLM_TEXT_PLAN_REGENERATE", "true")
        briefs = [
            _brief(concept="a garment"),
            _brief(concept="a bean"),
            _brief(concept="a spider"),
        ]
        first = {
            "plans": [
                {"brief": 1, "method": "embroidery", "carrier": "apparel"},
                {"brief": 2, "method": "carved", "carrier": "none"},
                {"brief": 3, "method": "typographic", "carrier": "none"},
            ]
        }
        second = {
            "plans": [
                {"brief": 1, "method": "painted", "carrier": "none"},
                {"brief": 2, "method": "sticker", "carrier": "none"},
                {"brief": 3, "method": "neon", "carrier": "none"},
            ]
        }
        llm = _LLM(first, second)

        planned = plan_batch_text_integration(llm, "test-model", briefs)

        assert len(llm.calls) == 2, "the model gets one more chance with the fit issues"
        assert planned[0].text_plan is not None
        assert planned[0].text_plan.method == "painted"
        assert planned[1].text_plan is not None
        assert planned[1].text_plan.method == "sticker"

    def test_an_invalid_plan_is_sanitised_when_regeneration_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        monkeypatch.setenv("LLM_TEXT_PLAN_REGENERATE", "false")
        llm = _LLM(
            {
                "plans": [
                    {"brief": 1, "method": "embroidery", "carrier": "apparel"},
                ]
            }
        )

        planned = plan_batch_text_integration(llm, "test-model", [_brief()])

        assert planned[0].text_plan is not None
        assert planned[0].text_plan.carrier == "none"
        assert planned[0].text_plan.method == "painted"

    def test_a_label_without_a_surface_gets_a_sensible_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        monkeypatch.setenv("LLM_TEXT_PLAN_REGENERATE", "false")
        llm = _LLM(
            {
                "plans": [
                    {
                        "brief": 1,
                        "method": "printed",
                        "carrier": "label",
                        "carrier_description": (
                            "a blank cream paper label wrapped around the front of the cup"
                        ),
                    }
                ]
            }
        )

        planned = plan_batch_text_integration(llm, "test-model", [_brief()])

        assert planned[0].text_plan is not None
        assert planned[0].text_plan.carrier == "label"
        assert planned[0].text_plan.surface == "paper"

    def test_a_floating_label_falls_back_to_painted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        monkeypatch.setenv("LLM_TEXT_PLAN_REGENERATE", "false")
        llm = _LLM(
            {
                "plans": [
                    {
                        "brief": 1,
                        "method": "printed",
                        "carrier": "label",
                        "carrier_description": "a blank label on a teacup",
                        "surface": "paper",
                    }
                ]
            }
        )

        planned = plan_batch_text_integration(llm, "test-model", [_brief()])

        assert planned[0].text_plan is not None
        assert planned[0].text_plan.carrier == "none"
        assert planned[0].text_plan.method == "painted"


class TestTheMaterials:
    @pytest.mark.parametrize(
        "method", ["painted", "printed", "sticker", "embroidery", "carved", "chalk", "neon"]
    )
    def test_every_material_draws_the_headline(self, method: str) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="hero_bottom",
                headline_text="GONE BUT NOT FORGOTTEN",
                font_color="#1B1B1B",
                text_plan=TextPlan(method=method),
            ),
        )

        assert _non_white(_image(result)) > 0, method

    def test_each_material_looks_different(self) -> None:
        outputs = {
            method: composite_text(
                _canvas(),
                LayoutSpec(
                    layout_type="hero_bottom",
                    headline_text="GONE BUT NOT FORGOTTEN",
                    font_color="#1B1B1B",
                    text_plan=TextPlan(method=method),
                ),
            )
            for method in ("typographic", "painted", "sticker", "embroidery", "neon")
        }

        assert len(set(outputs.values())) == len(outputs)


class TestTheCarrier:
    def _sign(self, color: tuple[int, int, int] = (139, 94, 60)) -> bytes:
        image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        draw = Image.new("RGBA", image.size, (0, 0, 0, 0))
        from PIL import ImageDraw

        ImageDraw.Draw(draw).rectangle([280, 700, 744, 860], fill=(*color, 255))
        return _bytes(Image.alpha_composite(image.convert("RGBA"), draw).convert("RGB"))

    def test_the_text_is_drawn_on_the_carrier(self) -> None:
        result = composite_text(
            self._sign(),
            LayoutSpec(
                layout_type="hero_bottom",
                headline_text="SLOW DOWN",
                font_color="#F5F0E6",
                text_plan=TextPlan(
                    method="painted",
                    carrier="sign",
                    region="bottom_center",
                    surface="wood",
                    ink="light",
                ),
            ),
        )

        image = _image(result)
        inside = image.crop((280, 700, 744, 860))
        outside = image.crop((0, 0, 1024, 400))

        assert len({pixel for pixel in inside.getdata()}) > 1, "the sign carries the text"
        assert len({pixel for pixel in outside.getdata()}) == 1, "the empty art is untouched"

    def test_a_carrier_anywhere_is_found(self) -> None:
        result = composite_text(
            self._sign(),
            LayoutSpec(
                layout_type="hero_bottom",
                headline_text="SLOW DOWN",
                font_color="#F5F0E6",
                text_plan=TextPlan(
                    method="painted",
                    carrier="sign",
                    region="any",
                    surface="wood",
                    ink="light",
                ),
            ),
        )

        inside = _image(result).crop((280, 700, 744, 860))

        assert len({pixel for pixel in inside.getdata()}) > 1, "the sign carries the text"

    def test_the_carrier_is_the_object_not_the_wall_behind_it(self) -> None:
        image = Image.new("RGB", (1024, 1024), (255, 0, 255))
        draw = ImageDraw.Draw(image)
        # A stone wall across the scene and a headstone standing in front of
        # it. They share the stone palette; the headstone is the carrier.
        draw.rectangle([0, 620, 1024, 690], fill=(140, 140, 135))
        draw.rectangle([370, 690, 590, 900], fill=(140, 140, 135))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = composite_text(
            buffer.getvalue(),
            LayoutSpec(
                layout_type="hero_top",
                headline_text="GONE BUT NOT FORGOTTEN",
                font_color="#1B1B1B",
                text_plan=TextPlan(
                    method="carved",
                    carrier="headstone",
                    region="bottom_center",
                    surface="stone",
                    ink="dark",
                ),
            ),
        )

        composed = np.asarray(_image(result))
        raw = np.asarray(image)
        changed = np.abs(composed.astype(np.int16) - raw.astype(np.int16)).sum(axis=2) > 30
        ys, xs = np.nonzero(changed)

        assert xs.min() >= 350 and xs.max() <= 610, "the words stay on the headstone"
        assert ys.min() >= 680, "the words are on the headstone, not the wall"

    def test_a_line_art_carrier_is_found_by_its_outline(self) -> None:
        image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        # The linocut presets draw a white carrier inside a black outline;
        # the grey stone palette cannot find it, so the enclosed white shape
        # has to.
        draw.rectangle([0, 680, 1024, 690], fill=(0, 0, 0))
        draw.rectangle([370, 690, 590, 900], outline=(0, 0, 0), width=10, fill=(255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = composite_text(
            buffer.getvalue(),
            LayoutSpec(
                layout_type="hero_top",
                headline_text="GONE BUT NOT FORGOTTEN",
                font_color="#1B1B1B",
                text_plan=TextPlan(
                    method="carved",
                    carrier="headstone",
                    region="bottom_center",
                    surface="stone",
                    ink="dark",
                ),
            ),
        )

        composed = np.asarray(_image(result))
        raw = np.asarray(image)
        changed = np.abs(composed.astype(np.int16) - raw.astype(np.int16)).sum(axis=2) > 30
        ys, xs = np.nonzero(changed)

        assert xs.min() >= 360 and xs.max() <= 600, "the words stay inside the outline"
        assert ys.min() >= 680, "the words are on the headstone"

    def test_a_carrier_outside_the_planned_region_is_still_found(self) -> None:
        image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        # The image model put the headstone on the right, while the plan said
        # bottom_center. The whole-canvas second chance finds it.
        draw.rectangle([700, 500, 900, 800], outline=(0, 0, 0), width=10, fill=(255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = composite_text(
            buffer.getvalue(),
            LayoutSpec(
                layout_type="hero_top",
                headline_text="GONE BUT NOT FORGOTTEN",
                font_color="#1B1B1B",
                text_plan=TextPlan(
                    method="carved",
                    carrier="headstone",
                    region="bottom_center",
                    surface="stone",
                    ink="dark",
                ),
            ),
        )

        composed = np.asarray(_image(result))
        raw = np.asarray(image)
        changed = np.abs(composed.astype(np.int16) - raw.astype(np.int16)).sum(axis=2) > 30
        ys, xs = np.nonzero(changed)

        assert xs.min() >= 680, "the words follow the carrier out of the planned region"
        assert ys.min() >= 490

    def test_a_carrier_that_is_not_there_falls_back_to_the_zone(self) -> None:
        result = composite_text(
            _canvas(),
            LayoutSpec(
                layout_type="hero_bottom",
                headline_text="SLOW DOWN",
                font_color="#1B1B1B",
                text_plan=TextPlan(
                    method="painted",
                    carrier="sign",
                    region="bottom_center",
                    surface="wood",
                    ink="dark",
                ),
            ),
        )

        assert _non_white(_image(result)) > 0, "a missing carrier must not lose the text"

    def test_a_dark_ink_on_a_dark_surface_is_made_light(self) -> None:
        result = composite_text(
            self._sign((35, 40, 38)),
            LayoutSpec(
                layout_type="hero_bottom",
                headline_text="SLOW DOWN",
                font_color="#101010",
                text_plan=TextPlan(
                    method="painted",
                    carrier="sign",
                    region="bottom_center",
                    surface="chalkboard",
                    ink="dark",
                ),
            ),
        )

        image = _image(result)
        inside = image.crop((280, 700, 744, 860))
        brightest = int(np.asarray(inside.convert("RGB")).sum(axis=2).max())

        assert brightest > 500, "the text has to stand out against the carrier"


class TestTheImagePrompt:
    def test_a_carrier_makes_the_look_allow_a_surface(self) -> None:
        brief = _brief()
        brief = _with_plan(brief, TextPlan(method="painted", carrier="sign", surface="wood"))

        negative = build_negative_prompt(brief)

        assert "no surface" not in negative
        assert "no lettering" in negative, "the carrier has to stay blank"
        assert "no lettering on signs" in negative, "other signs have to stay blank too"

    def test_without_a_carrier_the_look_still_forbids_a_surface(self) -> None:
        assert "no surface" in build_negative_prompt(_brief())

    def test_the_prompt_names_the_surface_the_compositor_will_look_for(self) -> None:
        brief = _with_plan(
            _brief(),
            TextPlan(method="carved", carrier="headstone", surface="stone"),
        )

        prompt = build_flux_prompt(brief).lower()

        assert "pale grey stone surface" in prompt

    def test_the_prompt_asks_for_the_blank_carrier_in_the_clear_band(self) -> None:
        brief = _with_plan(
            _brief(),
            TextPlan(
                method="painted",
                carrier="sign",
                surface="wood",
                carrier_description="a blank weathered wooden sign, no writing",
            ),
        )

        prompt = build_flux_prompt(brief).lower()

        assert "blank carrier" in prompt
        assert "blank weathered wooden sign" in prompt
        assert "no lettering" in prompt or "without any lettering" in prompt
        assert "plain background only" not in prompt


class TestTheStoredPlan:
    def test_a_brief_row_carries_its_plan_to_the_renderer(self) -> None:
        brief = _brief_from_row(
            {
                "concept": "A quiet graveyard",
                "product_type": "dtf_apparel",
                "scene_description": "a quiet graveyard in the mist",
                "text_plan": {
                    "method": "carved",
                    "carrier": "headstone",
                    "surface": "stone",
                    "ink": "light",
                },
            }
        )

        assert brief.text_plan is not None
        assert brief.text_plan.method == "carved"

    def test_the_batch_stores_the_plan_with_the_brief(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        monkeypatch.setattr(
            "store_listing.orchestration.design_briefs.BRIEF_MIX_REGENERATE_ONCE", False
        )
        monkeypatch.setattr(
            "store_listing.orchestration.design_briefs.BRIEF_TEXT_REGENERATE_ONCE", False
        )
        llm = _LLM(
            {
                "briefs": [
                    {
                        "concept": "A quiet graveyard",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                        "layout_type": "auto",
                        "headline_text": "GONE BUT NOT FORGOTTEN",
                        "scene_description": "a quiet graveyard in the mist",
                        "voice": "deadpan",
                        "style_preset": "hand_inked_woodcut",
                        "idea_device": "narrative",
                    }
                ]
            },
            {
                "plans": [
                    {
                        "brief": 1,
                        "method": "carved",
                        "carrier": "headstone",
                        "carrier_description": "a blank headstone",
                        "region": "bottom_center",
                        "surface": "stone",
                        "ink": "light",
                        "reason": "the words belong to the stone",
                        "scene_description": (
                            "a quiet graveyard with a blank headstone in the mist"
                        ),
                    }
                ]
            },
        )
        db = _DB()

        generate_design_briefs(
            cast(DesignBriefDBClient, db),
            llm,
            "test-model",
            today=date(2026, 10, 1),
            enabled=["dtf_apparel"],
        )

        stored = db.insert_design_brief.call_args.kwargs
        assert stored["text_plan"]["method"] == "carved"
        assert stored["scene_description"] == (
            "a quiet graveyard with a blank headstone in the mist"
        )


class TestTheReplan:
    def test_replanning_stores_the_new_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEXT_PLAN", "true")
        llm = _LLM(
            {
                "plans": [
                    {
                        "brief": 1,
                        "method": "painted",
                        "carrier": "none",
                        "scene_description": "a painted graveyard in the mist",
                    }
                ]
            }
        )
        db = _ReplanDB()

        result = replan_batch(llm, "test-model", cast(TextPlanDBClient, db), "batch-1")

        assert result["status"] == "success"
        assert result["updated"] == 1
        assert db.updated[0][0] == 1
        assert db.updated[0][1] is not None
        assert db.updated[0][1]["method"] == "painted"
        assert db.updated[0][2] == "a painted graveyard in the mist"

    def test_an_empty_batch_is_skipped(self) -> None:
        db = _ReplanDB(rows=[])

        result = replan_batch(_LLM(), "test-model", cast(TextPlanDBClient, db), "missing")

        assert result["status"] == "skipped"
        assert db.updated == []


class _ReplanDB:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = (
            rows
            if rows is not None
            else [
                {
                    "id": 1,
                    "concept": "a quiet graveyard",
                    "product_type": "dtf_apparel",
                    "layout_type": "hero_top",
                    "headline_text": "GONE BUT NOT FORGOTTEN",
                    "tagline_text": None,
                    "scene_description": "a quiet graveyard in the mist",
                    "voice": "deadpan",
                    "style_preset": "hand_inked_woodcut",
                    "text_plan": None,
                },
                {
                    "id": 2,
                    "concept": "a pattern",
                    "product_type": "dtf_apparel",
                    "layout_type": "full_bleed",
                    "headline_text": None,
                    "tagline_text": None,
                    "scene_description": "a repeating pattern",
                    "voice": "deadpan",
                    "style_preset": "hand_inked_woodcut",
                    "text_plan": None,
                },
            ]
        )
        self.updated: list[tuple[int, dict[str, Any] | None, str | None]] = []

    def list_briefs_by_batch(self, batch_id: str) -> list[dict[str, Any]]:
        return self.rows

    def update_brief_text_plan(
        self,
        brief_id: int,
        text_plan: dict[str, Any] | None,
        scene_description: str | None,
    ) -> None:
        self.updated.append((brief_id, text_plan, scene_description))


class _DB:
    """The database calls brief generation makes, without a database."""

    def __init__(self) -> None:
        self.insert_design_brief = _Recorder(1)
        self.insert_brief_source = _Recorder(1)

    def list_briefable_seeds(self, limit: int | None = None) -> list[BriefableCandidate]:
        return [
            BriefableCandidate(
                active_seed_id=1,
                query="funny graveyard shirt",
                source="google_trends",
                query_type="rising",
                score=80,
                delta=20,
                promotion_score=80,
                product_tags=["dtf_apparel"],
            )
        ]

    def list_active_materials(self, product_type: str | None = None) -> list[dict[str, Any]]:
        return []


class _Recorder:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.call_args: Any = None

    def __call__(self, **kwargs: Any) -> Any:
        self.call_args = _Call(kwargs)
        return self.value


@dataclass
class _Call:
    kwargs: dict[str, Any]


def _with_plan(brief: DesignBrief, plan: TextPlan) -> DesignBrief:
    from dataclasses import replace

    return replace(brief, text_plan=plan)


def _bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
