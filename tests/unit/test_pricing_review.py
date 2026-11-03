"""Tests for the optional LLM pricing pass."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.pricing import PricingPolicy
from store_listing.orchestration.pricing_review import (
    PositioningChoice,
    build_pricing_entries,
    build_pricing_prompt,
    choose_batch_positionings,
    parse_pricing_response,
    pricing_review_enabled,
)


def _material(
    *,
    substrate: str = "t-shirt",
    unit_cost: float = 4.50,
    sizes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "external_id": "bc3001-white",
        "name": "Bella Canvas 3001",
        "color": "White",
        "substrate_type": substrate,
        "unit_cost": unit_cost,
        "sizes": sizes or ["S", "M", "L", "XL", "2XL", "3XL"],
    }


def _brief(
    *,
    confidence: int = 80,
    product_type: str = "dtf_apparel",
    materials: list[dict[str, Any]] | None = None,
    layout_type: str = "emblem",
) -> DesignBrief:
    return DesignBrief(
        concept="Powered by Coffee",
        product_type=product_type,
        specific_products=["t-shirt"],
        audience="coffee lovers",
        visual_style="flat graphic",
        confidence=confidence,
        reasoning="evergreen",
        source_seed_ids=[],
        layout_type=layout_type,
        headline_text="COFFEE",
        recommended_materials=materials if materials is not None else [_material()],
    )


def _raw(content: dict[str, Any]) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps(content)
    parsed = MagicMock(choices=[choice])
    raw = MagicMock()
    raw.headers = {}
    raw.parse.return_value = parsed
    return raw


class TestEnabled:
    def test_the_pass_is_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_PRICING_REVIEW", raising=False)
        assert pricing_review_enabled() is False

    def test_the_pass_can_be_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PRICING_REVIEW", "true")
        assert pricing_review_enabled() is True


class TestEntries:
    def test_an_entry_carries_the_cost_and_the_print_size(self) -> None:
        entries = build_pricing_entries([_brief()], policy=PricingPolicy())

        assert len(entries) == 1
        entry = entries[0]
        assert entry["index"] == 1
        assert entry["confidence"] == 80
        assert entry["primary_material"]["unit_cost"] == 4.5
        assert entry["print_size"] == {
            "width_in": 12.0,
            "height_in": 12.0,
            "placement": "full_front",
        }
        assert entry["print_cost"] == pytest.approx(0.72)
        assert entry["estimated_retail"]["standard"] == 19.99
        assert entry["price_ceiling"] == 29.99
        assert entry["allowed_positionings"] == ["budget", "standard", "premium"]

    def test_a_non_dtf_brief_is_not_entered(self) -> None:
        entries = build_pricing_entries(
            [_brief(product_type="sublimation", materials=[_material(substrate="mug")])],
            policy=PricingPolicy(),
        )

        assert entries == []

    def test_a_brief_without_materials_is_not_entered(self) -> None:
        assert build_pricing_entries([_brief(materials=[])], policy=PricingPolicy()) == []

    def test_a_material_without_a_cost_is_not_entered(self) -> None:
        material = _material()
        material["unit_cost"] = None

        assert build_pricing_entries([_brief(materials=[material])], policy=PricingPolicy()) == []


class TestPrompt:
    def test_the_prompt_contains_the_cost_context_and_the_rules(self) -> None:
        entries = build_pricing_entries([_brief()], policy=PricingPolicy())

        messages = build_pricing_prompt(entries)
        system = messages[0]["content"]
        user = json.loads(messages[1]["content"])

        assert "budget" in system
        assert "standard" in system
        assert "premium" in system
        assert "confidence" in system.lower()
        assert user["briefs"][0]["price_ceiling"] == 29.99
        assert user["briefs"][0]["primary_material"]["unit_cost"] == 4.5


class TestParse:
    def test_a_valid_answer_is_read(self) -> None:
        text = json.dumps(
            {"positionings": [{"index": 1, "positioning": "premium", "reason": "Strong concept."}]}
        )

        choices, error = parse_pricing_response(text, expected_indexes={1})

        assert error is None
        assert choices[1] == PositioningChoice(1, "premium", "Strong concept.")

    def test_an_unknown_positioning_is_ignored(self) -> None:
        text = json.dumps({"positionings": [{"index": 1, "positioning": "luxury", "reason": "x"}]})

        choices, error = parse_pricing_response(text, expected_indexes={1})

        assert choices == {}
        assert error is not None

    def test_a_missing_choice_is_reported(self) -> None:
        text = json.dumps(
            {"positionings": [{"index": 1, "positioning": "standard", "reason": "x"}]}
        )

        choices, error = parse_pricing_response(text, expected_indexes={1, 2})

        assert list(choices) == [1]
        assert "missing" in (error or "")


class TestChoose:
    def test_disabled_uses_the_confidence_fallback(self) -> None:
        choices, issues = choose_batch_positionings(
            MagicMock(),
            "test-model",
            [_brief(confidence=30), _brief(confidence=80)],
            policy=PricingPolicy(),
            enabled=False,
        )

        assert issues == []
        assert choices[1].positioning == "budget"
        assert choices[2].positioning == "standard"

    def test_the_model_choice_is_used_when_enabled(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.return_value = _raw(
            {"positionings": [{"index": 1, "positioning": "premium", "reason": "Strong."}]}
        )

        choices, issues = choose_batch_positionings(
            llm,
            "test-model",
            [_brief(confidence=80)],
            policy=PricingPolicy(),
            enabled=True,
        )

        assert issues == []
        assert choices[1].positioning == "premium"
        assert choices[1].reason == "Strong."

    def test_a_failed_model_call_falls_back(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")

        choices, issues = choose_batch_positionings(
            llm,
            "test-model",
            [_brief(confidence=80)],
            policy=PricingPolicy(),
            enabled=True,
        )

        assert choices[1].positioning == "standard"
        assert issues

    def test_a_non_dtf_batch_does_not_call_the_model(self) -> None:
        llm = MagicMock()

        choices, issues = choose_batch_positionings(
            llm,
            "test-model",
            [_brief(product_type="sublimation", materials=[_material(substrate="mug")])],
            policy=PricingPolicy(),
            enabled=True,
        )

        assert choices == {}
        assert issues == []
        assert not llm.with_raw_response.create.called
