"""Tests for the marketplace listing-copy pass.

The scene description is an image prompt. This pass writes the copy a buyer
reads on the product page, and it must not turn into a description of the
picture.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.listing_copy import (
    ListingCopy,
    build_listing_copy_entries,
    build_listing_copy_prompt,
    fallback_listing_description,
    listing_copy_enabled,
    parse_listing_copy_response,
    write_batch_listing_copy,
)


def _material() -> dict[str, Any]:
    return {
        "external_id": "bc3001-white",
        "name": "Bella Canvas 3001",
        "color": "White",
        "substrate_type": "t-shirt",
        "unit_cost": 4.50,
        "sizes": ["S", "M", "L"],
    }


def _brief(
    *,
    concept: str = "Powered by Coffee",
    headline: str | None = "POWERED BY COFFEE",
    tagline: str | None = None,
    audience: str = "coffee lovers",
    specific_products: list[str] | None = None,
    materials: list[dict[str, Any]] | None = None,
) -> DesignBrief:
    return DesignBrief(
        concept=concept,
        product_type="dtf_apparel",
        specific_products=specific_products or ["t-shirt"],
        audience=audience,
        visual_style="warm cream and brown palette",
        confidence=80,
        reasoning="evergreen",
        source_seed_ids=[],
        headline_text=headline,
        tagline_text=tagline,
        scene_description="a steaming mug beside a notebook on a kitchen table",
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
        monkeypatch.delenv("LLM_LISTING_COPY", raising=False)
        assert listing_copy_enabled() is False

    def test_the_pass_can_be_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_LISTING_COPY", "true")
        assert listing_copy_enabled() is True


class TestEntries:
    def test_an_entry_carries_the_product_and_the_material(self) -> None:
        entries = build_listing_copy_entries([_brief()])

        assert entries[0]["index"] == 1
        assert entries[0]["headline"] == "POWERED BY COFFEE"
        assert entries[0]["primary_material"]["color"] == "White"
        assert entries[0]["specific_products"] == ["t-shirt"]

    def test_the_entry_does_not_carry_the_scene_prompt(self) -> None:
        entry = build_listing_copy_entries([_brief()])[0]

        assert "scene" not in entry
        assert "scene_description" not in entry


class TestPrompt:
    def test_the_prompt_says_amazon_and_forbids_scene_copy(self) -> None:
        prompt = build_listing_copy_prompt(build_listing_copy_entries([_brief()]))
        system = prompt[0]["content"].lower()

        assert "amazon" in system
        assert "product description" in system
        assert "scene" in system
        assert "illustration" in system
        assert "mockup" in system
        assert "price" in system
        assert "do not invent" in system
        assert "html" in system

    def test_the_prompt_has_a_plain_text_output_shape(self) -> None:
        prompt = build_listing_copy_prompt(build_listing_copy_entries([_brief()]))

        assert "listings" in prompt[0]["content"]
        assert "description" in prompt[0]["content"]


class TestParse:
    def test_a_valid_description_is_read(self) -> None:
        text = json.dumps(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "A white t-shirt for coffee lovers, "
                        "featuring a warm coffee-inspired design. Wear it on slow "
                        "mornings or give it to the friend who runs on espresso.",
                    }
                ]
            }
        )

        copies, error = parse_listing_copy_response(text, expected_indexes={1})

        assert error is None
        assert copies[1].description.startswith("A white t-shirt")

    def test_html_is_removed_and_whitespace_collapsed(self) -> None:
        text = json.dumps(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "<p>A white t-shirt for coffee lovers.</p>"
                        "  It is made for slow mornings and espresso runs.",
                    }
                ]
            }
        )

        copies, error = parse_listing_copy_response(text, expected_indexes={1})

        assert error is None
        assert "<p>" not in copies[1].description
        assert "  " not in copies[1].description

    def test_scene_language_is_rejected(self) -> None:
        text = json.dumps(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "This illustration shows a steaming mug beside a notebook.",
                    }
                ]
            }
        )

        copies, error = parse_listing_copy_response(text, expected_indexes={1})

        assert copies == {}
        assert error is not None

    def test_a_missing_listing_is_reported(self) -> None:
        text = json.dumps(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "A white t-shirt for coffee lovers, featuring a "
                        "warm coffee-inspired design for slow mornings.",
                    }
                ]
            }
        )

        copies, error = parse_listing_copy_response(text, expected_indexes={1, 2})

        assert list(copies) == [1]
        assert "missing" in (error or "")

    def test_too_short_is_rejected(self) -> None:
        text = json.dumps({"listings": [{"index": 1, "description": "A shirt."}]})

        copies, error = parse_listing_copy_response(text, expected_indexes={1})

        assert copies == {}
        assert error is not None


class TestFallback:
    def test_the_fallback_is_product_copy_not_a_scene(self) -> None:
        description = fallback_listing_description(_brief())

        assert "POWERED BY COFFEE" in description or "Powered by Coffee" in description
        assert "white" in description.lower()
        assert "t-shirt" in description.lower()
        assert "coffee lovers" in description.lower()
        assert "steaming mug" not in description.lower()

    def test_the_fallback_handles_a_missing_audience(self) -> None:
        description = fallback_listing_description(_brief(audience=""))

        assert description
        assert "designed for" not in description.lower()


class TestWriteBatch:
    def test_disabled_uses_the_fallback(self) -> None:
        briefs, issues = write_batch_listing_copy(
            MagicMock(), "test-model", [_brief()], enabled=False
        )

        assert issues == []
        assert briefs[0].listing_description
        assert "steaming mug" not in (briefs[0].listing_description or "").lower()

    def test_the_model_description_is_used(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.return_value = _raw(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "A white t-shirt for coffee lovers, featuring "
                        "a warm coffee-inspired design. Wear it on slow mornings or "
                        "give it to the friend who runs on espresso.",
                    }
                ]
            }
        )

        briefs, issues = write_batch_listing_copy(llm, "test-model", [_brief()], enabled=True)

        assert issues == []
        description = briefs[0].listing_description
        assert description is not None
        assert description.startswith("A white t-shirt")

    def test_a_failed_call_falls_back(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")

        briefs, issues = write_batch_listing_copy(llm, "test-model", [_brief()], enabled=True)

        assert issues
        assert briefs[0].listing_description
        assert "Printed on" in (briefs[0].listing_description or "")


def test_listing_copy_dataclass_holds_the_index_and_text() -> None:
    copy = ListingCopy(index=1, description="A white t-shirt for coffee lovers.")

    assert copy.index == 1
    assert copy.description.startswith("A white")
