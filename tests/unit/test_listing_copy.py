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
    build_listing_rewrite_prompt,
    fallback_listing_description,
    listing_copy_enabled,
    parse_listing_copy_response,
    regenerate_stored_listing,
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
        assert "color" not in (entries[0]["primary_material"] or {})
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
        assert "garment colour" in system
        assert "photos already show" in system
        assert "use the primary material's colour" not in system
        assert "a white t-shirt" not in system
        assert "a dark heather t-shirt" in system

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
        assert "white" not in description.lower()
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
                        "description": "For coffee lovers, a warm coffee-inspired design. "
                        "Wear it on slow mornings or give it to the friend who runs "
                        "on espresso.",
                    }
                ]
            }
        )

        briefs, issues = write_batch_listing_copy(llm, "test-model", [_brief()], enabled=True)

        assert issues == []
        description = briefs[0].listing_description
        assert description is not None
        assert description.startswith("For coffee lovers")

    def test_a_shirt_colour_opener_is_removed(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.return_value = _raw(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "A Dark Heather t-shirt for coffee lovers, featuring "
                        "a warm coffee-inspired design. Wear it on slow mornings or "
                        "give it to the friend who runs on espresso.",
                    }
                ]
            }
        )
        brief = _brief(
            materials=[
                {
                    "external_id": "bc3001-dark-heather",
                    "name": "Bella Canvas 3001",
                    "color": "Dark Heather",
                    "substrate_type": "t-shirt",
                    "unit_cost": 4.50,
                    "sizes": ["S", "M", "L"],
                }
            ]
        )

        briefs, issues = write_batch_listing_copy(llm, "test-model", [brief], enabled=True)

        assert issues == []
        description = briefs[0].listing_description or ""
        assert description.startswith("For coffee lovers")
        assert "dark heather" not in description.lower()
        assert not description.lower().startswith("a ")

    def test_a_jet_black_shirt_opener_is_removed(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.return_value = _raw(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "A jet black t-shirt featuring a small cross and a "
                        "short verse. Wear it on Sunday or give it to a friend.",
                    }
                ]
            }
        )
        brief = _brief(
            materials=[
                {
                    "external_id": "shirt-jet-black-m",
                    "name": "Bella Canvas 3001",
                    "color": "Jet Black",
                    "substrate_type": "t-shirt",
                    "unit_cost": 4.50,
                    "sizes": ["S", "M", "L"],
                }
            ]
        )

        briefs, _issues = write_batch_listing_copy(llm, "test-model", [brief], enabled=True)

        description = briefs[0].listing_description or ""
        assert description.startswith("Featuring a small cross")
        assert "jet black" not in description.lower()

    def test_an_excluded_description_falls_back(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.return_value = _raw(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "A white t-shirt for tarot readers, featuring "
                        "a card spread. Wear it to a reading or give it to the "
                        "friend who never misses a sitting.",
                    }
                ]
            }
        )

        briefs, issues = write_batch_listing_copy(llm, "test-model", [_brief()], enabled=True)

        assert issues == []
        description = briefs[0].listing_description or ""
        assert "tarot" not in description.lower()
        assert "Powered by Coffee" in description or "POWERED BY COFFEE" in description

    def test_a_halloween_description_is_kept(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.return_value = _raw(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": "A white t-shirt for Halloween, featuring a "
                        "smiling witch on a broomstick. Wear it to a costume party "
                        "or give it to the friend who loves the season.",
                    }
                ]
            }
        )

        briefs, _issues = write_batch_listing_copy(llm, "test-model", [_brief()], enabled=True)

        description = briefs[0].listing_description or ""
        assert "witch" in description.lower()

    def test_a_failed_call_falls_back(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")

        briefs, issues = write_batch_listing_copy(llm, "test-model", [_brief()], enabled=True)

        assert issues
        assert briefs[0].listing_description
        assert "Printed on" in (briefs[0].listing_description or "")


class TestRewrite:
    def test_the_prompt_carries_the_guidance_and_the_current_copy(self) -> None:
        prompt = build_listing_rewrite_prompt(
            _brief(),
            guidance="Mention Sunday mornings.",
            current="A Dark Heather t-shirt for coffee lovers, featuring a warm design.",
        )
        user = prompt[1]["content"].lower()
        system = prompt[0]["content"].lower()

        assert "mention sunday mornings." in user
        assert "dark heather" in user
        assert "garment colour" in system
        assert "guidance" in system

    def test_guidance_rewrites_the_description_and_drops_the_shirt_opener(self) -> None:
        db = MagicMock()
        db.get_listing_brief.return_value = {
            "id": 7,
            "concept": "Powered by Coffee",
            "product_type": "dtf_apparel",
            "specific_products": ["t-shirt"],
            "audience": "coffee lovers",
            "visual_style": "warm cream and brown palette",
            "headline_text": "POWERED BY COFFEE",
            "tagline_text": None,
            "kicker_text": None,
            "recommended_materials": [
                {
                    "external_id": "bc3001-dark-heather",
                    "name": "Bella Canvas 3001",
                    "color": "Dark Heather",
                    "substrate_type": "t-shirt",
                }
            ],
            "listing_description": (
                "A Dark Heather t-shirt for coffee lovers, featuring a warm design. "
                "Wear it on slow mornings."
            ),
        }
        llm = MagicMock()
        llm.with_raw_response.create.return_value = _raw(
            {
                "listings": [
                    {
                        "index": 1,
                        "description": (
                            "A Dark Heather t-shirt for coffee lovers on Sunday mornings, "
                            "featuring a warm design. Wear it after Mass or give it to the "
                            "friend who runs on espresso."
                        ),
                    }
                ]
            }
        )

        result = regenerate_stored_listing(
            db,
            llm,
            "test-model",
            7,
            "Mention Sunday mornings.",
        )

        description = result["listing_description"]
        assert result["status"] == "success"
        assert description.startswith("For coffee lovers on Sunday mornings")
        assert "dark heather" not in description.lower()
        db.update_brief_listing_description.assert_called_once_with(7, description)

    def test_excluded_guidance_is_not_sent_to_the_model(self) -> None:
        db = MagicMock()
        db.get_listing_brief.return_value = {
            "id": 7,
            "concept": "Powered by Coffee",
            "product_type": "dtf_apparel",
            "specific_products": ["t-shirt"],
            "audience": "coffee lovers",
            "visual_style": "warm",
            "headline_text": "POWERED BY COFFEE",
            "recommended_materials": [_material()],
            "listing_description": "For coffee lovers, a warm design for slow mornings.",
        }
        llm = MagicMock()

        result = regenerate_stored_listing(db, llm, "test-model", 7, "Make it a tarot shirt.")

        assert result["status"] == "rejected"
        llm.with_raw_response.create.assert_not_called()
        db.update_brief_listing_description.assert_not_called()

    def test_a_failed_rewrite_keeps_the_stored_description(self) -> None:
        current = "For coffee lovers, a warm design for slow mornings and espresso."
        db = MagicMock()
        db.get_listing_brief.return_value = {
            "id": 7,
            "concept": "Powered by Coffee",
            "product_type": "dtf_apparel",
            "specific_products": ["t-shirt"],
            "audience": "coffee lovers",
            "visual_style": "warm",
            "headline_text": "POWERED BY COFFEE",
            "recommended_materials": [_material()],
            "listing_description": current,
        }
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")

        result = regenerate_stored_listing(db, llm, "test-model", 7, "Make it shorter.")

        assert result["status"] == "unchanged"
        assert result["listing_description"] == current
        db.update_brief_listing_description.assert_not_called()


def test_listing_copy_dataclass_holds_the_index_and_text() -> None:
    copy = ListingCopy(index=1, description="A white t-shirt for coffee lovers.")

    assert copy.index == 1
    assert copy.description.startswith("A white")
