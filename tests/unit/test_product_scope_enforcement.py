"""Unit: the pipeline honours the product types that are in production.

The store focuses on one production method at a time, so the prompts, the
parsers, the material list and the image sweep have to agree on what is being
produced (issue #215).
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

from store_listing.orchestration.brief_prompts import (
    product_belongs_to,
    product_catalog,
)
from store_listing.orchestration.design_briefs import (
    BriefableCandidate,
    build_brief_prompt,
    generate_design_briefs,
    read_brief_response,
)


def _candidate(seed_id: int = 1, query: str = "funny t-shirt") -> BriefableCandidate:
    return BriefableCandidate(
        active_seed_id=seed_id,
        query=query,
        source="google",
        query_type="rising",
        score=100,
        delta=5000,
        promotion_score=5000,
        product_tags=["dtf_apparel"],
    )


def _brief(product_type: str = "dtf_apparel", products: list[str] | None = None) -> dict:
    return {
        "concept": "Coffee Beans",
        "product_type": product_type,
        "specific_products": products if products is not None else ["t-shirt"],
        "audience": "coffee lovers",
        "visual_style": "flat graphic",
        "confidence": 80,
        "reasoning": "a rising trend",
        "source_seed_ids": [1],
    }


def _response(briefs: list[dict]) -> str:
    return json.dumps({"briefs": briefs})


def _material() -> dict:
    return {
        "external_id": "c9924c1b-cherry-blossom",
        "name": "Core Adult",
        "brand": "Port & Company",
        "substrate_type": "dtf-apparel",
        "color": "Cherry Blossom",
        "sizes": ["S", "M", "L"],
        "material_composition": "100% cotton",
        "product_type": None,
        "sku": "C9924C1B",
        "unit_cost": 4.5,
    }


def _recommendation() -> dict:
    return {
        "external_id": "c9924c1b-cherry-blossom",
        "name": "Core Adult",
        "color": "Cherry Blossom",
        "substrate_type": "dtf-apparel",
        "unit_cost": 4.5,
    }


class TestProductCatalog:
    def test_the_catalog_lists_the_types_in_production(self) -> None:
        catalog = product_catalog(["dtf_apparel"]).lower()

        assert "dtf_apparel" in catalog
        assert "sublimation" not in catalog
        assert "sticker_vinyl" not in catalog

    def test_the_catalog_lists_every_type_when_none_is_given(self) -> None:
        catalog = product_catalog().lower()

        assert "dtf_apparel" in catalog
        assert "sublimation" in catalog
        assert "sticker_vinyl" in catalog


class TestProductBelongsTo:
    def test_a_t_shirt_is_dtf_apparel(self) -> None:
        assert product_belongs_to("t-shirt", "dtf_apparel")

    def test_a_mug_is_not_dtf_apparel(self) -> None:
        assert not product_belongs_to("mug", "dtf_apparel")

    def test_a_mug_is_sublimation(self) -> None:
        assert product_belongs_to("mug", "sublimation")

    def test_a_sticker_is_sticker_vinyl(self) -> None:
        assert product_belongs_to("vinyl decal", "sticker_vinyl")


class TestBriefPrompt:
    def test_the_prompt_only_offers_the_types_in_production(self) -> None:
        messages = build_brief_prompt(
            [_candidate()], [], date(2026, 10, 1), enabled=["dtf_apparel"]
        )

        system = messages[0]["content"].lower()
        assert "dtf_apparel" in system
        assert "sublimation" not in system
        assert "sticker_vinyl" not in system

    def test_the_prompt_offers_every_type_by_default(self) -> None:
        messages = build_brief_prompt([_candidate()], [], date(2026, 10, 1))

        system = messages[0]["content"].lower()
        assert "sublimation" in system
        assert "sticker_vinyl" in system


class TestParseBriefResponse:
    def test_a_brief_of_a_type_that_is_not_in_production_is_coerced(self) -> None:
        briefs, error = read_brief_response(_response([_brief("sublimation")]), ["dtf_apparel"])

        assert error is None
        assert len(briefs) == 1
        assert briefs[0].product_type == "dtf_apparel"

    def test_the_products_of_the_coerced_brief_are_filtered(self) -> None:
        briefs, _ = read_brief_response(
            _response([_brief("sublimation", products=["mug", "t-shirt"])]), ["dtf_apparel"]
        )

        assert briefs[0].specific_products == ["t-shirt"]

    def test_a_mug_does_not_survive_in_a_dtf_brief(self) -> None:
        briefs, _ = read_brief_response(
            _response([_brief("dtf_apparel", products=["mug"])]), ["dtf_apparel"]
        )

        assert briefs[0].specific_products == []

    def test_a_brief_of_a_disabled_type_is_dropped_when_several_types_are_in_production(
        self,
    ) -> None:
        briefs, _ = read_brief_response(
            _response([_brief("sublimation")]), ["dtf_apparel", "sticker_vinyl"]
        )

        assert briefs == []

    def test_a_brief_of_a_type_in_production_is_kept(self) -> None:
        briefs, _ = read_brief_response(_response([_brief("dtf_apparel")]), ["dtf_apparel"])

        assert len(briefs) == 1

    def test_without_a_scope_every_type_is_accepted(self) -> None:
        briefs, _ = read_brief_response(_response([_brief("sublimation")]), None)

        assert len(briefs) == 1
        assert briefs[0].product_type == "sublimation"


class TestMaterialRecommendations:
    def _materials(self) -> list[dict]:
        return [
            {
                "external_id": "bc3001-white",
                "name": "Bella Canvas 3001",
                "brand": "Bella Canvas",
                "substrate_type": "t-shirt",
                "color": "White",
                "sizes": ["S", "M", "L"],
                "material_composition": "100% cotton",
                "product_type": ["substrate", "dtf_apparel"],
                "sku": "BC3001-WHT",
                "unit_cost": 4.5,
            },
            {
                "external_id": "mug-white",
                "name": "11oz Mug",
                "brand": "Printify",
                "substrate_type": "mug",
                "color": "White",
                "sizes": None,
                "material_composition": None,
                "product_type": ["sublimation"],
                "sku": "MUG-WHT",
                "unit_cost": 8.0,
            },
        ]

    def test_a_recommendation_of_a_material_we_offered_is_kept(self) -> None:
        response = _response(
            [
                {
                    **_brief(),
                    "recommended_materials": [
                        {
                            "external_id": "bc3001-white",
                            "name": "Bella Canvas 3001",
                            "color": "White",
                            "substrate_type": "t-shirt",
                            "unit_cost": 4.5,
                        }
                    ],
                }
            ]
        )

        briefs, _ = read_brief_response(response, ["dtf_apparel"], materials=self._materials())

        assert len(briefs[0].recommended_materials or []) == 1

    def test_a_recommendation_the_model_invented_is_dropped(self) -> None:
        response = _response(
            [
                {
                    **_brief(),
                    "recommended_materials": [
                        {"external_id": "does-not-exist", "name": "Imaginary Shirt"},
                    ],
                }
            ]
        )

        briefs, _ = read_brief_response(response, ["dtf_apparel"], materials=self._materials())

        assert briefs[0].recommended_materials == []

    def test_a_recommendation_that_does_not_serve_the_type_is_dropped(self) -> None:
        response = _response(
            [
                {
                    **_brief(),
                    "recommended_materials": [
                        {"external_id": "mug-white", "name": "11oz Mug"},
                    ],
                }
            ]
        )

        briefs, _ = read_brief_response(response, ["dtf_apparel"], materials=self._materials())

        assert briefs[0].recommended_materials == []


class TestGenerateDesignBriefs:
    def _mock_db(self, materials: list[dict] | None = None) -> MagicMock:
        db = MagicMock()
        db.list_briefable_seeds.return_value = [_candidate()]
        db.list_active_materials.return_value = materials if materials is not None else []
        db.insert_design_brief.return_value = 1
        db.insert_brief_source.return_value = 1
        return db

    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        llm.with_raw_response.create.return_value = raw
        return llm

    def test_the_materials_of_the_types_in_production_are_offered(self) -> None:
        db = self._mock_db(materials=[_material()])
        llm = self._mock_llm(_response([_brief()]))

        generate_design_briefs(
            db, llm, "test-model", today=date(2026, 10, 1), enabled=["dtf_apparel"]
        )

        db.list_active_materials.assert_called_once_with(product_type="dtf_apparel")

    def test_every_material_is_offered_when_the_type_in_production_has_none(self) -> None:
        db = self._mock_db()
        db.list_active_materials.side_effect = [[], [_material()]]
        llm = self._mock_llm(_response([_brief()]))

        generate_design_briefs(
            db, llm, "test-model", today=date(2026, 10, 1), enabled=["dtf_apparel"]
        )

        calls = db.list_active_materials.call_args_list
        assert calls[0].kwargs == {"product_type": "dtf_apparel"}
        assert calls[1].kwargs == {}

    def test_the_materials_are_offered_to_the_model_when_the_type_has_none(self) -> None:
        db = self._mock_db()
        db.list_active_materials.side_effect = [[], [_material()]]
        llm = self._mock_llm(_response([_brief()]))

        generate_design_briefs(
            db, llm, "test-model", today=date(2026, 10, 1), enabled=["dtf_apparel"]
        )

        messages = llm.with_raw_response.create.call_args.kwargs["messages"]
        assert "Core Adult" in messages[1]["content"]

    def test_a_recommendation_survives_when_the_type_in_production_has_none(self) -> None:
        db = self._mock_db()
        db.list_active_materials.side_effect = [[], [_material()]]
        llm = self._mock_llm(
            _response([{**_brief(), "recommended_materials": [_recommendation()]}])
        )

        result = generate_design_briefs(
            db, llm, "test-model", today=date(2026, 10, 1), enabled=["dtf_apparel"]
        )

        assert result["briefs_created"] == 1
        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["recommended_materials"] == [_recommendation()]

    def test_every_material_is_offered_when_several_types_are_in_production(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_response([_brief()]))

        generate_design_briefs(
            db,
            llm,
            "test-model",
            today=date(2026, 10, 1),
            enabled=["dtf_apparel", "sublimation"],
        )

        db.list_active_materials.assert_called_once_with()
