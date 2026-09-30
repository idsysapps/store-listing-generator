"""Unit tests for LLM-driven design brief generation."""

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.design_briefs import (
    BriefableCandidate,
    build_brief_prompt,
    generate_design_briefs,
    parse_brief_response,
)


def _briefable(
    active_seed_id: int,
    query: str,
    source: str = "google",
    query_type: str = "rising",
    score: int = 100,
    delta: int = 5000,
    product_tags: list[str] | None = None,
) -> BriefableCandidate:
    return BriefableCandidate(
        active_seed_id=active_seed_id,
        query=query,
        source=source,
        query_type=query_type,
        score=score,
        delta=delta,
        promotion_score=5000,
        product_tags=product_tags or [],
    )


class TestBuildBriefPrompt:
    def test_returns_system_and_user_messages(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[{"name": "halloween", "days_until": 10}],
            today=date(2026, 10, 21),
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_seed_data(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(42, "dad jokes shirt", source="amazon")],
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert "amazon" in user_data["signals_by_source"]
        seeds = user_data["signals_by_source"]["amazon"]
        assert len(seeds) == 1
        assert seeds[0]["id"] == 42
        assert seeds[0]["query"] == "dad jokes shirt"

    def test_groups_seeds_by_source(self) -> None:
        messages = build_brief_prompt(
            candidates=[
                _briefable(1, "dad jokes shirt", source="amazon"),
                _briefable(2, "dad jokes 2026", source="google"),
                _briefable(3, "funny mugs", source="amazon"),
            ],
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert len(user_data["signals_by_source"]["amazon"]) == 2
        assert len(user_data["signals_by_source"]["google"]) == 1

    def test_includes_date_and_context(self) -> None:
        messages = build_brief_prompt(
            candidates=[],
            context=[{"name": "halloween", "days_until": 5}],
            today=date(2026, 10, 26),
        )

        user_data = json.loads(messages[1]["content"])
        assert user_data["date"] == "2026-10-26"
        assert user_data["upcoming_events"][0]["name"] == "halloween"

    def test_includes_product_tags_per_seed(self) -> None:
        messages = build_brief_prompt(
            candidates=[
                _briefable(1, "dad jokes shirt", product_tags=["dtf_apparel"]),
            ],
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        source_seeds = user_data["signals_by_source"]["google"]
        assert source_seeds[0]["product_tags"] == ["dtf_apparel"]

    def test_respects_max_seeds_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import store_listing.orchestration.design_briefs as mod

        monkeypatch.setattr(mod, "MAX_SEEDS_PER_BRIEF", 2)

        candidates = [
            _briefable(1, "a"),
            _briefable(2, "b"),
            _briefable(3, "c"),
        ]
        messages = build_brief_prompt(candidates, context=[], today=date(2026, 10, 21))

        user_data = json.loads(messages[1]["content"])
        total = sum(len(seeds) for seeds in user_data["signals_by_source"].values())
        assert total == 2


class TestParseBriefResponse:
    def test_parses_well_formed_json(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Dad Jokes That'll Get You Divorced",
                        "product_type": "dtf_apparel",
                        "specific_products": ["t-shirt", "hoodie"],
                        "audience": "men 30-50, gift buyers",
                        "visual_style": "bold serif text",
                        "confidence": 85,
                        "reasoning": "Amazon buyer intent + Google rising",
                        "source_seed_ids": [1, 2],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Dad Jokes That'll Get You Divorced"
        assert result[0].product_type == "dtf_apparel"
        assert result[0].specific_products == ["t-shirt", "hoodie"]
        assert result[0].confidence == 85
        assert result[0].source_seed_ids == [1, 2]

    def test_handles_json_with_surrounding_text(self) -> None:
        response = 'Here is my analysis:\n{"briefs": [{"concept": "Test", "product_type": "dtf_apparel", "confidence": 50, "source_seed_ids": [1]}]}\nDone.'

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Test"

    def test_handles_malformed_json(self) -> None:
        result = parse_brief_response("this is not json at all")
        assert result == []

    def test_handles_empty_response(self) -> None:
        result = parse_brief_response("")
        assert result == []

    def test_filters_invalid_product_types(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Good brief",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    },
                    {
                        "concept": "Bad brief",
                        "product_type": "invalid_type",
                        "confidence": 50,
                        "source_seed_ids": [2],
                    },
                ]
            }
        )

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Good brief"

    def test_filters_briefs_without_concept(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    },
                    {
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [2],
                    },
                ]
            }
        )

        result = parse_brief_response(response)
        assert result == []

    def test_clamps_confidence_to_valid_range(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Over",
                        "product_type": "dtf_apparel",
                        "confidence": 150,
                        "source_seed_ids": [1],
                    },
                    {
                        "concept": "Under",
                        "product_type": "sublimation",
                        "confidence": -10,
                        "source_seed_ids": [2],
                    },
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].confidence == 100
        assert result[1].confidence == 0

    def test_filters_non_integer_source_ids(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1, "bad", None, 2],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].source_seed_ids == [1, 2]

    def test_handles_missing_optional_fields(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Minimal",
                        "product_type": "sticker_vinyl",
                        "confidence": 70,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].specific_products == []
        assert result[0].audience == ""
        assert result[0].visual_style == ""
        assert result[0].reasoning == ""

    def test_non_list_briefs_returns_empty(self) -> None:
        response = json.dumps({"briefs": "not a list"})
        assert parse_brief_response(response) == []

    def test_non_dict_brief_entries_skipped(self) -> None:
        response = json.dumps({"briefs": ["not a dict", 42]})
        assert parse_brief_response(response) == []

    def test_parses_layout_fields(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Born To Be Spooky",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_top",
                        "headline_text": "Born To Be Spooky",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "text_top"
        assert result[0].headline_text == "Born To Be Spooky"
        assert result[0].tagline_text is None

    def test_parses_text_top_bottom_layout(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Rescue Dog Halloween",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                        "layout_type": "text_top_bottom",
                        "headline_text": "Adopted My",
                        "tagline_text": "Costume Too",
                        "font_color": "#FF6600",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "text_top_bottom"
        assert result[0].headline_text == "Adopted My"
        assert result[0].tagline_text == "Costume Too"
        assert result[0].font_color == "#FF6600"

    def test_defaults_layout_to_full_bleed(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Abstract Pattern",
                        "product_type": "sublimation",
                        "confidence": 70,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "full_bleed"
        assert result[0].headline_text is None
        assert result[0].tagline_text is None

    def test_invalid_layout_type_defaults_to_full_bleed(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                        "layout_type": "invalid_layout",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "full_bleed"


class TestGenerateDesignBriefs:
    def _mock_db(self, candidates: list[BriefableCandidate] | None = None) -> MagicMock:
        db = MagicMock()
        db.list_briefable_seeds.return_value = candidates or []
        db.insert_design_brief.return_value = 1
        db.insert_brief_source.return_value = 1
        return db

    def _mock_llm(self, response_json: dict) -> MagicMock:
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps(response_json)
        llm.create.return_value = MagicMock(choices=[choice])
        return llm

    def test_no_candidates_returns_early(self) -> None:
        db = self._mock_db()
        llm = MagicMock()

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "success"
        assert result["briefs_created"] == 0
        assert not llm.create.called

    def test_generates_briefs_from_llm(self) -> None:
        candidates = [_briefable(42, "dad jokes shirt", source="amazon")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Dad Jokes That'll Get You Divorced",
                        "product_type": "dtf_apparel",
                        "specific_products": ["t-shirt"],
                        "audience": "men 30-50",
                        "visual_style": "bold text",
                        "confidence": 85,
                        "reasoning": "buyer intent",
                        "source_seed_ids": [42],
                    }
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "success"
        assert result["briefs_created"] == 1
        db.insert_design_brief.assert_called_once()
        db.insert_brief_source.assert_called_once_with(brief_id=1, active_seed_id=42)

    def test_validates_source_ids_against_actual_candidates(self) -> None:
        candidates = [_briefable(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [42, 999],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        source_calls = db.insert_brief_source.call_args_list
        assert len(source_calls) == 1
        assert source_calls[0].kwargs["active_seed_id"] == 42

    def test_handles_llm_error_gracefully(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        llm.create.side_effect = RuntimeError("API down")

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "API down" in result["error"]

    def test_skips_briefs_with_invalid_product_type(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Bad",
                        "product_type": "invalid",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["briefs_created"] == 0
        assert not db.insert_design_brief.called

    def test_inserts_brief_source_for_each_valid_seed_id(self) -> None:
        candidates = [
            _briefable(10, "dad jokes"),
            _briefable(20, "funny mugs"),
        ]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Multi-source brief",
                        "product_type": "dtf_apparel",
                        "confidence": 75,
                        "source_seed_ids": [10, 20],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        source_calls = db.insert_brief_source.call_args_list
        assert len(source_calls) == 2
        seed_ids = {c.kwargs["active_seed_id"] for c in source_calls}
        assert seed_ids == {10, 20}

    def test_batch_id_is_todays_date(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["batch_id"] == "2026-10-01"

    def test_passes_model_to_insert(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["llm_model"] == "test-model"
