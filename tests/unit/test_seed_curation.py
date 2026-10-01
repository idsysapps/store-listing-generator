"""Unit tests for LLM-driven seed curation."""

import json
from datetime import date
from unittest.mock import MagicMock

from store_listing.orchestration.promotion import ActiveSeed, SeedCandidate
from store_listing.orchestration.seed_curation import (
    PRODUCT_TAGS,
    CurationResult,
    build_curation_prompt,
    curate_seeds,
    parse_curation_response,
)


def _candidate(
    cid: int,
    query: str,
    source: str = "google",
    query_type: str = "rising",
    score: int = 100,
    delta: int = 5000,
) -> SeedCandidate:
    return SeedCandidate(
        id=cid,
        query=query,
        source=source,
        query_type=query_type,
        score=score,
        delta=delta,
        promotion_score=5000,
        status="pending",
    )


def _active(query: str, score: int = 0) -> ActiveSeed:
    return ActiveSeed(id=1, query=query, promotion_score=score)


class TestBuildCurationPrompt:
    def test_returns_system_and_user_messages(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "dad jokes shirt")],
            active_seeds=[_active("hoodie")],
            cross_seed_counts={"dad jokes shirt": 2},
            context=[{"name": "halloween", "days_until": 10}],
            today=date(2026, 10, 21),
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_candidate_data(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(42, "dad jokes shirt")],
            active_seeds=[],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert len(user_data["candidates"]) == 1
        assert user_data["candidates"][0]["id"] == 42
        assert user_data["candidates"][0]["query"] == "dad jokes shirt"

    def test_includes_date_and_context(self) -> None:
        messages = build_curation_prompt(
            candidates=[],
            active_seeds=[],
            cross_seed_counts={},
            context=[{"name": "halloween", "days_until": 5}],
            today=date(2026, 10, 26),
        )

        user_data = json.loads(messages[1]["content"])
        assert user_data["date"] == "2026-10-26"
        assert user_data["upcoming_events"][0]["name"] == "halloween"

    def test_includes_current_seeds(self) -> None:
        messages = build_curation_prompt(
            candidates=[],
            active_seeds=[_active("hoodie"), _active("gift")],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert user_data["current_seeds"] == ["hoodie", "gift"]

    def test_includes_cross_source_count(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "mom shirt")],
            active_seeds=[],
            cross_seed_counts={"mom shirt": 3},
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert user_data["candidates"][0]["cross_source_count"] == 3


class TestParseCurationResponse:
    def test_parses_well_formed_json(self) -> None:
        response = json.dumps(
            {
                "promote": [42],
                "reject": [99],
                "product_tags": {"42": ["dtf_apparel"]},
                "pivot": [{"to": "vintage hoodie", "product_tags": ["dtf_apparel"]}],
                "event_seeds": [{"seed": "halloween candy shirt", "product_tags": ["dtf_apparel"]}],
            }
        )

        result = parse_curation_response(response)

        assert result.promote == [42]
        assert result.reject == [99]
        assert len(result.pivot) == 1
        assert result.pivot[0].to_seed == "vintage hoodie"
        assert len(result.event_seeds) == 1
        assert result.event_seeds[0].seed == "halloween candy shirt"

    def test_handles_json_with_surrounding_text(self) -> None:
        response = 'Here is my analysis:\n{"promote": [1], "reject": []}\nDone.'

        result = parse_curation_response(response)

        assert result.promote == [1]

    def test_handles_malformed_json(self) -> None:
        result = parse_curation_response("this is not json at all")

        assert result.promote == []
        assert result.reject == []
        assert result.pivot == []
        assert result.event_seeds == []

    def test_handles_empty_response(self) -> None:
        result = parse_curation_response("")

        assert result == CurationResult()

    def test_filters_invalid_promote_ids(self) -> None:
        response = json.dumps({"promote": [1, "bad", None, 2], "reject": []})

        result = parse_curation_response(response)

        assert result.promote == [1, 2]

    def test_filters_incomplete_pivots(self) -> None:
        response = json.dumps(
            {
                "promote": [],
                "reject": [],
                "pivot": [
                    {"to": "b", "product_tags": ["dtf_apparel"]},
                    {"from": "c"},
                    "not a dict",
                ],
            }
        )

        result = parse_curation_response(response)

        assert len(result.pivot) == 1
        assert result.pivot[0].to_seed == "b"

    def test_non_dict_reasoning_ignored(self) -> None:
        response = json.dumps({"promote": [], "reject": [], "reasoning": "not a dict"})

        result = parse_curation_response(response)

        assert result.reasoning == {}


class TestCurateSeeds:
    def _mock_db(self, candidates=None, active_seeds=None) -> MagicMock:
        db = MagicMock()
        db.list_pending_candidates.return_value = candidates or []
        db.list_active_seeds.return_value = active_seeds or []
        db.cross_seed_counts.return_value = {}
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1
        return db

    def _mock_llm(self, response_json: dict) -> MagicMock:
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps(response_json)
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        llm.with_raw_response.create.return_value = raw
        return llm

    def test_no_candidates_returns_early(self) -> None:
        db = self._mock_db()
        llm = MagicMock()

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "success"
        assert result["promoted"] == 0
        assert not llm.with_raw_response.create.called

    def test_promotes_candidates_from_llm(self) -> None:
        candidates = [_candidate(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm({"promote": [42], "reject": []})

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["promoted"] == 1
        db.insert_active_seed.assert_any_call(
            query="dad jokes shirt", promotion_score=5000, candidate_id=42
        )
        db.promote_candidate.assert_called_with(42)

    def test_rejects_candidates_from_llm(self) -> None:
        candidates = [_candidate(99, "compression socks")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm({"promote": [], "reject": [99]})

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["rejected"] == 1
        db.reject_candidate.assert_called_with(99)

    def test_inserts_pivot_seeds(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "promote": [],
                "reject": [],
                "pivot": [{"to": "vintage band hoodie", "product_tags": ["dtf_apparel"]}],
            }
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["pivots"] == 1
        db.insert_active_seed.assert_any_call(
            query="vintage band hoodie", promotion_score=0, candidate_id=None
        )

    def test_inserts_event_seeds(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "promote": [],
                "reject": [],
                "event_seeds": [{"seed": "halloween candy shirt", "product_tags": ["dtf_apparel"]}],
            }
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["event_seeds"] == 1
        db.insert_active_seed.assert_any_call(
            query="halloween candy shirt", promotion_score=0, candidate_id=None
        )

    def test_logs_all_actions(self) -> None:
        candidates = [_candidate(42, "dad jokes"), _candidate(99, "socks")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm({"promote": [42], "reject": [99]})

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        log_calls = db.insert_curation_log.call_args_list
        actions = [c.kwargs["action"] for c in log_calls]
        assert "promote" in actions
        assert "reject" in actions
        assert all(c.kwargs["llm_model"] == "test-model" for c in log_calls)

    def test_handles_llm_error_gracefully(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "API down" in result["error"]

    def test_ignores_unknown_candidate_ids(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm({"promote": [999], "reject": [888]})

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["promoted"] == 0
        assert result["rejected"] == 0
        assert not db.promote_candidate.called
        assert not db.reject_candidate.called

    def test_skips_duplicate_pivot_seeds(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        db.insert_active_seed.return_value = None
        llm = self._mock_llm(
            {"promote": [], "reject": [], "pivot": [{"to": "b", "product_tags": ["dtf_apparel"]}]}
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["pivots"] == 0

    def test_empty_choices_returns_error(self) -> None:
        candidates = [_candidate(1, "test seed")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = MagicMock(choices=[])
        llm.with_raw_response.create.return_value = raw

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "empty choices" in result["error"]

    def test_none_choices_returns_error(self) -> None:
        candidates = [_candidate(1, "test seed")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = MagicMock(choices=None)
        llm.with_raw_response.create.return_value = raw

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "empty choices" in result["error"]


class TestProductTags:
    def test_product_tags_contains_three_categories(self) -> None:
        assert "dtf_apparel" in PRODUCT_TAGS
        assert "sublimation" in PRODUCT_TAGS
        assert "sticker_vinyl" in PRODUCT_TAGS
        assert len(PRODUCT_TAGS) == 3


class TestParseProductTags:
    def test_parses_product_tags_on_promoted(self) -> None:
        response = json.dumps(
            {
                "promote": [42],
                "reject": [],
                "product_tags": {"42": ["dtf_apparel"]},
            }
        )
        result = parse_curation_response(response)
        assert result.product_tags == {"42": ["dtf_apparel"]}

    def test_filters_invalid_tags(self) -> None:
        response = json.dumps(
            {
                "promote": [42],
                "reject": [],
                "product_tags": {"42": ["dtf_apparel", "furniture", "sublimation"]},
            }
        )
        result = parse_curation_response(response)
        assert result.product_tags == {"42": ["dtf_apparel", "sublimation"]}

    def test_missing_product_tags_defaults_empty(self) -> None:
        response = json.dumps({"promote": [42], "reject": []})
        result = parse_curation_response(response)
        assert result.product_tags == {}

    def test_non_dict_product_tags_ignored(self) -> None:
        response = json.dumps({"promote": [], "reject": [], "product_tags": "not a dict"})
        result = parse_curation_response(response)
        assert result.product_tags == {}

    def test_pivot_product_tags_parsed(self) -> None:
        response = json.dumps(
            {
                "promote": [],
                "reject": [],
                "pivot": [
                    {
                        "to": "vintage band hoodie",
                        "product_tags": ["dtf_apparel"],
                    }
                ],
            }
        )
        result = parse_curation_response(response)
        assert result.pivot[0].product_tags == ["dtf_apparel"]

    def test_event_seed_product_tags_parsed(self) -> None:
        response = json.dumps(
            {
                "promote": [],
                "reject": [],
                "event_seeds": [
                    {
                        "seed": "halloween mug",
                        "product_tags": ["sublimation"],
                    }
                ],
            }
        )
        result = parse_curation_response(response)
        assert result.event_seeds[0].product_tags == ["sublimation"]

    def test_pivot_without_product_tags_defaults_empty(self) -> None:
        response = json.dumps(
            {
                "promote": [],
                "reject": [],
                "pivot": [{"to": "b"}],
            }
        )
        result = parse_curation_response(response)
        assert result.pivot[0].product_tags == []

    def test_event_seed_without_product_tags_defaults_empty(self) -> None:
        response = json.dumps(
            {
                "promote": [],
                "reject": [],
                "event_seeds": [{"seed": "x"}],
            }
        )
        result = parse_curation_response(response)
        assert result.event_seeds[0].product_tags == []


class TestCurateSeedsProductTags:
    def _mock_db(self, candidates=None, active_seeds=None) -> MagicMock:
        db = MagicMock()
        db.list_pending_candidates.return_value = candidates or []
        db.list_active_seeds.return_value = active_seeds or []
        db.cross_seed_counts.return_value = {}
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1
        return db

    def _mock_llm(self, response_json: dict) -> MagicMock:
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps(response_json)
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        llm.with_raw_response.create.return_value = raw
        return llm

    def test_promoted_seed_gets_product_tags(self) -> None:
        candidates = [_candidate(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "promote": [42],
                "reject": [],
                "product_tags": {"42": ["dtf_apparel"]},
            }
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_called_once()
        call_args = db.insert_product_tags.call_args
        assert call_args.kwargs["active_seed_id"] == 1
        assert call_args.kwargs["tags"] == ["dtf_apparel"]

    def test_pivot_seed_gets_product_tags(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "promote": [],
                "reject": [],
                "pivot": [
                    {
                        "to": "vintage hoodie",
                        "product_tags": ["dtf_apparel"],
                    }
                ],
            }
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_called_once()
        call_args = db.insert_product_tags.call_args
        assert call_args.kwargs["active_seed_id"] == 1
        assert call_args.kwargs["tags"] == ["dtf_apparel"]

    def test_event_seed_gets_product_tags(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "promote": [],
                "reject": [],
                "event_seeds": [
                    {
                        "seed": "halloween mug",
                        "product_tags": ["sublimation"],
                    }
                ],
            }
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_called_once()
        call_args = db.insert_product_tags.call_args
        assert call_args.kwargs["active_seed_id"] == 1
        assert call_args.kwargs["tags"] == ["sublimation"]

    def test_no_tags_means_no_insert(self) -> None:
        candidates = [_candidate(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm({"promote": [42], "reject": []})

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_not_called()

    def test_prompt_includes_product_tags_instruction(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "dad jokes shirt")],
            active_seeds=[],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        assert "product_tags" in messages[0]["content"]
        assert "dtf_apparel" in messages[0]["content"]
