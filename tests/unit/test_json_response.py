"""Unit: reading the JSON an LLM actually sent.

The model answers with JSON, but it does not always answer with *only* JSON:
it appends a note, repeats the batch, or splits it in two. A parser that
slices from the first `{` to the last `}` reads two JSON values as one and
throws the whole answer away (issue #209).
"""

from __future__ import annotations

from store_listing.orchestration.json_response import (
    extract_json_objects,
    merge_items,
)


class TestExtractJsonObjects:
    def test_finds_the_object_in_a_response_with_text_around_it(self) -> None:
        text = 'Here you go:\n{"briefs": []}\nHope that helps.'

        assert extract_json_objects(text) == [{"briefs": []}]

    def test_finds_every_object_in_a_repeated_answer(self) -> None:
        text = '{"a": 1}\nsome prose\n{"b": 2}'

        assert extract_json_objects(text) == [{"a": 1}, {"b": 2}]

    def test_ignores_a_truncated_object(self) -> None:
        text = '{"a": 1, "b": [2, 3'

        assert extract_json_objects(text) == []

    def test_keeps_braces_that_are_inside_a_string(self) -> None:
        text = '{"concept": "a brace } and a { in the text"}'

        assert extract_json_objects(text) == [{"concept": "a brace } and a { in the text"}]

    def test_keeps_a_quoted_brace_from_ending_the_value_early(self) -> None:
        text = '{"concept": "not } the end", "x": 1} trailing }'

        assert extract_json_objects(text) == [{"concept": "not } the end", "x": 1}]

    def test_finds_nothing_in_prose(self) -> None:
        assert extract_json_objects("I could not do it, sorry.") == []

    def test_reads_a_json_object_with_single_quotes(self) -> None:
        # Some models answer with single quotes; the curation parser used to
        # cope with that.
        assert extract_json_objects("{'a': 1}") == [{"a": 1}]

    def test_ignores_a_value_that_is_not_an_object(self) -> None:
        assert extract_json_objects("[1, 2, 3]") == []


class TestMergeItems:
    def test_merges_the_items_of_every_object(self) -> None:
        objects = [{"briefs": [1]}, {"briefs": [2]}, {"other": [3]}]

        assert merge_items(objects, "briefs") == [1, 2]

    def test_ignores_an_item_that_is_not_a_list(self) -> None:
        objects = [{"briefs": {"a": 1}}, {"briefs": [2]}]

        assert merge_items(objects, "briefs") == [2]

    def test_an_object_without_the_key_contributes_nothing(self) -> None:
        assert merge_items([{"a": 1}], "briefs") == []
