"""The personal custom design: search, one line, one print-ready graphic."""

from __future__ import annotations

import json

from store_listing.orchestration.personal_design import (
    PersonalDesign,
    SearchNote,
    arrangement_line,
    graphic_prompt,
    rearrange,
    research_personal_design,
    resolve_lettering,
)


def test_customer_written_words_are_the_only_line() -> None:
    line = resolve_lettering(
        customer_line="Get there eventually",
        proposed_line="Mesa so very, very thankful!",
        voice_found=True,
    )

    assert line == "Get there eventually"


def test_a_voice_in_the_notes_allows_one_new_line() -> None:
    line = resolve_lettering(
        customer_line=None,
        proposed_line="Mesa so very, very thankful!",
        voice_found=True,
    )

    assert line == "Mesa so very, very thankful!"


def test_no_voice_leaves_the_shirt_without_a_line() -> None:
    line = resolve_lettering(
        customer_line="  ",
        proposed_line="Mesa so very, very thankful!",
        voice_found=False,
    )

    assert line is None


def test_the_graphic_is_a_magenta_key_and_names_its_line_once() -> None:
    prompt = graphic_prompt(
        subjects="a tall Gungan with long ears beside a round freighter",
        lettering="Mesa so very, very thankful!",
        arrangement=arrangement_line(1),
        facts=("Jar Jar speaks in a Gungan voice.",),
    )

    assert "solid pure magenta" in prompt
    assert "no backdrop" in prompt
    assert prompt.count("Mesa so very, very thankful!") == 1
    assert "photograph of the place" not in prompt
    assert "Jar Jar speaks in a Gungan voice." in prompt


def test_a_second_pass_keeps_the_facts_and_changes_the_arrangement() -> None:
    design = PersonalDesign(
        prompt=graphic_prompt(
            subjects="a tall Gungan beside a round freighter",
            lettering="Mesa so very, very thankful!",
            arrangement=arrangement_line(1),
            facts=("Jar Jar speaks in a Gungan voice.",),
        ),
        lettering="Mesa so very, very thankful!",
        reference_urls=("https://example.com/jar.jpg",),
        facts=("Jar Jar speaks in a Gungan voice.",),
        subjects="a tall Gungan beside a round freighter",
    )

    again = rearrange(design, seed=2)

    assert again.prompt != design.prompt
    assert "Jar Jar speaks in a Gungan voice." in again.prompt
    assert again.lettering == design.lettering
    assert again.reference_urls == design.reference_urls


def test_research_writes_one_line_when_the_notes_have_a_voice() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["jar jar speech"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": True,
                    "lettering": "Mesa so very, very thankful!",
                    "subjects": "a tall Gungan with long ears beside a round freighter",
                    "references": [{"url": "https://example.com/jar.jpg", "keep": True}],
                }
            ),
        ]
    )
    note = SearchNote(
        query="jar jar speech",
        title="Jar Jar",
        text="Jar Jar speaks in a Gungan voice.",
        image_url="https://example.com/jar.jpg",
    )

    design = research_personal_design(
        "hollywood studios, jar jar, thanksgiving",
        complete=lambda _messages: next(answers),
        search=lambda _query: [note],
        seed=1,
    )

    assert design.lettering == "Mesa so very, very thankful!"
    assert design.prompt.count("Mesa so very, very thankful!") == 1
    assert "solid pure magenta" in design.prompt
    assert design.reference_urls == ("https://example.com/jar.jpg",)
    assert any("Jar Jar speaks in a Gungan voice." in fact for fact in design.facts)


def test_research_drops_a_line_the_notes_cannot_support() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["thanksgiving table"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": "Mesa so very, very thankful!",
                    "subjects": "a harvest table",
                    "references": [],
                }
            ),
        ]
    )

    design = research_personal_design(
        "thanksgiving table",
        complete=lambda _messages: next(answers),
        search=lambda _query: [],
        seed=1,
    )

    assert design.lettering is None
    assert "thankful" not in design.prompt


def test_research_keeps_at_most_six_references() -> None:
    refs = [{"url": f"https://example.com/{index}.jpg", "keep": True} for index in range(8)]
    answers = iter(
        [
            json.dumps({"queries": ["ship"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a round freighter",
                    "references": refs,
                }
            ),
        ]
    )

    design = research_personal_design(
        "a round freighter",
        complete=lambda _messages: next(answers),
        search=lambda _query: [],
        seed=4,
    )

    assert len(design.reference_urls) == 6


def test_a_rejected_reference_is_not_kept() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["ship"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a round freighter",
                    "references": [
                        {"url": "https://example.com/crowd.jpg", "keep": False},
                        {"url": "https://example.com/ship.jpg", "keep": True},
                    ],
                }
            ),
        ]
    )

    design = research_personal_design(
        "a round freighter",
        complete=lambda _messages: next(answers),
        search=lambda _query: [
            SearchNote("ship", "Ship", "A round freighter.", "https://example.com/crowd.jpg")
        ],
        seed=1,
    )

    assert design.reference_urls == ("https://example.com/ship.jpg",)
