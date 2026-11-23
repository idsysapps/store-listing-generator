"""The personal custom design: search, one line, one print-ready graphic."""

from __future__ import annotations

import json

import httpx
import pytest

from store_listing.orchestration.personal_design import (
    ArticleOutline,
    ArticleSection,
    PersonalDesign,
    PhotoLook,
    SearchNote,
    arrangement_line,
    character_from_notes,
    character_role,
    fetch_reference,
    graphic_prompt,
    look_at_reference,
    photo_look_from_answer,
    quoted_lettering,
    rearrange,
    research_personal_design,
    resolve_lettering,
    search_commons,
    search_duckduckgo,
    search_personal,
    search_wikidata,
    search_wikipedia,
    section_lines,
    specified_characters,
)


def _complete_from(answers):
    def complete(messages: list[dict[str, str]]) -> str:
        system = ""
        if messages and isinstance(messages[0], dict):
            system = str(messages[0].get("content") or "")
        if "List up to three concrete visual subjects" in system:
            return json.dumps({"subjects": []})
        return next(answers)

    return complete


def _scripted_complete(*lines: str, deepen: list[str] | None = None):
    """LLM answers for research tests; deepen is answered from the system prompt."""
    answers = iter(lines)
    deepen_json = json.dumps({"subjects": list(deepen or [])})

    def complete(messages: list[dict[str, str]]) -> str:
        system = ""
        if messages and isinstance(messages[0], dict):
            system = str(messages[0].get("content") or "")
        if "List up to three concrete visual subjects" in system:
            return deepen_json
        return next(answers)

    return complete


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
        complete=_complete_from(answers),
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
        complete=_complete_from(answers),
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
        complete=_complete_from(answers),
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
        complete=_complete_from(answers),
        search=lambda _query: [
            SearchNote("ship", "Ship", "A round freighter.", "https://example.com/crowd.jpg")
        ],
        seed=1,
    )

    assert design.reference_urls == ("https://example.com/ship.jpg",)


def test_a_search_keeps_a_photo_and_drops_a_logo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["piprop"] == "thumbnail"
        assert request.url.params["gsrsearch"] == "park"
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "title": "Park",
                            "extract": "A theme park.",
                            "thumbnail": {"source": "https://upload.wikimedia.org/park.jpg"},
                        },
                        "2": {
                            "title": "Logo",
                            "extract": "A studio logo.",
                            "thumbnail": {"source": "https://upload.wikimedia.org/logo.svg"},
                        },
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikipedia("park", client=client)
    urls = [note.image_url for note in notes]

    assert "https://upload.wikimedia.org/park.jpg" in urls
    assert "https://upload.wikimedia.org/logo.svg" not in urls


def test_double_quotes_are_the_shirt_line() -> None:
    assert quoted_lettering('slow moving with text "Get There Eventually"') == (
        "Get There Eventually"
    )
    assert quoted_lettering('slow moving with text ""Get There Eventually""') == (
        "Get There Eventually"
    )


def test_curly_quotes_are_the_shirt_line() -> None:
    assert quoted_lettering("slow moving with text “Get There Eventually”") == (
        "Get There Eventually"
    )


def test_a_name_before_character_is_that_character() -> None:
    names = specified_characters(
        'Judy Hopps character showing slow moving with text "Get There Eventually"'
    )

    assert names == ("Judy Hopps",)


def test_a_name_after_character_is_that_character() -> None:
    assert specified_characters("the character Judy Hopps, slow moving") == ("Judy Hopps",)
    assert specified_characters("characters Judy Hopps and Nick Wilde") == (
        "Judy Hopps",
        "Nick Wilde",
    )


def test_plural_characters_are_not_one_specific_character() -> None:
    assert specified_characters("lifelike Disney Zootopia characters") == ()
    assert specified_characters("Zootopia sleuth character") == ()
    assert character_role("lifelike Disney Zootopia characters") is None
    assert character_role("Judy Hopps character") is None


def test_a_cast_list_keeps_every_bullet() -> None:
    wikitext = """
== Voice cast ==
{{Multiple image
| footer = [[Someone]]
}}
{{castlist|
* Raymond S. Persi as:
** Flash Slothmore, the "fastest" [[three-toed sloth]] in the DMV.
* [[Kristen Bell]] as Priscilla Tripletoe, a sloth and Flash's co-worker.<ref name="k" />
}}
"""

    lines = section_lines(wikitext)

    assert 'Flash Slothmore, the "fastest" three-toed sloth in the DMV.' in lines
    assert "Priscilla Tripletoe, a sloth and Flash's co-worker." in lines
    assert "castlist" not in lines
    assert "<ref" not in lines


def test_a_role_is_the_search_for_an_unnamed_character() -> None:
    assert (
        character_role(
            'Zootopia sleuth character showing slow moving with text "Get There Eventually"'
        )
        == "Zootopia sleuth character"
    )


def test_one_named_result_is_the_character() -> None:
    notes = [
        SearchNote(
            "Zootopia sleuth character",
            "Flash Slothmore",
            "Flash Slothmore is a sloth.",
            "https://example.com/flash.jpg",
        )
    ]

    assert character_from_notes("Zootopia sleuth character", notes) == "Flash Slothmore"


def test_a_crowd_of_names_is_not_resolved() -> None:
    notes = [
        SearchNote(
            "Zootopia sleuth character",
            "Zootopia",
            "Judy Hopps, Nick Wilde and Flash Slothmore appear.",
            None,
        )
    ]

    assert character_from_notes("Zootopia sleuth character", notes) is None


def _article(title: str, *headings: tuple[str, str]) -> ArticleOutline:
    return ArticleOutline(
        title,
        tuple(ArticleSection(heading, index) for heading, index in headings),
    )


def _design_answer(subjects: str) -> str:
    return json.dumps(
        {
            "customer_line": None,
            "voice_found": False,
            "lettering": None,
            "subjects": subjects,
            "references": [],
        }
    )


def test_research_names_the_character_from_the_article() -> None:
    sentence = 'Zootopia sleuth character showing slow moving with text "Get There Eventually"'
    answers = iter(
        [
            json.dumps({"title": "Zootopia"}),
            json.dumps({"section": "Voice cast"}),
            json.dumps(
                {
                    "name": "Flash Slothmore",
                    "description": "the fastest three-toed sloth in the DMV.",
                    "enough": True,
                }
            ),
            json.dumps({"queries": []}),
            json.dumps({"accept": True}),
            _design_answer("a grey rabbit beside an orange tiger"),
        ]
    )
    queries: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries.append(query)
        if query == "Flash Slothmore":
            return [
                SearchNote(
                    query,
                    "Flash Slothmore",
                    "Flash Slothmore is a sloth in Zootopia.",
                    "https://example.com/flash-ref.jpg",
                )
            ]
        return [
            SearchNote(
                query,
                "Winnie the Pooh (franchise)",
                "Pooh's Super Sleuth Christmas Movie.",
                "https://example.com/pooh.jpg",
            )
        ]

    design = research_personal_design(
        sentence,
        complete=_complete_from(answers),
        search=search,
        seed=1,
        open_page=lambda title: _article(title, ("Voice cast", "2")),
        read_page=lambda _title, _index: (
            'Flash Slothmore, the "fastest" three-toed sloth in the DMV.'
        ),
    )

    assert queries == ["Flash Slothmore"]
    assert "Flash Slothmore" in design.subjects
    assert "not a different character" in design.subjects
    assert "Judy Hopps" not in design.subjects
    assert "https://example.com/pooh.jpg" not in design.reference_urls
    assert design.reference_urls[0] == "https://example.com/flash-ref.jpg"
    assert design.lettering == "Get There Eventually"


def test_a_reference_search_rejects_another_character_and_keeps_flash() -> None:
    sentence = 'Zootopia sleuth character showing slow moving with text "Get There Eventually"'
    answers = iter(
        [
            json.dumps({"title": "Zootopia"}),
            json.dumps({"section": "Voice cast"}),
            json.dumps(
                {
                    "name": "Flash Slothmore",
                    "description": "the fastest three-toed sloth in the DMV.",
                    "enough": True,
                }
            ),
            json.dumps({"queries": []}),
            json.dumps({"accept": False, "query": "Flash Slothmore sloth"}),
            json.dumps({"accept": True}),
            _design_answer("a sloth"),
        ]
    )
    queries: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries.append(query)
        if query == "Flash Slothmore sloth":
            return [
                SearchNote(
                    query,
                    "Flash Slothmore",
                    "Flash Slothmore is a sloth in Zootopia.",
                    "https://example.com/flash-ref.jpg",
                )
            ]
        return [
            SearchNote(
                query,
                "Winnie the Pooh (franchise)",
                "Pooh's Super Sleuth Christmas Movie.",
                "https://example.com/pooh.jpg",
            )
        ]

    design = research_personal_design(
        sentence,
        complete=_complete_from(answers),
        search=search,
        seed=1,
        open_page=lambda title: _article(title, ("Voice cast", "2")),
        read_page=lambda _title, _index: "Flash Slothmore, a sloth in the DMV.",
    )

    assert queries == ["Flash Slothmore", "Flash Slothmore sloth"]
    assert design.reference_urls == ("https://example.com/flash-ref.jpg",)
    assert "Winnie" not in design.subjects
    assert "Flash Slothmore" in design.subjects


def test_a_missing_article_title_is_rejected_and_the_next_title_is_opened() -> None:
    sentence = 'Zootopia sleuth character showing slow moving with text "Get There Eventually"'
    opened: list[str] = []
    answers = iter(
        [
            json.dumps({"title": "Clauddius Bullfrog"}),
            json.dumps({"title": "Zootopia"}),
            json.dumps({"section": "Voice cast"}),
            json.dumps(
                {
                    "name": "Flash Slothmore",
                    "description": "a sloth in the DMV.",
                    "enough": True,
                }
            ),
            json.dumps({"queries": []}),
            json.dumps({"accept": True}),
            _design_answer("a sloth"),
        ]
    )

    def open_page(title: str) -> ArticleOutline | None:
        opened.append(title)
        if title == "Zootopia":
            return _article(title, ("Voice cast", "2"))
        return None

    design = research_personal_design(
        sentence,
        complete=_complete_from(answers),
        search=lambda query: [
            SearchNote(query, query, f"{query} is a sloth.", "https://example.com/flash.jpg")
        ],
        seed=1,
        open_page=open_page,
        read_page=lambda _title, _index: "Flash Slothmore, a sloth.",
    )

    assert opened == ["Clauddius Bullfrog", "Zootopia"]
    assert "Flash Slothmore" in design.subjects


def test_an_unresolved_role_is_researched_and_not_replaced_by_a_guess() -> None:
    # The model knows Judy Hopps. The article names nobody, so that guess
    # must not be searched or drawn, and the loose role is not a search.
    prompts: list[str] = []

    def complete(messages: list[dict[str, str]]) -> str:
        prompts.append(messages[0]["content"])
        if "List up to three concrete visual subjects" in messages[0]["content"]:
            return json.dumps({"subjects": []})
        if "Wikipedia article" in messages[0]["content"]:
            return json.dumps({"title": "Nope"})
        if "web searches" in messages[0]["content"]:
            return json.dumps({"queries": ["Judy Hopps", "slow moving"]})
        return _design_answer("Judy Hopps, a rabbit police officer")

    queries: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries.append(query)
        return [
            SearchNote(
                query,
                "Zootopia",
                "Judy Hopps, Nick Wilde and Flash Slothmore appear.",
                None,
            )
        ]

    design = research_personal_design(
        'Zootopia sleuth character showing slow moving with text "Get There Eventually"',
        complete=complete,
        search=search,
        seed=1,
        open_page=lambda _title: None,
    )

    assert "Zootopia sleuth character" not in queries
    assert "Judy Hopps" not in queries
    assert "slow moving" in queries
    assert "Judy Hopps" not in design.subjects
    assert "Judy Hopps" not in design.prompt
    assert "Zootopia sleuth character" in design.subjects
    assert any("Do not guess a name." in prompt for prompt in prompts)
    assert any("Do not invent a character the notes do not name." in prompt for prompt in prompts)


def test_research_prints_the_quoted_line_and_locks_the_named_character() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["zootopia movie"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": "zoogether sleuth",
                    "subjects": "a grey rabbit beside an orange tiger detective",
                    "references": [{"url": "https://example.com/poster.jpg", "keep": False}],
                }
            ),
        ]
    )
    queries: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries.append(query)
        if query == "Judy Hopps":
            return [
                SearchNote(
                    query,
                    "Judy Hopps",
                    "Judy Hopps is a rabbit police officer.",
                    "https://example.com/judy.jpg",
                ),
                SearchNote(
                    query,
                    "1953 British Mount Everest expedition",
                    "A mountaineering expedition.",
                    "https://example.com/mountain.jpg",
                ),
            ]
        return [
            SearchNote(query, "Zootopia", "A film.", "https://example.com/film.jpg"),
        ]

    design = research_personal_design(
        'Judy Hopps character showing slow moving with text "Get There Eventually"',
        complete=_complete_from(answers),
        search=search,
        seed=1,
        open_page=lambda title: (_ for _ in ()).throw(AssertionError(title)),
    )

    assert design.customer_lettering is True
    assert design.lettering == "Get There Eventually"
    assert "Judy Hopps" in design.subjects
    assert "not a different character" in design.subjects
    assert "Get There Eventually" not in design.prompt
    assert "No lettering" in design.prompt
    assert queries[0] == "Judy Hopps"
    assert design.reference_urls[0] == "https://example.com/judy.jpg"
    assert "https://example.com/mountain.jpg" not in design.reference_urls
    assert "mountaineering" not in " ".join(design.facts)


def test_duckduckgo_adds_a_short_note_and_a_picture() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == '"Judy Hopps"'
        return httpx.Response(
            200,
            json={
                "Heading": "Judy Hopps",
                "AbstractText": "Judy Hopps is a rabbit police officer.",
                "Image": "https://example.com/judy.png",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_duckduckgo("Judy Hopps", client=client)

    assert notes[0].title == "Judy Hopps"
    assert notes[0].image_url == "https://example.com/judy.png"


def test_commons_keeps_a_character_picture_and_drops_a_logo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["gsrnamespace"] == "6"
        assert request.url.params["gsrsearch"] == '"Judy Hopps"'
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "title": "File:Judy Hopps.png",
                            "imageinfo": [
                                {
                                    "thumburl": "https://upload.wikimedia.org/judy.png",
                                    "mime": "image/png",
                                }
                            ],
                        },
                        "2": {
                            "title": "File:Studio logo.svg",
                            "imageinfo": [
                                {
                                    "thumburl": "https://upload.wikimedia.org/logo.svg",
                                    "mime": "image/svg+xml",
                                }
                            ],
                        },
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_commons("Judy Hopps", client=client)

    assert [note.image_url for note in notes] == ["https://upload.wikimedia.org/judy.png"]


def test_personal_search_asks_every_source(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def note(source: str) -> list[SearchNote]:
        seen.append(source)
        return [SearchNote("Judy Hopps", source, f"From {source}.", None)]

    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.search_wikipedia",
        lambda query, client=None: note("wikipedia"),
    )
    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.search_duckduckgo",
        lambda query, client=None: note("duckduckgo"),
    )
    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.search_commons",
        lambda query, client=None: note("commons"),
    )
    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.search_wikidata",
        lambda query, client=None: note("wikidata"),
    )

    notes = search_personal("Judy Hopps")

    assert seen == ["wikipedia", "duckduckgo", "commons", "wikidata"]
    assert [note.title for note in notes] == ["wikipedia", "duckduckgo", "commons", "wikidata"]


def test_a_reference_download_retries_a_rate_limit_and_rejects_a_logo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.time.sleep", lambda _seconds: None
    )
    jpeg = b"\xff\xd8\xff" + b"photo"
    photo_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        photo_calls["n"] += 1
        if photo_calls["n"] == 1:
            return httpx.Response(429, content=b"slow", headers={"content-type": "text/html"})
        return httpx.Response(200, content=jpeg, headers={"content-type": "image/jpeg"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert fetch_reference("https://example.com/logo.svg", client=client) is None
        assert fetch_reference("https://example.com/park.jpg", client=client) == jpeg


def test_a_logo_answer_is_not_a_subject() -> None:
    logo = photo_look_from_answer('{"look": "a studio logo", "kind": "logo"}')
    mountain = photo_look_from_answer('{"look": "a snowy mountain", "kind": "subject"}')
    unread = photo_look_from_answer("not json")

    assert logo == PhotoLook("a studio logo", keep=False)
    assert mountain == PhotoLook("a snowy mountain", keep=True)
    assert unread == PhotoLook("", keep=True)


def test_a_vision_failure_is_an_empty_look() -> None:
    class _Completions:
        def create(self, **_kwargs: object) -> object:
            raise RuntimeError("vision is down")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    assert look_at_reference(b"img", client=_Client(), model="qwen_vl_ocr") == PhotoLook(
        "", keep=True
    )


def test_a_photo_look_fills_the_gaps_and_does_not_rename_the_mountain() -> None:
    sentence = "Animal Kingdom, Expedition Everest, Zootopia zoogether sleuth"
    answers = iter(
        [
            json.dumps({"queries": ["expedition everest"]}),
            json.dumps(
                {
                    "customer_line": sentence,
                    "voice_found": True,
                    "lettering": "Get there eventually",
                    "subjects": "a roller coaster train",
                    "additions": "a rabbit detective",
                    "references": [
                        {"url": "https://example.com/mountain.jpg", "keep": True},
                        {"url": "https://example.com/logo.jpg", "keep": True},
                    ],
                }
            ),
        ]
    )
    seen: list[list[dict[str, str]]] = []

    def complete(messages: list[dict[str, str]]) -> str:
        seen.append(messages)
        system = messages[0]["content"] if messages else ""
        if "List up to three concrete visual subjects" in system:
            return json.dumps({"subjects": []})
        return next(answers)

    def look(image: bytes) -> PhotoLook:
        if image == b"logo":
            return PhotoLook("a studio logo", keep=False)
        return PhotoLook("a snowy mountain", keep=True)

    design = research_personal_design(
        sentence,
        complete=complete,
        search=lambda _query: [
            SearchNote(
                "expedition everest",
                "Expedition Everest",
                "Expedition Everest is a steel roller coaster.",
                "https://example.com/mountain.jpg",
            ),
            SearchNote("logo", "Logo", "A studio logo.", "https://example.com/logo.jpg"),
        ],
        seed=1,
        look=look,
        fetch=lambda url: b"logo" if url.endswith("logo.jpg") else b"mountain",
        open_page=lambda title: (_ for _ in ()).throw(AssertionError(title)),
    )

    instructions = " ".join(message["content"] for messages in seen for message in messages)
    assert "Fill the gaps" in instructions
    assert "Photo shows: a snowy mountain" in instructions
    assert "a snowy mountain" in design.prompt
    assert "a rabbit detective" in design.prompt
    assert "roller" not in design.prompt
    assert "train" not in design.prompt
    assert design.lettering == "Get there eventually"
    assert design.prompt.count("Get there eventually") == 1
    assert "zoogether sleuth" not in design.prompt
    assert design.reference_urls == ("https://example.com/mountain.jpg",)


def test_deepen_searches_icons_from_place_notes_before_design() -> None:
    """Magic Kingdom notes name Mickey; deepen searches him before the design pass."""
    queries: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries.append(query)
        if query == "Magic Kingdom":
            return [
                SearchNote(
                    query,
                    "Magic Kingdom",
                    "Magic Kingdom is known for Mickey Mouse and Cinderella Castle.",
                    "https://example.com/park.jpg",
                )
            ]
        if query == "Mickey Mouse":
            return [
                SearchNote(
                    query,
                    "Mickey Mouse",
                    "Mickey Mouse is the iconic mouse mascot.",
                    "https://example.com/mickey.jpg",
                )
            ]
        return []

    design = research_personal_design(
        "Thanksgiving-themed Magic Kingdom",
        complete=_scripted_complete(
            json.dumps({"queries": ["Magic Kingdom"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "Mickey Mouse beside Cinderella Castle",
                    "references": [{"url": "https://example.com/mickey.jpg", "keep": True}],
                }
            ),
            deepen=["Mickey Mouse"],
        ),
        search=search,
        seed=1,
    )

    assert queries[0] == "Magic Kingdom"
    assert "Mickey Mouse" in queries
    assert queries.index("Magic Kingdom") < queries.index("Mickey Mouse")
    assert "Mickey Mouse" in design.subjects
    assert design.reference_urls[0] == "https://example.com/mickey.jpg"
    assert any("Mickey Mouse" in fact for fact in design.facts)


def test_a_failed_look_stays_on_the_article() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["expedition everest"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a roller coaster train",
                    "references": [{"url": "https://example.com/mountain.jpg", "keep": True}],
                }
            ),
        ]
    )

    design = research_personal_design(
        "Expedition Everest",
        complete=_complete_from(answers),
        search=lambda _query: [
            SearchNote(
                "expedition everest",
                "Expedition Everest",
                "Expedition Everest is a steel roller coaster.",
                "https://example.com/mountain.jpg",
            )
        ],
        seed=1,
        look=lambda _image: PhotoLook("", keep=True),
        fetch=lambda _url: b"mountain",
    )

    assert "roller coaster train" in design.prompt
    assert "steel roller coaster" in design.prompt


# ---------------------------------------------------------------------------
# Wikidata search (issue #344)
# ---------------------------------------------------------------------------


def _wikidata_handler(
    *,
    qid: str = "Q676555",
    label: str = "Francis of Assisi",
    description: str = "Italian Catholic saint and friar",
    p18_filename: str | None = "StFrancis_part.jpg",
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        action = request.url.params.get("action")
        if action == "wbsearchentities":
            item: dict = {"id": qid, "label": label, "description": description}
            return httpx.Response(200, json={"search": [item]})
        if action == "wbgetentities":
            claims: dict = {}
            if p18_filename:
                claims["P18"] = [{"mainsnak": {"datavalue": {"value": p18_filename}}}]
            return httpx.Response(200, json={"entities": {qid: {"claims": claims}}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_wikidata_returns_label_description_and_raster_image() -> None:
    with httpx.Client(transport=_wikidata_handler()) as client:
        notes = search_wikidata("Francis of Assisi", client=client)

    assert len(notes) == 1
    assert notes[0].title == "Francis of Assisi"
    assert notes[0].text == "Italian Catholic saint and friar"
    assert notes[0].image_url is not None
    assert "StFrancis_part.jpg" in (notes[0].image_url or "")


def test_wikidata_skips_svg_logo_image() -> None:
    with httpx.Client(transport=_wikidata_handler(p18_filename="GreenBayPackers.svg")) as client:
        notes = search_wikidata("Green Bay Packers", client=client)

    assert len(notes) == 1
    assert notes[0].image_url is None


def test_wikidata_returns_description_only_when_no_p18() -> None:
    with httpx.Client(transport=_wikidata_handler(p18_filename=None)) as client:
        notes = search_wikidata("Green Bay Packers", client=client)

    assert len(notes) == 1
    assert notes[0].text == "Italian Catholic saint and friar"
    assert notes[0].image_url is None


def test_wikidata_empty_when_search_returns_no_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"search": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikidata("xyzzy not a thing", client=client)

    assert notes == []


def test_wikidata_empty_when_items_lack_description() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        action = request.url.params.get("action")
        if action == "wbsearchentities":
            return httpx.Response(200, json={"search": [{"id": "Q1", "label": "Thing"}]})
        return httpx.Response(200, json={"entities": {"Q1": {"claims": {}}}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikidata("Thing", client=client)

    assert notes == []


def test_wikidata_empty_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikidata("anything", client=client)

    assert notes == []


# ---------------------------------------------------------------------------
# Seasonal/holiday context preservation (issue #343)
# ---------------------------------------------------------------------------


def test_thanksgiving_holiday_survives_into_prompt_when_llm_drops_it() -> None:
    """When the LLM subjects omit a holiday the customer named, the prompt still contains it."""
    design = research_personal_design(
        "Thanksgiving in Disney's Magic Kingdom",
        complete=_scripted_complete(
            json.dumps({"queries": ["Magic Kingdom"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "Cinderella Castle beside a balloon",
                    "additions": None,
                    "references": [],
                }
            ),
        ),
        search=lambda _query: [
            SearchNote(
                "Magic Kingdom",
                "Magic Kingdom",
                "Magic Kingdom is a Disney theme park featuring Cinderella Castle.",
                None,
            )
        ],
        seed=0,
    )

    assert "Thanksgiving" in design.prompt or "thanksgiving" in design.prompt.lower()


def test_christmas_holiday_survives_into_prompt_when_llm_drops_it() -> None:
    design = research_personal_design(
        "Christmas at Rockefeller Center",
        complete=_scripted_complete(
            json.dumps({"queries": ["Rockefeller Center"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a tall tower in Manhattan",
                    "additions": None,
                    "references": [],
                }
            ),
        ),
        search=lambda _query: [
            SearchNote(
                "Rockefeller Center",
                "Rockefeller Center",
                "Rockefeller Center is a complex of buildings in midtown Manhattan.",
                None,
            )
        ],
        seed=0,
    )

    assert "Christmas" in design.prompt or "christmas" in design.prompt.lower()


def test_no_duplicate_holiday_injection_when_subjects_already_name_it() -> None:
    """When subjects already mention Thanksgiving, the fallback phrase is not injected."""
    design = research_personal_design(
        "Thanksgiving turkey design",
        complete=_scripted_complete(
            json.dumps({"queries": ["Thanksgiving turkey"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a Thanksgiving turkey with autumn leaves",
                    "additions": None,
                    "references": [],
                }
            ),
        ),
        search=lambda _query: [
            SearchNote(
                "Thanksgiving turkey",
                "Turkey",
                "A turkey is a traditional Thanksgiving dish.",
                None,
            )
        ],
        seed=0,
    )

    assert "The scene is Thanksgiving-themed" not in design.subjects


# ---------------------------------------------------------------------------
# 429 retry on search sources (issue #348 or similar)
# ---------------------------------------------------------------------------


def test_wikipedia_retries_on_429_and_returns_notes() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "title": "Thanksgiving",
                            "extract": "Thanksgiving is an annual holiday.",
                            "thumbnail": {"source": "https://example.com/turkey.jpg"},
                        }
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikipedia("Thanksgiving", client=client)

    assert len(calls) == 2
    assert notes[0].title == "Thanksgiving"


def test_wikipedia_returns_empty_after_all_retries_exhausted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikipedia("Thanksgiving", client=client)

    assert notes == []


def test_commons_retries_on_429_and_returns_notes() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "title": "File:Thanksgiving.jpg",
                            "imageinfo": [{"thumburl": "https://upload.wikimedia.org/turkey.jpg"}],
                        }
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_commons("Thanksgiving", client=client)

    assert len(calls) == 2
    assert notes[0].image_url == "https://upload.wikimedia.org/turkey.jpg"


def test_wikidata_retries_on_429_and_returns_notes() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        action = request.url.params.get("action")
        if action == "wbsearchentities":
            if len(calls) == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(
                200,
                json={
                    "search": [
                        {
                            "id": "Q4166",
                            "label": "Thanksgiving",
                            "description": "annual national holiday in the United States",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"entities": {"Q4166": {"claims": {}}}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikidata("Thanksgiving", client=client)

    assert notes[0].title == "Thanksgiving"
    assert notes[0].text == "annual national holiday in the United States"


# ---------------------------------------------------------------------------
# Multi-element scenes: all named things must appear (issue #349-ish)
# ---------------------------------------------------------------------------


def test_named_element_missing_from_llm_subjects_is_injected_into_prompt() -> None:
    """When the LLM omits a searched element from subjects, it is added back as a scene note."""
    queries_seen: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries_seen.append(query)
        note_map = {
            "Mickey Mouse": SearchNote(
                "Mickey Mouse", "Mickey Mouse", "Mickey is a cartoon mouse.", None
            ),
            "Tron": SearchNote("Tron", "Tron", "Tron is a sci-fi Disney film.", None),
            "Space Mountain": SearchNote(
                "Space Mountain", "Space Mountain", "Space Mountain is a roller coaster.", None
            ),
        }
        return [note_map[query]] if query in note_map else []

    design = research_personal_design(
        "Mickey Mouse, tron, space mountain",
        complete=_scripted_complete(
            json.dumps({"queries": ["Mickey Mouse", "Tron", "Space Mountain"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "Mickey Mouse waving cheerfully",
                    "additions": None,
                    "references": [],
                }
            ),
        ),
        search=search,
        seed=0,
    )

    subjects_lower = design.subjects.lower()
    assert "tron" in subjects_lower
    assert "space mountain" in subjects_lower


def test_all_named_elements_appear_in_prompt_when_llm_includes_them() -> None:
    """When the LLM correctly includes all elements, no duplicate injection happens."""
    design = research_personal_design(
        "Mickey Mouse, tron, space mountain",
        complete=_scripted_complete(
            json.dumps({"queries": ["Mickey Mouse", "Tron", "Space Mountain"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": ("Mickey Mouse in Tron costume speeding away from Space Mountain"),
                    "additions": None,
                    "references": [],
                }
            ),
        ),
        search=lambda q: [SearchNote(q, q, f"{q} is a thing.", None)],
        seed=0,
    )

    assert "weave in" not in design.subjects.lower()
    assert "mickey mouse" in design.subjects.lower()
    assert "tron" in design.subjects.lower()
    assert "space mountain" in design.subjects.lower()


def test_max_queries_allows_more_than_four_searches() -> None:
    """With six named things, all six should be searched (MAX_QUERIES >= 6)."""
    from store_listing.orchestration.personal_design import MAX_QUERIES

    assert MAX_QUERIES >= 6


# ---------------------------------------------------------------------------
# Lock all customer-named query subjects, not just deepen icons (issue #352)
# ---------------------------------------------------------------------------


def test_all_named_queries_appear_in_lock_instruction() -> None:
    """When customer names Mickey Mouse, Tron, and Space Mountain, all three are locked."""
    design = research_personal_design(
        "Disney's Magic Kingdom - Mickey Mouse, Tron, Space Mountain",
        complete=_scripted_complete(
            json.dumps({"queries": ["Mickey Mouse", "Tron", "Space Mountain", "Magic Kingdom"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    # LLM adds Cinderella Castle which triggers _guessed_names fallback
                    "subjects": "Mickey Mouse at Cinderella Castle with Space Mountain behind",
                    "additions": None,
                    "references": [],
                }
            ),
            deepen=["Mickey Mouse"],
        ),
        search=lambda q: [SearchNote(q, q, f"{q} is a thing.", None)],
        seed=0,
    )

    subjects_lower = design.subjects.lower()
    assert "mickey mouse" in subjects_lower
    assert "tron" in subjects_lower
    assert "space mountain" in subjects_lower
    # All three should appear in the lock/draw instruction, not just Mickey Mouse
    assert subjects_lower.count("draw") == 1
    assert "mickey mouse" in subjects_lower[subjects_lower.index("draw") :]
    assert "tron" in subjects_lower[subjects_lower.index("draw") :]
    assert "space mountain" in subjects_lower[subjects_lower.index("draw") :]


def test_guessed_names_fallback_still_locks_all_customer_named_elements() -> None:
    """When _guessed_names fires, the lock still covers all customer-named subjects."""
    design = research_personal_design(
        "Mickey Mouse, Tron",
        complete=_scripted_complete(
            json.dumps({"queries": ["Mickey Mouse", "Tron"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    # Cinderella Castle is not in the customer sentence → guessed
                    "subjects": "Mickey Mouse and Cinderella Castle",
                    "additions": None,
                    "references": [],
                }
            ),
        ),
        search=lambda q: [SearchNote(q, q, f"{q} is a thing.", None)],
        seed=0,
    )

    # _guessed_names triggers because Cinderella Castle is not in the sentence.
    # The lock draw instruction must cover both Mickey Mouse AND Tron.
    subjects_lower = design.subjects.lower()
    draw_portion = subjects_lower[subjects_lower.index("draw") :]
    assert "mickey mouse" in draw_portion
    assert "tron" in draw_portion
