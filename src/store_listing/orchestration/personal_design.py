"""Research a personal custom sentence into one DTF shirt graphic.

The customer's sentence stays the request. A search supplies what the named
things look like and how a subject speaks. The lettering rules below decide
whether one line is printed. FLUX.2 dev draws the graphic; this module does
not call it.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx

from store_listing.orchestration.json_response import extract_json_objects

logger = logging.getLogger(__name__)

MAX_REFERENCES: int = 6
MAX_QUERIES: int = 6
_USER_AGENT = "store-listing-generator/1.0 (personal design research)"
_QUOTED = re.compile(r"“([^”\n]{1,120})”|\"([^\"\n]{1,120})\"")
_NAME = r"[A-Z][\w'.-]*"
_NAME_SEQ = rf"{_NAME}(?:\s+{_NAME}){{0,3}}"

_ARRANGEMENTS: tuple[str, ...] = (
    "The subject is large on the left and the other objects are smaller beside it.",
    "The subject is centered and the other objects sit above and to one side.",
    "The subject is on the right and the other objects are clustered to the left.",
)

_QUERY_SYSTEM = (
    "You choose web searches for one shirt design. The customer's sentence is the "
    'request. Return JSON only: {"queries": ["..."]}. Search for the places, '
    "rides, shows, and objects, and for how a named speaker talks. Give each "
    "place, ride, show, and character its own query, and search a character by "
    "that character's name alone. A named place such as Magic Kingdom must be "
    "searched by that place name so the article can name the icons that belong "
    "there. A place or a ride is not a movie. Do not merge two names into one "
    "search. At most four queries. Do not guess a name. Search the role the "
    "customer wrote, and do not search a character you are naming from memory. "
    "If the sentence names a holiday or seasonal event such as Thanksgiving or "
    "Christmas, give that holiday its own search query so its imagery appears "
    "in the notes."
)

_DESIGN_SYSTEM = (
    "You write a creative brief for a DTF shirt illustrator. Return JSON only with "
    "customer_line (the phrase inside double quotes, copied exactly, or null; "
    "a list of names is not a line), "
    "voice_found (true only when the notes show how the subject speaks), "
    "lettering (one short line that completes the idea in that voice, or null), "
    "subjects (a creative brief in flat 2D cartoon vector style — bold outlines, "
    "clean color fills, no photorealism; direct the illustrator in one or two "
    "active-voice sentences: one primary character or element is the focal point "
    "anchoring the center foreground, supporting characters radiate outward and "
    "upward in a pyramid or cluster composition; name what each element is doing "
    "and how they interact; weave any holiday or seasonal theme into costume or "
    "props rather than adding it on top; write as a director not a photo caption; "
    "do not include phrases like 'the photos only show' or 'as the notes indicate'; "
    "no sky, no ground, no crowd, no backdrop — isolated elements on a flat field; "
    "keep a character the customer named in those exact words), "
    "additions (what the sentence asks for that the notes do not cover, or null), "
    'and references (up to six objects {"url", "keep"}). '
    "Fill the gaps. A role such as sleuth is a character in the picture, not a "
    "word on the shirt. Do not invent a character the notes do not name. When "
    "the notes name a character that belongs to a place the customer named, "
    "that character may be the shirt subject. If the notes do not name one "
    "character, keep the customer's words and do not substitute a character "
    "from memory. A logo, a poster, a map, a costume crowd, a tight crop, or a "
    "photo full of unrelated people is keep false. "
    "Do not invent a line when voice_found is false. Do not print the customer's "
    "sentence. Do not replace words the customer wrote when they asked for those "
    "words on the shirt."
)

_LOOK_PROMPT = (
    "Describe this one reference photo for a shirt graphic. Return JSON only: "
    '{"look": "the main thing in the frame, then the rest", '
    '"kind": "subject" or "logo" or "poster" or "crowd" or "unrelated"}. '
    "kind is subject when one character, animal, or object is the clear focus — "
    "including movie or animation scenes where one character is prominent "
    "(cinematic bars and CGI quality do not change this); "
    "kind is logo for a logo or wordmark; "
    "kind is poster for a promotional poster, advertisement, or title card with text; "
    "kind is crowd when no single subject stands out; "
    "kind is unrelated when the image has nothing to do with the query."
)

_PAGE_SYSTEM = (
    "Choose the Wikipedia article to open. The title must be a film, show, place, "
    "or ride named in the customer's sentence. Do not invent a name. "
    'Return JSON only: {"title": "article title"}.'
)

_SECTION_SYSTEM = (
    "The customer described one character in a film. Choose the section that lists "
    "the characters and says what each one is. Copy the heading exactly. "
    'Return JSON only: {"section": "heading"}.'
)

_CHARACTER_SYSTEM = (
    "Identify the one character the customer described, using only this section. "
    'Return JSON only: {"name": "...", "description": "...", "enough": true} '
    'or {"enough": false, "section": "a different exact heading"} '
    "if this section does not contain them."
)

_PICTURE_SYSTEM = (
    "You check search notes for pictures of one character. "
    'Return JSON only: {"accept": true} '
    'or {"accept": false, "query": "one better search"}. '
    "Accept only when the notes are about that character. "
    "A page that merely shares one word is not it. "
    "When you reject, query is one new search for that character."
)

_DEEPEN_SYSTEM = (
    "You read search notes about one shirt design request. Return JSON only: "
    '{"subjects": ["Name", ...]}. List up to three concrete visual subjects the '
    "notes say a named place, park, show, or work is known for. Prefer the most "
    "iconic character when the customer named a park (for example Mickey Mouse "
    "for Magic Kingdom when the notes support that). Copy names from the notes "
    "only. Do not invent a name the notes do not contain. An empty list is allowed."
)

MAX_DEEPEN_SUBJECTS: int = 3


@dataclass(frozen=True)
class PhotoLook:
    """What one reference photo actually shows."""

    description: str
    keep: bool = True


@dataclass(frozen=True)
class SearchNote:
    """One visual note from a search."""

    query: str
    title: str
    text: str
    image_url: str | None = None
    look: str | None = None
    look_keep: bool = True


@dataclass(frozen=True)
class ArticleSection:
    """One heading in a Wikipedia article."""

    heading: str
    index: str


@dataclass(frozen=True)
class ArticleOutline:
    """A Wikipedia page and the headings the model may choose."""

    title: str
    sections: tuple[ArticleSection, ...]


@dataclass(frozen=True)
class PersonalDesign:
    """The graphic a personal sentence became."""

    prompt: str
    lettering: str | None
    reference_urls: tuple[str, ...]
    facts: tuple[str, ...]
    subjects: str
    customer_lettering: bool = False
    search_query_count: int = 0
    note_count: int = 0
    search_errors: tuple[str, ...] = ()


def research_has_sources(design: PersonalDesign) -> bool:
    """Whether research found facts or reference photos to anchor the graphic."""
    return bool(design.facts) or bool(design.reference_urls)


def resolve_lettering(
    *,
    customer_line: str | None,
    proposed_line: str | None,
    voice_found: bool,
) -> str | None:
    """The one line for the shirt, or none.

    Words the customer already wrote stay as written. A new line is allowed
    only when the search showed how the subject speaks.
    """
    written = (customer_line or "").strip()
    if written:
        return written
    if not voice_found:
        return None
    proposed = (proposed_line or "").strip()
    return proposed or None


def quoted_lettering(sentence: str) -> str | None:
    """The shirt line the customer put in double quotes, copied exactly."""
    lines: list[str] = []
    for match in _QUOTED.finditer(sentence or ""):
        text = next(group for group in match.groups() if group)
        cleaned = " ".join(text.split())
        if cleaned and cleaned not in lines:
            lines.append(cleaned)
    if not lines:
        return None
    return " ".join(lines)


def _without_quotes(sentence: str) -> str:
    """The sentence with the shirt line taken out of the picture description."""
    cleaned = _QUOTED.sub(" ", sentence or "")
    cleaned = re.sub(r"\bwith text\b", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _names_in(fragment: str) -> list[str]:
    """Title-case names at the start of a fragment, split on and/or."""
    names: list[str] = []
    for part in re.split(r"\s+(?:and|or)\s+|,\s*|&\s*", fragment):
        match = re.match(rf"\s*({_NAME_SEQ})\b", part)
        if match is None:
            break
        name = " ".join(match.group(1).split())
        if name:
            names.append(name)
    return names


def specified_characters(sentence: str) -> tuple[str, ...]:
    """Characters the phrase names, not a generic 'characters' mention.

    ``Judy Hopps character`` and ``the character Judy Hopps`` name one
    character. ``Zootopia characters`` does not.
    """
    bare = _without_quotes(sentence)
    found: list[str] = []
    for match in re.finditer(
        r"\b(characters?)\b\s+(?:named\s+|called\s+)?",
        bare,
        flags=re.IGNORECASE,
    ):
        names = _names_in(bare[match.end() :])
        if match.group(1).lower() == "character":
            names = names[:1]
        found.extend(names)
    for match in re.finditer(r"\bcharacter\b", bare, flags=re.IGNORECASE):
        words = re.findall(r"[A-Za-z][\w'.-]*", bare[: match.start()])
        picked: list[str] = []
        for word in reversed(words):
            if word[:1].isupper():
                picked.append(word)
            else:
                break
        if picked:
            found.append(" ".join(reversed(picked)))
    unique: list[str] = []
    seen = {name.lower() for name in unique}
    for name in found:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return tuple(unique)


_ROLE_STOP = frozenset(
    {"a", "an", "the", "showing", "with", "text", "and", "or", "on", "in", "of", "for"}
)
_TITLE_FILLER = _ROLE_STOP | frozenset(
    {"list", "film", "movie", "series", "character", "characters"}
)


def character_role(sentence: str) -> str | None:
    """A singular character described by a role, ready to be searched.

    ``Zootopia sleuth character`` is a search. ``Judy Hopps character`` is
    already a name, and ``Zootopia characters`` is not one character.
    """
    if specified_characters(sentence):
        return None
    bare = _without_quotes(sentence)
    match = re.search(r"\bcharacter\b", bare, flags=re.IGNORECASE)
    if match is None or re.match(r"characters\b", bare[match.start() :], flags=re.IGNORECASE):
        return None
    words = re.findall(r"[A-Za-z][\w'.-]*", bare[: match.start()])
    picked: list[str] = []
    for word in reversed(words):
        if word.lower() in _ROLE_STOP:
            break
        picked.append(word)
        if len(picked) == 5:
            break
    if not picked or all(word[:1].isupper() for word in picked):
        return None
    return f"{' '.join(reversed(picked))} character"


def _proper_names(text: str, query_words: set[str]) -> list[str]:
    names: list[str] = []
    for match in re.finditer(rf"\b({_NAME_SEQ})\b", text or ""):
        kept = [
            word
            for word in match.group(1).split()
            if word.lower() not in query_words and word.lower() not in _TITLE_FILLER
        ]
        name = " ".join(kept)
        if name and name.lower() not in {item.lower() for item in names}:
            names.append(name)
    return names


def character_from_notes(query: str, notes: Sequence[SearchNote]) -> str | None:
    """The one character a role search names, or none when the notes disagree.

    A page titled Flash Slothmore is that character. A film page that lists
    Judy Hopps, Nick Wilde and Flash Slothmore is not a decision.
    """
    query_words = set(_title_text(query).split())
    titled: list[str] = []
    for note in notes:
        for name in _proper_names(note.title, query_words):
            if name.lower() not in {item.lower() for item in titled}:
                titled.append(name)
    if len(titled) == 1:
        return titled[0]
    if titled:
        return None
    mentioned: list[str] = []
    for note in notes:
        for name in _proper_names(note.text, query_words):
            if name.lower() not in {item.lower() for item in mentioned}:
                mentioned.append(name)
    if len(mentioned) == 1:
        return mentioned[0]
    return None


def _known_names(sentence: str, characters: Sequence[str]) -> set[str]:
    """Names the customer wrote, plus a name the notes already resolved."""
    known = {name.lower() for name in characters}
    known.update(name.lower() for name in _proper_names(sentence, set()))
    return known


def _guessed_names(text: str, known: set[str], sentence: str) -> list[str]:
    """Proper names in ``text`` that neither the customer nor the notes supplied.

    A name the customer wrote in lower case, such as ``jar jar``, is still
    the customer's name.
    """
    folded = sentence.lower()
    return [
        name
        for name in _proper_names(text, set())
        if " " in name and name.lower() not in known and name.lower() not in folded
    ]


def _facts_without_guesses(facts: Sequence[str], known: set[str], sentence: str) -> tuple[str, ...]:
    """Facts that do not hand the picture a character nobody resolved."""
    return tuple(fact for fact in facts if not _guessed_names(fact, known, sentence))


def lock_characters(subjects: str, characters: Sequence[str]) -> str:
    """The picture description, with every named character still in it."""
    text = " ".join((subjects or "").split())
    missing = [name for name in characters if name.lower() not in text.lower()]
    if missing:
        text = f"{text} The character is {', '.join(missing)}.".strip()
    if characters:
        names = ", ".join(characters)
        text = (
            f"{text} Draw {names} as that character, "
            "not a different character and not a generic substitute."
        )
    return " ".join(text.split())


def _title_text(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def matching_notes(query: str, notes: Sequence[SearchNote]) -> list[SearchNote]:
    """Notes whose title is the thing that was searched.

    A query for one ride also returns the mountain. When any title contains
    the query, the others are a different subject and are dropped.
    """
    phrase = _title_text(query)
    exact = [note for note in notes if phrase and phrase in _title_text(note.title)]
    return exact or list(notes)


def _add_query(queries: list[str], query: str) -> None:
    cleaned = " ".join(query.split())
    if not cleaned:
        return
    if any(cleaned.lower() == existing.lower() for existing in queries):
        return
    queries.append(cleaned)


# Holiday and seasonal terms that may appear in a customer sentence.
# Used only to detect when the LLM-generated subjects dropped the event name;
# the search pipeline fetches the actual visual content, the LLM decides the design.
_HOLIDAY_TERMS: frozenset[str] = frozenset(
    {
        "thanksgiving",
        "christmas",
        "easter",
        "halloween",
        "advent",
        "lent",
        "valentine",
        "st. patrick",
        "st patrick",
        "independence day",
        "fourth of july",
        "new year",
        "pentecost",
        "epiphany",
        "all saints",
        "all souls",
        "corpus christi",
        "good friday",
        "ash wednesday",
    }
)


def _missing_holidays(sentence: str, subjects: str) -> list[str]:
    """Holiday names the customer wrote that the LLM-generated subjects omit."""
    lower_sentence = sentence.lower()
    lower_subjects = subjects.lower()
    return [
        term for term in _HOLIDAY_TERMS if term in lower_sentence and term not in lower_subjects
    ]


def _missing_query_subjects(
    queries: Sequence[str],
    subjects: str,
    characters: Sequence[str],
    sentence: str,
) -> list[str]:
    """Named elements from searched queries absent from both subjects and the character lock.

    Holiday terms are handled separately by _missing_holidays.
    The sentence itself (when used as a fallback query) is never a missing element.
    """
    lower_subjects = subjects.lower()
    char_keys = {c.lower() for c in characters}
    sentence_lower = sentence.lower().strip()
    missing: list[str] = []
    for query in queries:
        q_lower = query.lower().strip()
        if not q_lower or q_lower == sentence_lower:
            continue
        if any(holiday in q_lower for holiday in _HOLIDAY_TERMS):
            continue
        if q_lower in char_keys:
            continue
        if q_lower in lower_subjects:
            continue
        missing.append(query)
    return missing


def _query_lock_subjects(
    queries: Sequence[str],
    characters: Sequence[str],
    sentence: str,
) -> list[str]:
    """Customer-named proper-noun queries not yet in the character lock.

    The queries the LLM chose represent everything the customer named.
    Adding them all to the lock ensures the image model receives an explicit
    instruction to include every named subject, not just what deepen discovered.
    Holiday terms and the sentence-as-fallback query are excluded.
    """
    char_keys = {c.lower() for c in characters}
    sentence_lower = sentence.lower().strip()
    result: list[str] = []
    for query in queries:
        q = query.strip()
        q_lower = q.lower()
        if not q or q_lower == sentence_lower:
            continue
        if any(holiday in q_lower for holiday in _HOLIDAY_TERMS):
            continue
        if q_lower in char_keys:
            continue
        if any(word[:1].isupper() for word in q.split()):
            result.append(q)
    return result


def arrangement_line(seed: int) -> str:
    """One arrangement of the same facts. A new seed is a new arrangement."""
    return _ARRANGEMENTS[seed % len(_ARRANGEMENTS)]


def graphic_prompt(
    *,
    subjects: str,
    lettering: str | None,
    arrangement: str,
    facts: Sequence[str] = (),
    leave_text_blank: bool = False,
    style: str = "",
) -> str:
    """The ComfyUI prompt: a flat magenta key, the subjects, and one line."""
    fact_text = " ".join(fact.strip() for fact in facts if fact and fact.strip())
    known = f"Known facts: {fact_text} " if fact_text else ""
    look = f"{style.strip()} " if style.strip() else ""
    if lettering:
        line = f' One line of lettering, spelled exactly: "{lettering}". No other text.'
    elif leave_text_blank:
        line = " No lettering and no words in the picture."
    else:
        line = ""
    placed = arrangement.strip()
    if placed and not placed.endswith("."):
        placed = f"{placed}."
    return (
        "Shirt print artwork. A solid pure magenta field fills every pixel that is "
        "not the subject, one flat colour edge to edge. No street, no sky, no ground, "
        "no buildings, no crowd, no place photograph, no frame, no backdrop. "
        f"{known}{look}{subjects.strip()} {placed}{line}"
    )


def rearrange(design: PersonalDesign, seed: int) -> PersonalDesign:
    """The same facts and line in a different arrangement."""
    prompt = graphic_prompt(
        subjects=design.subjects,
        lettering=None if design.customer_lettering else design.lettering,
        arrangement=arrangement_line(seed),
        facts=design.facts,
        leave_text_blank=design.customer_lettering,
    )
    return replace(design, prompt=prompt)


def _json_string(parsed: Any, key: str) -> str:
    if not isinstance(parsed, dict):
        return ""
    value = parsed.get(key)
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _section_named(outline: ArticleOutline, heading: str) -> ArticleSection | None:
    for section in outline.sections:
        if section.heading.lower() == heading.lower():
            return section
    return None


def _read_section_text(read_page: Callable[[str, str], str], title: str, index: str) -> str:
    try:
        return read_page(title, index)
    except Exception:  # noqa: BLE001
        logger.warning("The article section could not be read")
        return ""


def _choose_section(
    complete: Callable[[list[dict[str, str]]], str],
    sentence: str,
    outline: ArticleOutline,
) -> ArticleSection | None:
    listing = "\n".join(f"- {section.heading}" for section in outline.sections)
    raw = complete(
        [
            {"role": "system", "content": _SECTION_SYSTEM},
            {"role": "user", "content": f"Sentence: {sentence}\n\nHeadings:\n{listing}"},
        ]
    )
    return _section_named(outline, _json_string(_first_object(raw), "section"))


def _picture_check(raw: str) -> tuple[bool, str | None]:
    """Whether the notes are about the character, and a replacement search."""
    parsed = _first_object(raw)
    if not isinstance(parsed, dict) or "accept" not in parsed:
        return True, None
    if parsed.get("accept") is True:
        return True, None
    query = _json_string(parsed, "query")
    return False, query or None


def _ask_picture_check(
    complete: Callable[[list[dict[str, str]]], str],
    character: str,
    query: str,
    notes: Sequence[SearchNote],
) -> tuple[bool, str | None]:
    listed = "\n".join(_note_line(note) for note in notes)
    raw = complete(
        [
            {"role": "system", "content": _PICTURE_SYSTEM},
            {
                "role": "user",
                "content": f"Character: {character}\nQuery: {query}\n\nNotes:\n{listed}",
            },
        ]
    )
    return _picture_check(raw)


def _read_character(
    complete: Callable[[list[dict[str, str]]], str],
    sentence: str,
    heading: str,
    text: str,
) -> tuple[str | None, str | None, str | None]:
    """The character, a description, and another heading when this section is not enough."""
    raw = complete(
        [
            {"role": "system", "content": _CHARACTER_SYSTEM},
            {
                "role": "user",
                "content": f"Sentence: {sentence}\n\nSection: {heading}\n\n{text[:14000]}",
            },
        ]
    )
    parsed = _first_object(raw)
    if not isinstance(parsed, dict):
        return None, None, None
    if parsed.get("enough") is False:
        retry = _json_string(parsed, "section")
        return None, None, retry or None
    name = _json_string(parsed, "name")
    if not name or name.lower() == "null":
        return None, None, None
    return name, _json_string(parsed, "description"), None


def _character_from_work(
    sentence: str,
    complete: Callable[[list[dict[str, str]]], str],
    open_page: Callable[[str], ArticleOutline | None],
    read_page: Callable[[str, str], str],
) -> tuple[str | None, str | None]:
    """The character the model finds in the work the customer named.

    A title that is not a Wikipedia page is sent back once. A section that
    does not contain the character is replaced once. The role words are not
    a search.
    """
    note = "No page has been opened yet."
    outline: ArticleOutline | None = None
    for _attempt in range(2):
        raw = complete(
            [
                {"role": "system", "content": _PAGE_SYSTEM},
                {"role": "user", "content": f"Sentence: {sentence}\n\n{note}"},
            ]
        )
        title = _json_string(_first_object(raw), "title")
        if not title:
            return None, None
        try:
            outline = open_page(title)
        except Exception:  # noqa: BLE001
            logger.warning("The article could not be opened for %r", title)
            outline = None
        if outline is not None and outline.sections:
            break
        note = f"No Wikipedia page is titled {title!r}. Choose a different name from the sentence."
        outline = None
    if outline is None:
        return None, None

    heading = _choose_section(complete, sentence, outline)
    if heading is None:
        return None, None
    text = _read_section_text(read_page, outline.title, heading.index)
    name, description, retry = _read_character(complete, sentence, heading.heading, text)
    if name:
        return name, description
    if not retry or retry.lower() == heading.heading.lower():
        return None, None
    other = _section_named(outline, retry)
    if other is None:
        return None, None
    text = _read_section_text(read_page, outline.title, other.index)
    name, description, _again = _read_character(complete, sentence, other.heading, text)
    if not name:
        return None, None
    return name, description


def research_personal_design(
    sentence: str,
    *,
    complete: Callable[[list[dict[str, str]]], str],
    search: Callable[[str], list[SearchNote]],
    seed: int,
    look: Callable[[bytes], PhotoLook] | None = None,
    fetch: Callable[[str], bytes | None] | None = None,
    open_page: Callable[[str], ArticleOutline | None] | None = None,
    read_page: Callable[[str, str], str] | None = None,
) -> PersonalDesign:
    """Search the sentence, look at the photos, and turn them into one graphic.

    ``complete`` is one Qwen call. ``search`` is one query. ``look`` describes
    one downloaded photo. ``open_page`` and ``read_page`` read the work the
    customer named. All of them are injected so a test never touches the
    network. A missing look keeps the text-only path.
    """
    named = list(specified_characters(sentence))
    role = character_role(sentence)
    quotes = quoted_lettering(sentence)
    notes: list[SearchNote] = []
    character_notes: list[SearchNote] = []
    seen: list[SearchNote] = []
    search_errors: list[str] = []

    def _accepted(query: str, found: list[SearchNote]) -> list[SearchNote]:
        """Notes about this character. A different character is searched once more."""
        if resolved is None or not found:
            return found
        accept, replacement = _ask_picture_check(complete, resolved, query, found)
        if accept:
            return found
        if not replacement or replacement.lower() == query.lower():
            return []
        try:
            raw = search(replacement)
        except Exception:  # noqa: BLE001
            logger.error("Personal design search failed for %r", replacement)
            search_errors.append(f"search failed for {replacement!r}")
            raw = []
        retried = matching_notes(replacement, raw)
        if not retried:
            return []
        accept_again, _again = _ask_picture_check(complete, resolved, replacement, retried)
        return retried if accept_again else []

    def _found(query: str) -> list[SearchNote]:
        try:
            raw = search(query)
        except Exception:  # noqa: BLE001
            logger.error("Personal design search failed for %r", query)
            search_errors.append(f"search failed for {query!r}")
            raw = []
        if not raw:
            logger.error("Personal design search returned no notes for %r", query)
            search_errors.append(f"no notes for {query!r}")
        matched = matching_notes(query, raw)
        if resolved and query.lower() in character_keys:
            # A rejected picture never becomes a reference. Other searches still
            # show every result to the photo check, including a logo.
            kept = _accepted(query, matched)
            seen.extend(kept)
            return kept
        seen.extend(raw)
        return matched

    resolved: str | None = None
    if role:
        # The role words are not a search. Separate words match a different work.
        resolved, description = _character_from_work(
            sentence,
            complete,
            open_page or open_article,
            read_page or read_section,
        )
        if resolved and description:
            character_notes.append(SearchNote(query=resolved, title=resolved, text=description))
            notes.extend(character_notes)
    characters = list(named)
    if resolved is not None and resolved.lower() not in {name.lower() for name in characters}:
        characters.insert(0, resolved)
    known = _known_names(sentence, characters)
    queries: list[str] = []
    for name in characters:
        _add_query(queries, name)
    for query in _queries(complete, sentence):
        if len(queries) >= MAX_QUERIES:
            break
        cleaned = query.strip().lower()
        if role and cleaned in {role.lower(), sentence.strip().lower()}:
            continue
        if _guessed_names(query, known, sentence):
            continue
        _add_query(queries, query)
    # Guessed LLM queries can all be filtered out (brief 344). Always search the
    # customer's sentence so a named place such as Magic Kingdom still has notes.
    if not queries and sentence.strip():
        _add_query(queries, sentence.strip())

    character_keys = {name.lower() for name in characters}
    # Knowledge pass: text extracts first. Reference photos are fetched after
    # the deepen step names the icons those notes support (Magic Kingdom → Mickey).
    for query in queries:
        found = _found(query)
        if query.lower() in character_keys:
            character_notes.extend(found)
        notes.extend(found)

    icons = _deepen_subjects(complete, sentence, notes)
    for icon in icons:
        if icon.lower() not in {name.lower() for name in characters}:
            characters.append(icon)
    known = _known_names(sentence, characters)
    character_keys = {name.lower() for name in characters}
    for icon in icons[:MAX_DEEPEN_SUBJECTS]:
        if any(icon.lower() == existing.lower() for existing in queries):
            continue
        found = _found(icon)
        _add_query(queries, icon)
        if icon.lower() in character_keys:
            character_notes.extend(found)
        notes.extend(found)

    # Lock every customer-named proper-noun query, not just what deepen discovered.
    # The LLM chose the queries because the customer named those things; all of them
    # belong in lock_characters so the image model receives an explicit instruction
    # for each one. Notes already fetched for these queries are promoted to
    # character_notes so their reference images are prioritised.
    for extra in _query_lock_subjects(queries, characters, sentence):
        characters.append(extra)
        char_key = extra.lower()
        for note in notes:
            if note.query.lower() == char_key and note.image_url and note not in character_notes:
                character_notes.append(note)
    character_keys = {name.lower() for name in characters}
    known = _known_names(sentence, characters)

    if look is not None:
        seen = _attach_looks(seen, look, fetch or fetch_reference)
        by_url = {note.image_url: note for note in seen if note.image_url}

        def _with_look(note: SearchNote) -> SearchNote:
            if note.image_url and note.image_url in by_url:
                return by_url[note.image_url]
            return note

        notes = [_with_look(note) for note in notes]
        character_notes = [_with_look(note) for note in character_notes]
    reference_notes = seen if look is not None else notes

    facts = _facts_without_guesses(
        tuple(_fact(note) for note in notes if _fact(note)),
        known,
        sentence,
    )
    design = _read_design(complete, sentence, notes)
    picture = _without_quotes(sentence).strip() or sentence.strip()
    stats = {
        "search_query_count": len(queries),
        "note_count": len(notes),
        "search_errors": tuple(search_errors),
    }
    if design is None:
        subjects = lock_characters(picture, characters)
        missed = _missing_holidays(sentence, subjects)
        if missed:
            holiday_text = " and ".join(t.title() for t in missed)
            subjects = f"{subjects} The scene is {holiday_text}-themed."
        missed_elements = _missing_query_subjects(queries, subjects, characters, sentence)
        if missed_elements:
            elements_text = ", ".join(missed_elements)
            subjects = f"{subjects} Also weave in {elements_text}."
        return PersonalDesign(
            prompt=graphic_prompt(
                subjects=subjects,
                lettering=None,
                arrangement=arrangement_line(seed),
                facts=facts,
                leave_text_blank=quotes is not None,
            ),
            lettering=quotes,
            reference_urls=_merged_urls(character_notes, ()),
            facts=facts,
            subjects=subjects,
            customer_lettering=quotes is not None,
            **stats,
        )

    lettering = (
        quotes
        if quotes
        else resolve_lettering(
            customer_line=_requested_line(design["customer_line"], sentence),
            proposed_line=design["lettering"],
            voice_found=bool(design["voice_found"]),
        )
    )
    urls = _merged_urls(character_notes, _reference_urls(design["references"], reference_notes))
    subject, from_photos = _drawn_subject(
        _without_quotes(design["subjects"]).strip() or picture,
        design["additions"],
        _kept_looks(reference_notes, urls),
    )
    if _guessed_names(subject, known, sentence) and not from_photos:
        # A photo may describe what it shows. A name the notes did not return
        # is the model choosing a character, so the customer's words stay.
        subject = picture
    subject = lock_characters(subject, characters)
    # Fallback: if the LLM still dropped a holiday/season the customer named,
    # append its name so the image model knows the thematic context.
    # The search pipeline fetches the visual content; this only reinstates the name.
    missed = _missing_holidays(sentence, subject)
    if missed:
        holiday_text = " and ".join(t.title() for t in missed)
        subject = f"{subject} The scene is {holiday_text}-themed."
    # Fallback: if any named element the customer listed (and that was searched)
    # is absent from subjects, weave it back in so the image model includes it.
    missed_elements = _missing_query_subjects(queries, subject, characters, sentence)
    if missed_elements:
        elements_text = ", ".join(missed_elements)
        subject = f"{subject} Also weave in {elements_text}."
    image_facts = () if from_photos else facts
    return PersonalDesign(
        prompt=graphic_prompt(
            subjects=subject,
            lettering=None if quotes else lettering,
            arrangement=arrangement_line(seed),
            facts=image_facts,
            leave_text_blank=quotes is not None,
        ),
        lettering=lettering,
        reference_urls=urls,
        facts=image_facts,
        subjects=subject,
        customer_lettering=quotes is not None,
        **stats,
    )


def section_lines(wikitext: str) -> str:
    """The bullets of one section, including a cast list wrapped in a template.

    The cast is ``{{castlist| ... }}``. Dropping that template drops the names
    inside it. A reference and a link's markup are not part of the line.
    """
    lines: list[str] = []
    for raw in (wikitext or "").splitlines():
        if not raw.lstrip().startswith("*"):
            continue
        line = re.sub(r"<ref\b[^>]*/>", "", raw)
        line = re.sub(r"<ref\b[^>]*>.*?</ref>", "", line, flags=re.DOTALL)
        line = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", line)
        line = re.sub(r"'{2,}", "", line)
        line = re.sub(r"\*+", "", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def open_article(title: str, *, client: httpx.Client | None = None) -> ArticleOutline | None:
    """The page and its headings, or None when Wikipedia has no such page."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = http.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "redirects": "1", "titles": title, "format": "json"},
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        pages = (response.json().get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), None) if isinstance(pages, dict) else None
        if not isinstance(page, dict) or "missing" in page:
            return None
        resolved = str(page.get("title") or title).strip()
        sections_response = http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": resolved,
                "prop": "sections",
                "format": "json",
            },
            headers={"User-Agent": _USER_AGENT},
        )
        sections_response.raise_for_status()
        raw_sections = (sections_response.json().get("parse") or {}).get("sections") or []
        sections = tuple(
            ArticleSection(str(item.get("line") or "").strip(), str(item.get("index") or ""))
            for item in raw_sections
            if isinstance(item, dict) and str(item.get("line") or "").strip() and item.get("index")
        )
        if not sections:
            return None
        return ArticleOutline(resolved, sections)
    except (httpx.HTTPError, ValueError, StopIteration):
        logger.warning("Wikipedia article could not be opened for %r", title)
        return None
    finally:
        if owns_client:
            http.close()


def read_section(title: str, index: str, *, client: httpx.Client | None = None) -> str:
    """The flattened lines of one section. A failed call is an empty string."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": title,
                "prop": "wikitext",
                "section": index,
                "format": "json",
            },
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        wikitext = ((response.json().get("parse") or {}).get("wikitext") or {}).get("*")
        if not isinstance(wikitext, str):
            return ""
        return section_lines(wikitext)
    except (httpx.HTTPError, ValueError):
        logger.warning("Wikipedia section could not be read")
        return ""
    finally:
        if owns_client:
            http.close()


def bound_query(query: str) -> str:
    """Several words are one phrase. Separate words match a different page."""
    words = query.split()
    if len(words) < 2:
        return " ".join(words)
    phrase = " ".join(words)
    if phrase.startswith('"') and phrase.endswith('"'):
        return phrase
    return f'"{phrase}"'


_SEARCH_ATTEMPTS = 3


def _search_get(
    http: httpx.Client,
    url: str,
    *,
    params: dict,
    headers: dict,
) -> httpx.Response:
    """GET with backoff retry on 429 and 503. Raises HTTPError on other failures."""
    response = http.get(url, params=params, headers=headers)
    for attempt in range(1, _SEARCH_ATTEMPTS):
        if response.status_code not in (429, 503):
            break
        time.sleep(_retry_delay(response, attempt))
        response = http.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response


def search_wikipedia(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """Short visual notes from Wikipedia. A failed call is an empty list."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = _search_get(
            http,
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": bound_query(query),
                "gsrlimit": "3",
                "prop": "extracts|pageimages",
                "exintro": "1",
                "explaintext": "1",
                "exchars": "400",
                "piprop": "thumbnail",
                "pithumbsize": "1024",
                "format": "json",
            },
            headers={"User-Agent": "store-listing-generator/1.0 (personal design research)"},
        )
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        logger.warning("Wikipedia search failed for %r", query)
        return []
    finally:
        if owns_client:
            http.close()

    pages = (payload.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return []
    notes: list[SearchNote] = []
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "").strip()
        text = str(page.get("extract") or "").strip()
        original = page.get("thumbnail") or page.get("original")
        source = original.get("source") if isinstance(original, dict) else None
        image_url = source if isinstance(source, str) and _is_raster_url(source) else None
        if title and text:
            notes.append(SearchNote(query=query, title=title, text=text, image_url=image_url))
    return notes


def search_duckduckgo(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """One short note from DuckDuckGo. A failed call is an empty list."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = http.get(
            "https://api.duckduckgo.com/",
            params={
                "q": bound_query(query),
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            },
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        logger.warning("DuckDuckGo search failed for %r", query)
        return []
    finally:
        if owns_client:
            http.close()

    if not isinstance(payload, dict):
        return []
    title = str(payload.get("Heading") or "").strip()
    text = str(payload.get("AbstractText") or "").strip()
    image = payload.get("Image")
    image_url = image if isinstance(image, str) and _is_raster_url(image) else None
    if not title or not text:
        return []
    return [SearchNote(query=query, title=title, text=text, image_url=image_url)]


def search_commons(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """Character and place pictures from Wikimedia Commons."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = _search_get(
            http,
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": bound_query(query),
                "gsrnamespace": "6",
                "gsrlimit": "4",
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "iiurlwidth": "1024",
                "format": "json",
            },
            headers={"User-Agent": _USER_AGENT},
        )
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        logger.warning("Commons search failed for %r", query)
        return []
    finally:
        if owns_client:
            http.close()

    pages = (payload.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return []
    notes: list[SearchNote] = []
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "").removeprefix("File:").strip()
        info = page.get("imageinfo")
        thumb = info[0] if isinstance(info, list) and info else {}
        source = thumb.get("thumburl") if isinstance(thumb, dict) else None
        image_url = source if isinstance(source, str) and _is_raster_url(source) else None
        if title and image_url:
            notes.append(SearchNote(query=query, title=title, text=title, image_url=image_url))
    return notes


def search_wikidata(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """Entity description and portrait from Wikidata. A failed call is an empty list."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        r = _search_get(
            http,
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "limit": "3",
                "format": "json",
            },
            headers={"User-Agent": _USER_AGENT},
        )
        results = r.json().get("search") or []
        if not isinstance(results, list):
            return []
        top = next(
            (
                item
                for item in results
                if isinstance(item, dict) and item.get("label") and item.get("description")
            ),
            None,
        )
        if top is None:
            return []
        qid = str(top.get("id") or "").strip()
        label = " ".join(str(top.get("label") or "").split())
        description = " ".join(str(top.get("description") or "").split())
        if not qid or not label or not description:
            return []

        image_url: str | None = None
        try:
            r2 = _search_get(
                http,
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "claims",
                    "format": "json",
                },
                headers={"User-Agent": _USER_AGENT},
            )
            entity = (r2.json().get("entities") or {}).get(qid) or {}
            p18 = (entity.get("claims") or {}).get("P18") or []
            if isinstance(p18, list) and p18:
                filename = p18[0]["mainsnak"]["datavalue"]["value"]
                if isinstance(filename, str) and filename.strip():
                    name = filename.strip().replace(" ", "_")
                    if _is_raster_url(name):
                        image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{name}"
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            pass

        return [SearchNote(query=query, title=label, text=description, image_url=image_url)]
    except (httpx.HTTPError, ValueError):
        logger.warning("Wikidata search failed for %r", query)
        return []
    finally:
        if owns_client:
            http.close()


_GOOGLE_IMAGES_ACTOR = "bvAQMqCbp6wE53JzK"
_GOOGLE_IMAGES_RESULTS = 50  # minimum enforced by actor; we cap output to MAX_REFERENCES
_GOOGLE_IMAGES_KEEP = MAX_REFERENCES  # only keep this many to avoid flooding references
_IMAGE_TITLE_CHECK_SYSTEM = (
    "You check if an image title is relevant to a search query. "
    'Return JSON only: {"accept": true} when the title describes the same '
    "subject as the query, or "
    '{"accept": false} when it is clearly a different subject. '
    "A query for 'Tron Disney character' should accept Disney Tron art and "
    "reject a Swedish historical painting that happens to be named Tron."
)


def _image_title_accepted(
    query: str,
    title: str,
    complete: Callable[[list[dict[str, str]]], str],
) -> bool:
    """Whether the LLM thinks this image title matches the search query."""
    raw = complete(
        [
            {"role": "system", "content": _IMAGE_TITLE_CHECK_SYSTEM},
            {"role": "user", "content": f"Query: {query}\nTitle: {title}"},
        ]
    )
    for obj in _first_object(raw), *(extract_json_objects(raw)):
        if isinstance(obj, dict) and "accept" in obj:
            return bool(obj["accept"])
    return True  # default to accept when response cannot be parsed


def search_google_images(
    query: str,
    *,
    apify_client_cls: Any | None = None,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
    look: Callable[[bytes], PhotoLook] | None = None,
    fetch: Callable[[str], bytes | None] | None = None,
) -> list[SearchNote]:
    """Reference images from Google Image Search via Apify (johnvc/google-images-api).

    Fires only as a fallback when Wikimedia sources return no raster images.
    Cost: $0.0001/image × 50-image minimum = $0.005 per call.
    Returns an empty list when APIFY_API_TOKEN is not set or the call fails.

    When ``complete`` is provided each candidate title is checked with the LLM
    before the image is accepted. The function iterates through Apify results
    lazily, stopping as soon as MAX_REFERENCES good images are found.

    When ``look`` is provided each image is downloaded and checked with the
    vision LLM to verify it is a clean character-art reference (kind=subject),
    not a logo, poster, Etsy SVG, or unrelated photo. ``fetch`` defaults to
    ``fetch_reference`` when not supplied.
    """
    import os

    token = os.environ.get("APIFY_API_TOKEN", "")
    if not token:
        return []
    try:
        if apify_client_cls is not None:
            make_client = apify_client_cls
        else:
            from apify_client import ApifyClient  # type: ignore[import-not-found]

            def make_client(tok: str) -> Any:
                return ApifyClient(tok)

        client = make_client(token)
        run = client.actor(_GOOGLE_IMAGES_ACTOR).call(
            run_input={
                "queries": [query],
                "maxResultsPerQuery": _GOOGLE_IMAGES_RESULTS,
                "gl": "us",
                "hl": "en",
            },
        )
        if run is None:
            return []
        dataset_id = (
            run.get("defaultDatasetId") if isinstance(run, dict) else run.default_dataset_id
        )
        items = client.dataset(dataset_id).list_items().items
    except Exception:  # noqa: BLE001
        logger.warning("Google Images (Apify) search failed for %r", query)
        return []

    notes: list[SearchNote] = []
    for item in items or []:
        if len(notes) >= _GOOGLE_IMAGES_KEEP:
            break
        if not isinstance(item, dict):
            continue
        url = item.get("imageUrl")
        if not isinstance(url, str) or not _is_raster_url(url):
            continue
        if len(notes) >= _GOOGLE_IMAGES_KEEP:
            break
        title = str(item.get("title") or query).strip()
        if complete is not None and not _image_title_accepted(query, title, complete):
            continue
        if look is not None:
            _fetch = fetch if fetch is not None else fetch_reference
            image_data = _fetch(url)
            if image_data is None:
                continue
            try:
                photo = look(image_data)
            except Exception:  # noqa: BLE001
                logger.warning("Vision gate failed for %r", url)
                continue
            if not photo.keep:
                continue
        source = str(item.get("source") or "").strip()
        text = f"{title} ({source})" if source else title
        # Pre-fill look so _attach_looks skips the slow vision-LLM download step.
        # Google Images results are pre-validated; vision description adds no value.
        notes.append(SearchNote(query=query, title=title, text=text, image_url=url, look=title))
    return notes


_GEMINI_MODEL = "gemini-3.8-flash"
_GEMINI_PROMPT = (
    "In 2-3 sentences describe what {query} looks like visually — "
    "distinctive colours, features, costume, or setting. "
    "Focus on visual details useful for illustration. Be specific."
)
_GEMINI_ATTEMPTS = 3


def search_gemini(
    query: str,
    *,
    genai_client_cls: Any | None = None,
) -> list[SearchNote]:
    """Visually descriptive research text from Gemini with Google Search grounding.

    Returns one SearchNote with a synthesised visual description grounded in
    current web results. No image URL is returned; Gemini does not expose
    image URLs via the API. Returns an empty list when GEMINI_API_KEY is not
    set or the call fails.

    Retries up to three times on 429 rate-limit responses with backoff, since
    the pipeline fires one Gemini call per named subject in quick succession.
    Uses v1beta explicitly so Google Search grounding is available.
    """
    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []
    try:
        from google.genai import types  # type: ignore[import-not-found]

        if genai_client_cls is not None:
            make_client = genai_client_cls
        else:
            from google import genai  # type: ignore[import-not-found]

            def make_client(key: str) -> Any:
                return genai.Client(
                    api_key=key,
                    http_options={"api_version": "v1beta"},
                )

        client = make_client(api_key)
        last_exc: Exception | None = None
        for attempt in range(1, _GEMINI_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=_GEMINI_PROMPT.format(query=query),
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        max_output_tokens=1000,
                    ),
                )
                break
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                if "429" in exc_str and attempt < _GEMINI_ATTEMPTS:
                    wait = min(4.0 * attempt, 15.0)
                    time.sleep(wait)
                    continue
                raise
        else:
            raise last_exc  # type: ignore[misc]

        text = (response.text or "").strip()
        if not text:
            return []
        title = query
        try:
            chunks = response.candidates[0].grounding_metadata.grounding_chunks or []
            if chunks and chunks[0].web:
                title = str(chunks[0].web.title or query).strip() or query
        except (AttributeError, IndexError, TypeError):
            pass
        return [SearchNote(query=query, title=title, text=text, image_url=None)]
    except Exception:  # noqa: BLE001
        logger.warning("Gemini search failed for %r", query)
        return []


def search_personal(
    query: str,
    *,
    client: httpx.Client | None = None,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
    look: Callable[[bytes], PhotoLook] | None = None,
) -> list[SearchNote]:
    """Wikipedia, Commons, Wikidata, Gemini, and Google Images notes for one query.

    Gemini with Google Search grounding supplies synthesised visual descriptions.
    Google Images via Apify fires only as a fallback when Wikimedia sources return
    no raster images, keeping cost within budget (~$0.005 per fallback call).

    When ``complete`` is provided it is forwarded to search_google_images for
    LLM-based title relevance checking of Apify results.
    When ``look`` is provided it is used as a vision gate to validate that
    each Apify image is a clean character-art reference.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        notes: list[SearchNote] = []
        for search in (search_wikipedia, search_commons, search_wikidata):
            notes.extend(search(query, client=http))
        notes.extend(search_gemini(query))
        if not any(note.image_url for note in notes):
            notes.extend(search_google_images(query, complete=complete, look=look))
        return notes
    finally:
        if owns_client:
            http.close()


_RASTER_TYPES = ("image/jpeg", "image/jpg", "image/png", "image/webp")
_FETCH_ATTEMPTS = 3


def _is_raster_url(url: str) -> bool:
    path = url.split("?", 1)[0].lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".webp"))


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """How long to wait after a rate limit. The server's number wins."""
    header = response.headers.get("retry-after", "").strip()
    try:
        requested = float(header)
    except ValueError:
        requested = float(attempt * 4)
    return min(max(requested, 1.0), 15.0)


def fetch_reference(url: str, *, client: httpx.Client | None = None) -> bytes | None:
    """The bytes of one reference photo, or None when it cannot be fetched.

    A logo is not a photo. A rate limit is waited out and tried again.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=20.0, follow_redirects=True)
    try:
        if not _is_raster_url(url):
            return None
        for attempt in range(1, _FETCH_ATTEMPTS + 1):
            try:
                response = http.get(
                    url,
                    headers={
                        "User-Agent": "store-listing-generator/1.0 (personal design research)"
                    },
                )
            except httpx.HTTPError:
                logger.warning("Reference image could not be fetched")
                return None
            if response.status_code in (429, 503) and attempt < _FETCH_ATTEMPTS:
                time.sleep(_retry_delay(response, attempt))
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPError:
                logger.warning("Reference image could not be fetched")
                return None
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type not in _RASTER_TYPES or not response.content:
                return None
            return response.content
        return None
    finally:
        if owns_client:
            http.close()


def photo_look_from_answer(text: str) -> PhotoLook:
    """The photo description in a vision answer. An unreadable answer is empty."""
    parsed = _first_object(text)
    if not isinstance(parsed, dict):
        return PhotoLook("", keep=True)
    look = parsed.get("look")
    description = look.strip() if isinstance(look, str) else ""
    kind = str(parsed.get("kind") or "subject").strip().lower()
    return PhotoLook(description, keep=kind == "subject")


def look_at_reference(image: bytes, *, client: Any, model: str) -> PhotoLook:
    """What one reference photo shows. A failure is an empty look.

    The brief still continues. An empty look leaves that photo on the text path.
    """
    from store_listing.orchestration.text_validation import encode_image_for_vision

    try:
        data_url = encode_image_for_vision(image)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _LOOK_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        logger.warning("The reference photo could not be described")
        return PhotoLook("", keep=True)
    return photo_look_from_answer(content)


def _attach_looks(
    notes: Sequence[SearchNote],
    look: Callable[[bytes], PhotoLook],
    fetch: Callable[[str], bytes | None],
) -> list[SearchNote]:
    """Describe up to six photos. A failed look leaves the note unchanged."""
    described = 0
    updated: list[SearchNote] = []
    for note in notes:
        if described >= MAX_REFERENCES or not note.image_url or note.look is not None:
            updated.append(note)
            continue
        described += 1
        try:
            data = fetch(note.image_url)
        except Exception:  # noqa: BLE001
            logger.warning("Reference image could not be fetched")
            updated.append(note)
            continue
        if not data:
            updated.append(note)
            continue
        try:
            photo = look(data)
        except Exception:  # noqa: BLE001
            logger.warning("The reference photo could not be described")
            updated.append(note)
            continue
        description = photo.description.strip()
        if not description and photo.keep:
            updated.append(note)
            continue
        updated.append(
            replace(note, look=description or None, look_keep=bool(description and photo.keep))
        )
    return updated


def _requested_line(customer_line: str | None, sentence: str) -> str | None:
    """A list of topics is the request, not the words on the shirt."""
    written = (customer_line or "").strip()
    if not written:
        return None
    if "," in sentence and written.casefold() == sentence.strip().casefold():
        return None
    return written


def _kept_looks(notes: Sequence[SearchNote], urls: Sequence[str]) -> tuple[str, ...]:
    by_url = {note.image_url: note.look for note in notes if note.image_url and note.look}
    return tuple(by_url[url] for url in urls if url in by_url)


def _drawn_subject(
    model_subjects: str, additions: str | None, looks: Sequence[str]
) -> tuple[str, bool]:
    """The shirt subject. A photo look wins over the article's name for it."""
    shown = [item.strip().rstrip(".") for item in looks if item and item.strip()]
    if not shown:
        return model_subjects.strip(), False
    extra = (additions or "").strip().rstrip(".")
    if extra:
        shown.append(extra)
    return ". ".join(shown) + ".", True


def _deepen_subjects(
    complete: Callable[[list[dict[str, str]]], str],
    sentence: str,
    notes: Sequence[SearchNote],
) -> list[str]:
    """Icons the notes say a place or work is known for, before picture search."""
    if not any(_fact(note) for note in notes):
        return []
    note_text = "\n".join(_note_line(note) for note in notes)
    raw = complete(
        [
            {"role": "system", "content": _DEEPEN_SYSTEM},
            {"role": "user", "content": f"Sentence: {sentence}\n\nNotes:\n{note_text}"},
        ]
    )
    parsed = _first_object(raw)
    if not isinstance(parsed, dict):
        return []
    subjects = parsed.get("subjects")
    if not isinstance(subjects, list):
        return []
    cleaned: list[str] = []
    for item in subjects:
        name = " ".join(str(item).split())
        if not name:
            continue
        if name.lower() in {existing.lower() for existing in cleaned}:
            continue
        cleaned.append(name)
    return cleaned[:MAX_DEEPEN_SUBJECTS]


def _queries(complete: Callable[[list[dict[str, str]]], str], sentence: str) -> list[str]:
    raw = complete(
        [
            {"role": "system", "content": _QUERY_SYSTEM},
            {"role": "user", "content": sentence},
        ]
    )
    parsed = _first_object(raw)
    queries = parsed.get("queries") if isinstance(parsed, dict) else None
    if not isinstance(queries, list):
        return [sentence.strip()]
    cleaned = [str(query).strip() for query in queries if str(query).strip()]
    return cleaned[:MAX_QUERIES] or [sentence.strip()]


def _read_design(
    complete: Callable[[list[dict[str, str]]], str],
    sentence: str,
    notes: Sequence[SearchNote],
) -> dict[str, Any] | None:
    note_text = "\n".join(_note_line(note) for note in notes)
    raw = complete(
        [
            {"role": "system", "content": _DESIGN_SYSTEM},
            {"role": "user", "content": f"Sentence: {sentence}\n\nNotes:\n{note_text}"},
        ]
    )
    parsed = _first_object(raw)
    if not isinstance(parsed, dict):
        return None
    subjects = parsed.get("subjects")
    if not isinstance(subjects, str):
        return None
    customer_line = parsed.get("customer_line")
    proposed = parsed.get("lettering")
    references = parsed.get("references")
    additions = parsed.get("additions")
    return {
        "customer_line": customer_line if isinstance(customer_line, str) else None,
        "voice_found": bool(parsed.get("voice_found")),
        "lettering": proposed if isinstance(proposed, str) else None,
        "subjects": subjects,
        "additions": additions if isinstance(additions, str) else None,
        "references": references if isinstance(references, list) else [],
    }


def _note_line(note: SearchNote) -> str:
    line = f"- {note.title}: {note.text}"
    if note.look:
        return f"{line} Photo shows: {note.look}"
    if note.image_url:
        return f"{line} image: {note.image_url}"
    return line


def _reference_urls(references: list[Any], notes: Sequence[SearchNote]) -> tuple[str, ...]:
    rejected = {note.image_url for note in notes if note.image_url and not note.look_keep}
    chosen: list[Any] = references
    if not chosen:
        chosen = [
            {"url": note.image_url, "keep": True}
            for note in notes
            if note.image_url and note.image_url not in rejected
        ]
    urls: list[str] = []
    for entry in chosen:
        if not isinstance(entry, dict) or entry.get("keep") is False:
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip() or url.strip() in rejected:
            continue
        if url.strip() not in urls:
            urls.append(url.strip())
    return tuple(urls[:MAX_REFERENCES])


def _merged_urls(character_notes: Sequence[SearchNote], chosen: Sequence[str]) -> tuple[str, ...]:
    """Character pictures first, then the other references, at most six."""
    urls: list[str] = []
    for note in character_notes:
        url = note.image_url
        if isinstance(url, str) and url.strip() and note.look_keep and url.strip() not in urls:
            urls.append(url.strip())
    for url in chosen:
        if url not in urls:
            urls.append(url)
    return tuple(urls[:MAX_REFERENCES])


def _fact(note: SearchNote) -> str:
    text = note.text.strip()
    if not text or text.lower() == note.title.strip().lower():
        return ""
    sentence = text.split(". ", 1)[0].strip()
    if sentence and not sentence.endswith("."):
        sentence = f"{sentence}."
    return f"{note.title}: {sentence}" if sentence else note.title


def _first_object(text: str) -> Any:
    for value in extract_json_objects(text or ""):
        if isinstance(value, dict):
            return value
    return None
