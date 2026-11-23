"""Research a personal custom sentence into one DTF shirt graphic.

The customer's sentence stays the request. A search supplies what the named
things look like and how a subject speaks. The lettering rules below decide
whether one line is printed. FLUX.2 dev draws the graphic; this module does
not call it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx

from store_listing.orchestration.json_response import extract_json_objects

logger = logging.getLogger(__name__)

MAX_REFERENCES: int = 6
MAX_QUERIES: int = 4

_ARRANGEMENTS: tuple[str, ...] = (
    "The subject is large on the left and the other objects are smaller beside it.",
    "The subject is centered and the other objects sit above and to one side.",
    "The subject is on the right and the other objects are clustered to the left.",
)

_QUERY_SYSTEM = (
    "You choose web searches for one shirt design. The customer's sentence is the "
    'request. Return JSON only: {"queries": ["..."]}. Search for the places, '
    "rides, shows, and objects, and for how a named speaker talks. Give each "
    "place, ride, show, and character its own query. A place or a ride is not "
    "a movie. Do not merge two names into one search. At most four queries."
)

_DESIGN_SYSTEM = (
    "You turn search notes into one DTF shirt graphic. Return JSON only with "
    "customer_line (a phrase the customer asked to print, or null; a list of "
    "names is not a line), "
    "voice_found (true only when the notes show how the subject speaks), "
    "lettering (one short line that completes the idea in that voice, or null), "
    "subjects (what the photos show; the article does not rename the photo), "
    "additions (what the sentence asks for that the photos do not show, or null), "
    'and references (up to six objects {"url", "keep"}). '
    "Fill the gaps. A role such as sleuth is a character in the picture, not a "
    "word on the shirt. Name the shirt subject from the photo. Do not replace "
    "the photo with the article's object. A logo, a poster, a map, a costume "
    "crowd, a tight crop, or a photo full of unrelated people is keep false. "
    "Do not invent a line when voice_found is false. Do not print the customer's "
    "sentence. Do not replace words the customer wrote when they asked for those "
    "words on the shirt."
)

_LOOK_PROMPT = (
    "Describe this one reference photo for a shirt graphic. Return JSON only: "
    '{"look": "the main thing in the frame, then the rest", '
    '"kind": "subject" or "logo" or "poster" or "crowd" or "unrelated"}. '
    "kind is logo for a logo or a wordmark, poster for a poster or a title card, "
    "crowd when the frame is full of people, and unrelated when it is not the subject."
)


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
class PersonalDesign:
    """The graphic a personal sentence became."""

    prompt: str
    lettering: str | None
    reference_urls: tuple[str, ...]
    facts: tuple[str, ...]
    subjects: str


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


def arrangement_line(seed: int) -> str:
    """One arrangement of the same facts. A new seed is a new arrangement."""
    return _ARRANGEMENTS[seed % len(_ARRANGEMENTS)]


def graphic_prompt(
    *,
    subjects: str,
    lettering: str | None,
    arrangement: str,
    facts: Sequence[str] = (),
) -> str:
    """The ComfyUI prompt: a flat magenta key, the subjects, and one line."""
    fact_text = " ".join(fact.strip() for fact in facts if fact and fact.strip())
    known = f"Known facts: {fact_text} " if fact_text else ""
    line = ""
    if lettering:
        line = f' One line of lettering, spelled exactly: "{lettering}". No other text.'
    placed = arrangement.strip()
    if placed and not placed.endswith("."):
        placed = f"{placed}."
    return (
        "Shirt print artwork. A solid pure magenta field fills every pixel that is "
        "not the subject, one flat colour edge to edge. No street, no sky, no ground, "
        "no buildings, no crowd, no place photograph, no frame, no backdrop. "
        f"{known}{subjects.strip()} {placed}{line}"
    )


def rearrange(design: PersonalDesign, seed: int) -> PersonalDesign:
    """The same facts and line in a different arrangement."""
    prompt = graphic_prompt(
        subjects=design.subjects,
        lettering=design.lettering,
        arrangement=arrangement_line(seed),
        facts=design.facts,
    )
    return replace(design, prompt=prompt)


def research_personal_design(
    sentence: str,
    *,
    complete: Callable[[list[dict[str, str]]], str],
    search: Callable[[str], list[SearchNote]],
    seed: int,
    look: Callable[[bytes], PhotoLook] | None = None,
    fetch: Callable[[str], bytes | None] | None = None,
) -> PersonalDesign:
    """Search the sentence, look at the photos, and turn them into one graphic.

    ``complete`` is one Qwen call. ``search`` is one query. ``look`` describes
    one downloaded photo. All three are injected so a test never touches the
    network. A missing look keeps the text-only path.
    """
    queries = _queries(complete, sentence)
    notes: list[SearchNote] = []
    for query in queries:
        try:
            found = search(query)
        except Exception:  # noqa: BLE001
            logger.warning("Personal design search failed for %r", query)
            found = []
        notes.extend(found)
    if look is not None:
        notes = _attach_looks(notes, look, fetch or fetch_reference)

    facts = tuple(_fact(note) for note in notes if _fact(note))
    design = _read_design(complete, sentence, notes)
    if design is None:
        return PersonalDesign(
            prompt=graphic_prompt(
                subjects=sentence.strip(),
                lettering=None,
                arrangement=arrangement_line(seed),
                facts=facts,
            ),
            lettering=None,
            reference_urls=(),
            facts=facts,
            subjects=sentence.strip(),
        )

    lettering = resolve_lettering(
        customer_line=_requested_line(design["customer_line"], sentence),
        proposed_line=design["lettering"],
        voice_found=bool(design["voice_found"]),
    )
    urls = _reference_urls(design["references"], notes)
    subject, from_photos = _drawn_subject(
        design["subjects"].strip() or sentence.strip(),
        design["additions"],
        _kept_looks(notes, urls),
    )
    image_facts = () if from_photos else facts
    return PersonalDesign(
        prompt=graphic_prompt(
            subjects=subject,
            lettering=lettering,
            arrangement=arrangement_line(seed),
            facts=image_facts,
        ),
        lettering=lettering,
        reference_urls=urls,
        facts=image_facts,
        subjects=subject,
    )


def search_wikipedia(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """Short visual notes from Wikipedia. A failed call is an empty list."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
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
        response.raise_for_status()
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
        if described >= MAX_REFERENCES or not note.image_url:
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


def _fact(note: SearchNote) -> str:
    text = note.text.strip()
    if not text:
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
