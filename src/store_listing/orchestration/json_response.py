"""Reading the JSON an LLM actually sent.

The model answers with JSON, but it does not always answer with *only* JSON:
it appends a note, repeats the batch, or splits it in two. A parser that
slices the answer from the first `{` to the last `}` reads two JSON values as
one, `json.loads` fails with "Extra data", and the whole batch is thrown
away without a word in the logs (issue #209).

The reader below finds every complete JSON value in the text, so a parser can
merge what it finds instead of losing the answer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMResponseParseError(Exception):
    """The model answered, but the answer could not be read."""


def _end_of_value(text: str, start: int) -> int | None:
    """The index just past the JSON value that starts at `start`.

    Only the bracket type that opened the value is counted, and a bracket
    inside a string is ignored: a `}` in a headline must not end the value.
    """
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _loads(fragment: str) -> Any | None:
    """The value of a JSON fragment, or None when it is not JSON.

    Some models answer with single quotes; the curation parser coped with
    that, and the answer is worth keeping.
    """
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass
    if '"' not in fragment and "'" in fragment:
        try:
            return json.loads(fragment.replace("'", '"'))
        except json.JSONDecodeError:
            return None
    return None


def extract_json_values(text: str) -> list[Any]:
    """Every complete JSON value in the text, in the order they appear."""
    values: list[Any] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char not in "{[":
            index += 1
            continue
        end = _end_of_value(text, index)
        if end is None:
            # A truncated value: there is no complete JSON here.
            index += 1
            continue
        value = _loads(text[index:end])
        if value is not None:
            values.append(value)
        index = end
    return values


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object in the text, in the order they appear."""
    return [value for value in extract_json_values(text) if isinstance(value, dict)]


def merge_items(objects: list[dict[str, Any]], key: str) -> list[Any]:
    """The items under `key` in every object, in order.

    A model that split the batch in two answers with two objects, and a model
    that repeated the batch answers with two objects as well. Both are read
    as one list.
    """
    items: list[Any] = []
    for obj in objects:
        value = obj.get(key)
        if isinstance(value, list):
            items.extend(value)
    return items
