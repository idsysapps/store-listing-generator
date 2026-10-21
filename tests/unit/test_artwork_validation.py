"""Tests for the artwork gate.

The text gate reads the words back out of an image. Nothing checked the
picture itself, so a brief whose scene mentioned a bag produced an image of a
bag, and that bag was uploaded as the print file (issue #195).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

from PIL import Image

from store_listing.orchestration.artwork_validation import (
    ARTWORK,
    PRODUCT,
    UNKNOWN,
    ArtworkValidator,
    parse_verdict,
)


def _image_bytes() -> bytes:
    image = Image.new("RGBA", (32, 32), (255, 0, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _client(answer: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = answer
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


class TestParseVerdict:
    def test_artwork_is_recognised(self) -> None:
        assert parse_verdict("ARTWORK") == ARTWORK

    def test_product_is_recognised(self) -> None:
        assert parse_verdict("PRODUCT") == PRODUCT

    def test_the_answer_is_case_insensitive(self) -> None:
        assert parse_verdict("artwork") == ARTWORK
        assert parse_verdict("product") == PRODUCT

    def test_the_first_word_wins(self) -> None:
        assert parse_verdict("ARTWORK, not a product") == ARTWORK
        assert parse_verdict("PRODUCT, not artwork") == PRODUCT

    def test_an_answer_with_the_keyword_later_is_read(self) -> None:
        assert parse_verdict("The image is ARTWORK") == ARTWORK

    def test_an_unreadable_answer_is_unknown(self) -> None:
        assert parse_verdict("I cannot tell") == UNKNOWN

    def test_an_empty_answer_is_unknown(self) -> None:
        assert parse_verdict("") == UNKNOWN


class TestArtworkValidator:
    def test_an_artwork_image_is_accepted(self) -> None:
        validator = ArtworkValidator(client=_client("ARTWORK"), model="vl")

        result = validator.validate(_image_bytes())

        assert result.verdict == ARTWORK
        assert result.is_product is False
        assert result.error is None

    def test_a_product_image_is_reported(self) -> None:
        validator = ArtworkValidator(client=_client("PRODUCT"), model="vl")

        result = validator.validate(_image_bytes())

        assert result.verdict == PRODUCT
        assert result.is_product is True

    def test_an_unreadable_answer_is_unknown_and_not_a_product(self) -> None:
        validator = ArtworkValidator(client=_client("hmm"), model="vl")

        result = validator.validate(_image_bytes())

        assert result.verdict == UNKNOWN
        assert result.is_product is False

    def test_a_client_error_is_reported_and_never_raises(self) -> None:
        client = _client("ARTWORK")
        client.chat.completions.create.side_effect = RuntimeError("vision is down")
        validator = ArtworkValidator(client=client, model="vl")

        result = validator.validate(_image_bytes())

        assert result.verdict == UNKNOWN
        assert result.is_product is False
        assert result.error is not None

    def test_the_image_is_sent_to_the_vision_model(self) -> None:
        client = _client("ARTWORK")
        validator = ArtworkValidator(client=client, model="my-vl")

        validator.validate(_image_bytes())

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "my-vl"
        content = call_kwargs["messages"][0]["content"]
        image_parts = [part for part in content if part["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_the_prompt_asks_for_one_word(self) -> None:
        client = _client("ARTWORK")
        validator = ArtworkValidator(client=client, model="vl")

        validator.validate(_image_bytes())

        call_kwargs = client.chat.completions.create.call_args.kwargs
        text = call_kwargs["messages"][0]["content"][0]["text"]
        assert "ARTWORK" in text
        assert "PRODUCT" in text
