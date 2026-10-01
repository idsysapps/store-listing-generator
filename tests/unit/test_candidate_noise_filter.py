"""Unit tests for seed candidate noise filtering."""

import pytest

from store_listing.ingest.trends.google_trends import _is_candidate_noise


class TestIsCandidateNoise:
    @pytest.mark.parametrize(
        "query",
        [
            "nervous",
            "going",
            "plane",
            "juliana",
            "com",
        ],
    )
    def test_single_word_is_noise(self, query: str) -> None:
        assert _is_candidate_noise(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "dad jokes shirt",
            "funny coffee mug",
            "fall decor",
        ],
    )
    def test_multi_word_is_not_noise(self, query: str) -> None:
        assert _is_candidate_noise(query) is False

    def test_very_short_is_noise(self) -> None:
        assert _is_candidate_noise("ab") is True

    def test_very_long_is_noise(self) -> None:
        assert _is_candidate_noise("a " + "word " * 20) is True

    def test_html_entities_is_noise(self) -> None:
        assert _is_candidate_noise("it&#39;s giving spooky tee") is True

    def test_url_is_noise(self) -> None:
        assert _is_candidate_noise("check out http://example.com") is True

    def test_www_is_noise(self) -> None:
        assert _is_candidate_noise("visit www.shop.com") is True

    def test_email_is_noise(self) -> None:
        assert _is_candidate_noise("contact us user@example.com") is True

    @pytest.mark.parametrize(
        "query",
        [
            "taylor swift eras tour",
            "disney princess shirt",
            "marvel avengers mug",
            "pokemon sticker pack",
            "star wars baby onesie",
            "harry potter tumbler",
            "nfl team hoodie",
        ],
    )
    def test_licensed_ip_is_noise(self, query: str) -> None:
        assert _is_candidate_noise(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "how to make candles",
            "tutorial crochet blanket",
            "review best tumblers",
            "unboxing mystery box",
        ],
    )
    def test_tutorial_content_is_noise(self, query: str) -> None:
        assert _is_candidate_noise(query) is True

    def test_diy_with_space_is_noise(self) -> None:
        assert _is_candidate_noise("diy fall wreath") is True

    def test_diy_in_middle_of_word_is_not_noise(self) -> None:
        assert _is_candidate_noise("muddy boots shirt") is False

    def test_valid_product_concept_passes(self) -> None:
        assert _is_candidate_noise("skeleton coffee mug") is False
        assert _is_candidate_noise("funny dad hoodie") is False
        assert _is_candidate_noise("halloween cat sticker") is False
