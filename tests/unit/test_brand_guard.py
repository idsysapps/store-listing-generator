"""Unit tests for the brand guard.

The guard is used in two places: the trend queries that are harvested, and the
text the brief model wrote. A trend query that names a brand is noise, and a
headline that names one is a trademark we would print on a shirt (issue #225).
"""

from store_listing.orchestration.brand_guard import licensed_brand


class TestLicensedBrand:
    def test_a_theme_park_name_is_caught(self) -> None:
        # The brief that started this: the headline named the park.
        assert licensed_brand("Thanksgiving in the Magic Kingdom") is not None

    def test_the_name_that_is_caught_is_reported(self) -> None:
        assert licensed_brand("The Magic Kingdom") == "Magic Kingdom"

    def test_a_brand_name_in_a_tagline_is_caught(self) -> None:
        assert licensed_brand("Feast at Disneyland") is not None

    def test_the_sport_leagues_are_caught(self) -> None:
        assert licensed_brand("Super Bowl Sunday") is not None
        assert licensed_brand("NBA on the court") is not None

    def test_an_ordinary_line_is_not_caught(self) -> None:
        assert licensed_brand("Gobble Til The Drawbridge Drops") is None

    def test_a_word_that_merely_contains_a_brand_name_is_not_caught(self) -> None:
        # "Legacy" contains "lego" and "Marvellous" contains "marvel": the
        # guard matches whole names, not parts of a word.
        assert licensed_brand("Legacy Code") is None
        assert licensed_brand("A Marvellous Day") is None

    def test_an_empty_text_has_no_brand(self) -> None:
        assert licensed_brand("") is None
