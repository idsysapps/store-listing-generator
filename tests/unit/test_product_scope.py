"""Unit: the product type scope.

The store focuses on one production method at a time, so the pipeline needs
one answer to "which product types do we produce right now". The stored
setting wins over the environment, so the scope can be changed without a
redeploy.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from store_listing.orchestration.product_scope import (
    ENV_VAR,
    KNOWN_PRODUCT_TYPES,
    SETTING_KEY,
    enabled_product_types,
    parse_product_types,
)


class TestParseProductTypes:
    def test_reads_a_comma_separated_list(self) -> None:
        assert parse_product_types("dtf_apparel,sublimation") == ["dtf_apparel", "sublimation"]

    def test_ignores_the_whitespace_around_a_name(self) -> None:
        assert parse_product_types(" dtf_apparel , sublimation ") == [
            "dtf_apparel",
            "sublimation",
        ]

    def test_drops_a_name_the_pipeline_does_not_know(self) -> None:
        assert parse_product_types("dtf_apparel,screen_print") == ["dtf_apparel"]

    def test_drops_a_repeated_name(self) -> None:
        assert parse_product_types("dtf_apparel,dtf_apparel") == ["dtf_apparel"]

    def test_an_empty_value_names_nothing(self) -> None:
        assert parse_product_types("") == []
        assert parse_product_types(None) == []


class TestEnabledProductTypes:
    def test_the_environment_names_the_types(self) -> None:
        with patch.dict(os.environ, {ENV_VAR: "dtf_apparel"}):
            assert enabled_product_types() == ["dtf_apparel"]

    def test_without_a_scope_every_known_type_is_produced(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert enabled_product_types() == list(KNOWN_PRODUCT_TYPES)

    def test_the_stored_setting_wins_over_the_environment(self) -> None:
        db = MagicMock()
        db.get_app_setting.return_value = "sublimation"

        with patch.dict(os.environ, {ENV_VAR: "dtf_apparel"}):
            assert enabled_product_types(db) == ["sublimation"]

    def test_the_stored_setting_is_read_by_its_key(self) -> None:
        db = MagicMock()
        db.get_app_setting.return_value = "dtf_apparel"

        enabled_product_types(db)

        db.get_app_setting.assert_called_once_with(SETTING_KEY)

    def test_a_stored_setting_that_names_nothing_falls_back_to_the_environment(self) -> None:
        db = MagicMock()
        db.get_app_setting.return_value = "screen_print"

        with patch.dict(os.environ, {ENV_VAR: "dtf_apparel"}):
            assert enabled_product_types(db) == ["dtf_apparel"]

    def test_a_stored_setting_that_is_empty_falls_back_to_the_environment(self) -> None:
        db = MagicMock()
        db.get_app_setting.return_value = ""

        with patch.dict(os.environ, {ENV_VAR: "dtf_apparel"}):
            assert enabled_product_types(db) == ["dtf_apparel"]

    def test_without_an_environment_a_stored_setting_is_used(self) -> None:
        db = MagicMock()
        db.get_app_setting.return_value = "dtf_apparel,sticker_vinyl"

        with patch.dict(os.environ, {}, clear=True):
            assert enabled_product_types(db) == ["dtf_apparel", "sticker_vinyl"]
