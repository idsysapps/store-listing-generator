"""Which product types the pipeline produces right now.

The store focuses on one production method at a time: the DTF apparel
batches are the ones that sell, and the sublimation ones wait. The scope is
resolved in one place, so the prompts, the parsers and the queries all agree
on what is being produced.

The scope comes from the database when a row is there, from the environment
otherwise, and covers every known type when neither is set.
"""

from __future__ import annotations

import logging
import os
from typing import Final, Protocol

logger = logging.getLogger(__name__)

KNOWN_PRODUCT_TYPES: Final[tuple[str, ...]] = ("dtf_apparel", "sublimation", "sticker_vinyl")
SETTING_KEY: Final[str] = "enabled_product_types"
ENV_VAR: Final[str] = "ENABLED_PRODUCT_TYPES"


class ProductScopeDBClient(Protocol):
    """The setting store the scope is read from."""

    def get_app_setting(self, key: str) -> str | None: ...


def parse_product_types(raw: str | None) -> list[str]:
    """The product types named in a setting or an environment variable.

    A name the pipeline does not know is dropped with a warning: a typo in a
    setting must not silently narrow what is produced.
    """
    if not raw:
        return []

    types: list[str] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value not in KNOWN_PRODUCT_TYPES:
            logger.warning("Unknown product type %r, ignored", value)
            continue
        if value not in types:
            types.append(value)
    return types


def enabled_product_types(db_client: ProductScopeDBClient | None = None) -> list[str]:
    """The product types that are produced right now.

    The stored setting wins over the environment, so the scope can be changed
    without a redeploy. A setting that names nothing usable falls back to the
    environment, and an environment that names nothing falls back to every
    known type.
    """
    if db_client is not None:
        try:
            stored = db_client.get_app_setting(SETTING_KEY)
        except Exception as e:  # noqa: BLE001
            # A database that cannot be reached must not stop the pipeline:
            # the environment is the fallback.
            logger.warning("Could not read the stored product type scope: %s", e)
            stored = None
        if stored is not None:
            types = parse_product_types(stored)
            if types:
                return types
            logger.warning(
                "The stored product type scope names no known type, falling back to the environment"
            )

    types = parse_product_types(os.environ.get(ENV_VAR))
    if types:
        return types

    return list(KNOWN_PRODUCT_TYPES)
