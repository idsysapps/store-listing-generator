"""Unit: the chart ships the product type scope.

The store focuses on one production method at a time, so the chart has to
carry the scope the pipeline reads (issue #216).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELM_DIR = REPO_ROOT / "helm" / "store-listing"
VALUES_YAML = HELM_DIR / "values.yaml"
CONFIGMAP_YAML = HELM_DIR / "templates" / "configmap.yaml"


def test_the_configmap_ships_the_scope() -> None:
    configmap = CONFIGMAP_YAML.read_text()

    assert "ENABLED_PRODUCT_TYPES:" in configmap


def test_the_chart_defines_the_scope() -> None:
    values = VALUES_YAML.read_text()

    match = re.search(r"^\s*enabledProductTypes:\s*\"?([^\"\n]*)\"?\s*$", values, re.MULTILINE)
    assert match, "values.yaml must define enabledProductTypes"
    assert match.group(1).strip() == "", "the chart produces every known type by default"


def test_the_migration_creates_the_settings_table() -> None:
    configmap = CONFIGMAP_YAML.read_text()

    assert "CREATE TABLE IF NOT EXISTS app_settings" in configmap
