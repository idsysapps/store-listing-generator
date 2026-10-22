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
APP_DEPLOYMENT_YAML = HELM_DIR / "templates" / "app" / "deployment.yaml"
TREND_SERVICE_YAML = HELM_DIR / "templates" / "trend-service" / "deployment.yaml"


def test_the_configmap_ships_the_scope() -> None:
    configmap = CONFIGMAP_YAML.read_text()

    assert "ENABLED_PRODUCT_TYPES:" in configmap


def test_the_worker_is_told_the_scope() -> None:
    # The key was in the ConfigMap and in no container, so a cluster that set
    # the value produced every known type (issue #234).
    deployment = APP_DEPLOYMENT_YAML.read_text()

    assert "- name: ENABLED_PRODUCT_TYPES" in deployment


def test_the_api_is_told_the_scope() -> None:
    deployment = TREND_SERVICE_YAML.read_text()

    assert "- name: ENABLED_PRODUCT_TYPES" in deployment


def test_the_scope_is_only_passed_when_the_chart_sets_it() -> None:
    # The key is absent from the ConfigMap when the value is empty, and a
    # container that names a missing key does not start.
    for path in (APP_DEPLOYMENT_YAML, TREND_SERVICE_YAML):
        deployment = path.read_text()

        assert "if .Values.config.enabledProductTypes" in deployment


def test_the_chart_defines_the_scope() -> None:
    values = VALUES_YAML.read_text()

    match = re.search(r"^\s*enabledProductTypes:\s*\"?([^\"\n]*)\"?\s*$", values, re.MULTILINE)
    assert match, "values.yaml must define enabledProductTypes"
    assert match.group(1).strip() == "", "the chart produces every known type by default"


def test_the_migration_creates_the_settings_table() -> None:
    configmap = CONFIGMAP_YAML.read_text()

    assert "CREATE TABLE IF NOT EXISTS app_settings" in configmap
