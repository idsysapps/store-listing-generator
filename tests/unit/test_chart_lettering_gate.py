"""Unit: the chart carries the lettering-gate knobs the image task needs.

The artwork must not carry words: the image model cannot spell, and the text
of the design is drawn on the artwork afterwards with a font. The gate that
enforces it, and how often the design is regenerated because of it, are
deployment settings (issue #208).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELM_DIR = REPO_ROOT / "helm" / "store-listing"
VALUES_YAML = HELM_DIR / "values.yaml"
DEPLOYMENT_YAML = HELM_DIR / "templates" / "app" / "deployment.yaml"

# env key -> (values.yaml key, code default)
KNOBS = {
    "ARTWORK_TEXT_VALIDATION_ENABLED": ("artworkTextValidationEnabled", "true"),
    "MAX_LETTERING_REGENERATION_ATTEMPTS": ("maxLetteringRegenerationAttempts", "1"),
}


def _deployment_entry(key: str) -> tuple[str, str]:
    match = re.search(
        rf"- name: {key}\s*\n\s*value: \{{\{{ \.Values\.config\.(\w+) \| default \"([^\"]+)\"",
        DEPLOYMENT_YAML.read_text(),
    )
    assert match, f"deployment.yaml must expose {key}"
    return match.group(1), match.group(2)


def _values_entry(key: str) -> str:
    match = re.search(rf"^\s*{key}:\s*\"?([^\"\n]+)\"?\s*$", VALUES_YAML.read_text(), re.MULTILINE)
    assert match, f"values.yaml must define {key}"
    return match.group(1).strip()


def test_the_deployment_exposes_every_lettering_knob() -> None:
    for env_key in KNOBS:
        assert _deployment_entry(env_key)


def test_the_chart_default_matches_the_code_default() -> None:
    for env_key, (values_key, code_default) in KNOBS.items():
        values_default, template_default = _deployment_entry(env_key)
        assert values_default == values_key
        assert _values_entry(values_key) == code_default
        assert template_default == code_default


def test_both_containers_get_the_lettering_knobs() -> None:
    deployment = DEPLOYMENT_YAML.read_text()
    for env_key in KNOBS:
        assert deployment.count(f"- name: {env_key}") == 2
