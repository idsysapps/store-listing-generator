"""Unit: the chart points at a model the DGX proxy actually serves.

The cluster pointed at `qwen2.5:32b-instruct-q4_K_M`, a model the proxy does
not serve, so every brief and curation call failed and the logs never said
which model was missing (issue #194). The model names the proxy serves live in
`deploy/dgx/openai-proxy.py` (`MODEL_ROUTES`), so the chart is checked against
that file: a config that names a model the proxy cannot route is a rollout
that breaks every LLM task.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELM_DIR = REPO_ROOT / "helm" / "store-listing"
VALUES_YAML = HELM_DIR / "values.yaml"
PROD_VALUES_YAML = HELM_DIR / "prod-values.yaml"
CONFIGMAP_YAML = HELM_DIR / "templates" / "configmap.yaml"
PROXY_PY = REPO_ROOT / "deploy" / "dgx" / "openai-proxy.py"


def _served_models() -> set[str]:
    """The models the DGX proxy routes to a backend."""
    source = PROXY_PY.read_text()
    routes = re.findall(r'"(?P<name>[\w.\-:]+)":\s*"http://localhost', source)
    assert routes, "the proxy must declare its model routes"
    return set(routes)


def _values_entry(text: str, key: str) -> str | None:
    match = re.search(
        rf"^\s*{key}:\s*(?:\"(?P<quoted>[^\"\n]*)\"|(?P<plain>[^\"\n]+?))\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        return None
    if match.group("quoted") is not None:
        return match.group("quoted").strip()
    return match.group("plain").strip()


def test_the_configmap_ships_both_model_names() -> None:
    configmap = CONFIGMAP_YAML.read_text()

    assert "LLM_MODEL:" in configmap
    assert "LLM_CURATION_MODEL:" in configmap


def test_the_chart_defines_the_curation_model() -> None:
    assert _values_entry(VALUES_YAML.read_text(), "llmCurationModel") is not None


def test_the_production_values_point_at_a_model_the_proxy_serves() -> None:
    values = PROD_VALUES_YAML.read_text()
    served = _served_models()

    for key in ("llmModel", "llmCurationModel"):
        model = _values_entry(values, key)
        assert model is not None, f"prod-values.yaml must set {key}"
        assert model in served, (
            f"{key} is {model!r}, which the DGX proxy does not serve; it serves {sorted(served)}"
        )
