"""Unit: the store-listing chart keeps postgres data on uninstall.

Two contracts guard our GitOps chain (chart source -> published tgz ->
cluster helm release):

1. The postgres PVC must carry `helm.sh/resource-policy: keep` so `helm
   uninstall` orphans the volume instead of deleting postgres data.
2. The `app` deployment's memory limit must be 1Gi so the generator has room
   for large LLM/trend payloads.

These are asserted against the chart source because every master publish
re-derives the chart from it; a template/values regression here would ship to
the cluster as-is.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELM_DIR = REPO_ROOT / "helm" / "store-listing"
VALUES_YAML = HELM_DIR / "values.yaml"
PVC_TEMPLATE = HELM_DIR / "templates" / "postgres" / "pvc.yaml"

KEEP_ANNOTATION = "helm.sh/resource-policy: keep"

# Scoped to the `app:` block so postgres's own 1Gi limit can't satisfy it.
APP_LIMIT_RE = re.compile(
    r"^app:\n(?P<body>(?:[ \t].*\n|\n)*?)[^\s#]",
    re.MULTILINE,
)


def _top_level_block(text: str, key: str) -> str:
    """Return the indented body of a top-level `key:` mapping (no PyYAML)."""
    m = re.search(rf"^{re.escape(key)}:\n(?P<body>(?:[ \t].*\n|\n)*)", text, re.MULTILINE)
    assert m, f"missing top-level `{key}:` in {VALUES_YAML}"
    return m.group("body")


def test_postgres_pvc_has_keep_resource_policy() -> None:
    """GREEN: uninstalling the chart must not delete the postgres volume."""
    assert KEEP_ANNOTATION in PVC_TEMPLATE.read_text(), (
        "pvc.yaml must annotate the PVC with "
        "`helm.sh/resource-policy: keep` so `helm uninstall` orphans it"
    )


def test_app_memory_limit_is_two_gi() -> None:
    """GREEN: the app container gets a 2Gi memory ceiling."""
    values = VALUES_YAML.read_text()
    app_block = _top_level_block(values, "app")
    limits_block = _top_level_block(app_block, "    limits")
    m = re.search(r"^\s*memory:\s*(?P<mem>.+)$", limits_block, re.MULTILINE)
    assert m, "app.resources.limits.memory must be set"
    limit = m.group("mem").strip()
    qty = re.fullmatch(r"(\d+)([MG])i", limit)
    assert qty, f"memory limit must be an Mi/Gi quantity, got: {limit!r}"
    assert (qty.group(1), qty.group(2)) == ("2", "G"), (
        f"app memory limit must be 2Gi, got: {limit!r}"
    )
