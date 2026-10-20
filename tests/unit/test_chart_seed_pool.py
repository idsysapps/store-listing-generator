"""Unit: the chart carries the seed-pool sizing knobs the brief task needs.

The brief task runs from a pool of curated trend seeds. The pool is capped and
trimmed by the curation step, so the cap, the briefable reserve and the reuse
cooldown are all deployment settings: they must be settable per cluster without
a code change, and the chart default must match the code default so a cluster
that never overrides them behaves the same as the source tree (issue #192).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELM_DIR = REPO_ROOT / "helm" / "store-listing"
VALUES_YAML = HELM_DIR / "values.yaml"
CONFIGMAP_YAML = HELM_DIR / "templates" / "configmap.yaml"

# key -> (values.yaml key, code default)
KNOBS = {
    "LLM_MAX_ACTIVE_SEEDS": ("llmMaxActiveSeeds", "200"),
    "LLM_MIN_BRIEFABLE_SEEDS": ("llmMinBriefableSeeds", "30"),
    "LLM_BRIEF_SEED_COOLDOWN_DAYS": ("llmBriefSeedCooldownDays", "14"),
}


def _configmap_entry(key: str) -> str:
    match = re.search(rf"^\s*{key}:\s*(.+)$", CONFIGMAP_YAML.read_text(), re.MULTILINE)
    assert match, f"configmap.yaml must expose {key}"
    return match.group(1).strip()


def _values_entry(key: str) -> str:
    match = re.search(rf"^\s*{key}:\s*\"?([^\"\n]+)\"?\s*$", VALUES_YAML.read_text(), re.MULTILINE)
    assert match, f"values.yaml must define {key}"
    return match.group(1).strip()


def test_configmap_exposes_every_seed_pool_knob() -> None:
    for env_key in KNOBS:
        assert _configmap_entry(env_key)


def test_configmap_defaults_match_the_code_defaults() -> None:
    for env_key, (_values_key, default) in KNOBS.items():
        entry = _configmap_entry(env_key)
        assert default in entry, f"{env_key} must default to {default}, got {entry}"


def test_values_define_every_seed_pool_knob() -> None:
    for values_key, default in KNOBS.values():
        assert _values_entry(values_key) == default
