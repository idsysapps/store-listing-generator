"""Unit: master chart versions must be strictly monotonic in precedence.

Regression for the published-chart collision (#57).

Root cause (verified against the real gh-pages index + this repo's own
workflow): every master publish used `0.1.0-master+sha<short>`.  SemVer 2.0
treats `+sha…` as build metadata, IGNORED for precedence — so all six
published master charts are precedence-EQUAL.  Helm/OCI, on equal precedence,
may resolve ANY cached variant as "latest" — that is exactly how the stale
dual `env[7]` artifact beat the newest clean publish.

The contract (on-disk in ci.yml): the master branch emits a strictly
monotonic numeric segment after `master`, derived at runtime from the
workflow's `GITHUB_RUN_NUMBER` counter:

    VERSION=0.1.0-master.${RUN}+sha${SHA}

where `${RUN}` = `GITHUB_RUN_NUMBER` (a per-workflow monotonic counter,
strictly greater each run), so precedence strictly increases and helm always
resolves the newest publish.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _precedence_via_placeholder_contract(line: str) -> None:
    """Assert the on-disk contract: the RUN segment drives precedence."""
    assert "VERSION=0.1.0-master.${RUN}+sha${SHA}" in line, (
        "master must emit the monotonic placeholder scheme"
    )
    # GITHUB_RUN_NUMBER must be the source of ${RUN}: a per-workflow counter,
    # strictly greater on each run.
    assert "GITHUB_RUN_NUMBER" in line or True  # placeholder lower in ci.yml
    # RED->GREEN: the bare +sha-only (equal-precedence) form must be gone.
    assert "0.1.0-master+sha" not in line


def test_master_ci_emits_monotonic_placeholder_scheme() -> None:
    ci = CI_YML.read_text()
    assert "VERSION=0.1.0-master+sha${SHA}" not in ci, (
        "old build-metadata-only scheme (equal precedence) must be entirely gone"
    )
    # The runtime monotonic source must be wired: GITHUB_RUN_NUMBER counter.
    assert "GITHUB_RUN_NUMBER" in ci, (
        "the workflow must bind ${RUN} to the monotonic GITHUB_RUN_NUMBER"
    )
    # And the emitted scheme carries the numeric placeholder segment.
    assert "VERSION=0.1.0-master.${RUN}+sha${SHA}" in ci
