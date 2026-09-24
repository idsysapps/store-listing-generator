"""Unit: master chart versions must be strictly monotonic in precedence.

Regression for the published-chart collision (#57).

Root cause (verified against the real gh-pages index + this repo's own
workflow): every master publish used `0.1.0-master+sha<short>`.  SemVer 2.0
treats `+sha...` as build metadata, IGNORED for precedence — so all six
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

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# The master branch emits the numeric segment either as a literal (a concrete
# re-publish) or as the ${RUN} placeholder resolved at workflow runtime.
MASTER_VERSION_RE = re.compile(
    r"VERSION=0\.1\.0-master\.(?P<run>\d+|\$\{RUN\})\+sha(?P<sha>[0-9a-f]+)",
    re.MULTILINE,
)


def _run_number(version: str) -> int:
    """The precedence-relevant numeric segment (build metadata is ignored)."""
    m = MASTER_VERSION_RE.search(version)
    assert m, f"must match VERSION=0.1.0-master.N+sha scheme: {version!r}"
    run = m.group("run")
    assert run.isdigit(), f"run segment must be numeric at publish: {run!r}"
    return int(run)


def test_master_ci_uses_monotonic_build_scheme() -> None:
    """GREEN: no bare +sha-only (equal-precedence) publish may be emitted."""
    ci = CI_YML.read_text()
    # The old bare metadata-only form must be entirely gone.
    assert "VERSION=0.1.0-master+sha${SHA}" not in ci, (
        "old build-metadata-only scheme (equal precedence) must be gone"
    )
    # GITHUB_RUN_NUMBER is the strictly-monotonic counter driving ${RUN}.
    assert "GITHUB_RUN_NUMBER" in ci, (
        "the workflow must bind ${RUN} to the monotonic GITHUB_RUN_NUMBER"
    )


def test_master_precedence_strictly_increases_between_runs() -> None:
    """Newer runs must resolve strictly above older ones (no stale tie)."""
    newer = _run_number("VERSION=0.1.0-master.5+shaacf4090")
    older = _run_number("VERSION=0.1.0-master.4+sha6d516b2")
    assert newer > older
