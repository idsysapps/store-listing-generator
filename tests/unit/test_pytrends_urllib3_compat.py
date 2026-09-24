"""Unit: the trend scrape must work with the resolved HTTP stack.

Regression for the failed manual scrape on the cluster: every seed failed with

    Retry.__init__() got an unexpected keyword argument 'method_whitelist'

Root cause: pytrends 4.9.2 builds its retry via `requests.packages.urllib3`
(`Retry(..., method_whitelist=...)`). urllib3 >= 2 renamed that kwarg to
`allowed_methods`, so a fresh resolution of our `>=` constraints picks a
urllib3 that raises TypeError on every pytrends request. The deploy image
resolves deps from the same `>=` constraints, so it shipped broken.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# urllib3's Retry kwarg that pytrends relies on was renamed in v2.0.
RENAMED_KWARG = "method_whitelist"


def test_pyproject_pins_urllib3_below_v2() -> None:
    """GREEN: resolution must not float onto urllib3 >= 2 (pytrends break)."""
    pyproject = PYPROJECT.read_text()
    m = re.search(r'"urllib3\s*<\s*2"', pyproject)
    assert m is not None, (
        "pyproject must pin urllib3<2 so pytrends 4.9.2's method_whitelist "
        "retry arg survives resolution (fixes the cluster scrape TypeError)"
    )


def test_resolved_retry_accepts_pytrends_kwarg() -> None:
    """The installed urllib3 Retry must still accept method_whitelist."""
    # pytrends reaches `Retry` via `requests.packages.urllib3`, a vendoring
    # shim over the real `urllib3` import; use the real module for pyright.
    import inspect

    from urllib3.util.retry import Retry

    sig = inspect.signature(Retry.__init__)
    assert RENAMED_KWARG in sig.parameters, (
        "resolved urllib3 Retry must accept `method_whitelist` "
        f"(pytrends 4.9.2 needs it); got params: {list(sig.parameters)}"
    )
