import pytest


@pytest.fixture(autouse=True)
def _disable_source_health_github(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests hermetic: harvesters do not open GitHub issues unless enabled.

    A test that exercises the health wiring opts back in with
    ``monkeypatch.setenv("SOURCE_HEALTH_ENABLED", "true")``.
    """
    monkeypatch.setenv("SOURCE_HEALTH_ENABLED", "false")


@pytest.fixture(autouse=True)
def _disable_text_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests hermetic: briefs do not make a second LLM call unless enabled.

    Text planning is a separate LLM pass, so the tests that exercise it opt in
    with ``monkeypatch.setenv("LLM_TEXT_PLAN", "true")``.
    """
    monkeypatch.setenv("LLM_TEXT_PLAN", "false")


@pytest.fixture(autouse=True)
def _disable_copy_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests hermetic: the copy editor is a separate LLM pass.

    Tests that exercise the editor opt in with
    ``monkeypatch.setenv("LLM_COPY_REVIEW", "true")``.
    """
    monkeypatch.setenv("LLM_COPY_REVIEW", "false")
