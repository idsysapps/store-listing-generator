"""How long one design is expected to take.

A personal brief researches the subject before a FLUX.2 graphic, about four
minutes of sampling plus the background cut. A sellable brief skips that
research and draws a much shorter Flux Schnell graphic. Once a sampling step
count arrives, the sampling guess is replaced by the pace of the steps so far.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CHANNEL = "image-generation"

# Measured on the DGX: a referenced 20-step FLUX.2 graphic is about four
# minutes. Flux Schnell is a few seconds of sampling. The cut is BiRefNet.
# The earlier phases are one model call, a handful of searches, and the
# photo looks that happen before the image is queued.
_WRITING_SECONDS = 40.0
_RESEARCH_SECONDS = 50.0
_STUDY_SECONDS = 40.0
_PERSONAL_PLAN_SECONDS = 30.0
_SELLABLE_PLAN_SECONDS = 35.0
_PERSONAL_SAMPLE_SECONDS = 240.0
_SELLABLE_SAMPLE_SECONDS = 20.0
_CUT_SECONDS = 40.0

_PERSONAL_PHASES: tuple[tuple[str, float], ...] = (
    ("writing", _WRITING_SECONDS),
    ("researching", _RESEARCH_SECONDS),
    ("studying", _STUDY_SECONDS),
    ("planning", _PERSONAL_PLAN_SECONDS),
    ("sampling", _PERSONAL_SAMPLE_SECONDS),
    ("cutting", _CUT_SECONDS),
)
_SELLABLE_PHASES: tuple[tuple[str, float], ...] = (
    ("writing", _WRITING_SECONDS),
    ("planning", _SELLABLE_PLAN_SECONDS),
    ("sampling", _SELLABLE_SAMPLE_SECONDS),
    ("cutting", _CUT_SECONDS),
)


def status_key(brief_id: int) -> str:
    """The Redis key that holds the latest status for one brief."""
    return f"{CHANNEL}:status:{brief_id}"


def clock_key(brief_id: int) -> str:
    """The Redis key that holds when work on one brief started."""
    return f"{CHANNEL}:clock:{brief_id}"


def estimate(
    *,
    elapsed: float,
    phase: str,
    personal: bool,
    step: int = 0,
    total_steps: int = 0,
    sample_elapsed: float | None = None,
    phase_elapsed: float | None = None,
) -> tuple[float, float]:
    """The expected total and the time still left, in seconds."""
    spent = max(0.0, elapsed)
    if phase in {"done", "failed"}:
        return _round(spent), 0.0

    in_phase = _time_in_phase(
        spent,
        phase,
        sample_elapsed=sample_elapsed,
        phase_elapsed=phase_elapsed,
    )
    remaining = _remaining(
        phase,
        in_phase,
        personal=personal,
        step=step,
        total_steps=total_steps,
    )
    return _round(spent + remaining), _round(remaining)


def image_status(
    *,
    brief_id: int,
    phase: str,
    elapsed: float,
    personal: bool,
    step: int = 0,
    total_steps: int = 0,
    sample_elapsed: float | None = None,
    phase_elapsed: float | None = None,
) -> ImageStatus:
    """One status message for a brief."""
    total, remaining = estimate(
        elapsed=elapsed,
        phase=phase,
        personal=personal,
        step=step,
        total_steps=total_steps,
        sample_elapsed=sample_elapsed,
        phase_elapsed=phase_elapsed,
    )
    return ImageStatus(
        brief_id=brief_id,
        phase=phase,
        step=step,
        total_steps=total_steps,
        elapsed_seconds=_round(max(0.0, elapsed)),
        estimated_total_seconds=total,
        remaining_seconds=remaining,
    )


@dataclass(frozen=True)
class ImageStatus:
    """The payload published for the GraphQL subscription."""

    brief_id: int
    phase: str
    step: int
    total_steps: int
    elapsed_seconds: float
    estimated_total_seconds: float
    remaining_seconds: float

    @property
    def channel(self) -> str:
        return CHANNEL

    def to_json(self) -> str:
        return json.dumps(
            {
                "briefId": self.brief_id,
                "phase": self.phase,
                "step": self.step,
                "totalSteps": self.total_steps,
                "elapsedSeconds": self.elapsed_seconds,
                "estimatedTotalSeconds": self.estimated_total_seconds,
                "remainingSeconds": self.remaining_seconds,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> ImageStatus:
        value: dict[str, Any] = json.loads(payload)
        return cls(
            brief_id=int(value["briefId"]),
            phase=str(value["phase"]),
            step=int(value["step"]),
            total_steps=int(value["totalSteps"]),
            elapsed_seconds=float(value["elapsedSeconds"]),
            estimated_total_seconds=float(value["estimatedTotalSeconds"]),
            remaining_seconds=float(value["remainingSeconds"]),
        )


def publish_status(client: Any, status: ImageStatus) -> None:
    """Publish one status message and keep it for a subscriber that connects late."""
    payload = status.to_json()
    client.publish(CHANNEL, payload)
    setter = getattr(client, "set", None)
    if setter is None:
        return
    try:
        setter(status_key(status.brief_id), payload)
    except Exception:
        logger.warning("Image generation status was not stored", exc_info=True)


def publishing(client: Any) -> Callable[[ImageStatus], None]:
    """A publisher that keeps the render going when Redis is briefly down."""

    def publish(status: ImageStatus) -> None:
        try:
            publish_status(client, status)
        except Exception:
            logger.warning("Image generation status was not published", exc_info=True)

    return publish


def load_clock(client: Any, brief_id: int) -> float | None:
    """When work on this brief started, or None when this render stands alone."""
    try:
        raw = client.get(clock_key(brief_id))
    except Exception:
        logger.warning("Image generation clock was not read", exc_info=True)
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def save_clock(client: Any, brief_id: int, started: float) -> None:
    """Remember when work on this brief started so the image task can continue it."""
    try:
        client.set(clock_key(brief_id), str(started))
    except Exception:
        logger.warning("Image generation clock was not stored", exc_info=True)


class GenerationWatch:
    """Turns phase changes into status messages for one brief.

    A Redis client lets a later task continue the clock this watch started.
    Without one, or when no clock is stored, the watch starts at the phase it
    is given. A regeneration does that: it begins at sampling.
    """

    def __init__(
        self,
        brief_id: int,
        *,
        personal: bool,
        publish: Callable[[ImageStatus], None],
        clock: Callable[[], float] | None = None,
        client: Any | None = None,
        started: float | None = None,
    ) -> None:
        self.brief_id = brief_id
        self.personal = personal
        self._publish = publish
        self._clock = clock or time.time
        self._phase: str | None = None
        self._sample_elapsed: float | None = None
        self._started = self._resolve_start(started, client)
        self._phase_started = self._clock()

    def phase(self, phase: str, step: int = 0, total_steps: int = 0) -> None:
        now = self._clock()
        if phase != self._phase:
            if phase == "sampling":
                self._sample_elapsed = None
            elif phase == "cutting" and self._sample_elapsed is None:
                self._sample_elapsed = now - self._started
            self._phase = phase
            self._phase_started = now
        elapsed = now - self._started
        self._publish(
            image_status(
                brief_id=self.brief_id,
                phase=phase,
                elapsed=elapsed,
                personal=self.personal,
                step=step,
                total_steps=total_steps,
                sample_elapsed=self._sample_elapsed,
                phase_elapsed=now - self._phase_started,
            )
        )

    def _resolve_start(self, started: float | None, client: Any | None) -> float:
        if started is not None:
            return started
        if client is not None:
            loaded = load_clock(client, self.brief_id)
            if loaded is not None:
                return loaded
        start = self._clock()
        if client is not None:
            save_clock(client, self.brief_id, start)
        return start


def _phases(personal: bool) -> tuple[tuple[str, float], ...]:
    return _PERSONAL_PHASES if personal else _SELLABLE_PHASES


def _time_in_phase(
    spent: float,
    phase: str,
    *,
    sample_elapsed: float | None,
    phase_elapsed: float | None,
) -> float:
    if phase_elapsed is not None:
        return max(0.0, phase_elapsed)
    if phase == "cutting" and sample_elapsed is not None:
        return max(0.0, spent - sample_elapsed)
    return spent


def _remaining(
    phase: str,
    in_phase: float,
    *,
    personal: bool,
    step: int,
    total_steps: int,
) -> float:
    phases = _phases(personal)
    names = [name for name, _budget in phases]
    if phase not in names:
        return 0.0
    index = names.index(phase)
    remaining = 0.0
    for cursor, (name, budget) in enumerate(phases):
        if cursor < index:
            continue
        current = budget
        if name == "sampling" and step > 0 and total_steps > 0 and phase == "sampling":
            current = in_phase / step * total_steps
        if cursor == index:
            remaining += max(0.0, current - in_phase)
        else:
            remaining += current
    return remaining


def _round(value: float) -> float:
    return round(value, 1)
