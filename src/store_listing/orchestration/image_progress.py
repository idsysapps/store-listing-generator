"""How long one image generation is expected to take.

A personal FLUX.2 graphic is about four minutes of sampling plus the
background cut. A sellable Flux Schnell graphic is much shorter. Once a step
count arrives, the guess is replaced by the pace of the steps so far.
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
_PERSONAL_SAMPLE_SECONDS = 240.0
_SELLABLE_SAMPLE_SECONDS = 20.0
_CUT_SECONDS = 40.0


def estimate(
    *,
    elapsed: float,
    phase: str,
    personal: bool,
    step: int = 0,
    total_steps: int = 0,
    sample_elapsed: float | None = None,
) -> tuple[float, float]:
    """The expected total and the time still left, in seconds."""
    spent = max(0.0, elapsed)
    if phase in {"done", "failed"}:
        return _round(spent), 0.0

    sample_budget = _PERSONAL_SAMPLE_SECONDS if personal else _SELLABLE_SAMPLE_SECONDS
    if phase == "cutting":
        sampled = sample_elapsed if sample_elapsed is not None else min(spent, sample_budget)
        cut_spent = max(0.0, spent - sampled)
        if cut_spent <= _CUT_SECONDS:
            return _round(sampled + _CUT_SECONDS), _round(_CUT_SECONDS - cut_spent)
        return _round(spent), 0.0

    if step > 0 and total_steps > 0:
        sample_estimate = spent / step * total_steps
        remaining = max(0.0, sample_estimate - spent) + _CUT_SECONDS
        return _round(sample_estimate + _CUT_SECONDS), _round(remaining)

    if spent <= sample_budget:
        remaining = sample_budget - spent + _CUT_SECONDS
        return _round(sample_budget + _CUT_SECONDS), _round(remaining)
    return _round(spent + _CUT_SECONDS), _round(_CUT_SECONDS)


def image_status(
    *,
    brief_id: int,
    phase: str,
    elapsed: float,
    personal: bool,
    step: int = 0,
    total_steps: int = 0,
    sample_elapsed: float | None = None,
) -> ImageStatus:
    """One status message for a brief."""
    total, remaining = estimate(
        elapsed=elapsed,
        phase=phase,
        personal=personal,
        step=step,
        total_steps=total_steps,
        sample_elapsed=sample_elapsed,
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
    """Publish one status message. The client is a Redis connection."""
    client.publish(CHANNEL, status.to_json())


def publishing(client: Any) -> Callable[[ImageStatus], None]:
    """A publisher that keeps the render going when Redis is briefly down."""

    def publish(status: ImageStatus) -> None:
        try:
            publish_status(client, status)
        except Exception:
            logger.warning("Image generation status was not published", exc_info=True)

    return publish


class GenerationWatch:
    """Turns phase changes into status messages for one brief."""

    def __init__(
        self,
        brief_id: int,
        *,
        personal: bool,
        publish: Callable[[ImageStatus], None],
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.brief_id = brief_id
        self.personal = personal
        self._publish = publish
        self._clock = clock or time.monotonic
        self._started = self._clock()
        self._sample_elapsed: float | None = None

    def phase(self, phase: str) -> None:
        elapsed = self._clock() - self._started
        if phase == "sampling":
            self._sample_elapsed = None
        elif phase == "cutting" and self._sample_elapsed is None:
            self._sample_elapsed = elapsed
        self._publish(
            image_status(
                brief_id=self.brief_id,
                phase=phase,
                elapsed=elapsed,
                personal=self.personal,
                sample_elapsed=self._sample_elapsed,
            )
        )


def _round(value: float) -> float:
    return round(value, 1)
