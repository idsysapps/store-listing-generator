"""Benchmark the image model at different step counts.

The step count of the image model was a constant in the client, so nobody
could measure whether the hardware and the prompt style do better at more or
fewer steps (issue #210). This harness renders a fixed set of briefs at
several step counts with the same seeds, and reports what each run cost:

- whether the artwork passed the artwork gate (the picture is the design, not
  the product it is printed on)
- whether the lettering passed the lettering gate (the artwork carries no
  words of its own)
- how long the render took
- where the images were written, so they can be compared side by side

It talks to the ComfyUI on the DGX, so it runs where the DGX runs:

    python -m store_listing.orchestration.benchmark_steps

The result is written as JSON, and the images are kept next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from store_listing.orchestration.artwork_validation import (
    ArtworkTextValidator,
    ArtworkValidator,
)
from store_listing.orchestration.comfyui_client import ComfyUIClient
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.image_generation import build_flux_prompt

logger = logging.getLogger(__name__)

DEFAULT_STEP_COUNTS = (4, 6, 8)


@dataclass(frozen=True)
class BenchmarkBrief:
    """One brief of the benchmark set."""

    name: str
    product_type: str
    scene_description: str
    headline_text: str | None = None
    tagline_text: str | None = None
    font_color: str = "#000000"
    layout_type: str = "full_bleed"

    def to_design_brief(self) -> DesignBrief:
        return DesignBrief(
            concept=self.name,
            product_type=self.product_type,
            specific_products=[],
            audience="",
            visual_style="",
            confidence=0,
            reasoning="",
            source_seed_ids=[],
            layout_type=self.layout_type,
            headline_text=self.headline_text,
            tagline_text=self.tagline_text,
            font_color=self.font_color,
            scene_description=self.scene_description,
        )


# The benchmark set: DTF apparel and sublimation, with and without art, so
# the measurement covers the layouts the pipeline actually produces.
BENCHMARK_BRIEFS: tuple[BenchmarkBrief, ...] = (
    BenchmarkBrief(
        name="Coffee Beans",
        product_type="dtf_apparel",
        scene_description="scattered roasted coffee beans in a flat graphic",
    ),
    BenchmarkBrief(
        name="Ocean Waves",
        product_type="dtf_apparel",
        scene_description="stylised ocean waves in a flat graphic",
    ),
    BenchmarkBrief(
        name="Mountain Sunrise",
        product_type="dtf_apparel",
        scene_description="a mountain range at sunrise in a flat graphic",
        headline_text="Worth the Climb",
        layout_type="text_top",
    ),
    BenchmarkBrief(
        name="Funny Cats",
        product_type="dtf_apparel",
        scene_description="three cats sitting on a windowsill in a flat graphic",
    ),
    BenchmarkBrief(
        name="Retro Sunset",
        product_type="sublimation",
        scene_description="a retro sun over a desert in a flat graphic",
    ),
    BenchmarkBrief(
        name="Space Turtle",
        product_type="sublimation",
        scene_description="a turtle floating in space in a flat graphic",
    ),
    BenchmarkBrief(
        name="Floral Pattern",
        product_type="sublimation",
        scene_description="a repeating floral pattern in a flat graphic",
    ),
    BenchmarkBrief(
        name="Pizza Slice",
        product_type="sticker_vinyl",
        scene_description="a slice of pizza in a flat graphic",
    ),
    BenchmarkBrief(
        name="Skateboard",
        product_type="sticker_vinyl",
        scene_description="a skateboard in a flat graphic",
    ),
    BenchmarkBrief(
        name="Forest Animals",
        product_type="dtf_apparel",
        scene_description="a fox and a badger in a forest in a flat graphic",
    ),
)


@dataclass
class BenchmarkResult:
    """What one render cost."""

    brief: str
    product_type: str
    steps: int
    cfg: float
    seed: int
    seconds: float
    artwork_ok: bool | None
    lettering_ok: bool | None
    error: str | None
    image_path: str | None


def _client() -> ComfyUIClient:
    return ComfyUIClient(
        base_url=os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188"),
        timeout_seconds=float(os.environ.get("COMFYUI_TIMEOUT_SECONDS", "300")),
        poll_interval_seconds=float(os.environ.get("COMFYUI_POLL_INTERVAL_SECONDS", "1.0")),
        unet_name=os.environ.get("COMFYUI_UNET_NAME", "flux1-schnell.safetensors"),
    )


def _validators() -> tuple[ArtworkValidator | None, ArtworkTextValidator | None]:
    from store_listing.orchestration.llm_client import (
        get_artwork_vision_model,
        get_vision_client,
    )

    client = get_vision_client()
    model = get_artwork_vision_model()
    return ArtworkValidator(client=client, model=model), ArtworkTextValidator(
        client=client, model=model
    )


def run(
    briefs: tuple[BenchmarkBrief, ...] = BENCHMARK_BRIEFS,
    step_counts: tuple[int, ...] = DEFAULT_STEP_COUNTS,
    cfg: float = 1.0,
    seed: int = 20261021,
    output_dir: Path = Path("benchmark-output"),
) -> list[BenchmarkResult]:
    """Render every brief at every step count and report what it cost."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _client()
    artwork_validator, lettering_validator = _validators()

    results: list[BenchmarkResult] = []
    try:
        for brief in briefs:
            design_brief = brief.to_design_brief()
            prompt = build_flux_prompt(design_brief)
            for steps in step_counts:
                started = time.monotonic()
                error: str | None = None
                artwork_ok: bool | None = None
                lettering_ok: bool | None = None
                image_path: str | None = None

                try:
                    result = client.generate_image(prompt=prompt, steps=steps, cfg=cfg, seed=seed)
                    if result.image_bytes is None:
                        error = result.error or "no image"
                    else:
                        image_path = str(
                            output_dir / f"{brief.name.replace(' ', '_')}-{steps}steps.png"
                        )
                        Path(image_path).write_bytes(result.image_bytes)
                        if artwork_validator is not None:
                            artwork = artwork_validator.validate(result.image_bytes)
                            artwork_ok = not artwork.is_product
                        if lettering_validator is not None:
                            lettering = lettering_validator.validate(result.image_bytes)
                            lettering_ok = not lettering.has_text
                except Exception as e:  # noqa: BLE001
                    error = str(e)

                seconds = time.monotonic() - started
                results.append(
                    BenchmarkResult(
                        brief=brief.name,
                        product_type=brief.product_type,
                        steps=steps,
                        cfg=cfg,
                        seed=seed,
                        seconds=round(seconds, 2),
                        artwork_ok=artwork_ok,
                        lettering_ok=lettering_ok,
                        error=error,
                        image_path=image_path,
                    )
                )
                logger.info(
                    "%s at %d steps: %.1fs, artwork=%s, lettering=%s",
                    brief.name,
                    steps,
                    seconds,
                    artwork_ok,
                    lettering_ok,
                )
    finally:
        client.close()

    return results


def summarise(results: list[BenchmarkResult]) -> dict[str, Any]:
    """The numbers the issue asks for, per step count."""
    summary: dict[str, Any] = {}
    for steps in sorted({result.steps for result in results}):
        at_steps = [result for result in results if result.steps == steps]
        rendered = [result for result in at_steps if result.error is None]
        summary[str(steps)] = {
            "rendered": len(rendered),
            "failed": len(at_steps) - len(rendered),
            "artwork_passed": sum(1 for result in rendered if result.artwork_ok),
            "lettering_passed": sum(1 for result in rendered if result.lettering_ok),
            "seconds_total": round(sum(result.seconds for result in at_steps), 2),
            "seconds_per_image": (
                round(sum(result.seconds for result in at_steps) / len(at_steps), 2)
                if at_steps
                else 0
            ),
        }
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEP_COUNTS))
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-output"))
    args = parser.parse_args()

    results = run(
        step_counts=tuple(args.steps),
        cfg=args.cfg,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    summary = summarise(results)

    report = {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"the images and the report are in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
