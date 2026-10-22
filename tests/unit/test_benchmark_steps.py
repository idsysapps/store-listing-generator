"""Unit: the step benchmark harness.

The step count of the image model was a constant, so nobody could measure
whether the hardware and the prompt style do better at more or fewer steps
(issue #210). The harness renders a fixed set of briefs at several step counts
with the same seeds and reports what each run cost.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from store_listing.orchestration import benchmark_steps
from store_listing.orchestration.benchmark_steps import (
    BENCHMARK_BRIEFS,
    BenchmarkBrief,
    BenchmarkResult,
    run,
    summarise,
)


def _result(steps: int, seconds: float, artwork: bool | None = True, error: str | None = None):
    return BenchmarkResult(
        brief="Coffee Beans",
        product_type="dtf_apparel",
        steps=steps,
        cfg=1.0,
        seed=1,
        seconds=seconds,
        artwork_ok=artwork,
        lettering_ok=artwork,
        error=error,
        image_path=None,
    )


class TestTheBenchmarkSet:
    def test_it_covers_the_product_types_and_the_layouts(self) -> None:
        product_types = {brief.product_type for brief in BENCHMARK_BRIEFS}
        layouts = {brief.layout_type for brief in BENCHMARK_BRIEFS}

        assert "dtf_apparel" in product_types
        assert "sublimation" in product_types
        assert "sticker_vinyl" in product_types
        assert "full_bleed" in layouts
        assert "text_top" in layouts

    def test_every_brief_has_a_scene_to_draw(self) -> None:
        assert all(brief.scene_description for brief in BENCHMARK_BRIEFS)

    def test_a_brief_becomes_a_design_brief(self) -> None:
        brief = BenchmarkBrief(
            name="Coffee Beans",
            product_type="dtf_apparel",
            scene_description="scattered roasted coffee beans in a flat graphic",
        )

        design_brief = brief.to_design_brief()

        assert design_brief.concept == "Coffee Beans"
        assert design_brief.product_type == "dtf_apparel"
        assert design_brief.scene_description is not None


class TestTheRun:
    def test_every_brief_is_rendered_at_every_step_count(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.generate_image.return_value = MagicMock(image_bytes=b"png", error=None)

        with (
            patch.object(benchmark_steps, "_client", return_value=client),
            patch.object(benchmark_steps, "_validators", return_value=(None, None)),
        ):
            results = run(
                briefs=(BENCHMARK_BRIEFS[0], BENCHMARK_BRIEFS[1]),
                step_counts=(4, 8),
                output_dir=tmp_path,
            )

        assert len(results) == 4
        assert client.generate_image.call_count == 4

    def test_every_render_uses_the_same_seed(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.generate_image.return_value = MagicMock(image_bytes=b"png", error=None)

        with (
            patch.object(benchmark_steps, "_client", return_value=client),
            patch.object(benchmark_steps, "_validators", return_value=(None, None)),
        ):
            run(
                briefs=(BENCHMARK_BRIEFS[0],),
                step_counts=(4, 6, 8),
                seed=99,
                output_dir=tmp_path,
            )

        seeds = {call.kwargs["seed"] for call in client.generate_image.call_args_list}
        assert seeds == {99}

    def test_a_failed_render_is_reported(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.generate_image.return_value = MagicMock(image_bytes=None, error="no image")

        with (
            patch.object(benchmark_steps, "_client", return_value=client),
            patch.object(benchmark_steps, "_validators", return_value=(None, None)),
        ):
            results = run(briefs=(BENCHMARK_BRIEFS[0],), step_counts=(4,), output_dir=tmp_path)

        assert results[0].error == "no image"

    def test_the_images_are_written_next_to_the_report(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.generate_image.return_value = MagicMock(image_bytes=b"png", error=None)

        with (
            patch.object(benchmark_steps, "_client", return_value=client),
            patch.object(benchmark_steps, "_validators", return_value=(None, None)),
        ):
            results = run(briefs=(BENCHMARK_BRIEFS[0],), step_counts=(4,), output_dir=tmp_path)

        assert Path(results[0].image_path or "").exists()


class TestTheSummary:
    def test_it_counts_the_gates_and_the_time_per_step_count(self) -> None:
        summary = summarise(
            [
                _result(4, 10.0),
                _result(4, 12.0),
                _result(8, 20.0),
            ]
        )

        assert summary["4"]["rendered"] == 2
        assert summary["4"]["artwork_passed"] == 2
        assert summary["4"]["lettering_passed"] == 2
        assert summary["4"]["seconds_total"] == 22.0
        assert summary["4"]["seconds_per_image"] == 11.0
        assert summary["8"]["rendered"] == 1

    def test_a_failed_render_is_counted_as_failed(self) -> None:
        summary = summarise([_result(4, 1.0, error="no image")])

        assert summary["4"]["failed"] == 1
        assert summary["4"]["rendered"] == 0
