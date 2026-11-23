"""Time to completion for one image generation."""

from store_listing.orchestration.image_progress import (
    CHANNEL,
    GenerationWatch,
    ImageStatus,
    estimate,
    image_status,
    publish_status,
    publishing,
    status_key,
)


def test_a_personal_render_starts_with_about_four_minutes_plus_the_cut() -> None:
    total, remaining = estimate(elapsed=0, phase="sampling", personal=True)

    assert total == 280
    assert remaining == 280


def test_sampling_counts_down_and_keeps_the_cut_in_the_total() -> None:
    total, remaining = estimate(elapsed=10, phase="sampling", personal=True)

    assert total == 280
    assert remaining == 270


def test_a_step_count_replaces_the_guess() -> None:
    total, remaining = estimate(
        elapsed=50,
        phase="sampling",
        personal=True,
        step=5,
        total_steps=20,
    )

    assert total == 240
    assert remaining == 190


def test_running_past_the_budget_keeps_the_cut_ahead() -> None:
    total, remaining = estimate(elapsed=250, phase="sampling", personal=True)

    assert total == 290
    assert remaining == 40


def test_a_long_cut_reports_the_time_already_spent() -> None:
    total, remaining = estimate(
        elapsed=260,
        phase="cutting",
        personal=True,
        sample_elapsed=200,
    )

    assert total == 260
    assert remaining == 0


def test_the_cut_counts_down_from_the_moment_sampling_finished() -> None:
    total, remaining = estimate(
        elapsed=210,
        phase="cutting",
        personal=True,
        sample_elapsed=200,
    )

    assert total == 240
    assert remaining == 30


def test_done_reports_the_actual_total_and_nothing_left() -> None:
    total, remaining = estimate(elapsed=196, phase="done", personal=True)

    assert total == 196
    assert remaining == 0


def test_a_sellable_render_uses_the_short_budget() -> None:
    total, remaining = estimate(elapsed=0, phase="sampling", personal=False)

    assert total == 60
    assert remaining == 60


def test_a_personal_brief_starts_with_research_and_the_render() -> None:
    total, remaining = estimate(elapsed=0, phase="writing", personal=True)

    # writing 40, research 50, study 40, planning 30, sampling 240, cut 40
    assert total == 440
    assert remaining == 440


def test_writing_counts_down_and_keeps_the_later_work() -> None:
    total, remaining = estimate(elapsed=10, phase="writing", personal=True)

    assert total == 440
    assert remaining == 430


def test_a_long_writing_phase_keeps_the_later_work_ahead() -> None:
    total, remaining = estimate(elapsed=50, phase="writing", personal=True)

    assert total == 450
    assert remaining == 400


def test_a_sellable_brief_skips_research_and_study() -> None:
    total, remaining = estimate(elapsed=0, phase="writing", personal=False)

    # writing 40, planning 35, sampling 20, cut 40
    assert total == 135
    assert remaining == 135


def test_research_counts_the_searches_without_changing_the_budget() -> None:
    status = image_status(
        brief_id=7,
        phase="researching",
        elapsed=40,
        personal=True,
        step=2,
        total_steps=4,
        phase_elapsed=0,
    )

    assert status.step == 2
    assert status.total_steps == 4
    assert status.remaining_seconds == 400
    assert status.estimated_total_seconds == 440


def test_status_json_uses_the_names_the_subscription_reads() -> None:
    status = image_status(
        brief_id=12,
        phase="sampling",
        elapsed=10,
        personal=True,
    )

    payload = status.to_json()
    restored = ImageStatus.from_json(payload)

    assert status.channel == CHANNEL
    assert restored.brief_id == 12
    assert restored.phase == "sampling"
    assert restored.estimated_total_seconds == 280
    assert restored.remaining_seconds == 270
    assert '"briefId": 12' in payload
    assert '"estimatedTotalSeconds": 280.0' in payload or '"estimatedTotalSeconds": 280' in payload


def test_the_watch_counts_from_the_start_and_freezes_sampling_at_the_cut() -> None:
    published: list[ImageStatus] = []
    now = {"t": 100.0}
    watch = GenerationWatch(
        7,
        personal=True,
        publish=published.append,
        clock=lambda: now["t"],
    )

    watch.phase("sampling")
    now["t"] = 150.0
    watch.phase("cutting")
    now["t"] = 160.0
    watch.phase("done")

    assert published[0].remaining_seconds == 280
    assert published[1].phase == "cutting"
    assert published[1].estimated_total_seconds == 90
    assert published[1].remaining_seconds == 40
    assert published[2].estimated_total_seconds == 60
    assert published[2].remaining_seconds == 0


class _MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.messages: list[tuple[str, str]] = []

    def publish(self, channel: str, payload: str) -> None:
        self.messages.append((channel, payload))

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def get(self, key: str) -> str | None:
        return self.values.get(key)


def test_publish_keeps_the_latest_status_for_a_late_subscriber() -> None:
    client = _MemoryRedis()
    status = image_status(brief_id=12, phase="writing", elapsed=0, personal=True)

    publish_status(client, status)

    assert client.messages[0][0] == CHANNEL
    stored = ImageStatus.from_json(client.values[status_key(12)])
    assert stored.phase == "writing"
    assert stored.remaining_seconds == 440


def test_the_image_task_keeps_the_original_elapsed_time() -> None:
    client = _MemoryRedis()
    now = {"t": 1000.0}
    published: list[ImageStatus] = []
    first = GenerationWatch(
        7,
        personal=True,
        publish=published.append,
        clock=lambda: now["t"],
        client=client,
    )
    first.phase("writing")

    now["t"] = 1200.0
    published.clear()
    second = GenerationWatch(
        7,
        personal=True,
        publish=published.append,
        clock=lambda: now["t"],
        client=client,
    )
    second.phase("sampling")

    assert published[0].phase == "sampling"
    assert published[0].elapsed_seconds == 200
    assert published[0].remaining_seconds == 280


def test_a_redis_outage_does_not_stop_the_render() -> None:
    class Down:
        def publish(self, channel: str, payload: str) -> None:
            raise ConnectionError(channel)

    publishing(Down())(image_status(brief_id=1, phase="sampling", elapsed=0, personal=False))
