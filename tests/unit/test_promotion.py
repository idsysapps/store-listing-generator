from store_listing.orchestration.promotion import (
    MAX_ACTIVE_SEEDS,
    STARTER_SEEDS,
    ActiveSeed,
    SeedCandidate,
    candidates_to_promote,
    compute_promotion_score,
    enforce_cap,
)


def candidate(
    cid: int,
    query: str,
    source: str = "google",
    query_type: str = "rising",
    score: int = 0,
    delta: int = 0,
    status: str = "pending",
) -> SeedCandidate:
    return SeedCandidate(
        id=cid,
        query=query,
        source=source,
        query_type=query_type,
        score=score,
        delta=delta,
        promotion_score=compute_promotion_score(source, query_type, score, delta),
        status=status,
    )


class TestComputePromotionScore:
    def test_google_rising_uses_delta(self) -> None:
        assert compute_promotion_score("google", "rising", score=10, delta=250000) == 250000

    def test_google_top_uses_score(self) -> None:
        assert compute_promotion_score("google", "top", score=90, delta=0) == 90

    def test_non_google_sources_are_zero_for_now(self) -> None:
        assert compute_promotion_score("tiktok", "hashtag", score=500, delta=99) == 0
        assert compute_promotion_score("amazon", "bsr_riser", score=80, delta=60) == 0


class TestCandidatesToPromote:
    def test_rising_delta_at_threshold_promotes(self) -> None:
        results = candidates_to_promote(
            [candidate(1, "mom shirt", delta=5000)],
            cross_seed_counts={},
        )
        assert results == [1]

    def test_rising_delta_below_threshold_stays_pending(self) -> None:
        assert (
            candidates_to_promote(
                [candidate(1, "mom shirt", delta=4999)],
                cross_seed_counts={},
            )
            == []
        )

    def test_top_cross_seed_promotes(self) -> None:
        results = candidates_to_promote(
            [candidate(1, "pickleball", query_type="top", score=80)],
            cross_seed_counts={"pickleball": 2},
        )
        assert results == [1]

    def test_top_single_seed_stays_pending(self) -> None:
        assert (
            candidates_to_promote(
                [candidate(1, "pickleball", query_type="top", score=80)],
                cross_seed_counts={"pickleball": 1},
            )
            == []
        )

    def test_non_pending_candidates_are_ignored(self) -> None:
        assert (
            candidates_to_promote(
                [candidate(1, "past", delta=5000, status="promoted")],
                cross_seed_counts={},
            )
            == []
        )

    def test_tiktok_and_amazon_sources_not_promoted_yet(self) -> None:
        assert (
            candidates_to_promote(
                [
                    candidate(1, "viral", source="tiktok", query_type="hashtag", score=1000),
                    candidate(2, "bsr", source="amazon", query_type="bsr_riser", score=1000),
                ],
                cross_seed_counts={},
            )
            == []
        )


class TestEnforceCap:
    def test_no_archive_when_under_cap(self) -> None:
        active = [ActiveSeed(1, "a", 10), ActiveSeed(2, "b", 20)]
        assert enforce_cap(active, incoming=2) == []

    def test_archives_lowest_when_over_cap(self) -> None:
        active = [ActiveSeed(1, "a", 10), ActiveSeed(2, "b", 20)]
        # len(active)=2 + incoming 1 = 3; max would be exceeded only if len active == cap.
        assert enforce_cap(active, incoming=2, max_active=3) == ["a"]

    def test_archives_multiple_when_far_over(self) -> None:
        active = [
            ActiveSeed(1, "lowest", 1),
            ActiveSeed(2, "mid", 50),
            ActiveSeed(3, "highest", 100),
        ]
        assert enforce_cap(active, incoming=1, max_active=2) == ["lowest", "mid"]

    def test_starting_at_cap_keeps_only_best(self) -> None:
        active = [ActiveSeed(1, "l", 5), ActiveSeed(2, "h", 99)]
        archived = enforce_cap(active, incoming=2, max_active=3)
        assert archived == ["l"]


class TestConstants:
    def test_max_active_seeds_is_fifty(self) -> None:
        assert MAX_ACTIVE_SEEDS == 50

    def test_starter_seeds_are_fifteen(self) -> None:
        assert len(STARTER_SEEDS) == 15
