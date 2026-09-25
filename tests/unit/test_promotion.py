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

    def test_tiktok_hashtag_uses_virality_plays(self) -> None:
        assert compute_promotion_score("tiktok", "hashtag", score=150000, delta=12) == 150000

    def test_tiktok_sound_uses_play_traction(self) -> None:
        assert compute_promotion_score("tiktok", "sound", score=120000, delta=15) == 120000

    def test_pinterest_search_uses_repins(self) -> None:
        assert compute_promotion_score("pinterest", "search", score=600, delta=4) == 600

    def test_pinterest_board_uses_repins(self) -> None:
        assert compute_promotion_score("pinterest", "board", score=1200, delta=9) == 1200

    def test_marketplace_search_uses_score(self) -> None:
        assert compute_promotion_score("amazon", "search", score=1, delta=0) == 1
        assert compute_promotion_score("etsy", "search", score=1, delta=0) == 1

    def test_amazon_bsr_riser_remains_zero_until_amazon_ingester(self) -> None:
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

    def test_tiktok_hashtag_virality_promotes(self) -> None:
        results = candidates_to_promote(
            [
                candidate(
                    1,
                    "#pickleballgift",
                    source="tiktok",
                    query_type="hashtag",
                    score=150000,
                    delta=6,
                )
            ],
            cross_seed_counts={"#pickleballgift": 1},
        )
        assert results == [1]

    def test_tiktok_hashtag_below_virality_stays_pending(self) -> None:
        assert (
            candidates_to_promote(
                [
                    candidate(
                        1, "#corgi", source="tiktok", query_type="hashtag", score=500, delta=100
                    )
                ],
                cross_seed_counts={},
            )
            == []
        )

    def test_tiktok_cross_seed_promotes_regardless_of_virality(self) -> None:
        results = candidates_to_promote(
            [candidate(1, "#momsvg", source="tiktok", query_type="hashtag", score=1000, delta=1)],
            cross_seed_counts={"#momsvg": 2},
        )
        assert results == [1]

    def test_tiktok_sound_usage_promotes(self) -> None:
        results = candidates_to_promote(
            [
                candidate(
                    1, "sunset tones", source="tiktok", query_type="sound", score=120000, delta=12
                )
            ],
            cross_seed_counts={},
        )
        assert results == [1]

    def test_tiktok_sound_below_usage_stays_pending(self) -> None:
        assert (
            candidates_to_promote(
                [candidate(1, "lo fi", source="tiktok", query_type="sound", score=120000, delta=4)],
                cross_seed_counts={},
            )
            == []
        )

    def test_pinterest_search_traction_promotes(self) -> None:
        results = candidates_to_promote(
            [
                candidate(
                    1,
                    "pickleball mom shirt",
                    source="pinterest",
                    query_type="search",
                    score=600,
                    delta=4,
                )
            ],
            cross_seed_counts={},
        )
        assert results == [1]

    def test_pinterest_search_below_repins_stays_pending(self) -> None:
        assert (
            candidates_to_promote(
                [
                    candidate(
                        1, "corgi svg", source="pinterest", query_type="search", score=400, delta=10
                    )
                ],
                cross_seed_counts={},
            )
            == []
        )

    def test_pinterest_board_traction_promotes(self) -> None:
        results = candidates_to_promote(
            [
                candidate(
                    1,
                    "pickleball gifts",
                    source="pinterest",
                    query_type="board",
                    score=1200,
                    delta=7,
                )
            ],
            cross_seed_counts={},
        )
        assert results == [1]

    def test_amazon_cross_seed_suggestion_promotes(self) -> None:
        results = candidates_to_promote(
            [candidate(1, "mom shirt svg", source="amazon", query_type="search", score=1, delta=0)],
            cross_seed_counts={"mom shirt svg": 2},
        )
        assert results == [1]

    def test_etsy_single_seed_suggestion_stays_pending(self) -> None:
        assert (
            candidates_to_promote(
                [
                    candidate(
                        1, "pickleball mug", source="etsy", query_type="search", score=1, delta=0
                    )
                ],
                cross_seed_counts={"pickleball mug": 1},
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
