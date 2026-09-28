from store_listing.orchestration.promotion import (
    MAX_ACTIVE_SEEDS,
    STARTER_SEEDS,
    ActiveSeed,
    SeedCandidate,
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
    def test_google_rising_capped_at_10000(self) -> None:
        assert compute_promotion_score("google", "rising", score=10, delta=250000) == 10000

    def test_google_rising_uses_delta(self) -> None:
        assert compute_promotion_score("google", "rising", score=10, delta=5000) == 5000

    def test_google_top_scaled_by_100(self) -> None:
        assert compute_promotion_score("google", "top", score=90, delta=0) == 9000

    def test_google_top_max_maps_to_10000(self) -> None:
        assert compute_promotion_score("google", "top", score=100, delta=0) == 10000

    def test_tiktok_hashtag_divided_by_1000(self) -> None:
        assert compute_promotion_score("tiktok", "hashtag", score=150000, delta=12) == 150

    def test_tiktok_sound_divided_by_1000(self) -> None:
        assert compute_promotion_score("tiktok", "sound", score=120000, delta=15) == 120

    def test_tiktok_capped_at_10000(self) -> None:
        assert compute_promotion_score("tiktok", "hashtag", score=20_000_000, delta=1) == 10000

    def test_pinterest_search_uses_repins(self) -> None:
        assert compute_promotion_score("pinterest", "search", score=600, delta=4) == 600

    def test_pinterest_board_uses_repins(self) -> None:
        assert compute_promotion_score("pinterest", "board", score=1200, delta=9) == 1200

    def test_marketplace_search_scaled_by_500(self) -> None:
        assert compute_promotion_score("amazon", "search", score=1, delta=0) == 500
        assert compute_promotion_score("etsy", "search", score=1, delta=0) == 500

    def test_x_hashtag_divided_by_100(self) -> None:
        assert compute_promotion_score("x", "hashtag", score=150000, delta=8) == 1500

    def test_x_keyword_divided_by_100(self) -> None:
        assert compute_promotion_score("x", "search", score=250000, delta=12) == 2500

    def test_x_capped_at_10000(self) -> None:
        assert compute_promotion_score("x", "hashtag", score=2_000_000, delta=1) == 10000

    def test_reddit_search_capped(self) -> None:
        assert compute_promotion_score("reddit", "search", score=2000, delta=50) == 2000

    def test_reddit_subreddit_capped(self) -> None:
        assert compute_promotion_score("reddit", "subreddit", score=50000, delta=10) == 10000

    def test_youtube_video_divided_by_100k(self) -> None:
        assert compute_promotion_score("youtube", "video", score=500_000_000, delta=100) == 5000

    def test_youtube_hashtag_divided_by_100k(self) -> None:
        assert compute_promotion_score("youtube", "hashtag", score=1_000_000_000, delta=10) == 10000

    def test_youtube_capped_at_10000(self) -> None:
        assert compute_promotion_score("youtube", "video", score=2_000_000_000, delta=1) == 10000

    def test_instagram_hashtag_divided_by_1000(self) -> None:
        assert compute_promotion_score("instagram", "hashtag", score=5_000_000, delta=20) == 5000

    def test_amazon_bsr_riser_remains_zero_until_amazon_ingester(self) -> None:
        assert compute_promotion_score("amazon", "bsr_riser", score=80, delta=60) == 0


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
