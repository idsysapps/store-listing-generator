"""Unit tests for trend discovery that does not start from the seed pool.

Every harvester used to be fed the active seeds, so a new trend could only
be an expansion of a seed that was itself derived from an earlier seed. A
holiday reproduced itself and a non-seasonal idea could only enter the pool
through the calendar (issue #201).
"""

from unittest.mock import MagicMock

from store_listing.ingest.trends.discovery import (
    DISCOVERY_CATEGORIES,
    DiscoveryHarvester,
    DiscoveryRequest,
    TrendingTopic,
    parse_trending_rss,
)
from store_listing.orchestration.promotion import (
    compute_novelty_score,
    echoes_seed,
)

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:ht="https://trends.google.com/trending/rss" version="2.0">
  <channel>
    <item>
      <title>National Dog Day</title>
      <ht:approx_traffic>50K+</ht:approx_traffic>
      <link>https://trends.google.com/trending?geo=US</link>
    </item>
    <item>
      <title>Crochet Blanket</title>
      <ht:approx_traffic>20K+</ht:approx_traffic>
    </item>
  </channel>
</rss>
"""


class TestParseTrendingRss:
    def test_reads_the_trending_topics(self) -> None:
        topics = parse_trending_rss(RSS)

        assert [topic.title for topic in topics] == ["National Dog Day", "Crochet Blanket"]

    def test_reads_the_traffic(self) -> None:
        topics = parse_trending_rss(RSS)

        assert topics[0].traffic == "50K+"

    def test_an_empty_feed_reads_as_no_topics(self) -> None:
        assert parse_trending_rss("<rss><channel></channel></rss>") == []

    def test_a_broken_feed_reads_as_no_topics(self) -> None:
        assert parse_trending_rss("not xml at all") == []


class TestDiscoveryHarvester:
    def _db(self) -> MagicMock:
        db = MagicMock()
        db.insert_trend_query.return_value = 1
        db.insert_trend_score.return_value = 1
        db.upsert_seed_candidate.return_value = 1
        return db

    def test_the_trending_topics_are_stored_as_discoveries(self) -> None:
        db = self._db()
        trending = MagicMock()
        trending.trending.return_value = [TrendingTopic("National Dog Day", "50K+")]
        amazon = MagicMock()
        amazon.suggest.return_value = []
        harvester = DiscoveryHarvester(db_client=db, trending=trending, amazon=amazon)

        harvester.harvest_and_store(DiscoveryRequest(categories=("pets",)))

        source = db.insert_trend_query.call_args.kwargs["source"]
        assert source == "discovery"

    def test_a_discovery_names_the_category_it_came_from(self) -> None:
        db = self._db()
        trending = MagicMock()
        trending.trending.return_value = []
        amazon = MagicMock()
        amazon.suggest.return_value = ["dog shirt"]
        harvester = DiscoveryHarvester(db_client=db, trending=trending, amazon=amazon)

        harvester.harvest_and_store(DiscoveryRequest(categories=("pets",)))

        assert db.insert_trend_query.call_args.kwargs["seed_keyword"] == "pets"

    def test_the_marketplace_prefixes_are_searched(self) -> None:
        db = self._db()
        trending = MagicMock()
        trending.trending.return_value = []
        amazon = MagicMock()
        amazon.suggest.return_value = ["dog shirt", "dog mug"]
        harvester = DiscoveryHarvester(db_client=db, trending=trending, amazon=amazon)

        harvester.harvest_and_store(DiscoveryRequest(categories=("pets",)))

        prefixes = [call.args[0] for call in amazon.suggest.call_args_list]
        assert prefixes == list(DISCOVERY_CATEGORIES["pets"])

    def test_a_failing_source_does_not_stop_the_harvest(self) -> None:
        db = self._db()
        trending = MagicMock()
        trending.trending.side_effect = RuntimeError("boom")
        amazon = MagicMock()
        amazon.suggest.return_value = ["dog shirt"]
        harvester = DiscoveryHarvester(db_client=db, trending=trending, amazon=amazon)

        results, failed = harvester.harvest_and_store(DiscoveryRequest(categories=("pets",)))

        assert results
        assert failed


class TestNoveltyScoring:
    def test_a_candidate_that_repeats_its_seed_is_ranks_lower(self) -> None:
        base = compute_novelty_score("amazon", "search", score=2, delta=0)
        echo = compute_novelty_score(
            "amazon", "search", score=2, delta=0, query="funny shirt", seed="funny shirt"
        )

        assert echo < base

    def test_a_candidate_that_adds_something_is_not_an_echo(self) -> None:
        assert not echoes_seed("funny shirt for runners", "funny shirt")

    def test_a_candidate_that_only_repeats_its_seed_is_an_echo(self) -> None:
        assert echoes_seed("funny shirt", "funny shirt")

    def test_a_discovery_ranks_higher_than_the_same_trend(self) -> None:
        trend = compute_novelty_score("google", "top", score=50, delta=0)
        discovery = compute_novelty_score("discovery", "top", score=50, delta=0)

        assert discovery > trend

    def test_the_score_stays_in_range(self) -> None:
        score = compute_novelty_score("discovery", "top", score=100, delta=0)

        assert 0 <= score <= 10000
