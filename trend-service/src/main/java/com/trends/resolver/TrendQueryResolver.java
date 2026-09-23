package com.trends.resolver;

import com.trends.domain.TrendQuery;
import com.trends.domain.TrendScore;
import com.trends.dto.*;
import com.trends.repository.TrendQueryRepository;
import io.quarkus.security.Authenticated;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.Query;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@GraphQLApi
@ApplicationScoped
public class TrendQueryResolver {

    private static final Logger LOG = Logger.getLogger(TrendQueryResolver.class);

    @Inject
    TrendQueryRepository trendQueryRepository;

    @Authenticated
    @Query("trendQueries")
    public TrendQueryConnection getTrendQueries(int limit, int offset) {
        LOG.debugf("Fetching trend queries: limit=%d, offset=%d", limit, offset);

        List<TrendQuery> queries = trendQueryRepository.findPaginated(limit, offset);
        long totalCount = trendQueryRepository.countAll();

        List<TrendQueryEdge> edges = queries.stream()
                .map(q -> new TrendQueryEdge(q, trendQueryRepository.encodeCursor(q.id)))
                .collect(Collectors.toList());

        PageInfo pageInfo = new PageInfo(
                offset + limit < totalCount,
                offset > 0,
                (int) totalCount
        );

        return new TrendQueryConnection(edges, pageInfo);
    }

    @Authenticated
    @Query("trendSummary")
    public TrendSummary getTrendSummary(String query) {
        LOG.debugf("Fetching trend summary for query: %s", query);

        TrendQuery trendQuery = trendQueryRepository.findByQuery(query);
        if (trendQuery == null) {
            return null;
        }

        List<TrendScore> latestScores = trendQueryRepository.findLatestScoresByQueryId(trendQuery.id, 10);

        if (latestScores.isEmpty()) {
            return new TrendSummary(query, 0, 0, List.of());
        }

        TrendScore latest = latestScores.get(0);
        Integer velocity = latest.delta != null ? latest.delta : 0;

        Map<String, Integer> regionScores = latestScores.stream()
                .collect(Collectors.groupingBy(
                        ts -> ts.region != null ? ts.region : "Unknown",
                        Collectors.collectingAndThen(
                                Collectors.maxBy(Comparator.comparingInt(ts -> ts.score != null ? ts.score : 0)),
                                opt -> opt.map(ts -> ts.score).orElse(0)
                        )
                ));

        List<RegionScore> topRegions = regionScores.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(5)
                .map(e -> new RegionScore(e.getKey(), e.getValue()))
                .collect(Collectors.toList());

        return new TrendSummary(query, latest.score, velocity, topRegions);
    }

    @Authenticated
    @Query("topTrends")
    public List<TrendSummary> getTopTrends(int limit) {
        LOG.debugf("Fetching top trends: limit=%d", limit);

        List<TrendQuery> allQueries = trendQueryRepository.listAll();
        List<TrendSummary> summaries = new ArrayList<>();

        for (TrendQuery tq : allQueries) {
            TrendSummary summary = getTrendSummary(tq.query);
            if (summary != null && summary.getLatestScore() != null && summary.getLatestScore() > 0) {
                summaries.add(summary);
            }
        }

        return summaries.stream()
                .sorted((a, b) -> {
                    int scoreCompare = b.getLatestScore().compareTo(a.getLatestScore());
                    if (scoreCompare != 0) return scoreCompare;
                    return b.getVelocity().compareTo(a.getVelocity());
                })
                .limit(limit)
                .collect(Collectors.toList());
    }

    @Authenticated
    @Query("trendHistory")
    public List<TrendScore> getTrendHistory(String query, int days) {
        LOG.debugf("Fetching trend history: query=%s, days=%d", query, days);
        return trendQueryRepository.findScoresByQuery(query, days);
    }

    public List<TrendScore> getScores(TrendQuery trendQuery, int limit, int offset) {
        if (trendQuery == null || trendQuery.id == null) {
            return List.of();
        }
        return trendQueryRepository.findScoresByQueryId(trendQuery.id, limit, offset);
    }
}
