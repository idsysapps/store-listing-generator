package com.trends.resolver;

import com.trends.domain.DesignBrief;
import com.trends.domain.TrendQuery;
import com.trends.domain.TrendScore;
import com.trends.dto.*;
import com.trends.repository.DesignBriefRepository;
import com.trends.repository.TrendQueryRepository;
import io.quarkus.security.Authenticated;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.Query;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.Arrays;
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

    @Inject
    DesignBriefRepository designBriefRepository;

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
            return new TrendSummary(query, 0L, 0L, List.of(), null);
        }

        TrendScore latest = latestScores.get(0);
        Long velocity = latest.delta != null ? latest.delta : 0L;
        String source = latest.source != null ? latest.source : "google";

        Map<String, Long> regionScores = latestScores.stream()
                .collect(Collectors.groupingBy(
                        ts -> ts.region != null ? ts.region : "Unknown",
                        Collectors.collectingAndThen(
                                Collectors.maxBy(Comparator.comparingLong(ts -> ts.score != null ? ts.score : 0L)),
                                opt -> opt.map(ts -> ts.score).orElse(0L)
                        )
                ));

        List<RegionScore> topRegions = regionScores.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(5)
                .map(e -> new RegionScore(e.getKey(), e.getValue()))
                .collect(Collectors.toList());

        String seedKeyword = trendQuery.seedKeyword;

        List<DesignBriefSummary> briefs = designBriefRepository.findBySeedKeyword(seedKeyword)
                .stream()
                .map(this::toDesignBriefSummary)
                .collect(Collectors.toList());

        return new TrendSummary(query, seedKeyword, latest.score, velocity, topRegions, source, null, briefs);
    }

    @Authenticated
    @Query("topTrends")
    public List<TrendSummary> getTopTrends(int limit, boolean hasDesignBriefs) {
        LOG.debugf("Fetching top trends: limit=%d, hasDesignBriefs=%s", limit, hasDesignBriefs);

        Map<String, Long> maxScoresBySource = trendQueryRepository.findMaxScoreBySource();

        int fetchLimit = hasDesignBriefs ? limit * 5 : limit;
        List<TrendQuery> topQueries = trendQueryRepository.findTopByLatestScore(fetchLimit);
        List<TrendSummary> summaries = new ArrayList<>();

        for (TrendQuery tq : topQueries) {
            if (summaries.size() >= limit) {
                break;
            }
            TrendSummary summary = getTrendSummary(tq.query);
            if (summary != null) {
                if (hasDesignBriefs && summary.getDesignBriefs().isEmpty()) {
                    continue;
                }
                Long maxForSource = maxScoresBySource.getOrDefault(
                        summary.getSource(), summary.getLatestScore());
                if (maxForSource > 0 && summary.getLatestScore() != null) {
                    summary.setNormalizedScore(
                            (int) (summary.getLatestScore() * 100 / maxForSource));
                }
                summaries.add(summary);
            }
        }

        return summaries;
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

    private DesignBriefSummary toDesignBriefSummary(DesignBrief brief) {
        List<String> products = brief.specificProducts != null
                ? Arrays.asList(brief.specificProducts)
                : List.of();

        List<String> sourceSeeds = brief.sources != null
                ? brief.sources.stream()
                        .map(s -> s.activeSeed.query)
                        .collect(Collectors.toList())
                : List.of();

        return new DesignBriefSummary(
                brief.id, brief.concept, brief.productType,
                products, brief.audience, brief.visualStyle,
                brief.confidence, brief.reasoning, brief.llmModel,
                brief.batchId, brief.createdAt, sourceSeeds,
                brief.imageKeyRaw, brief.imageKeyTransparent);
    }
}
