package com.trends.resolver;

import com.trends.config.ImageStorageConfig;
import com.trends.domain.DesignBrief;
import com.trends.domain.TrendQuery;
import com.trends.domain.TrendScore;
import com.trends.dto.TrendSummary;
import com.trends.repository.DesignBriefRepository;
import com.trends.repository.TrendQueryRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class TrendQueryResolverTest {

    @Mock
    TrendQueryRepository trendQueryRepository;

    @Mock
    DesignBriefRepository designBriefRepository;

    @Mock
    ImageStorageConfig imageStorageConfig;

    @InjectMocks
    TrendQueryResolver trendQueryResolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        when(designBriefRepository.findBySeedKeyword(anyString())).thenReturn(List.of());
    }

    @Test
    void testGetTrendSummary_WhenQueryNotFound_ReturnsNull() {
        when(trendQueryRepository.findByQuery("nonexistent")).thenReturn(null);

        TrendSummary result = trendQueryResolver.getTrendSummary("nonexistent");

        assertNull(result);
        verify(trendQueryRepository).findByQuery("nonexistent");
    }

    @Test
    void testGetTrendSummary_WhenQueryExists_ReturnsSummary() {
        TrendQuery trendQuery = new TrendQuery("hoodie", "funny hoodie");
        trendQuery.id = 1;

        TrendScore score1 = new TrendScore(trendQuery, 75L, 10L, "US");
        TrendScore score2 = new TrendScore(trendQuery, 60L, 5L, "CA");

        when(trendQueryRepository.findByQuery("funny hoodie")).thenReturn(trendQuery);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(score1, score2));

        TrendSummary result = trendQueryResolver.getTrendSummary("funny hoodie");

        assertNotNull(result);
        assertEquals("funny hoodie", result.getQuery());
        assertEquals(75L, result.getLatestScore());
        assertEquals(10L, result.getVelocity());
        assertFalse(result.getTopRegions().isEmpty());
    }

    @Test
    void testGetTopTrends_ReturnsSortedByScoreAndVelocity() {
        TrendQuery tq1 = new TrendQuery("hoodie", "cool hoodie");
        tq1.id = 1;

        TrendQuery tq2 = new TrendQuery("tshirt", "funny tshirt");
        tq2.id = 2;

        when(trendQueryRepository.findTopByLatestScore(10)).thenReturn(List.of(tq2, tq1));
        when(trendQueryRepository.findMaxScoreBySource()).thenReturn(Map.of("google", 100L));

        TrendScore hoodieScore = new TrendScore(tq1, 50L, 5L, "US");
        when(trendQueryRepository.findByQuery("cool hoodie")).thenReturn(tq1);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(hoodieScore));

        TrendScore tshirtScore = new TrendScore(tq2, 100L, 20L, "US");
        when(trendQueryRepository.findByQuery("funny tshirt")).thenReturn(tq2);
        when(trendQueryRepository.findLatestScoresByQueryId(2, 10)).thenReturn(List.of(tshirtScore));

        List<TrendSummary> result = trendQueryResolver.getTopTrends(10, false);

        assertEquals(2, result.size());
        assertEquals("funny tshirt", result.get(0).getQuery());
        assertEquals("cool hoodie", result.get(1).getQuery());
    }

    @Test
    void testGetTopTrends_NormalizesScoresWithinSource() {
        TrendQuery tq1 = new TrendQuery("hoodie", "cool hoodie");
        tq1.id = 1;

        when(trendQueryRepository.findTopByLatestScore(10)).thenReturn(List.of(tq1));
        when(trendQueryRepository.findMaxScoreBySource()).thenReturn(
                Map.of("youtube", 1000000L, "google", 100L));

        TrendScore youtubeScore = new TrendScore(tq1, 500000L, 5L, "US", "youtube");
        when(trendQueryRepository.findByQuery("cool hoodie")).thenReturn(tq1);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(youtubeScore));

        List<TrendSummary> result = trendQueryResolver.getTopTrends(10, false);

        assertEquals(1, result.size());
        assertEquals(50, result.get(0).getNormalizedScore());
    }

    @Test
    void testGetTrendSummary_ExposesSourceFromLatestScore() {
        TrendQuery trendQuery = new TrendQuery("hoodie", "funny hoodie");
        trendQuery.id = 1;

        TrendScore score1 = new TrendScore(trendQuery, 75L, 10L, "US", "google");
        TrendScore score2 = new TrendScore(trendQuery, 60L, 5L, "CA");

        when(trendQueryRepository.findByQuery("funny hoodie")).thenReturn(trendQuery);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(score1, score2));

        TrendSummary result = trendQueryResolver.getTrendSummary("funny hoodie");

        assertEquals("google", result.getSource());
    }

    @Test
    void testGetTrendSummary_IncludesSeedKeyword() {
        TrendQuery trendQuery = new TrendQuery("hoodie", "funny hoodie");
        trendQuery.id = 1;

        TrendScore score = new TrendScore(trendQuery, 75L, 10L, "US", "youtube");

        when(trendQueryRepository.findByQuery("funny hoodie")).thenReturn(trendQuery);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(score));

        TrendSummary result = trendQueryResolver.getTrendSummary("funny hoodie");

        assertEquals("hoodie", result.getSeedKeyword());
        assertNotNull(result.getDesignBriefs());
    }

    @Test
    void testGetTopTrends_HasDesignBriefsTrue_FiltersOnlyWithBriefs() {
        TrendQuery tq1 = new TrendQuery("hoodie", "cool hoodie");
        tq1.id = 1;

        TrendQuery tq2 = new TrendQuery("tshirt", "funny tshirt");
        tq2.id = 2;

        when(trendQueryRepository.findTopByLatestScore(50)).thenReturn(List.of(tq1, tq2));
        when(trendQueryRepository.findMaxScoreBySource()).thenReturn(Map.of("google", 100L));

        TrendScore hoodieScore = new TrendScore(tq1, 50L, 5L, "US");
        when(trendQueryRepository.findByQuery("cool hoodie")).thenReturn(tq1);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(hoodieScore));

        TrendScore tshirtScore = new TrendScore(tq2, 100L, 20L, "US");
        when(trendQueryRepository.findByQuery("funny tshirt")).thenReturn(tq2);
        when(trendQueryRepository.findLatestScoresByQueryId(2, 10)).thenReturn(List.of(tshirtScore));

        // Only tq1 has design briefs
        DesignBrief brief = new DesignBrief();
        brief.id = 1;
        brief.concept = "Cool Hoodie Design";
        brief.productType = "dtf_apparel";
        brief.confidence = 80;
        brief.sources = List.of();
        when(designBriefRepository.findBySeedKeyword("hoodie")).thenReturn(List.of(brief));
        when(designBriefRepository.findBySeedKeyword("tshirt")).thenReturn(List.of());

        List<TrendSummary> result = trendQueryResolver.getTopTrends(10, true);

        assertEquals(1, result.size());
        assertEquals("cool hoodie", result.get(0).getQuery());
    }

    @Test
    void testGetTopTrends_HasDesignBriefsFalse_ReturnsAll() {
        TrendQuery tq1 = new TrendQuery("hoodie", "cool hoodie");
        tq1.id = 1;

        when(trendQueryRepository.findTopByLatestScore(10)).thenReturn(List.of(tq1));
        when(trendQueryRepository.findMaxScoreBySource()).thenReturn(Map.of("google", 100L));

        TrendScore hoodieScore = new TrendScore(tq1, 50L, 5L, "US");
        when(trendQueryRepository.findByQuery("cool hoodie")).thenReturn(tq1);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(hoodieScore));

        List<TrendSummary> result = trendQueryResolver.getTopTrends(10, false);

        assertEquals(1, result.size());
    }

    @Test
    void testGetScores_WithNullTrendQuery_ReturnsEmptyList() {
        List<com.trends.domain.TrendScore> result = trendQueryResolver.getScores(null, 10, 0);
        assertTrue(result.isEmpty());
    }
}
