package com.trends.resolver;

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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class TrendQueryResolverTest {

    @Mock
    TrendQueryRepository trendQueryRepository;

    @Mock
    DesignBriefRepository designBriefRepository;

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

        TrendScore hoodieScore = new TrendScore(tq1, 50L, 5L, "US");
        when(trendQueryRepository.findByQuery("cool hoodie")).thenReturn(tq1);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(hoodieScore));

        TrendScore tshirtScore = new TrendScore(tq2, 100L, 20L, "US");
        when(trendQueryRepository.findByQuery("funny tshirt")).thenReturn(tq2);
        when(trendQueryRepository.findLatestScoresByQueryId(2, 10)).thenReturn(List.of(tshirtScore));

        List<TrendSummary> result = trendQueryResolver.getTopTrends(10);

        assertEquals(2, result.size());
        assertEquals("funny tshirt", result.get(0).getQuery());
        assertEquals("cool hoodie", result.get(1).getQuery());
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
    void testGetScores_WithNullTrendQuery_ReturnsEmptyList() {
        List<com.trends.domain.TrendScore> result = trendQueryResolver.getScores(null, 10, 0);
        assertTrue(result.isEmpty());
    }
}
