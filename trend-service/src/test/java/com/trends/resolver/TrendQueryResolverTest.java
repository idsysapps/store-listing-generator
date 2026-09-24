package com.trends.resolver;

import com.trends.domain.TrendQuery;
import com.trends.domain.TrendScore;
import com.trends.dto.TrendSummary;
import com.trends.repository.TrendQueryRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class TrendQueryResolverTest {

    @Mock
    TrendQueryRepository trendQueryRepository;

    @InjectMocks
    TrendQueryResolver trendQueryResolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
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

        TrendScore score1 = new TrendScore(trendQuery, 75, 10, "US");
        TrendScore score2 = new TrendScore(trendQuery, 60, 5, "CA");

        when(trendQueryRepository.findByQuery("funny hoodie")).thenReturn(trendQuery);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(score1, score2));

        TrendSummary result = trendQueryResolver.getTrendSummary("funny hoodie");

        assertNotNull(result);
        assertEquals("funny hoodie", result.getQuery());
        assertEquals(75, result.getLatestScore());
        assertEquals(10, result.getVelocity());
        assertFalse(result.getTopRegions().isEmpty());
    }

    @Test
    void testGetTopTrends_ReturnsSortedByScoreAndVelocity() {
        TrendQuery tq1 = new TrendQuery("hoodie", "cool hoodie");
        tq1.id = 1;

        TrendQuery tq2 = new TrendQuery("tshirt", "funny tshirt");
        tq2.id = 2;

        when(trendQueryRepository.listAll()).thenReturn(List.of(tq1, tq2));

        TrendScore hoodieScore = new TrendScore(tq1, 50, 5, "US");
        when(trendQueryRepository.findByQuery("cool hoodie")).thenReturn(tq1);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(hoodieScore));

        TrendScore tshirtScore = new TrendScore(tq2, 100, 20, "US");
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

        TrendScore score1 = new TrendScore(trendQuery, 75, 10, "US", "google");
        TrendScore score2 = new TrendScore(trendQuery, 60, 5, "CA");

        when(trendQueryRepository.findByQuery("funny hoodie")).thenReturn(trendQuery);
        when(trendQueryRepository.findLatestScoresByQueryId(1, 10)).thenReturn(List.of(score1, score2));

        TrendSummary result = trendQueryResolver.getTrendSummary("funny hoodie");

        assertEquals("google", result.getSource());
    }

    @Test
    void testGetScores_WithNullTrendQuery_ReturnsEmptyList() {
        List<com.trends.domain.TrendScore> result = trendQueryResolver.getScores(null, 10, 0);
        assertTrue(result.isEmpty());
    }
}
