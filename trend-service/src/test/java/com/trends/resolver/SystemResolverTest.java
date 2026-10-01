package com.trends.resolver;

import com.trends.domain.SourceHealth;
import com.trends.dto.SourceHealthSummary;
import com.trends.dto.SystemStatus;
import com.trends.event.CeleryTaskDispatcher;
import com.trends.repository.SourceHealthRepository;
import io.smallrye.mutiny.Uni;
import jakarta.persistence.EntityManager;
import jakarta.persistence.Query;
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

class SystemResolverTest {

    @Mock
    EntityManager entityManager;

    @Mock
    SourceHealthRepository sourceHealthRepository;

    @Mock
    CeleryTaskDispatcher celeryTaskDispatcher;

    @InjectMocks
    SystemResolver systemResolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    private Query mockNativeQuery(long result) {
        Query query = mock(Query.class);
        when(query.getSingleResult()).thenReturn(result);
        return query;
    }

    @Test
    void testSystemStatus_ReturnsCorrectCounts() {
        Query pendingQ = mockNativeQuery(23760L);
        Query promotedQ = mockNativeQuery(1786L);
        Query archivedQ = mockNativeQuery(0L);
        Query activeSeedsQ = mockNativeQuery(50L);
        Query briefsTotalQ = mockNativeQuery(8L);
        Query briefsWithImagesQ = mockNativeQuery(7L);
        Query briefsPendingImagesQ = mockNativeQuery(1L);
        Query briefsPendingRegenQ = mockNativeQuery(2L);

        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM seed_candidates WHERE status = 'pending'"))
                .thenReturn(pendingQ);
        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM seed_candidates WHERE status = 'promoted'"))
                .thenReturn(promotedQ);
        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM seed_candidates WHERE status = 'archived'"))
                .thenReturn(archivedQ);
        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM active_seeds WHERE archived_at IS NULL"))
                .thenReturn(activeSeedsQ);
        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM design_briefs"))
                .thenReturn(briefsTotalQ);
        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM design_briefs WHERE image_key_raw IS NOT NULL"))
                .thenReturn(briefsWithImagesQ);
        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM design_briefs WHERE image_key_raw IS NULL"))
                .thenReturn(briefsPendingImagesQ);
        when(entityManager.createNativeQuery("SELECT COUNT(*) FROM design_briefs WHERE regeneration_feedback IS NOT NULL AND image_key_raw IS NULL"))
                .thenReturn(briefsPendingRegenQ);

        SystemStatus status = systemResolver.getSystemStatus();

        assertEquals(23760, status.getCandidatesPending());
        assertEquals(1786, status.getCandidatesPromoted());
        assertEquals(0, status.getCandidatesArchived());
        assertEquals(50, status.getActiveSeedsCount());
        assertEquals(8, status.getDesignBriefsTotal());
        assertEquals(7, status.getDesignBriefsWithImages());
        assertEquals(1, status.getDesignBriefsPendingImages());
        assertEquals(2, status.getDesignBriefsPendingRegen());
    }

    @Test
    void testSourceHealth_ReturnsSummaries() {
        SourceHealth sh = new SourceHealth();
        sh.source = "google";
        sh.consecutiveFailures = 0;
        sh.consecutiveEmpty = 1;
        sh.lastError = null;
        sh.lastErrorAt = null;
        sh.lastSuccessAt = OffsetDateTime.now();
        sh.issueOpen = false;
        sh.lastIssueUrl = null;

        when(sourceHealthRepository.listAll()).thenReturn(List.of(sh));

        List<SourceHealthSummary> result = systemResolver.getSourceHealth();

        assertEquals(1, result.size());
        assertEquals("google", result.get(0).getSource());
        assertEquals(0, result.get(0).getConsecutiveFailures());
        assertEquals(1, result.get(0).getConsecutiveEmpty());
        assertFalse(result.get(0).isIssueOpen());
    }

    @Test
    void testSourceHealth_EmptyList() {
        when(sourceHealthRepository.listAll()).thenReturn(List.of());

        List<SourceHealthSummary> result = systemResolver.getSourceHealth();

        assertTrue(result.isEmpty());
    }

    @Test
    void testTriggerCuration_DispatchesTaskAndReturnsId() {
        when(celeryTaskDispatcher.dispatchTask(anyString()))
                .thenReturn(Uni.createFrom().item("test-task-id-123"));

        String taskId = systemResolver.triggerCuration();

        assertNotNull(taskId);
        verify(celeryTaskDispatcher).dispatchTask(
                "store_listing.orchestration.tasks.curate_seeds_task");
    }

    @Test
    void testTriggerDesignBriefs_DispatchesCorrectTask() {
        when(celeryTaskDispatcher.dispatchTask(anyString()))
                .thenReturn(Uni.createFrom().item("test-task-id-456"));

        String taskId = systemResolver.triggerDesignBriefs();

        assertNotNull(taskId);
        verify(celeryTaskDispatcher).dispatchTask(
                "store_listing.orchestration.tasks.generate_design_briefs_task");
    }

    @Test
    void testTriggerImageGeneration_DispatchesCorrectTask() {
        when(celeryTaskDispatcher.dispatchTask(anyString()))
                .thenReturn(Uni.createFrom().item("test-task-id-789"));

        String taskId = systemResolver.triggerImageGeneration();

        assertNotNull(taskId);
        verify(celeryTaskDispatcher).dispatchTask(
                "store_listing.orchestration.tasks.generate_design_images_task");
    }
}
