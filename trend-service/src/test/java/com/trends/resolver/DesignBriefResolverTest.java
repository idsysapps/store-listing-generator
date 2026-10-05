package com.trends.resolver;

import com.trends.config.ImageStorageConfig;
import com.trends.domain.ActiveSeed;
import com.trends.domain.DesignBrief;
import com.trends.domain.DesignBriefSource;
import com.trends.dto.DesignBriefConnection;
import com.trends.dto.DesignBriefSummary;
import com.trends.event.CeleryTaskDispatcher;
import com.trends.repository.DesignBriefRepository;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class DesignBriefResolverTest {

    @Mock
    DesignBriefRepository designBriefRepository;

    @Mock
    ImageStorageConfig imageStorageConfig;

    @Mock
    CeleryTaskDispatcher celeryTaskDispatcher;

    @InjectMocks
    DesignBriefResolver designBriefResolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        when(celeryTaskDispatcher.dispatchGenerateSingleBriefImage(anyInt()))
                .thenReturn(Uni.createFrom().voidItem());
    }

    private DesignBrief makeBrief(int id, String concept, String productType) {
        DesignBrief brief = new DesignBrief();
        brief.id = id;
        brief.concept = concept;
        brief.productType = productType;
        brief.specificProducts = new String[]{"t-shirt", "hoodie"};
        brief.audience = "men 30-50";
        brief.visualStyle = "bold serif text";
        brief.confidence = 85;
        brief.reasoning = "buyer intent + trending";
        brief.llmModel = "llama-3.3-70b-versatile";
        brief.batchId = "2026-10-01";
        brief.createdAt = OffsetDateTime.now();
        brief.sources = List.of();
        return brief;
    }

    private DesignBriefConnection queryBriefs(String productType, String batchId, int limit) {
        return designBriefResolver.getDesignBriefs(
                productType, batchId, null, null, null, null, null, null, null, limit, 0);
    }

    @Test
    void testGetDesignBriefs_ReturnsRecent() {
        DesignBrief brief = makeBrief(1, "Dad Jokes Shirt", "dtf_apparel");
        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);

        DesignBriefConnection result = queryBriefs(null, null, 10);

        assertEquals(1, result.getItems().size());
        assertEquals("Dad Jokes Shirt", result.getItems().get(0).getConcept());
        assertEquals("dtf_apparel", result.getItems().get(0).getProductType());
        assertEquals(85, result.getItems().get(0).getConfidence());
        assertEquals(List.of("t-shirt", "hoodie"), result.getItems().get(0).getSpecificProducts());
        assertEquals(1, result.getPageInfo().getTotalCount());
        assertFalse(result.getPageInfo().getHasNextPage());
        assertFalse(result.getPageInfo().getHasPreviousPage());
    }

    @Test
    void testGetDesignBriefs_FiltersByProductType() {
        DesignBrief brief = makeBrief(1, "Coffee Mug Design", "sublimation");
        when(designBriefRepository.findWithFilters(
                "sublimation", null, null, null, null, null, null, null, null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                "sublimation", null, null, null, null, null, null, null))
                .thenReturn(1L);

        DesignBriefConnection result = queryBriefs("sublimation", null, 10);

        assertEquals(1, result.getItems().size());
        assertEquals("sublimation", result.getItems().get(0).getProductType());
    }

    @Test
    void testGetDesignBriefs_FiltersByBatchId() {
        DesignBrief brief = makeBrief(1, "Test Brief", "dtf_apparel");
        when(designBriefRepository.findWithFilters(
                null, "2026-10-01", null, null, null, null, null, null, null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                null, "2026-10-01", null, null, null, null, null, null))
                .thenReturn(1L);

        DesignBriefConnection result = queryBriefs(null, "2026-10-01", 10);

        assertEquals(1, result.getItems().size());
    }

    @Test
    void testGetDesignBriefs_IncludesSourceSeeds() {
        DesignBrief brief = makeBrief(1, "Dad Jokes Shirt", "dtf_apparel");

        ActiveSeed seed1 = new ActiveSeed();
        seed1.id = 10;
        seed1.query = "dad jokes";

        ActiveSeed seed2 = new ActiveSeed();
        seed2.id = 20;
        seed2.query = "funny t-shirt";

        DesignBriefSource src1 = new DesignBriefSource();
        src1.activeSeed = seed1;
        DesignBriefSource src2 = new DesignBriefSource();
        src2.activeSeed = seed2;

        brief.sources = List.of(src1, src2);

        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);

        DesignBriefConnection result = queryBriefs(null, null, 10);

        assertEquals(1, result.getItems().size());
        assertEquals(List.of("dad jokes", "funny t-shirt"), result.getItems().get(0).getSourceSeeds());
    }

    @Test
    void testGetDesignBriefs_EmptySourcesWhenNone() {
        DesignBrief brief = makeBrief(1, "Solo Brief", "sticker_vinyl");
        brief.sources = List.of();

        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);

        DesignBriefConnection result = queryBriefs(null, null, 10);

        assertEquals(1, result.getItems().size());
        assertTrue(result.getItems().get(0).getSourceSeeds().isEmpty());
    }

    @Test
    void testGetDesignBriefs_NullSpecificProducts() {
        DesignBrief brief = makeBrief(1, "Null Products", "dtf_apparel");
        brief.specificProducts = null;

        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);

        DesignBriefConnection result = queryBriefs(null, null, 10);

        assertEquals(1, result.getItems().size());
        assertTrue(result.getItems().get(0).getSpecificProducts().isEmpty());
    }

    @Test
    void testGetDesignBriefs_IncludesImageKeysAndUrls() {
        DesignBrief brief = makeBrief(1, "Skeleton Yoga", "dtf_apparel");
        brief.imageKeyRaw = "designs/2026/09/dtf_apparel/1_raw.png";
        brief.imageKeyTransparent = "designs/2026/09/dtf_apparel/1_transparent.png";

        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);
        when(imageStorageConfig.buildUrl("designs/2026/09/dtf_apparel/1_raw.png"))
                .thenReturn("https://minio.local:9000/designs/designs/2026/09/dtf_apparel/1_raw.png");
        when(imageStorageConfig.buildUrl("designs/2026/09/dtf_apparel/1_transparent.png"))
                .thenReturn("https://minio.local:9000/designs/designs/2026/09/dtf_apparel/1_transparent.png");

        DesignBriefConnection result = queryBriefs(null, null, 10);

        assertEquals(1, result.getItems().size());
        assertEquals("designs/2026/09/dtf_apparel/1_raw.png", result.getItems().get(0).getImageKeyRaw());
        assertEquals("designs/2026/09/dtf_apparel/1_transparent.png", result.getItems().get(0).getImageKeyTransparent());
        assertNotNull(result.getItems().get(0).getImageUrl());
        assertNotNull(result.getItems().get(0).getImageTransparentUrl());
    }

    @Test
    void testGetDesignBriefs_NullImageKeysWhenNotGenerated() {
        DesignBrief brief = makeBrief(1, "No Image Brief", "sublimation");
        brief.imageKeyRaw = null;
        brief.imageKeyTransparent = null;

        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);

        DesignBriefConnection result = queryBriefs(null, null, 10);

        assertEquals(1, result.getItems().size());
        assertNull(result.getItems().get(0).getImageKeyRaw());
        assertNull(result.getItems().get(0).getImageKeyTransparent());
    }

    @Test
    void testGetDesignBriefs_PaginationHasNextPage() {
        DesignBrief brief = makeBrief(1, "Page 1 Brief", "dtf_apparel");
        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(25L);

        DesignBriefConnection result = queryBriefs(null, null, 10);

        assertEquals(1, result.getItems().size());
        assertTrue(result.getPageInfo().getHasNextPage());
        assertFalse(result.getPageInfo().getHasPreviousPage());
        assertEquals(25, result.getPageInfo().getTotalCount());
    }

    @Test
    void testGetDesignBriefs_PaginationWithOffset() {
        DesignBrief brief = makeBrief(2, "Page 2 Brief", "dtf_apparel");
        when(designBriefRepository.findRecentPaginated(10, 10)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(25L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, null, null, null, null, null, null, null, 10, 10);

        assertEquals(1, result.getItems().size());
        assertTrue(result.getPageInfo().getHasNextPage());
        assertTrue(result.getPageInfo().getHasPreviousPage());
        assertEquals(25, result.getPageInfo().getTotalCount());
    }

    @Test
    void testGetDesignBriefs_LimitZeroReturnsCountOnly() {
        when(designBriefRepository.countWithFilters(
                null, null, null, null, null, null, null, null))
                .thenReturn(42L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, null, null, null, null, null, null, null, 0, 0);

        assertTrue(result.getItems().isEmpty());
        assertEquals(42, result.getPageInfo().getTotalCount());
        verify(designBriefRepository, never()).findRecentPaginated(anyInt(), anyInt());
    }

    @Test
    void testGetDesignBriefs_LimitCappedAt100() {
        when(designBriefRepository.findRecentPaginated(100, 0)).thenReturn(List.of());
        when(designBriefRepository.countAll()).thenReturn(0L);

        designBriefResolver.getDesignBriefs(
                null, null, null, null, null, null, null, null, null, 500, 0);

        verify(designBriefRepository).findRecentPaginated(100, 0);
    }

    @Test
    void testRequestRegeneration_ClearsImageKeysAndStoresFeedback() {
        DesignBrief brief = makeBrief(1, "Skeleton Yoga", "dtf_apparel");
        brief.imageKeyRaw = "designs/2026/09/dtf_apparel/1_raw.png";
        brief.imageKeyTransparent = "designs/2026/09/dtf_apparel/1_transparent.png";

        when(designBriefRepository.findById(1L)).thenReturn(brief);

        DesignBriefSummary result = designBriefResolver.requestRegeneration(
                1, "Make the skeleton more cartoonish, use brighter colors");

        assertNotNull(result);
        assertNull(brief.imageKeyRaw);
        assertNull(brief.imageKeyTransparent);
        assertEquals("Make the skeleton more cartoonish, use brighter colors",
                brief.regenerationFeedback);
        verify(designBriefRepository).persist(brief);
    }

    @Test
    void testRequestRegeneration_ReturnsNullForMissingBrief() {
        when(designBriefRepository.findById(999L)).thenReturn(null);

        DesignBriefSummary result = designBriefResolver.requestRegeneration(
                999, "some feedback");

        assertNull(result);
    }

    @Test
    void testRequestRegeneration_DispatchesCeleryTask() {
        DesignBrief brief = makeBrief(1, "Skeleton Yoga", "dtf_apparel");
        brief.imageKeyRaw = "designs/2026/09/dtf_apparel/1_raw.png";
        brief.imageKeyTransparent = "designs/2026/09/dtf_apparel/1_transparent.png";

        when(designBriefRepository.findById(1L)).thenReturn(brief);

        designBriefResolver.requestRegeneration(1, "brighter colors");

        verify(celeryTaskDispatcher).dispatchGenerateSingleBriefImage(1);
    }

    @Test
    void testRequestRegeneration_DoesNotDispatchForMissingBrief() {
        when(designBriefRepository.findById(999L)).thenReturn(null);

        designBriefResolver.requestRegeneration(999, "some feedback");

        verify(celeryTaskDispatcher, never()).dispatchGenerateSingleBriefImage(anyInt());
    }

    @Test
    void testRequestRegeneration_WorksWithNullFeedback() {
        DesignBrief brief = makeBrief(1, "Skeleton Yoga", "dtf_apparel");
        brief.imageKeyRaw = "designs/2026/09/dtf_apparel/1_raw.png";

        when(designBriefRepository.findById(1L)).thenReturn(brief);

        DesignBriefSummary result = designBriefResolver.requestRegeneration(1, null);

        assertNotNull(result);
        assertNull(brief.imageKeyRaw);
        assertNull(brief.regenerationFeedback);
    }

    // --- Advanced filter tests ---

    @Test
    void testGetDesignBriefs_FiltersByDateRange() {
        DesignBrief brief = makeBrief(1, "Holiday Shirt", "dtf_apparel");
        when(designBriefRepository.findWithFilters(
                null, null, "2026-09-01", "2026-09-30",
                null, null, null, null, null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                null, null, "2026-09-01", "2026-09-30",
                null, null, null, null))
                .thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, "2026-09-01", "2026-09-30",
                null, null, null, null, null, 10, 0);

        assertEquals(1, result.getItems().size());
    }

    @Test
    void testGetDesignBriefs_FiltersByMinConfidence() {
        DesignBrief brief = makeBrief(1, "High Confidence", "dtf_apparel");
        brief.confidence = 90;
        when(designBriefRepository.findWithFilters(
                null, null, null, null, 80, null, null, null, null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                null, null, null, null, 80, null, null, null))
                .thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, null, null, 80, null, null, null, null, 10, 0);

        assertEquals(1, result.getItems().size());
        assertEquals(90, result.getItems().get(0).getConfidence());
    }

    @Test
    void testGetDesignBriefs_FiltersByConfidenceRange() {
        DesignBrief brief = makeBrief(1, "Mid Confidence", "dtf_apparel");
        brief.confidence = 75;
        when(designBriefRepository.findWithFilters(
                null, null, null, null, 70, 80, null, null, null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                null, null, null, null, 70, 80, null, null))
                .thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, null, null, 70, 80, null, null, null, 10, 0);

        assertEquals(1, result.getItems().size());
    }

    @Test
    void testGetDesignBriefs_FiltersBySourceSeeds() {
        DesignBrief brief = makeBrief(1, "Dad Jokes Shirt", "dtf_apparel");
        when(designBriefRepository.findWithFilters(
                null, null, null, null, null, null,
                List.of("dad jokes", "funny shirts"), null, null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                null, null, null, null, null, null,
                List.of("dad jokes", "funny shirts"), null))
                .thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, null, null, null, null,
                List.of("dad jokes", "funny shirts"), null, null, 10, 0);

        assertEquals(1, result.getItems().size());
    }

    @Test
    void testGetDesignBriefs_FiltersBySpecificProducts() {
        DesignBrief brief = makeBrief(1, "Mug Design", "sublimation");
        when(designBriefRepository.findWithFilters(
                null, null, null, null, null, null, null,
                List.of("mug", "tumbler"), null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                null, null, null, null, null, null, null, null))
                .thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, null, null, null, null, null,
                List.of("mug", "tumbler"), null, 10, 0);

        assertEquals(1, result.getItems().size());
    }

    @Test
    void testGetDesignBriefs_FiltersByAudienceContains() {
        DesignBrief brief = makeBrief(1, "Yoga Design", "dtf_apparel");
        brief.audience = "yoga enthusiasts who love dark humor";
        when(designBriefRepository.findWithFilters(
                null, null, null, null, null, null, null, null, "yoga", 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                null, null, null, null, null, null, null, "yoga"))
                .thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                null, null, null, null, null, null, null, null, "yoga", 10, 0);

        assertEquals(1, result.getItems().size());
    }

    @Test
    void testGetDesignBriefs_CombinedFilters() {
        DesignBrief brief = makeBrief(1, "Premium DTF", "dtf_apparel");
        brief.confidence = 90;
        when(designBriefRepository.findWithFilters(
                "dtf_apparel", null, "2026-09-01", null, 80, null, null, null, null, 10, 0))
                .thenReturn(List.of(brief));
        when(designBriefRepository.countWithFilters(
                "dtf_apparel", null, "2026-09-01", null, 80, null, null, null))
                .thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                "dtf_apparel", null, "2026-09-01", null, 80, null, null, null, null, 10, 0);

        assertEquals(1, result.getItems().size());
        assertEquals("dtf_apparel", result.getItems().get(0).getProductType());
    }

    @Test
    void testGetDesignBriefs_EmptyStringsTreatedAsNull() {
        DesignBrief brief = makeBrief(1, "Test Brief", "dtf_apparel");
        when(designBriefRepository.findRecentPaginated(10, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                "", "", "", "", null, null, null, null, "", 10, 0);

        assertEquals(1, result.getItems().size());
        verify(designBriefRepository).findRecentPaginated(10, 0);
    }

    @Test
    void testGetDesignBriefs_EmptyStringListsTreatedAsNull() {
        DesignBrief brief = makeBrief(1, "Test Brief", "dtf_apparel");
        when(designBriefRepository.findRecentPaginated(20, 0)).thenReturn(List.of(brief));
        when(designBriefRepository.countAll()).thenReturn(1L);

        DesignBriefConnection result = designBriefResolver.getDesignBriefs(
                "", "", "", "", null, null,
                List.of(""), List.of(""), "", 20, 0);

        assertEquals(1, result.getItems().size());
        verify(designBriefRepository).findRecentPaginated(20, 0);
    }
}
