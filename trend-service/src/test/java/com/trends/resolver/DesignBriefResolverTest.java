package com.trends.resolver;

import com.trends.domain.ActiveSeed;
import com.trends.domain.DesignBrief;
import com.trends.domain.DesignBriefSource;
import com.trends.dto.DesignBriefSummary;
import com.trends.repository.DesignBriefRepository;
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

    @InjectMocks
    DesignBriefResolver designBriefResolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
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

    @Test
    void testGetDesignBriefs_ReturnsRecent() {
        DesignBrief brief = makeBrief(1, "Dad Jokes Shirt", "dtf_apparel");
        when(designBriefRepository.findRecent(10)).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs(null, null, 10);

        assertEquals(1, result.size());
        assertEquals("Dad Jokes Shirt", result.get(0).getConcept());
        assertEquals("dtf_apparel", result.get(0).getProductType());
        assertEquals(85, result.get(0).getConfidence());
        assertEquals(List.of("t-shirt", "hoodie"), result.get(0).getSpecificProducts());
    }

    @Test
    void testGetDesignBriefs_FiltersByProductType() {
        DesignBrief brief = makeBrief(1, "Coffee Mug Design", "sublimation");
        when(designBriefRepository.findByProductType("sublimation", 10)).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs("sublimation", null, 10);

        assertEquals(1, result.size());
        assertEquals("sublimation", result.get(0).getProductType());
        verify(designBriefRepository).findByProductType("sublimation", 10);
        verify(designBriefRepository, never()).findRecent(anyInt());
    }

    @Test
    void testGetDesignBriefs_FiltersByBatchId() {
        DesignBrief brief = makeBrief(1, "Test Brief", "dtf_apparel");
        when(designBriefRepository.findByBatchId("2026-10-01")).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs(null, "2026-10-01", 10);

        assertEquals(1, result.size());
        verify(designBriefRepository).findByBatchId("2026-10-01");
        verify(designBriefRepository, never()).findRecent(anyInt());
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

        when(designBriefRepository.findRecent(10)).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs(null, null, 10);

        assertEquals(1, result.size());
        assertEquals(List.of("dad jokes", "funny t-shirt"), result.get(0).getSourceSeeds());
    }

    @Test
    void testGetDesignBriefs_EmptySourcesWhenNone() {
        DesignBrief brief = makeBrief(1, "Solo Brief", "sticker_vinyl");
        brief.sources = List.of();

        when(designBriefRepository.findRecent(10)).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs(null, null, 10);

        assertEquals(1, result.size());
        assertTrue(result.get(0).getSourceSeeds().isEmpty());
    }

    @Test
    void testGetDesignBriefs_NullSpecificProducts() {
        DesignBrief brief = makeBrief(1, "Null Products", "dtf_apparel");
        brief.specificProducts = null;

        when(designBriefRepository.findRecent(10)).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs(null, null, 10);

        assertEquals(1, result.size());
        assertTrue(result.get(0).getSpecificProducts().isEmpty());
    }

    @Test
    void testGetDesignBriefs_IncludesImageKeys() {
        DesignBrief brief = makeBrief(1, "Skeleton Yoga", "dtf_apparel");
        brief.imageKeyRaw = "designs/2026/09/dtf_apparel/1_raw.png";
        brief.imageKeyTransparent = "designs/2026/09/dtf_apparel/1_transparent.png";

        when(designBriefRepository.findRecent(10)).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs(null, null, 10);

        assertEquals(1, result.size());
        assertEquals("designs/2026/09/dtf_apparel/1_raw.png", result.get(0).getImageKeyRaw());
        assertEquals("designs/2026/09/dtf_apparel/1_transparent.png", result.get(0).getImageKeyTransparent());
    }

    @Test
    void testGetDesignBriefs_NullImageKeysWhenNotGenerated() {
        DesignBrief brief = makeBrief(1, "No Image Brief", "sublimation");
        brief.imageKeyRaw = null;
        brief.imageKeyTransparent = null;

        when(designBriefRepository.findRecent(10)).thenReturn(List.of(brief));

        List<DesignBriefSummary> result = designBriefResolver.getDesignBriefs(null, null, 10);

        assertEquals(1, result.size());
        assertNull(result.get(0).getImageKeyRaw());
        assertNull(result.get(0).getImageKeyTransparent());
    }
}
