package com.trends.resolver;

import com.trends.domain.ActiveSeed;
import com.trends.domain.SeedProductTag;
import com.trends.dto.ActiveSeedSummary;
import com.trends.repository.ActiveSeedRepository;
import io.quarkus.security.identity.SecurityIdentity;
import org.eclipse.microprofile.graphql.GraphQLException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ActiveSeedResolverTest {

    @Mock
    ActiveSeedRepository activeSeedRepository;

    @Mock
    SecurityIdentity securityIdentity;

    @InjectMocks
    ActiveSeedResolver activeSeedResolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        when(securityIdentity.isAnonymous()).thenReturn(false);
    }

    @Test
    void testGetActiveSeeds_ReturnsAllActive() throws GraphQLException {
        ActiveSeed seed1 = new ActiveSeed();
        seed1.id = 1;
        seed1.query = "dad jokes shirt";
        seed1.promotionScore = 5000L;

        ActiveSeed seed2 = new ActiveSeed();
        seed2.id = 2;
        seed2.query = "halloween mug";
        seed2.promotionScore = 3000L;

        when(activeSeedRepository.findActiveSeeds(10)).thenReturn(List.of(seed1, seed2));

        List<ActiveSeedSummary> result = activeSeedResolver.getActiveSeeds(null, 10);

        assertEquals(2, result.size());
        assertEquals("dad jokes shirt", result.get(0).getQuery());
        assertEquals(5000L, result.get(0).getPromotionScore());
    }

    @Test
    void testGetActiveSeeds_FiltersByTag() throws GraphQLException {
        ActiveSeed seed1 = new ActiveSeed();
        seed1.id = 1;
        seed1.query = "dad jokes shirt";
        seed1.promotionScore = 5000L;

        when(activeSeedRepository.findActiveSeedsByTag("dtf_apparel", 10)).thenReturn(List.of(seed1));

        List<ActiveSeedSummary> result = activeSeedResolver.getActiveSeeds("dtf_apparel", 10);

        assertEquals(1, result.size());
        assertEquals("dad jokes shirt", result.get(0).getQuery());
        verify(activeSeedRepository).findActiveSeedsByTag("dtf_apparel", 10);
        verify(activeSeedRepository, never()).findActiveSeeds(anyInt());
    }

    @Test
    void testGetActiveSeeds_IncludesProductTags() throws GraphQLException {
        ActiveSeed seed = new ActiveSeed();
        seed.id = 1;
        seed.query = "dad jokes shirt";
        seed.promotionScore = 5000L;

        SeedProductTag tag = new SeedProductTag();
        tag.tag = "dtf_apparel";
        seed.productTags = List.of(tag);

        when(activeSeedRepository.findActiveSeeds(10)).thenReturn(List.of(seed));

        List<ActiveSeedSummary> result = activeSeedResolver.getActiveSeeds(null, 10);

        assertEquals(1, result.size());
        assertEquals(List.of("dtf_apparel"), result.get(0).getProductTags());
    }

    @Test
    void testGetActiveSeeds_EmptyTagsWhenNone() throws GraphQLException {
        ActiveSeed seed = new ActiveSeed();
        seed.id = 1;
        seed.query = "hoodie season";
        seed.promotionScore = 0L;
        seed.productTags = List.of();

        when(activeSeedRepository.findActiveSeeds(10)).thenReturn(List.of(seed));

        List<ActiveSeedSummary> result = activeSeedResolver.getActiveSeeds(null, 10);

        assertEquals(1, result.size());
        assertTrue(result.get(0).getProductTags().isEmpty());
    }

    @Test
    void testGetProductTags_ReturnsDistinctTags() throws GraphQLException {
        when(activeSeedRepository.findDistinctTags()).thenReturn(List.of("dtf_apparel", "sublimation", "sticker_vinyl"));

        List<String> result = activeSeedResolver.getProductTags();

        assertEquals(3, result.size());
        assertTrue(result.contains("dtf_apparel"));
        assertTrue(result.contains("sublimation"));
        assertTrue(result.contains("sticker_vinyl"));
    }

    @Test
    void testGetActiveSeeds_ThrowsAuthenticationRequired_WhenAnonymous() {
        when(securityIdentity.isAnonymous()).thenReturn(true);

        GraphQLException ex = assertThrows(GraphQLException.class,
                () -> activeSeedResolver.getActiveSeeds(null, 10));
        assertEquals("Authentication required", ex.getMessage());
    }
}
