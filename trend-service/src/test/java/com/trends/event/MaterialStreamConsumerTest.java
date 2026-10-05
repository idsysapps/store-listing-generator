package com.trends.event;

import com.trends.domain.FabricationMaterial;
import com.trends.repository.FabricationMaterialRepository;
import io.quarkus.redis.datasource.ReactiveRedisDataSource;
import io.smallrye.mutiny.Uni;
import io.vertx.mutiny.core.Vertx;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import io.vertx.mutiny.redis.client.Response;

import java.math.BigDecimal;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class MaterialStreamConsumerTest {

    @Mock
    ReactiveRedisDataSource redis;

    @Mock
    FabricationMaterialRepository materialRepository;

    @Mock
    Vertx vertx;

    @InjectMocks
    MaterialStreamConsumer consumer;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testProcessUpsertEvent_CreatesNewMaterial() {
        when(materialRepository.findByExternalId("bc3001-white")).thenReturn(null);

        consumer.processUpsertEvent(
                "bc3001-white", "Bella Canvas 3001", "Bella Canvas",
                "t-shirt", "White", "[\"S\",\"M\",\"L\",\"XL\"]",
                "100% ring-spun cotton", "dtf_apparel",
                "BC3001-WHT", "4.50", "true", null);

        ArgumentCaptor<FabricationMaterial> captor = ArgumentCaptor.forClass(FabricationMaterial.class);
        verify(materialRepository).persist(captor.capture());

        FabricationMaterial saved = captor.getValue();
        assertEquals("bc3001-white", saved.externalId);
        assertEquals("Bella Canvas 3001", saved.name);
        assertEquals("Bella Canvas", saved.brand);
        assertEquals("t-shirt", saved.substrateType);
        assertEquals("White", saved.color);
        assertEquals("100% ring-spun cotton", saved.materialComposition);
        assertEquals("dtf_apparel", saved.productType);
        assertEquals("BC3001-WHT", saved.sku);
        assertEquals(new BigDecimal("4.50"), saved.unitCost);
        assertTrue(saved.active);
        assertNotNull(saved.sizes);
    }

    @Test
    void testProcessUpsertEvent_UpdatesExistingMaterial() {
        FabricationMaterial existing = new FabricationMaterial();
        existing.id = 1;
        existing.externalId = "bc3001-white";
        existing.name = "Old Name";
        existing.unitCost = new BigDecimal("3.00");

        when(materialRepository.findByExternalId("bc3001-white")).thenReturn(existing);

        consumer.processUpsertEvent(
                "bc3001-white", "Bella Canvas 3001", "Bella Canvas",
                "t-shirt", "White", null,
                "100% cotton", "dtf_apparel",
                "BC3001-WHT", "4.50", "true", null);

        assertEquals("Bella Canvas 3001", existing.name);
        assertEquals(new BigDecimal("4.50"), existing.unitCost);
        assertEquals("Bella Canvas", existing.brand);
        assertNotNull(existing.updatedAt);
        verify(materialRepository, never()).persist(any(FabricationMaterial.class));
    }

    @Test
    void testProcessDeactivateEvent_DeactivatesMaterial() {
        consumer.processDeactivateEvent("bc3001-white");

        verify(materialRepository).deactivateByExternalId("bc3001-white");
    }

    @Test
    void testProcessUpsertEvent_ParsesSizesJson() {
        when(materialRepository.findByExternalId("test-001")).thenReturn(null);

        consumer.processUpsertEvent(
                "test-001", "Test Shirt", null,
                "t-shirt", "Black", "[\"S\",\"M\",\"L\"]",
                null, "dtf_apparel",
                null, null, "true", null);

        ArgumentCaptor<FabricationMaterial> captor = ArgumentCaptor.forClass(FabricationMaterial.class);
        verify(materialRepository).persist(captor.capture());

        FabricationMaterial saved = captor.getValue();
        assertNotNull(saved.sizes);
        assertEquals(3, saved.sizes.length);
        assertEquals("S", saved.sizes[0]);
        assertEquals("M", saved.sizes[1]);
        assertEquals("L", saved.sizes[2]);
    }

    @Test
    void testProcessUpsertEvent_HandlesNullSizes() {
        when(materialRepository.findByExternalId("test-002")).thenReturn(null);

        consumer.processUpsertEvent(
                "test-002", "Test Mug", null,
                "mug", "White", null,
                null, "sublimation",
                null, null, "true", null);

        ArgumentCaptor<FabricationMaterial> captor = ArgumentCaptor.forClass(FabricationMaterial.class);
        verify(materialRepository).persist(captor.capture());

        assertNull(captor.getValue().sizes);
    }

    @Test
    void testProcessUpsertEvent_DefaultsActiveToTrue() {
        when(materialRepository.findByExternalId("test-003")).thenReturn(null);

        consumer.processUpsertEvent(
                "test-003", "Test Item", null,
                "t-shirt", "Red", null,
                null, "dtf_apparel",
                null, null, null, null);

        ArgumentCaptor<FabricationMaterial> captor = ArgumentCaptor.forClass(FabricationMaterial.class);
        verify(materialRepository).persist(captor.capture());

        assertTrue(captor.getValue().active);
    }
}
