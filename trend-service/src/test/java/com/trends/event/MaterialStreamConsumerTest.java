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
import static org.mockito.AdditionalMatchers.aryEq;
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
                "100% ring-spun cotton", "[\"substrate\",\"dtf_apparel\"]",
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
        assertArrayEquals(new String[] { "substrate", "dtf_apparel" }, saved.productType);
        assertEquals("BC3001-WHT", saved.sku);
        assertEquals(new BigDecimal("4.50"), saved.unitCost);
        assertTrue(saved.active);
        assertNotNull(saved.sizes);
    }

    @Test
    void testProcessUpsertEvent_StoresProductTypesAsAnArray() {
        // Cost-book publishes the product types a material can serve as a
        // JSON array, so the column holds the array and not one string
        // (issue #186).
        when(materialRepository.findByExternalId("bc3001-white")).thenReturn(null);

        consumer.processUpsertEvent(
                "bc3001-white", "Bella Canvas 3001", "Bella Canvas",
                "t-shirt", "White", null,
                "100% cotton", "[\"substrate\",\"dtf_apparel\"]",
                "BC3001-WHT", "4.50", "true", null);

        ArgumentCaptor<FabricationMaterial> captor = ArgumentCaptor.forClass(FabricationMaterial.class);
        verify(materialRepository).persist(captor.capture());

        assertArrayEquals(new String[] { "substrate", "dtf_apparel" }, captor.getValue().productType);
    }

    @Test
    void testProcessUpsertEvent_StoresASingleProductTypeAsAnArray() {
        when(materialRepository.findByExternalId("bc3001-white")).thenReturn(null);

        consumer.processUpsertEvent(
                "bc3001-white", "Bella Canvas 3001", "Bella Canvas",
                "t-shirt", "White", null,
                "100% cotton", "dtf_apparel",
                "BC3001-WHT", "4.50", "true", null);

        ArgumentCaptor<FabricationMaterial> captor = ArgumentCaptor.forClass(FabricationMaterial.class);
        verify(materialRepository).persist(captor.capture());

        assertArrayEquals(new String[] { "dtf_apparel" }, captor.getValue().productType);
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
                "100% cotton", "[\"substrate\",\"dtf_apparel\"]",
                "BC3001-WHT", "4.50", "true", null);

        assertEquals("Bella Canvas 3001", existing.name);
        assertEquals(new BigDecimal("4.50"), existing.unitCost);
        assertEquals("Bella Canvas", existing.brand);
        assertArrayEquals(new String[] { "substrate", "dtf_apparel" }, existing.productType);
        assertNotNull(existing.updatedAt);
        verify(materialRepository, never()).persist(any(FabricationMaterial.class));
    }

    @Test
    void testProcessUpsertEvent_CreatesMaterialWithoutOptionalFields() {
        // Cost-book publishes a field only when the material actually carries it, so a
        // catalog entry without a product type arrives without substrate_type and
        // product_type. Those columns have to accept a null.
        when(materialRepository.findByExternalId("c9924c1b-true-celadon")).thenReturn(null);

        consumer.processUpsertEvent(
                "c9924c1b-true-celadon", "Core Adult", "Port & Company",
                null, "True Celadon", null,
                null, null,
                "PC54-TRUE-CELADON-L", "2.39", "true", null);

        ArgumentCaptor<FabricationMaterial> captor = ArgumentCaptor.forClass(FabricationMaterial.class);
        verify(materialRepository).persist(captor.capture());

        FabricationMaterial saved = captor.getValue();
        assertNull(saved.substrateType, "substrate_type is optional in the event contract");
        assertNull(saved.productType, "product_type is optional in the event contract");
        assertNull(saved.sizes, "sizes are optional in the event contract");
        assertEquals("Core Adult", saved.name);
        assertEquals("Port & Company", saved.brand);
        assertEquals("True Celadon", saved.color);
        assertEquals("PC54-TRUE-CELADON-L", saved.sku);
        assertEquals(new BigDecimal("2.39"), saved.unitCost);
        assertTrue(saved.active);
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

    @Test
    void testClaimPendingMessages_ClaimsTheEntriesThatWereNeverAcknowledged() {
        // A message that failed to process stays pending, and a plain read
        // never delivers a pending entry again, so the material was lost for
        // good (issue #182).
        when(redis.execute(eq("XAUTOCLAIM"), any(String[].class)))
                .thenReturn(Uni.createFrom().nullItem());

        consumer.claimPendingMessages();

        verify(redis).execute(eq("XAUTOCLAIM"), eq("materials:events"), eq("trend-service"),
                eq("consumer-1"), anyString(), eq("0-0"), eq("COUNT"), eq("10"));
    }

    @Test
    void testProcessClaimedResponse_RetriesTheClaimedMessage() {
        Response fields = mock(Response.class);
        Response id = mock(Response.class);
        when(id.toString()).thenReturn("1700000000000-0");
        Response deliveries = mock(Response.class);
        when(deliveries.toLong()).thenReturn(2L);

        Response entry = mock(Response.class);
        when(entry.get(0)).thenReturn(id);
        when(entry.get(1)).thenReturn(fields);
        when(entry.get(2)).thenReturn(deliveries);
        when(entry.size()).thenReturn(3);

        Response entries = mock(Response.class);
        when(entries.size()).thenReturn(1);
        when(entries.get(0)).thenReturn(entry);

        Response claimed = mock(Response.class);
        when(claimed.size()).thenReturn(3);
        when(claimed.get(1)).thenReturn(entries);

        consumer.processClaimedResponse(claimed);

        verify(vertx).executeBlocking(any(java.util.concurrent.Callable.class));
    }

    @Test
    void testProcessClaimedResponse_AMessageThatKeepsFailingIsMovedAside() {
        // A poison message must not stay pending forever: it is moved to the
        // dead letter stream and acknowledged (issue #182).
        Response fields = mock(Response.class);
        when(fields.size()).thenReturn(0);
        Response id = mock(Response.class);
        when(id.toString()).thenReturn("1700000000000-0");
        Response deliveries = mock(Response.class);
        when(deliveries.toLong()).thenReturn(9L);

        Response entry = mock(Response.class);
        when(entry.get(0)).thenReturn(id);
        when(entry.get(1)).thenReturn(fields);
        when(entry.get(2)).thenReturn(deliveries);
        when(entry.size()).thenReturn(3);

        Response entries = mock(Response.class);
        when(entries.size()).thenReturn(1);
        when(entries.get(0)).thenReturn(entry);

        Response claimed = mock(Response.class);
        when(claimed.size()).thenReturn(3);
        when(claimed.get(1)).thenReturn(entries);

        when(redis.execute(eq("XADD"), any(String[].class)))
                .thenReturn(Uni.createFrom().nullItem());
        when(redis.execute(eq("XACK"), any(String[].class)))
                .thenReturn(Uni.createFrom().nullItem());

        consumer.processClaimedResponse(claimed);

        verify(redis).execute(eq("XADD"), aryEq(new String[] {
                "materials:events:dead", "*", "failed_deliveries", "9",
                "failed_message_id", "1700000000000-0" }));
        verify(redis).execute(eq("XACK"), eq("materials:events"), eq("trend-service"),
                eq("1700000000000-0"));
    }

    @Test
    void testLogPendingBacklog_ReadsThePendingCount() {
        when(redis.execute(eq("XPENDING"), any(String[].class)))
                .thenReturn(Uni.createFrom().nullItem());

        consumer.logPendingBacklog();

        verify(redis).execute(eq("XPENDING"), eq("materials:events"), eq("trend-service"));
    }
}
