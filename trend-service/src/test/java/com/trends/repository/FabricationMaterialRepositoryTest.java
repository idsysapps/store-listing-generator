package com.trends.repository;

import com.trends.domain.FabricationMaterial;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.MockitoAnnotations;
import org.mockito.Spy;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

import io.quarkus.hibernate.orm.panache.PanacheQuery;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class FabricationMaterialRepositoryTest {

    @Spy
    @InjectMocks
    FabricationMaterialRepository repository;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testFindByProductType_ReturnsActiveMatchingMaterials() {
        FabricationMaterial mat1 = createMaterial("bc3001-white", "Bella Canvas 3001", "dtf_apparel", true);
        FabricationMaterial mat2 = createMaterial("bc3001-grey", "Bella Canvas 3001", "dtf_apparel", true);

        @SuppressWarnings("unchecked")
        PanacheQuery<FabricationMaterial> mockQuery = mock(PanacheQuery.class);
        when(mockQuery.list()).thenReturn(List.of(mat1, mat2));
        doReturn(mockQuery).when(repository).find(
                eq("productType = ?1 AND active = true ORDER BY name, color"),
                eq("dtf_apparel"));

        List<FabricationMaterial> result = repository.findByProductType("dtf_apparel");

        assertEquals(2, result.size());
        assertEquals("bc3001-white", result.get(0).externalId);
        assertEquals("bc3001-grey", result.get(1).externalId);
    }

    @Test
    void testFindByExternalId_ReturnsMatchingMaterial() {
        FabricationMaterial mat = createMaterial("bc3001-white", "Bella Canvas 3001", "dtf_apparel", true);

        @SuppressWarnings("unchecked")
        PanacheQuery<FabricationMaterial> mockQuery = mock(PanacheQuery.class);
        when(mockQuery.firstResult()).thenReturn(mat);
        doReturn(mockQuery).when(repository).find(eq("externalId"), eq("bc3001-white"));

        FabricationMaterial result = repository.findByExternalId("bc3001-white");

        assertNotNull(result);
        assertEquals("bc3001-white", result.externalId);
        assertEquals("Bella Canvas 3001", result.name);
    }

    @Test
    void testFindByExternalId_ReturnsNullWhenNotFound() {
        @SuppressWarnings("unchecked")
        PanacheQuery<FabricationMaterial> mockQuery = mock(PanacheQuery.class);
        when(mockQuery.firstResult()).thenReturn(null);
        doReturn(mockQuery).when(repository).find(eq("externalId"), eq("nonexistent"));

        FabricationMaterial result = repository.findByExternalId("nonexistent");

        assertNull(result);
    }

    @Test
    void testFindAllActive_ReturnsOnlyActiveMaterials() {
        FabricationMaterial mat1 = createMaterial("bc3001-white", "Bella Canvas 3001", "dtf_apparel", true);
        FabricationMaterial mat2 = createMaterial("mug-white", "11oz Mug", "sublimation", true);

        @SuppressWarnings("unchecked")
        PanacheQuery<FabricationMaterial> mockQuery = mock(PanacheQuery.class);
        when(mockQuery.list()).thenReturn(List.of(mat1, mat2));
        doReturn(mockQuery).when(repository).find(
                eq("active = true ORDER BY productType, name, color"));

        List<FabricationMaterial> result = repository.findAllActive();

        assertEquals(2, result.size());
    }

    private FabricationMaterial createMaterial(String externalId, String name, String productType, boolean active) {
        FabricationMaterial mat = new FabricationMaterial();
        mat.externalId = externalId;
        mat.name = name;
        mat.productType = productType;
        mat.active = active;
        mat.substrateType = "t-shirt";
        mat.color = "White";
        mat.brand = "Test Brand";
        mat.sku = "TEST-SKU";
        mat.unitCost = new BigDecimal("4.50");
        mat.updatedAt = OffsetDateTime.now();
        mat.createdAt = OffsetDateTime.now();
        return mat;
    }
}
