package com.trends.repository;

import com.trends.domain.FabricationMaterial;
import jakarta.persistence.EntityManager;
import jakarta.persistence.Query;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
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

    @Mock
    EntityManager entityManager;

    @Spy
    @InjectMocks
    FabricationMaterialRepository repository;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testFindByProductType_MatchesAnyProductTypeOfTheMaterial() {
        // A material can serve several product types, so the column holds an
        // array and the filter has to match any of them (issue #186).
        FabricationMaterial mat = createMaterial("bc3001-white", "Bella Canvas 3001",
                new String[] { "substrate", "dtf_apparel" }, true);

        @SuppressWarnings("unchecked")
        Query mockQuery = mock(Query.class);
        when(entityManager.createNativeQuery(anyString(), eq(FabricationMaterial.class)))
                .thenReturn(mockQuery);
        when(mockQuery.setParameter(eq(1), eq("dtf_apparel"))).thenReturn(mockQuery);
        when(mockQuery.setParameter(eq(2), eq(50))).thenReturn(mockQuery);
        when(mockQuery.getResultList()).thenReturn(List.of(mat));

        List<FabricationMaterial> result = repository.findByProductType("dtf_apparel", 50);

        assertEquals(1, result.size());
        assertEquals("bc3001-white", result.get(0).externalId);

        ArgumentCaptor<String> sql = ArgumentCaptor.forClass(String.class);
        verify(entityManager).createNativeQuery(sql.capture(), eq(FabricationMaterial.class));
        assertTrue(sql.getValue().contains("= ANY(product_type)"),
                "the filter has to match any product type of the material: " + sql.getValue());
    }

    @Test
    void testFindByExternalId_ReturnsMatchingMaterial() {
        FabricationMaterial mat = createMaterial("bc3001-white", "Bella Canvas 3001",
                new String[] { "dtf_apparel" }, true);

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
        FabricationMaterial mat1 = createMaterial("bc3001-white", "Bella Canvas 3001",
                new String[] { "dtf_apparel" }, true);
        FabricationMaterial mat2 = createMaterial("mug-white", "11oz Mug",
                new String[] { "sublimation" }, true);

        @SuppressWarnings("unchecked")
        Query mockQuery = mock(Query.class);
        when(entityManager.createNativeQuery(anyString(), eq(FabricationMaterial.class)))
                .thenReturn(mockQuery);
        when(mockQuery.setParameter(eq(1), eq(50))).thenReturn(mockQuery);
        when(mockQuery.getResultList()).thenReturn(List.of(mat1, mat2));

        List<FabricationMaterial> result = repository.findAllActive(50);

        assertEquals(2, result.size());
    }

    private FabricationMaterial createMaterial(String externalId, String name, String[] productType,
            boolean active) {
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
