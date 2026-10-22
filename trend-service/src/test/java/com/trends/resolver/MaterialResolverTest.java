package com.trends.resolver;

import com.trends.domain.FabricationMaterial;
import com.trends.dto.FabricationMaterialSummary;
import com.trends.repository.FabricationMaterialRepository;
import io.quarkus.security.identity.SecurityIdentity;
import org.eclipse.microprofile.graphql.GraphQLException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.math.BigDecimal;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MaterialResolverTest {

    @Mock
    FabricationMaterialRepository materialRepository;

    @Mock
    SecurityIdentity securityIdentity;

    @InjectMocks
    MaterialResolver resolver;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    private FabricationMaterial material() {
        FabricationMaterial mat = new FabricationMaterial();
        mat.id = 7;
        mat.externalId = "bc3001-white";
        mat.name = "Bella Canvas 3001";
        mat.brand = "Bella Canvas";
        mat.substrateType = "t-shirt";
        mat.color = "White";
        mat.sizes = new String[] { "S", "M", "L" };
        mat.materialComposition = "100% cotton";
        mat.productType = new String[] { "substrate", "dtf_apparel" };
        mat.unitCost = new BigDecimal("4.50");
        mat.active = true;
        return mat;
    }

    @Test
    void testGetFabricationMaterials_ReturnsTheMaterialsOfAProductType() throws GraphQLException {
        when(securityIdentity.isAnonymous()).thenReturn(false);
        when(materialRepository.findByProductType("dtf_apparel", 50))
                .thenReturn(List.of(material()));

        List<FabricationMaterialSummary> result =
                resolver.getFabricationMaterials("dtf_apparel", 50);

        assertEquals(1, result.size());
        FabricationMaterialSummary summary = result.get(0);
        assertEquals("bc3001-white", summary.getExternalId());
        assertEquals("Bella Canvas 3001", summary.getName());
        assertEquals(List.of("S", "M", "L"), summary.getSizes());
        assertEquals(List.of("substrate", "dtf_apparel"), summary.getProductTypes());
        assertEquals(new BigDecimal("4.50"), summary.getUnitCost());
    }

    @Test
    void testGetFabricationMaterials_WithoutAProductTypeReturnsEveryActiveMaterial()
            throws GraphQLException {
        when(securityIdentity.isAnonymous()).thenReturn(false);
        when(materialRepository.findAllActive(50)).thenReturn(List.of(material()));

        assertEquals(1, resolver.getFabricationMaterials(null, 50).size());

        verify(materialRepository).findAllActive(50);
    }

    @Test
    void testGetFabricationMaterials_RequiresAuthentication() {
        when(securityIdentity.isAnonymous()).thenReturn(true);

        assertThrows(Exception.class, () -> resolver.getFabricationMaterials("dtf_apparel", 50));
    }
}
