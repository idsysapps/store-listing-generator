package com.trends.resolver;

import com.trends.domain.FabricationMaterial;
import com.trends.dto.FabricationMaterialSummary;
import com.trends.repository.FabricationMaterialRepository;
import io.quarkus.security.identity.SecurityIdentity;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.graphql.DefaultValue;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.GraphQLException;
import org.eclipse.microprofile.graphql.Query;
import org.jboss.logging.Logger;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The materials the cost book offers.
 *
 * <p>A design brief says which substrate it should be printed on, and this is
 * the catalogue those substrates come from (issue #215).
 */
@GraphQLApi
@ApplicationScoped
public class MaterialResolver {

    private static final Logger LOG = Logger.getLogger(MaterialResolver.class);

    private static final int MAX_LIMIT = 200;

    @Inject
    FabricationMaterialRepository materialRepository;

    @Inject
    SecurityIdentity securityIdentity;

    @Query("fabricationMaterials")
    public List<FabricationMaterialSummary> getFabricationMaterials(
            String productType, @DefaultValue("50") int limit) throws GraphQLException {
        requireAuthentication();
        int effectiveLimit = Math.max(0, Math.min(limit, MAX_LIMIT));
        LOG.debugf("Fetching materials: productType=%s, limit=%d", productType, effectiveLimit);

        List<FabricationMaterial> materials;
        if (productType != null && !productType.isEmpty()) {
            materials = materialRepository.findByProductType(productType, effectiveLimit);
        } else {
            materials = materialRepository.findAllActive(effectiveLimit);
        }

        return materials.stream().map(this::toSummary).collect(Collectors.toList());
    }

    private FabricationMaterialSummary toSummary(FabricationMaterial material) {
        List<String> sizes = material.sizes != null ? Arrays.asList(material.sizes) : List.of();
        List<String> productTypes =
                material.productType != null ? Arrays.asList(material.productType) : List.of();

        return new FabricationMaterialSummary(
                material.id,
                material.externalId,
                material.name,
                material.brand,
                material.substrateType,
                material.color,
                sizes,
                material.materialComposition,
                productTypes,
                material.unitCost);
    }

    private void requireAuthentication() throws GraphQLException {
        if (securityIdentity == null || securityIdentity.isAnonymous()) {
            throw new GraphQLException("Authentication required");
        }
    }
}
