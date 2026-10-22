package com.trends.repository;

import com.trends.domain.FabricationMaterial;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.persistence.EntityManager;
import jakarta.transaction.Transactional;

import java.time.OffsetDateTime;
import java.util.List;

@ApplicationScoped
public class FabricationMaterialRepository implements PanacheRepository<FabricationMaterial> {

    /**
     * A material can serve several product types, so the product types are
     * stored as an array and the filter has to match any of them
     * (issue #186).
     */
    @Inject
    EntityManager entityManager;

    public List<FabricationMaterial> findByProductType(String productType, int limit) {
        return entityManager
                .createNativeQuery(
                        "SELECT * FROM fabrication_materials "
                                + "WHERE active = true AND ?1 = ANY(product_type) "
                                + "ORDER BY name, color LIMIT ?2",
                        FabricationMaterial.class)
                .setParameter(1, productType)
                .setParameter(2, limit)
                .getResultList();
    }

    public FabricationMaterial findByExternalId(String externalId) {
        return find("externalId", externalId).firstResult();
    }

    public List<FabricationMaterial> findAllActive(int limit) {
        return entityManager
                .createNativeQuery(
                        "SELECT * FROM fabrication_materials "
                                + "WHERE active = true ORDER BY product_type, name, color LIMIT ?1",
                        FabricationMaterial.class)
                .setParameter(1, limit)
                .getResultList();
    }

    @Transactional
    public void deactivateByExternalId(String externalId) {
        update("active = false, updatedAt = ?1 WHERE externalId = ?2",
                OffsetDateTime.now(), externalId);
    }
}
